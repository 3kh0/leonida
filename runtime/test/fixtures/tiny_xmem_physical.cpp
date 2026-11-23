#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <iterator>

extern "C" {
__declspec(dllimport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
__declspec(dllimport) BOOL WINAPI XMemAllocatePhysicalPages(uint64_t, uint64_t*, uint64_t*);
__declspec(dllimport) BOOL WINAPI XMemFreePhysicalPages(uint64_t, const uint64_t*);
__declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void*, uint64_t, const uint64_t*);
__declspec(dllimport) BOOL WINAPI XMemMapPhysicalPagesScatter(void* const*, uint32_t, const uint64_t*);
}

namespace {
constexpr SIZE_T page64k = 64 * 1024;
constexpr DWORD physical_reserve = 0x20402000;
constexpr DWORD placeholder_reserve = 0x42000;
constexpr DWORD placeholder_carve = 0x20406000;
constexpr uint64_t window_pages = 24;
constexpr uint64_t carve_pages = 16;
constexpr SIZE_T arena_bytes = SIZE_T{185} * 1024 * 1024;

struct Carve {
    SIZE_T offset;
    uint32_t tag;
};

constexpr Carve carves[] = {
    {0, 0x51000000},
    {carve_pages * page64k, 0x52000000},
    {SIZE_T{16} * 1024 * 1024, 0x53000000},
    {SIZE_T{64} * 1024 * 1024, 0x54000000},
    {arena_bytes - carve_pages * page64k, 0x55000000},
};

bool fail(const char* what) {
    std::printf("tiny_xmem_physical: FAIL %s (Win32 error %lu)\n", what, GetLastError());
    return false;
}

unsigned char* page_at(void* base, uint64_t index) { return static_cast<unsigned char*>(base) + index * page64k; }

void stamp(void* page, uint32_t tag) {
    auto* words = static_cast<volatile uint32_t*>(page);
    words[0] = tag;
    words[page64k / sizeof(uint32_t) - 1] = ~tag;
}

bool stamped(void* page, uint32_t tag) {
    auto* words = static_cast<volatile uint32_t*>(page);
    return words[0] == tag && words[page64k / sizeof(uint32_t) - 1] == ~tag;
}

bool distinct_and_nonzero(const uint64_t* pages, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        if (pages[i] == 0) return false;
        for (uint64_t j = 0; j < i; ++j) {
            if (pages[j] == pages[i]) return false;
        }
    }
    return true;
}

bool reserved_without_memory(void* address) {
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) == sizeof(info) && info.State == MEM_RESERVE;
}

bool window_sequence() {
    uint64_t pages[window_pages]{};
    uint64_t count = window_pages;
    if (!XMemAllocatePhysicalPages(0x280, &count, pages)) return fail("XMemAllocatePhysicalPages(0x280)");
    if (count != window_pages) return fail("page count written back");
    if (!distinct_and_nonzero(pages, count)) return fail("page numbers distinct and non-zero");

    void* const window = XMemVirtualAlloc(nullptr, window_pages * page64k, physical_reserve, 0x280, PAGE_READWRITE);
    if (!window) return fail("physical reserve");
    if (reinterpret_cast<uintptr_t>(window) % page64k != 0) return fail("physical reserve alignment");

    if (XMemMapPhysicalPages(window, count, pages) != window) return fail("XMemMapPhysicalPages");
    for (uint64_t i = 0; i < count; ++i) stamp(page_at(window, i), 0x41000000 + static_cast<uint32_t>(i));

    if (!XMemMapPhysicalPages(window, count, nullptr)) return fail("unmap with a null page array");

    uint64_t reversed[window_pages]{};
    for (uint64_t i = 0; i < count; ++i) reversed[i] = pages[count - 1 - i];
    if (XMemMapPhysicalPages(window, count, reversed) != window) return fail("remap in reverse order");
    for (uint64_t i = 0; i < count; ++i) {
        if (!stamped(page_at(window, i), 0x41000000 + static_cast<uint32_t>(count - 1 - i))) {
            return fail("a page keeps its contents across a remap");
        }
    }
    if (!XMemMapPhysicalPages(window, count, nullptr)) return fail("unmap before the scatter map");

    void* const addresses[] = {page_at(window, 9), page_at(window, 0), page_at(window, 17)};
    const uint64_t scattered[] = {pages[3], pages[20], pages[11]};
    if (!XMemMapPhysicalPagesScatter(addresses, static_cast<uint32_t>(std::size(addresses)), scattered)) {
        return fail("XMemMapPhysicalPagesScatter");
    }
    if (!stamped(addresses[0], 0x41000003) || !stamped(addresses[1], 0x41000014) ||
        !stamped(addresses[2], 0x4100000b)) {
        return fail("scatter map puts each page at its own address");
    }
    if (!XMemMapPhysicalPagesScatter(addresses, static_cast<uint32_t>(std::size(addresses)), nullptr)) {
        return fail("scatter unmap");
    }
    if (!XMemFreePhysicalPages(count, pages)) return fail("XMemFreePhysicalPages");
    return true;
}

bool placeholder_sequence() {
    auto* const arena =
        static_cast<unsigned char*>(XMemVirtualAlloc(nullptr, arena_bytes, placeholder_reserve, 7, PAGE_NOACCESS));
    if (!arena) return fail("placeholder reserve");
    if (reinterpret_cast<uintptr_t>(arena) % page64k != 0) return fail("placeholder reserve alignment");

    for (const Carve& carve : carves) {
        void* const wanted = arena + carve.offset;
        if (XMemVirtualAlloc(wanted, carve_pages * page64k, placeholder_carve, 0x380, PAGE_READWRITE) != wanted) {
            return fail("carving a physical range out of the placeholder");
        }
    }

    constexpr uint64_t total = carve_pages * std::size(carves);
    uint64_t pages[total]{};
    uint64_t count = total;
    if (!XMemAllocatePhysicalPages(0x380, &count, pages)) return fail("XMemAllocatePhysicalPages(0x380)");
    if (count != total) return fail("second batch page count written back");
    if (!distinct_and_nonzero(pages, count)) return fail("second batch page numbers distinct and non-zero");

    for (size_t c = 0; c < std::size(carves); ++c) {
        void* const base = arena + carves[c].offset;
        if (XMemMapPhysicalPages(base, carve_pages, pages + c * carve_pages) != base) {
            return fail("mapping pages into a carved range");
        }
        for (uint64_t i = 0; i < carve_pages; ++i) stamp(page_at(base, i), carves[c].tag + static_cast<uint32_t>(i));
    }
    for (const Carve& carve : carves) {
        for (uint64_t i = 0; i < carve_pages; ++i) {
            if (!stamped(page_at(arena + carve.offset, i), carve.tag + static_cast<uint32_t>(i))) {
                return fail("carved ranges do not overwrite each other");
            }
        }
    }
    if (!reserved_without_memory(arena + SIZE_T{8} * 1024 * 1024) ||
        !reserved_without_memory(arena + SIZE_T{100} * 1024 * 1024)) {
        return fail("the placeholder between carved ranges stays reserved");
    }

    for (const Carve& carve : carves) {
        if (!XMemMapPhysicalPages(arena + carve.offset, carve_pages, nullptr)) return fail("unmapping a carved range");
    }
    if (!XMemFreePhysicalPages(count, pages)) return fail("freeing the second batch");
    return true;
}
}  // namespace

int main() {
    if (!window_sequence() || !placeholder_sequence()) return 1;
    std::puts("tiny_xmem_physical: OK");
    return 0;
}
