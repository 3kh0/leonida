// Exercise the early EXE allocations through a mapped guest's real xmem.dll import.
// Keep the regions live together, using small synthetic sizes that require page rounding.
#include <windows.h>
#include <cstdint>
#include <cstdio>

extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD);

namespace {
constexpr SIZE_T page64k = 64 * 1024;
constexpr SIZE_T page2m = 2 * 1024 * 1024;
constexpr DWORD reserve64k = 0x20400000 | MEM_RESERVE;
constexpr DWORD commit2m = 0x20000000 | MEM_RESERVE | MEM_COMMIT;

struct Request {
    const char* name;
    SIZE_T bytes, page;
    DWORD type;
    ULONGLONG attributes;
    DWORD protection, state, host_protection;
};

constexpr Request requests[] = {
    {"CPU reservation", page64k + 1, page64k, reserve64k, 0x21, PAGE_READWRITE, MEM_RESERVE, PAGE_READWRITE},
    {"graphics reservation", 2 * page64k + 1, page64k, reserve64k, 0x3, 0x801, MEM_RESERVE, PAGE_NOACCESS},
    {"0x183 commitment", page2m + 1, page2m, commit2m, 0x183, 0x22004, MEM_COMMIT, PAGE_READWRITE},
    {"0x283 commitment", page2m - 1, page2m, commit2m, 0x283, 0x22004, MEM_COMMIT, PAGE_READWRITE},
    {"0x181 commitment", 2 * page2m + 1, page2m, commit2m, 0x181, 0x22004, MEM_COMMIT, PAGE_READWRITE},
};

bool check_region(const Request& request, void* base) {
    MEMORY_BASIC_INFORMATION info{};
    const SIZE_T rounded = (request.bytes + request.page - 1) & ~(request.page - 1);
    if (VirtualQuery(base, &info, sizeof(info)) != sizeof(info) ||
        reinterpret_cast<std::uintptr_t>(base) % request.page != 0 ||
        info.AllocationBase != base || info.BaseAddress != base || info.RegionSize != rounded ||
        info.State != request.state || info.AllocationProtect != request.host_protection ||
        (request.state == MEM_COMMIT && info.Protect != request.host_protection)) {
        std::printf("tiny_xmem_startup: FAIL %s query (size %zu, state %#lx, protection %#lx/%#lx)\n",
                    request.name, info.RegionSize, info.State, info.AllocationProtect, info.Protect);
        return false;
    }
    if (request.state == MEM_COMMIT) {
        auto* bytes = static_cast<volatile unsigned char*>(base);
        bytes[0] = 0x35;
        bytes[rounded - 1] = 0xa7;
        if (bytes[0] != 0x35 || bytes[rounded - 1] != 0xa7) {
            std::puts("tiny_xmem_startup: FAIL committed memory readback");
            return false;
        }
    }
    return true;
}
}  // namespace

int main() {
    void* bases[ARRAYSIZE(requests)]{};
    size_t count = 0;
    bool ok = true;
    for (const Request& request : requests) {
        void* base = XMemVirtualAlloc(nullptr, request.bytes, request.type, request.attributes, request.protection);
        if (!base) {
            std::printf("tiny_xmem_startup: FAIL %s allocation (Win32 error %lu)\n", request.name, GetLastError());
            ok = false;
            break;
        }
        bases[count++] = base;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!check_region(requests[i], bases[i])) ok = false;
    }
    while (count > 0) {
        --count;
        if (!VirtualFree(bases[count], 0, MEM_RELEASE)) {
            std::printf("tiny_xmem_startup: FAIL %s release (Win32 error %lu)\n", requests[count].name, GetLastError());
            ok = false;
        }
    }
    if (ok) std::puts("tiny_xmem_startup: OK");
    return ok ? 0 : 1;
}
