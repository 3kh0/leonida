// Unit tests for the descriptor hash table (set 0, binding 2 contract).
// Tests FNV-1a hashing, T#/S# key generation, probe resolution, dynamic resizing,
// GPU buffer serialization, and multithreaded insertion.
#include "../src/descriptor_table.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "assertion failed: %s at %s:%d\n", #condition, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

using namespace leonida;

int test_fnv1a_hash_matches_contract() {
    // Verified against vk_texture_test.cpp key_hash and rdna2spirv.py desc_probe
    const std::uint32_t zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::uint32_t expected_zero_hash = 2166136261u;
    for (int i = 0; i < 8; ++i) expected_zero_hash = (expected_zero_hash ^ 0) * 16777619u;
    EXPECT(hash_descriptor_key(zeros) == expected_zero_hash);

    const std::uint32_t sample_key[8] = {
        0x12345678, 0x9abcdef0, 0x11223344, 0x55667788,
        0x99aabbcc, 0xddeeff00, 0x01020304, 0x05060708
    };
    std::uint32_t expected_hash = 2166136261u;
    for (int i = 0; i < 8; ++i) expected_hash = (expected_hash ^ sample_key[i]) * 16777619u;
    EXPECT(hash_descriptor_key(sample_key) == expected_hash);
    return 0;
}

int test_ssharp_key_padding() {
    const std::uint32_t ssharp[4] = {0x92, 0xc00000, 0x14500000, 0x0};
    const auto key = make_ssharp_key(ssharp);
    EXPECT(key[0] == 0x92);
    EXPECT(key[1] == 0xc00000);
    EXPECT(key[2] == 0x14500000);
    EXPECT(key[3] == 0x0);
    EXPECT(key[4] == 0xFFFFFFFFu);
    EXPECT(key[5] == 0xFFFFFFFFu);
    EXPECT(key[6] == 0xFFFFFFFFu);
    EXPECT(key[7] == 0xFFFFFFFFu);
    return 0;
}

int test_basic_insertion_and_lookup() {
    DescriptorTable table(16);
    EXPECT(table.capacity() == 16);
    EXPECT(table.mask() == 15);
    EXPECT(table.size() == 0);

    const std::uint32_t tsharp[8] = {
        0x1000, 0x2000, 0x3000, 0x4000,
        0x5000, 0x6000, 0x7000, 0x8000
    };
    EXPECT(table.lookup(tsharp) == 0);
    EXPECT(!table.contains(tsharp));

    EXPECT(table.insert(tsharp, 1));
    EXPECT(table.size() == 1);
    EXPECT(table.contains(tsharp));
    EXPECT(table.lookup(tsharp) == 1);

    // Duplicate key updates in place
    EXPECT(table.insert(tsharp, 42));
    EXPECT(table.size() == 1);
    EXPECT(table.lookup(tsharp) == 42);

    // S# key
    const std::uint32_t ssharp[4] = {0xa, 0xb, 0xc, 0xd};
    const auto skey = make_ssharp_key(ssharp);
    EXPECT(table.insert(skey, 7));
    EXPECT(table.size() == 2);
    EXPECT(table.lookup(skey) == 7);
    EXPECT(table.lookup(tsharp) == 42);
    return 0;
}

int test_load_factor_and_growth() {
    // Start with 16 slots. Load factor threshold 50% means max 8 entries before growth.
    DescriptorTable table(16);
    EXPECT(table.capacity() == 16);

    std::vector<DescriptorKey> keys;
    for (std::uint32_t i = 1; i <= 20; ++i) {
        std::uint32_t k[8] = {i, i * 2, i * 3, i * 4, i * 5, i * 6, i * 7, i * 8};
        keys.push_back(make_tsharp_key(k));
        EXPECT(table.insert(keys.back(), i + 100));
    }

    EXPECT(table.size() == 20);
    EXPECT(table.capacity() >= 64); // Doubled from 16 -> 32 -> 64
    EXPECT(table.mask() == table.capacity() - 1);

    // Verify all 20 keys resolve to their exact indices after growths
    for (std::uint32_t i = 1; i <= 20; ++i) {
        EXPECT(table.lookup(keys[i - 1]) == i + 100);
    }
    return 0;
}

int test_insert_reports_mapping_changes() {
    DescriptorTable table;
    const DescriptorKey key{1, 2, 3, 4, 5, 6, 7, 8};
    using Result = DescriptorTable::InsertResult;

    EXPECT(table.insert_or_update(key.data(), 0) == Result::added);
    EXPECT(table.insert_or_update(key.data(), 0) == Result::unchanged);
    EXPECT(table.insert_or_update(key.data(), 17) == Result::replaced);
    EXPECT(table.insert_or_update(key.data(), 17) == Result::unchanged);
    EXPECT(table.lookup(key) == 17);
    EXPECT(table.size() == 1);
    return 0;
}

int test_probe_collision_resolution() {
    DescriptorTable table(64);

    // Create 4 distinct keys that share the exact same hash
    // We can forge keys that share hash or test linear collision handling
    std::uint32_t k1[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    std::uint32_t k2[8] = {2, 0, 0, 0, 0, 0, 0, 0};
    std::uint32_t k3[8] = {3, 0, 0, 0, 0, 0, 0, 0};

    EXPECT(table.insert(k1, 10));
    EXPECT(table.insert(k2, 20));
    EXPECT(table.insert(k3, 30));

    EXPECT(table.lookup(k1) == 10);
    EXPECT(table.lookup(k2) == 20);
    EXPECT(table.lookup(k3) == 30);
    return 0;
}

int test_gpu_buffer_serialization() {
    DescriptorTable table(16);
    const std::uint32_t t1[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    const std::uint32_t s1[4] = {1, 2, 3, 4};
    const auto skey = make_ssharp_key(s1);

    EXPECT(table.insert(t1, 5));
    EXPECT(table.insert(skey, 9));

    const auto gpu_buf = table.build_gpu_buffer(false);
    EXPECT(gpu_buf.size() == 1 + 16 * 10);
    EXPECT(gpu_buf[0] == 15); // mask

    // Verify lookup by traversing the generated buffer using the exact shader logic
    const auto shader_lookup = [&](const std::uint32_t key[8]) -> std::uint32_t {
        const std::uint32_t mask = gpu_buf[0];
        const std::uint32_t h = hash_descriptor_key(key);
        for (std::uint32_t k = 0; k < 8; ++k) {
            const std::uint32_t slot = (h + k) & mask;
            const std::size_t base = 1 + slot * 10;
            bool match = true;
            for (std::size_t j = 0; j < 8; ++j) {
                if (gpu_buf[base + j] != key[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return gpu_buf[base + 8]; // index
            }
        }
        return 0;
    };

    EXPECT(shader_lookup(t1) == 5);
    EXPECT(shader_lookup(skey.data()) == 9);

    const std::uint32_t missing[8] = {99, 99, 99, 99, 99, 99, 99, 99};
    EXPECT(shader_lookup(missing) == 0);

    // Test write_to_buffer
    std::vector<std::uint32_t> raw_copy(gpu_buf.size());
    table.write_to_buffer(raw_copy.data(), true);
    EXPECT(raw_copy[0] == 15);
    for (std::size_t s = 0; s < 16; ++s) {
        EXPECT(raw_copy[1 + s * 10 + 9] == 0); // clear_pad = true
    }
    return 0;
}

int test_multithreaded_insert_and_lookup() {
    DescriptorTable table(32);
    constexpr int kThreads = 4;
    constexpr int kPerThread = 50;

    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&table, t] {
            for (int i = 0; i < kPerThread; ++i) {
                const std::uint32_t id = static_cast<std::uint32_t>(t * kPerThread + i + 1);
                std::uint32_t key[8] = {id, id * 3, id * 7, id * 11, id * 13, id * 17, id * 19, id * 23};
                table.insert(key, id);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    EXPECT(table.size() == kThreads * kPerThread);
    for (int t = 0; t < kThreads; ++t) {
        for (int i = 0; i < kPerThread; ++i) {
            const std::uint32_t id = static_cast<std::uint32_t>(t * kPerThread + i + 1);
            std::uint32_t key[8] = {id, id * 3, id * 7, id * 11, id * 13, id * 17, id * 19, id * 23};
            EXPECT(table.lookup(key) == id);
        }
    }
    return 0;
}

int test_clear_resets_table() {
    DescriptorTable table(16);
    std::uint32_t k[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    table.insert(k, 100);
    EXPECT(table.size() == 1);
    EXPECT(table.lookup(k) == 100);

    table.clear();
    EXPECT(table.size() == 0);
    EXPECT(table.lookup(k) == 0);
    return 0;
}

int test_erase_and_reinsert() {
    DescriptorTable table(16);

    const std::uint32_t k1[8] = {1, 10, 100, 1000, 2, 20, 200, 2000};
    const std::uint32_t k2[8] = {3, 30, 300, 3000, 4, 40, 400, 4000};
    const std::uint32_t k3[8] = {5, 50, 500, 5000, 6, 60, 600, 6000};

    // Erasing a non-existent key returns false
    EXPECT(!table.erase(k1));

    EXPECT(table.insert(k1, 101));
    EXPECT(table.insert(k2, 102));
    EXPECT(table.insert(k3, 103));
    EXPECT(table.size() == 3);

    // Erase middle element
    EXPECT(table.erase(k2));
    EXPECT(table.size() == 2);
    EXPECT(!table.contains(k2));
    EXPECT(table.lookup(k2) == 0);
    EXPECT(table.contains(k1));
    EXPECT(table.lookup(k1) == 101);
    EXPECT(table.contains(k3));
    EXPECT(table.lookup(k3) == 103);

    // Double erase returns false
    EXPECT(!table.erase(k2));

    // Erase remaining entries
    EXPECT(table.erase(k1));
    EXPECT(table.size() == 1);
    EXPECT(!table.contains(k1));
    EXPECT(table.lookup(k1) == 0);
    EXPECT(table.contains(k3));
    EXPECT(table.lookup(k3) == 103);

    EXPECT(table.erase(k3));
    EXPECT(table.size() == 0);
    EXPECT(!table.contains(k3));
    EXPECT(table.lookup(k3) == 0);

    // Re-inserting into erased slots
    EXPECT(table.insert(k1, 201));
    EXPECT(table.size() == 1);
    EXPECT(table.contains(k1));
    EXPECT(table.lookup(k1) == 201);

    return 0;
}

int test_change_tracking() {
    DescriptorTable table(16);
    const std::uint32_t k1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::uint32_t k2[8] = {9, 10, 11, 12, 13, 14, 15, 16};

    // A fresh table has nothing to upload.
    DescriptorTable::Changes changes = table.take_changes();
    EXPECT(!changes.all && changes.slots.empty());

    // An insert marks exactly the slot it landed in, and read_slot agrees with build_gpu_buffer.
    EXPECT(table.insert(k1, 42));
    changes = table.take_changes();
    EXPECT(!changes.all && changes.slots.size() == 1);
    const std::uint32_t slot = changes.slots.front();
    const auto words = table.build_gpu_buffer();
    std::uint32_t read[10] = {};
    EXPECT(table.read_slot(slot, read));
    EXPECT(std::memcmp(read, &words[1 + slot * 10], sizeof(read)) == 0);
    EXPECT(read[8] == 42 && read[9] == kSlotOccupiedMarker);
    EXPECT(table.take_changes().slots.empty());  // taking consumes the list

    // Replacing an index marks the same slot; read_slot sees the new index.
    EXPECT(table.insert_or_update(k1, 43) == DescriptorTable::InsertResult::replaced);
    changes = table.take_changes();
    EXPECT(!changes.all && changes.slots.size() == 1 && changes.slots.front() == slot);
    EXPECT(table.read_slot(slot, read) && read[8] == 43);

    // Erasing marks only the slot it was in: no other entry moves.
    EXPECT(table.insert(k2, 7));
    table.take_changes();
    EXPECT(table.erase(k1));
    changes = table.take_changes();
    EXPECT(!changes.all && changes.slots.size() == 1 && changes.slots.front() == slot);

    // Growth changes the mask, so everything re-uploads.
    const std::size_t before = table.capacity();
    for (std::uint32_t i = 0; i < 200 && table.capacity() == before; ++i) {
        const std::uint32_t key[8] = {i, i * 2, i * 3, i * 4, i * 5, i * 6, i * 7, i * 8};
        table.insert(key, 1000 + i);
    }
    EXPECT(table.capacity() > before);
    EXPECT(table.take_changes().all);

    // Out-of-range and wrapping uint32 slot ranges are refused rather than read past the table.
    EXPECT(!table.read_slot(std::uint32_t(table.capacity()), read));
    EXPECT(!table.read_slots(UINT32_MAX, 1, read));
    EXPECT(!table.read_slots(UINT32_MAX - 2, 4, read));
    EXPECT(!table.read_slots(0, 1, nullptr));
    EXPECT(table.read_slots(0, 0, nullptr));

    // Direct write_to_buffer matches build_gpu_buffer byte-for-byte for both clear_pad modes.
    for (bool clear_pad : {false, true}) {
        const auto expected = table.build_gpu_buffer(clear_pad);
        std::vector<std::uint32_t> direct(table.size_dwords(), 0xccccccccu);
        table.write_to_buffer(direct.data(), clear_pad);
        EXPECT(direct == expected);
    }
    return 0;
}

int test_relocation_marks_both_slots() {
    DescriptorTable table(64);
    const std::uint32_t slots = std::uint32_t(table.capacity());
    // Eight keys in eight adjacent home slots across the end of the table; the ninth shares the first home and can
    // only fit by moving a resident (make_probe_slot), which changes two slots.
    std::array<DescriptorKey, kDescriptorProbes + 1> keys{};
    for (std::uint32_t i = 0; i < keys.size(); ++i) {
        auto& key = keys[i];
        key[0] = i + 1;
        const std::uint32_t target = (slots - 4 + i % kDescriptorProbes) & (slots - 1);
        while ((hash_descriptor_key(key) & (slots - 1)) != target) ++key[7];
    }
    for (std::uint32_t i = 0; i < kDescriptorProbes; ++i) EXPECT(table.insert(keys[i].data(), 100 + i));
    table.take_changes();
    EXPECT(table.insert_or_update(keys.back().data(), 999) == DescriptorTable::InsertResult::added_with_relocation);
    const DescriptorTable::Changes changes = table.take_changes();
    EXPECT(!changes.all);
    EXPECT(changes.slots.size() >= 2);  // the hole a resident moved into and the slot the new key took
    for (std::uint32_t i = 0; i < kDescriptorProbes; ++i) EXPECT(table.lookup(keys[i]) == 100 + i);
    EXPECT(table.lookup(keys.back()) == 999);
    return 0;
}

// Shaders keep probing the GPU copy of the table while an erase is uploaded without waiting for idle
// (BindlessSet::flush_needs_idle), so an erase must leave every other key's slot exactly as it was.
int test_erase_moves_no_other_entry() {
    DescriptorTable table(64);
    const std::uint32_t slots = std::uint32_t(table.capacity());
    // Five keys with one home: they fill five consecutive slots, the cluster the old erase compacted.
    std::array<DescriptorKey, 5> keys{};
    for (std::uint32_t i = 0; i < keys.size(); ++i) {
        auto& key = keys[i];
        key[0] = i + 1;
        while ((hash_descriptor_key(key) & (slots - 1)) != slots - 2) ++key[7];  // the cluster wraps the end
        EXPECT(table.insert_or_update(key.data(), 200 + i) == DescriptorTable::InsertResult::added);
    }
    table.take_changes();
    const auto before = table.build_gpu_buffer();

    EXPECT(table.erase(keys[1].data()));
    const DescriptorTable::Changes changes = table.take_changes();
    EXPECT(!changes.all && changes.slots.size() == 1);
    const std::uint32_t hole = changes.slots.front();
    const auto after = table.build_gpu_buffer();
    for (std::uint32_t s = 0; s < slots; ++s) {
        const bool same = std::memcmp(&before[1 + s * 10], &after[1 + s * 10], 10 * sizeof(std::uint32_t)) == 0;
        EXPECT(same == (s != hole));
    }
    for (std::uint32_t w = 0; w < 10; ++w) EXPECT(after[1 + hole * 10 + w] == 0);  // an empty slot: a miss
    for (std::uint32_t i = 0; i < keys.size(); ++i) EXPECT(table.lookup(keys[i]) == (i == 1 ? 0u : 200 + i));

    // Keys past the hole are still found after more erases, and a new key of the same home reuses the hole.
    EXPECT(table.erase(keys[0].data()));
    EXPECT(table.lookup(keys[4]) == 204 && table.lookup(keys[3]) == 203);
    DescriptorKey extra = keys[0];
    extra[0] = 100;
    while ((hash_descriptor_key(extra) & (slots - 1)) != slots - 2) ++extra[7];
    table.take_changes();
    EXPECT(table.insert_or_update(extra.data(), 300) == DescriptorTable::InsertResult::added);
    const DescriptorTable::Changes reuse = table.take_changes();
    EXPECT(reuse.slots.size() == 1 && reuse.slots.front() == slots - 2);  // the first hole of the probe window
    EXPECT(table.lookup(extra) == 300 && table.size() == 4);
    return 0;
}

int test_unresolvable_hash_collision() {
    DescriptorTable table(32);
    std::array<DescriptorKey, 9> keys{};
    for (std::uint32_t i = 0; i < keys.size(); ++i) {
        keys[i][0] = i;
        // Invert the final FNV multiplication to make every complete hash zero.
        std::uint32_t prefix = kFnvOffsetBasis;
        for (int word = 0; word < 7; ++word) prefix = (prefix ^ keys[i][word]) * kFnvPrime;
        keys[i][7] = prefix;
        EXPECT(hash_descriptor_key(keys[i]) == 0);
    }
    for (std::uint32_t i = 0; i < 8; ++i) table.insert(keys[i], i + 1);
    const auto before = table.build_gpu_buffer();
    bool rejected = false;
    try { table.insert(keys[8], 9); }
    catch (const std::length_error&) { rejected = true; }
    EXPECT(rejected && table.size() == 8 && table.capacity() == 32);
    EXPECT(table.build_gpu_buffer() == before);
    for (std::uint32_t i = 0; i < 8; ++i) EXPECT(table.lookup(keys[i]) == i + 1);
    EXPECT(table.insert_or_update(keys[0].data(), 99) == DescriptorTable::InsertResult::replaced);
    EXPECT(table.lookup(keys[0]) == 99);
    EXPECT(table.erase(keys[0]));
    EXPECT(table.insert(keys[8], 9) && table.lookup(keys[8]) == 9);
    return 0;
}

int test_capacity_overflow() {
    for (std::uint32_t requested : {0x80000001u, 0xffffffffu}) {
        bool rejected = false;
        try { DescriptorTable table(requested); }
        catch (const std::length_error&) { rejected = true; }
        EXPECT(rejected);
    }
    for (std::uint32_t requested : {0u, 1u, 16u, 17u, 31u, 32u}) {
        DescriptorTable table(requested);
        EXPECT(table.capacity() == (requested <= 16 ? 16 : 32));
    }
    return 0;
}

constexpr std::uint32_t fnv_prime_inverse() {
    std::uint32_t inv = 1u;
    for (int i = 0; i < 5; ++i) inv *= 2u - kFnvPrime * inv;
    return inv;
}
static_assert(kFnvPrime * fnv_prime_inverse() == 1u);

DescriptorKey make_key_with_hash(std::uint32_t id, std::uint32_t target_hash) {
    DescriptorKey key{};
    key[0] = id;
    std::uint32_t prefix = kFnvOffsetBasis;
    for (int word = 0; word < 7; ++word) prefix = (prefix ^ key[word]) * kFnvPrime;
    key[7] = prefix ^ (target_hash * fnv_prime_inverse());
    return key;
}

int test_unresolvable_collision_at_half_load_does_not_grow() {
    {
        DescriptorTable table(16);
        for (std::uint32_t i = 0; i < 8; ++i) {
            const DescriptorKey k = make_key_with_hash(i, 0u);
            EXPECT(hash_descriptor_key(k) == 0u);
            EXPECT(table.insert(k, i + 1));
        }
        table.take_changes();
        const auto before = table.build_gpu_buffer();

        bool rejected = false;
        try { table.insert(make_key_with_hash(8, 0u), 9); }
        catch (const std::length_error&) { rejected = true; }
        EXPECT(rejected);
        EXPECT(table.size() == 8);
        EXPECT(table.capacity() == 16);
        EXPECT(table.mask() == 15);
        EXPECT(table.build_gpu_buffer() == before);
        const auto changes = table.take_changes();
        EXPECT(!changes.all && changes.slots.empty());
    }
    {
        DescriptorTable table(32);
        for (std::uint32_t i = 0; i < 8; ++i) EXPECT(table.insert(make_key_with_hash(i, 0u), i + 1));
        for (std::uint32_t i = 0; i < 8; ++i) EXPECT(table.insert(make_key_with_hash(100 + i, 16u), i + 10));
        table.take_changes();
        const auto before = table.build_gpu_buffer();

        bool rejected = false;
        try { table.insert(make_key_with_hash(8, 0u), 99); }
        catch (const std::length_error&) { rejected = true; }
        EXPECT(rejected);
        EXPECT(table.size() == 16);
        EXPECT(table.capacity() == 32);
        EXPECT(table.build_gpu_buffer() == before);
        const auto changes = table.take_changes();
        EXPECT(!changes.all && changes.slots.empty());
    }
    return 0;
}

int test_all_changed_suppresses_redundant_slot_list() {
    {
        DescriptorTable table(16);
        const DescriptorKey k1 = make_key_with_hash(1, 0u);
        const DescriptorKey k2 = make_key_with_hash(2, 4u);
        EXPECT(table.insert(k1, 10));
        table.clear();
        EXPECT(table.insert(k1, 11));
        EXPECT(table.insert_or_update(k1.data(), 12) == DescriptorTable::InsertResult::replaced);
        EXPECT(table.insert(k2, 20));
        EXPECT(table.erase(k1));
        const auto changes = table.take_changes();
        EXPECT(changes.all);
        EXPECT(changes.slots.empty());
    }
    {
        DescriptorTable table(16);
        table.take_changes();
        for (std::uint32_t i = 0; i < 9; ++i) EXPECT(table.insert(make_key_with_hash(i, i * 2u), i + 1));
        EXPECT(table.capacity() == 32);
        EXPECT(table.insert(make_key_with_hash(99, 20u), 100));
        const auto changes = table.take_changes();
        EXPECT(changes.all);
        EXPECT(changes.slots.empty());
    }
    return 0;
}

int test_unresolvable_adjacent_hash_cluster_rejected_without_growth() {
    std::uint32_t next_id = 1;
    // 10 keys across adjacent hashes (2 at h=0, 8 at h=1) need 10 slots inside [0..8] (9 slots).
    {
        DescriptorTable table(32);
        EXPECT(table.insert(make_key_with_hash(next_id++, 0u), 10));
        for (int i = 0; i < 8; ++i) EXPECT(table.insert(make_key_with_hash(next_id++, 1u), 20u + i));
        table.take_changes();
        const auto before = table.build_gpu_buffer();

        bool rejected = false;
        try { table.insert(make_key_with_hash(next_id++, 0u), 99); }
        catch (const std::length_error&) { rejected = true; }
        EXPECT(rejected);
        EXPECT(table.size() == 9);
        EXPECT(table.capacity() == 32);
        EXPECT(table.build_gpu_buffer() == before);
        const auto changes = table.take_changes();
        EXPECT(!changes.all && changes.slots.empty());
    }
    // Multi-step cluster: 17 keys across {0, 4, 8} need 17 slots inside [0..15] (16 slots).
    {
        DescriptorTable table(64);
        for (int i = 0; i < 4; ++i) EXPECT(table.insert(make_key_with_hash(next_id++, 0u), 100u + i));
        for (int i = 0; i < 4; ++i) EXPECT(table.insert(make_key_with_hash(next_id++, 4u), 200u + i));
        for (int i = 0; i < 8; ++i) EXPECT(table.insert(make_key_with_hash(next_id++, 8u), 300u + i));
        table.take_changes();
        const auto before = table.build_gpu_buffer();

        bool rejected = false;
        try { table.insert(make_key_with_hash(next_id++, 0u), 999); }
        catch (const std::length_error&) { rejected = true; }
        EXPECT(rejected);
        EXPECT(table.size() == 16);
        EXPECT(table.capacity() == 64);
        EXPECT(table.build_gpu_buffer() == before);
        const auto changes = table.take_changes();
        EXPECT(!changes.all && changes.slots.empty());
    }
    // Wraparound in 32-bit hash space (0xFFFFFFFF -> 0).
    {
        DescriptorTable table(32);
        EXPECT(table.insert(make_key_with_hash(next_id++, 0xFFFFFFFFu), 10));
        for (int i = 0; i < 8; ++i) EXPECT(table.insert(make_key_with_hash(next_id++, 0u), 20u + i));
        table.take_changes();
        const auto before = table.build_gpu_buffer();

        bool rejected = false;
        try { table.insert(make_key_with_hash(next_id++, 0xFFFFFFFFu), 99); }
        catch (const std::length_error&) { rejected = true; }
        EXPECT(rejected);
        EXPECT(table.size() == 9);
        EXPECT(table.capacity() == 32);
        EXPECT(table.build_gpu_buffer() == before);
        const auto changes = table.take_changes();
        EXPECT(!changes.all && changes.slots.empty());
    }
    // False-positive guard: hash 32 collides at mod-32 slot 0, not in full 32-bit hash space; growing resolves it.
    {
        DescriptorTable table(32);
        std::vector<std::pair<DescriptorKey, std::uint32_t>> inserted;
        for (int i = 0; i < 8; ++i) {
            const DescriptorKey k = make_key_with_hash(next_id++, 1u);
            inserted.push_back({k, 100u + i});
            EXPECT(table.insert(k, 100u + i));
        }
        const DescriptorKey mod_alias = make_key_with_hash(next_id++, 32u);
        inserted.push_back({mod_alias, 500});
        EXPECT(table.insert(mod_alias, 500));
        const DescriptorKey zero_key = make_key_with_hash(next_id++, 0u);
        inserted.push_back({zero_key, 600});
        EXPECT(table.insert(zero_key, 600));
        EXPECT(table.capacity() == 64);
        EXPECT(table.size() == 10);
        for (const auto& [k, v] : inserted) EXPECT(table.lookup(k) == v);
    }
    return 0;
}

int test_multihop_relocation_fits_without_growth() {
    auto shader_lookup = [](const std::vector<std::uint32_t>& gpu_buf, const std::uint32_t key[8]) -> std::uint32_t {
        const std::uint32_t mask = gpu_buf[0];
        const std::uint32_t h = hash_descriptor_key(key);
        for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
            const std::uint32_t slot = (h + k) & mask;
            const std::size_t base = 1 + std::size_t(slot) * 10;
            if (gpu_buf[base + 9] == kSlotOccupiedMarker && std::memcmp(&gpu_buf[base], key, 8 * sizeof(std::uint32_t)) == 0)
                return gpu_buf[base + 8];
        }
        return 0;
    };
    auto apply_and_check = [&](DescriptorTable& table, std::vector<std::uint32_t>& gpu, std::size_t min_slots) -> int {
        const DescriptorTable::Changes changes = table.take_changes();
        EXPECT(!changes.all && changes.slots.size() >= min_slots);
        for (const std::uint32_t slot : changes.slots) EXPECT(table.read_slot(slot, &gpu[1 + std::size_t(slot) * 10]));
        EXPECT(gpu == table.build_gpu_buffer());
        return 0;
    };

    std::uint32_t next_id = 1;
    // Subcase A: 2-hop non-wrapping chain at capacity 32.
    {
        DescriptorTable table(32);
        std::vector<std::pair<DescriptorKey, std::uint32_t>> keys;
        auto put = [&](std::uint32_t h, std::uint32_t val) {
            const DescriptorKey k = make_key_with_hash(next_id++, h);
            keys.push_back({k, val});
            return table.insert_or_update(k.data(), val);
        };
        EXPECT(put(0u, 10) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(1u, 100u + i) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(8u, 200u + i) == DescriptorTable::InsertResult::added);
        table.take_changes();
        auto gpu = table.build_gpu_buffer();

        EXPECT(put(0u, 300) == DescriptorTable::InsertResult::added_with_relocation);
        EXPECT(table.capacity() == 32 && table.size() == 16);
        EXPECT(apply_and_check(table, gpu, 3) == 0);
        for (const auto& [k, v] : keys) {
            EXPECT(table.lookup(k) == v);
            EXPECT(shader_lookup(gpu, k.data()) == v);
        }
        EXPECT(put(24u, 400) == DescriptorTable::InsertResult::added);
        EXPECT(table.capacity() == 64 && table.size() == 17);
        for (const auto& [k, v] : keys) EXPECT(table.lookup(k) == v);
    }
    // Subcase B: 2-hop wrapping across both 0xFFFFFFFF->0 and table end 31->0 at capacity 32.
    {
        DescriptorTable table(32);
        std::vector<std::pair<DescriptorKey, std::uint32_t>> keys;
        auto put = [&](std::uint32_t h, std::uint32_t val) {
            const DescriptorKey k = make_key_with_hash(next_id++, h);
            keys.push_back({k, val});
            return table.insert_or_update(k.data(), val);
        };
        EXPECT(put(0xFFFFFFFDu, 10) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(0xFFFFFFFEu, 100u + i) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(5u, 200u + i) == DescriptorTable::InsertResult::added);
        table.take_changes();
        auto gpu = table.build_gpu_buffer();

        EXPECT(put(0xFFFFFFFDu, 300) == DescriptorTable::InsertResult::added_with_relocation);
        EXPECT(table.capacity() == 32 && table.size() == 16);
        EXPECT(apply_and_check(table, gpu, 3) == 0);
        for (const auto& [k, v] : keys) {
            EXPECT(table.lookup(k) == v);
            EXPECT(shader_lookup(gpu, k.data()) == v);
        }
    }
    // Subcase C: 3-hop chain at capacity 64.
    {
        DescriptorTable table(64);
        std::vector<std::pair<DescriptorKey, std::uint32_t>> keys;
        auto put = [&](std::uint32_t h, std::uint32_t val) {
            const DescriptorKey k = make_key_with_hash(next_id++, h);
            keys.push_back({k, val});
            return table.insert_or_update(k.data(), val);
        };
        EXPECT(put(0u, 10) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(1u, 100u + i) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(8u, 200u + i) == DescriptorTable::InsertResult::added);
        for (int i = 0; i < 7; ++i) EXPECT(put(15u, 300u + i) == DescriptorTable::InsertResult::added);
        table.take_changes();
        auto gpu = table.build_gpu_buffer();

        EXPECT(put(0u, 400) == DescriptorTable::InsertResult::added_with_relocation);
        EXPECT(table.capacity() == 64 && table.size() == 23);
        EXPECT(apply_and_check(table, gpu, 4) == 0);
        for (const auto& [k, v] : keys) {
            EXPECT(table.lookup(k) == v);
            EXPECT(shader_lookup(gpu, k.data()) == v);
        }
    }
    return 0;
}

int main() {
    EXPECT(test_unresolvable_collision_at_half_load_does_not_grow() == 0);
    EXPECT(test_all_changed_suppresses_redundant_slot_list() == 0);
    EXPECT(test_unresolvable_adjacent_hash_cluster_rejected_without_growth() == 0);
    EXPECT(test_multihop_relocation_fits_without_growth() == 0);
    EXPECT(test_unresolvable_hash_collision() == 0);
    EXPECT(test_capacity_overflow() == 0);
    EXPECT(test_fnv1a_hash_matches_contract() == 0);
    EXPECT(test_ssharp_key_padding() == 0);
    EXPECT(test_basic_insertion_and_lookup() == 0);
    EXPECT(test_insert_reports_mapping_changes() == 0);
    EXPECT(test_load_factor_and_growth() == 0);
    EXPECT(test_probe_collision_resolution() == 0);
    EXPECT(test_gpu_buffer_serialization() == 0);
    EXPECT(test_change_tracking() == 0);
    EXPECT(test_relocation_marks_both_slots() == 0);
    EXPECT(test_erase_moves_no_other_entry() == 0);
    EXPECT(test_multithreaded_insert_and_lookup() == 0);
    EXPECT(test_clear_resets_table() == 0);
    EXPECT(test_erase_and_reinsert() == 0);

    std::printf("descriptor_table_test: all tests passed (0 failures)\n");
    return 0;
}
