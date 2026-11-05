// Differential libFuzzer harness: simeon::FlatHashMapU64 against
// std::unordered_map, driven by a byte-coded op stream.
//
// Each op is one byte followed by an 8-byte key when the op needs one:
//   0 insert/increment  1 find  2 reserve(size + k)  3 clear  4 shrink_to_fit
// Low key bits are reused often so probing chains and collisions get covered.
//
// Checks: size, lookups and full iteration match the reference; iteration
// visits each key once; memory_bytes() covers at least the live entries.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

#include "simeon/flat_hash_map_u64.hpp"

namespace {

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "fuzz_flat_hash_map: %s\n", what);
    std::abort();
}

void require(bool ok, const char* what) {
    if (!ok)
        fail(what);
}

using Map = simeon::FlatHashMapU64<std::uint32_t>;
using Ref = std::unordered_map<std::uint64_t, std::uint32_t>;

void compare(const Map& m, const Ref& ref) {
    require(m.size() == ref.size(), "size mismatch");
    require(m.empty() == ref.empty(), "empty mismatch");
    require(m.memory_bytes() >= m.size() * (sizeof(Map::value_type) + 12),
            "memory_bytes below live entries");
    std::unordered_set<std::uint64_t> seen;
    for (const auto& [k, v] : m) {
        require(seen.insert(k).second, "iteration visited a key twice");
        auto it = ref.find(k);
        require(it != ref.end(), "iteration produced unknown key");
        require(it->second == v, "iteration value mismatch");
    }
    require(seen.size() == ref.size(), "iteration missed keys");
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    Map m;
    Ref ref;
    std::size_t i = 0;
    std::size_t ops = 0;
    while (i < size && ops < 4096) {
        const std::uint8_t op = data[i++] % 5;
        std::uint64_t key = 0;
        if (op <= 2) {
            const std::size_t take = std::min<std::size_t>(8, size - i);
            std::memcpy(&key, data + i, take);
            i += take;
            // Fold to a small key space half the time to force collisions.
            if (key & 1u)
                key &= 0x3FFu;
        }
        switch (op) {
            case 0:
                ++m[key];
                ++ref[key];
                break;
            case 1: {
                auto it = m.find(key);
                auto rit = ref.find(key);
                require((it == m.end()) == (rit == ref.end()), "find presence mismatch");
                if (rit != ref.end())
                    require(it->second == rit->second, "find value mismatch");
                break;
            }
            case 2:
                m.reserve(m.size() + (key & 0xFFFu));
                break;
            case 3:
                m.clear();
                ref.clear();
                break;
            case 4:
                m.shrink_to_fit();
                break;
        }
        if (++ops % 64 == 0)
            compare(m, ref);
    }
    compare(m, ref);
    return 0;
}
