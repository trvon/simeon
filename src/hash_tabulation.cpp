#include "simeon/hash_tabulation.hpp"

#include <array>
#include <cstdint>
#include <string_view>

#include "simeon/hasher.hpp" // splitmix64_mix

namespace simeon {

namespace {

// 4 primary tables (one per byte position mod 4) + 2 derived tables. Each
// table has 256 uint64 entries; ~6 KB total per seed. Materialized on first
// use per (thread, seed) pair and cached.
struct TabulationTables {
    std::array<std::array<std::uint64_t, 256>, 4> primary;
    std::array<std::array<std::uint64_t, 256>, 2> derived;
};

void fill_tables(std::uint64_t seed, TabulationTables& t) noexcept {
    // Seed -> 6 stream offsets. Each table entry is one splitmix64 mix of
    // the stream-offset seed, advanced by index.
    std::uint64_t s = seed ^ 0x636F66666565313FULL; // arbitrary salt
    auto fill = [&](std::array<std::uint64_t, 256>& tbl) {
        for (std::uint32_t i = 0; i < 256; ++i) {
            s = splitmix64_mix(s + i);
            tbl[i] = s;
        }
        s = splitmix64_mix(s ^ 0x9E3779B97F4A7C15ULL); // separate streams
    };
    for (auto& p : t.primary)
        fill(p);
    for (auto& d : t.derived)
        fill(d);
}

const TabulationTables& tables_for(std::uint64_t seed) noexcept {
    // Per-thread cache. Common case in simeon: a small fixed set of seeds
    // (one per HashFamily caller). Tiny array probe is faster than a hash
    // map at this size.
    constexpr std::size_t kSlots = 4;
    struct Slot {
        std::uint64_t seed = 0;
        bool valid = false;
        TabulationTables tables;
    };
    thread_local std::array<Slot, kSlots> cache{};
    thread_local std::size_t next_slot = 0;

    for (auto& slot : cache) {
        if (slot.valid && slot.seed == seed)
            return slot.tables;
    }
    Slot& target = cache[next_slot];
    next_slot = (next_slot + 1) % kSlots;
    target.seed = seed;
    fill_tables(seed, target.tables);
    target.valid = true;
    return target.tables;
}

} // namespace

std::uint64_t mixed_tabulation_hash(std::string_view s, std::uint64_t seed) noexcept {
    const TabulationTables& t = tables_for(seed);
    const auto* data = static_cast<const std::uint8_t*>(static_cast<const void*>(s.data()));
    const std::size_t n = s.size();

    // Primary mix: each byte XORs in T[i & 3][b].
    std::uint64_t h = 0;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= t.primary[i & 3][data[i]];
    }
    // Length mix so empty / short strings get distinct hashes per length.
    h ^= splitmix64_mix(static_cast<std::uint64_t>(n));

    // Derived mix: XOR in two derived bytes from the current hash. This is
    // the "twisted" / "mixed" step that lifts simple-tabulation 3-independence
    // to the 5-independence required for sparse-JL.
    h ^= t.derived[0][h & 0xFF];
    h ^= t.derived[1][(h >> 8) & 0xFF];

    return h;
}

} // namespace simeon
