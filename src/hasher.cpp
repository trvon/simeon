#include "simeon/hasher.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>

#include "simeon/hash_tabulation.hpp"

namespace simeon {

// Defined in src/hash_xxh64.cpp and src/hash_crc32c.cpp.
std::uint64_t xxh64_hash(std::string_view s, std::uint64_t seed) noexcept;
std::uint64_t crc32c_hash(std::string_view s, std::uint64_t seed) noexcept;

namespace {

[[nodiscard]] inline std::uint64_t load_u64_unaligned(const char* p) noexcept {
    std::array<std::byte, sizeof(std::uint64_t)> bytes{};
    std::memcpy(bytes.data(), p, bytes.size());
    return std::bit_cast<std::uint64_t>(bytes);
}

[[nodiscard]] inline std::uint64_t load_tail_le(const char* p, std::size_t n) noexcept {
    std::uint64_t tail = 0;
    for (std::size_t j = 0; j < n; ++j) {
        tail |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(p[j])) << (j * 8);
    }
    return tail;
}

std::uint64_t splitmix64_hash(std::string_view s, std::uint64_t seed) noexcept {
    std::uint64_t h = splitmix64_mix(seed ^ 0x9E3779B97F4A7C15ULL);
    std::size_t i = 0;
    const std::size_t n = s.size();
    while (i + 8 <= n) {
        const std::uint64_t chunk = load_u64_unaligned(s.data() + i);
        h = splitmix64_mix(h ^ chunk);
        i += 8;
    }
    if (i < n) {
        std::uint64_t tail = load_tail_le(s.data() + i, n - i);
        tail ^= static_cast<std::uint64_t>(n - i) << 56;
        h = splitmix64_mix(h ^ tail);
    }
    return splitmix64_mix(h ^ static_cast<std::uint64_t>(n));
}

} // namespace

std::uint64_t hash64(std::string_view s, std::uint64_t seed, HashFamily family) noexcept {
    switch (family) {
        case HashFamily::SplitMix64:
            return splitmix64_hash(s, seed);
        case HashFamily::XxHash64:
            return xxh64_hash(s, seed);
        case HashFamily::Crc32:
            return crc32c_hash(s, seed);
        case HashFamily::MixedTabulation:
            return mixed_tabulation_hash(s, seed);
    }
    return splitmix64_hash(s, seed);
}

} // namespace simeon
