// XXH64 implementation per the published xxHash spec (Y. Collet, BSD-2).
// Reference: https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
//
// This is a from-spec re-implementation: the public hash64() entry point
// returns one uint64_t for use as a feature-hash seed in the count-sketch,
// so we don't need streaming or canonical-form output.

#include <cstdint>
#include <cstring>
#include <string_view>

namespace simeon {

namespace {

constexpr std::uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
constexpr std::uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr std::uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
constexpr std::uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr std::uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;

inline std::uint64_t rotl64(std::uint64_t x, int r) noexcept {
    return (x << r) | (x >> (64 - r));
}

inline std::uint64_t read64(const std::uint8_t* p) noexcept {
    std::uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;  // host endian — XXH64 is defined little-endian; correct on x86/arm64.
}

inline std::uint32_t read32(const std::uint8_t* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline std::uint64_t round(std::uint64_t acc, std::uint64_t value) noexcept {
    acc += value * PRIME64_2;
    acc = rotl64(acc, 31);
    return acc * PRIME64_1;
}

inline std::uint64_t merge_accumulator(std::uint64_t acc, std::uint64_t val) noexcept {
    val = round(0, val);
    acc ^= val;
    return acc * PRIME64_1 + PRIME64_4;
}

}  // namespace

std::uint64_t xxh64_hash(std::string_view s, std::uint64_t seed) noexcept {
    const auto* p = reinterpret_cast<const std::uint8_t*>(s.data());
    const std::size_t len = s.size();
    const std::uint8_t* const end = p + len;

    std::uint64_t h;
    if (len >= 32) {
        std::uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
        std::uint64_t v2 = seed + PRIME64_2;
        std::uint64_t v3 = seed;
        std::uint64_t v4 = seed - PRIME64_1;
        const std::uint8_t* const limit = end - 32;
        while (p <= limit) {
            v1 = round(v1, read64(p));      p += 8;
            v2 = round(v2, read64(p));      p += 8;
            v3 = round(v3, read64(p));      p += 8;
            v4 = round(v4, read64(p));      p += 8;
        }
        h = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h = merge_accumulator(h, v1);
        h = merge_accumulator(h, v2);
        h = merge_accumulator(h, v3);
        h = merge_accumulator(h, v4);
    } else {
        h = seed + PRIME64_5;
    }

    h += static_cast<std::uint64_t>(len);

    while (p + 8 <= end) {
        h ^= round(0, read64(p));
        h = rotl64(h, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<std::uint64_t>(read32(p)) * PRIME64_1;
        h = rotl64(h, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<std::uint64_t>(*p) * PRIME64_5;
        h = rotl64(h, 11) * PRIME64_1;
        ++p;
    }

    h ^= h >> 33;
    h *= PRIME64_2;
    h ^= h >> 29;
    h *= PRIME64_3;
    h ^= h >> 32;
    return h;
}

}  // namespace simeon
