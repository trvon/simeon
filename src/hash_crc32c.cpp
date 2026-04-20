// CRC32C (Castagnoli polynomial 0x1EDC6F41) — produces a 32-bit checksum that
// is then mixed up to 64 bits via SplitMix64 finalization, since the public
// hash64() interface expects uint64_t and feature-hashing needs both an
// index nibble (low 32 bits) and a sign bit (bit 63).
//
// Hardware acceleration:
//   - x86_64 with SSE 4.2: _mm_crc32_u64 / _mm_crc32_u8
//   - aarch64 with __ARM_FEATURE_CRC32 (Apple M-series, Armv8.0+ with +crc):
//     __crc32cd / __crc32cb
//   - Fallback: byte-at-a-time table-driven (slice-by-1, ~1KB table).

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "simeon/hasher.hpp"  // for splitmix64_mix

#if defined(__SSE4_2__)
#include <nmmintrin.h>
#endif
#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace simeon {

namespace {

// Reversed Castagnoli polynomial.
constexpr std::uint32_t POLY = 0x82F63B78u;

constexpr std::array<std::uint32_t, 256> make_crc32c_table() {
    std::array<std::uint32_t, 256> t{};
    for (int i = 0; i < 256; ++i) {
        std::uint32_t c = static_cast<std::uint32_t>(i);
        for (int j = 0; j < 8; ++j) {
            c = (c >> 1) ^ ((c & 1u) ? POLY : 0u);
        }
        t[i] = c;
    }
    return t;
}

constexpr std::array<std::uint32_t, 256> CRC32C_TABLE = make_crc32c_table();

[[maybe_unused]] std::uint32_t crc32c_software(std::uint32_t crc, const std::uint8_t* p,
                                               std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        crc = CRC32C_TABLE[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc;
}

#if defined(__SSE4_2__)
std::uint32_t crc32c_hw(std::uint32_t crc, const std::uint8_t* p, std::size_t n) noexcept {
    while (n >= 8) {
        std::uint64_t chunk;
        std::memcpy(&chunk, p, 8);
        crc = static_cast<std::uint32_t>(_mm_crc32_u64(crc, chunk));
        p += 8;
        n -= 8;
    }
    while (n--) {
        crc = _mm_crc32_u8(crc, *p++);
    }
    return crc;
}
#elif defined(__ARM_FEATURE_CRC32)
std::uint32_t crc32c_hw(std::uint32_t crc, const std::uint8_t* p, std::size_t n) noexcept {
    while (n >= 8) {
        std::uint64_t chunk;
        std::memcpy(&chunk, p, 8);
        crc = __crc32cd(crc, chunk);
        p += 8;
        n -= 8;
    }
    while (n--) {
        crc = __crc32cb(crc, *p++);
    }
    return crc;
}
#endif

std::uint32_t crc32c(std::uint32_t crc_in, const std::uint8_t* p, std::size_t n) noexcept {
    std::uint32_t crc = ~crc_in;
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    crc = crc32c_hw(crc, p, n);
#else
    crc = crc32c_software(crc, p, n);
#endif
    return ~crc;
}

}  // namespace

std::uint64_t crc32c_hash(std::string_view s, std::uint64_t seed) noexcept {
    const auto* p = reinterpret_cast<const std::uint8_t*>(s.data());
    const std::uint32_t crc = crc32c(static_cast<std::uint32_t>(seed), p, s.size());
    // Expand 32-bit checksum to 64 bits via SplitMix64 finalization, mixing
    // in the high 32 bits of the seed and the length so different seeds /
    // lengths yield different 64-bit hashes even when the CRC collides.
    const std::uint64_t packed = (static_cast<std::uint64_t>(crc) << 32) ^ crc;
    return splitmix64_mix(packed ^ seed ^ (static_cast<std::uint64_t>(s.size()) << 24));
}

}  // namespace simeon
