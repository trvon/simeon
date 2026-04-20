#include <cassert>
#include <cstdint>
#include <set>
#include <string>

#include "simeon/hasher.hpp"

using simeon::HashFamily;
using simeon::hash64;

namespace simeon {
// Direct test access to the per-family kernels so we can pin canonical
// XXH64 test vectors against the implementation, independent of how
// hash64() composes them.
std::uint64_t xxh64_hash(std::string_view s, std::uint64_t seed) noexcept;
std::uint64_t crc32c_hash(std::string_view s, std::uint64_t seed) noexcept;
}  // namespace simeon

namespace {

void test_determinism() {
    const auto a = hash64("hello world", 0x1234, HashFamily::SplitMix64);
    const auto b = hash64("hello world", 0x1234, HashFamily::SplitMix64);
    assert(a == b);
}

void test_empty() {
    const auto a = hash64("", 0x1234, HashFamily::SplitMix64);
    const auto b = hash64("", 0x1234, HashFamily::SplitMix64);
    assert(a == b);
}

void test_seed_matters() {
    const auto a = hash64("hello", 0x1, HashFamily::SplitMix64);
    const auto b = hash64("hello", 0x2, HashFamily::SplitMix64);
    assert(a != b);
}

void test_distribution_sanity() {
    // Crude sanity: 1024 distinct inputs should produce many distinct hashes (collisions < 10).
    std::set<std::uint64_t> seen;
    for (int i = 0; i < 1024; ++i) {
        const std::string s = "item_" + std::to_string(i);
        seen.insert(hash64(s, 0xA5, HashFamily::SplitMix64));
    }
    assert(seen.size() > 1000);
}

void test_length_sensitivity() {
    const auto a = hash64("abc", 0x1, HashFamily::SplitMix64);
    const auto b = hash64("abcd", 0x1, HashFamily::SplitMix64);
    assert(a != b);
}

void test_prefix_discrimination() {
    const auto a = hash64("abcdefgh", 0x1, HashFamily::SplitMix64);
    const auto b = hash64("abcdefgi", 0x1, HashFamily::SplitMix64);
    assert(a != b);
}

// XXH64 test vectors. The first three are well-known canonical values; the
// remainder are pinned outputs of this implementation, validated against the
// reference spec on aarch64-apple-darwin. If the kernel drifts, these fail
// loudly — which would silently break any persisted index in the wild.
void test_xxh64_kat() {
    using simeon::xxh64_hash;
    // Canonical xxHash test vector — appears verbatim in the xxHash docs.
    assert(xxh64_hash("Nobody inspects the spammish repetition", 0) == 0xFBCEA83C8A378BF1ULL);
    assert(xxh64_hash("", 0) == 0xEF46DB3751D8E999ULL);
    assert(xxh64_hash("", 1) == 0xD5AFBA1336A3BE4BULL);
    // Pinned outputs of this implementation; cross-platform identical
    // because XXH64 is endianness-defined to little-endian and aarch64
    // / x86_64 both honor that via memcpy load.
    assert(xxh64_hash("abc", 0) == 0x44BC2CF5AD770999ULL);
    assert(xxh64_hash(std::string_view("\xab", 1), 0) == 0xC84670DA8B5B5EA8ULL);
    const std::string s14 = std::string(
        "\x9e\xff\x1f\xfb\x8f\x9f\x77\x9d\xee\xf6\xb1\x35\x97\x47", 14);
    assert(xxh64_hash(s14, 0) == 0xDE559473C4AA8821ULL);
}

void test_xxh64_basic_properties() {
    using simeon::xxh64_hash;
    // Determinism.
    assert(xxh64_hash("hello", 1) == xxh64_hash("hello", 1));
    // Seed sensitivity.
    assert(xxh64_hash("hello", 1) != xxh64_hash("hello", 2));
    // Length sensitivity.
    assert(xxh64_hash("abc", 0) != xxh64_hash("abcd", 0));
    // Distinguishes the 32-byte boundary path from the small-input path.
    const std::string under32(31, 'a');
    const std::string over32(33, 'a');
    assert(xxh64_hash(under32, 0) != xxh64_hash(over32, 0));
}

void test_xxh64_distribution_sanity() {
    using simeon::xxh64_hash;
    std::set<std::uint64_t> seen;
    for (int i = 0; i < 1024; ++i) {
        const std::string s = "item_" + std::to_string(i);
        seen.insert(xxh64_hash(s, 0xA5));
    }
    assert(seen.size() > 1000);
}

// CRC32C-as-uint64: we don't pin a CRC32C reference vector here because the
// public hash combines the 32-bit CRC with seed/length via SplitMix64 (so
// the output is not a raw CRC). Test the API contract instead: deterministic,
// seed-sensitive, length-sensitive, well distributed.
void test_crc32c_basic_properties() {
    using simeon::crc32c_hash;
    assert(crc32c_hash("hello", 1) == crc32c_hash("hello", 1));
    assert(crc32c_hash("hello", 1) != crc32c_hash("hello", 2));
    assert(crc32c_hash("abc", 0) != crc32c_hash("abcd", 0));
}

void test_crc32c_distribution_sanity() {
    using simeon::crc32c_hash;
    std::set<std::uint64_t> seen;
    for (int i = 0; i < 1024; ++i) {
        const std::string s = "item_" + std::to_string(i);
        seen.insert(crc32c_hash(s, 0xA5));
    }
    assert(seen.size() > 1000);
}

void test_all_families_route() {
    // hash64() must dispatch each family to a distinct kernel.
    const auto s = hash64("dispatch test", 42, HashFamily::SplitMix64);
    const auto x = hash64("dispatch test", 42, HashFamily::XxHash64);
    const auto c = hash64("dispatch test", 42, HashFamily::Crc32);
    const auto m = hash64("dispatch test", 42, HashFamily::MixedTabulation);
    assert(s != x);
    assert(s != c);
    assert(s != m);
    assert(x != c);
    assert(x != m);
    assert(c != m);
    // And each must be deterministic across calls.
    assert(hash64("dispatch test", 42, HashFamily::SplitMix64) == s);
    assert(hash64("dispatch test", 42, HashFamily::XxHash64) == x);
    assert(hash64("dispatch test", 42, HashFamily::Crc32) == c);
    assert(hash64("dispatch test", 42, HashFamily::MixedTabulation) == m);
}

void test_mixed_tabulation_basic_properties() {
    using F = HashFamily;
    // Determinism.
    assert(hash64("hello", 1, F::MixedTabulation) ==
           hash64("hello", 1, F::MixedTabulation));
    // Seed sensitivity.
    assert(hash64("hello", 1, F::MixedTabulation) !=
           hash64("hello", 2, F::MixedTabulation));
    // Length sensitivity.
    assert(hash64("abc", 0, F::MixedTabulation) !=
           hash64("abcd", 0, F::MixedTabulation));
    // Empty input is well-defined and seed-sensitive.
    assert(hash64("", 0, F::MixedTabulation) !=
           hash64("", 1, F::MixedTabulation));
}

void test_mixed_tabulation_distribution_sanity() {
    std::set<std::uint64_t> seen;
    for (int i = 0; i < 1024; ++i) {
        const std::string s = "item_" + std::to_string(i);
        seen.insert(hash64(s, 0xA5, HashFamily::MixedTabulation));
    }
    // Strong-uniformity hash on 1024 distinct inputs: collisions should be
    // negligible. Allow a tiny slack for birthday.
    assert(seen.size() > 1000);
}

// Mixed Tabulation is meant to be 5-independent across single-byte changes,
// so flipping each byte position in turn must always produce a different
// hash from the original. Stronger than the "distribution sanity" test
// above because it pins per-position avalanche.
void test_mixed_tabulation_avalanche_per_byte() {
    const std::string base(32, 'A');
    const auto base_h = hash64(base, 7, HashFamily::MixedTabulation);
    for (std::size_t i = 0; i < base.size(); ++i) {
        std::string mut = base;
        mut[i] = 'B';
        assert(hash64(mut, 7, HashFamily::MixedTabulation) != base_h);
    }
}

}  // namespace

int main() {
    test_determinism();
    test_empty();
    test_seed_matters();
    test_distribution_sanity();
    test_length_sensitivity();
    test_prefix_discrimination();
    test_xxh64_kat();
    test_xxh64_basic_properties();
    test_xxh64_distribution_sanity();
    test_crc32c_basic_properties();
    test_crc32c_distribution_sanity();
    test_all_families_route();
    test_mixed_tabulation_basic_properties();
    test_mixed_tabulation_distribution_sanity();
    test_mixed_tabulation_avalanche_per_byte();
    return 0;
}
