#pragma once

#include <cstdint>
#include <string_view>

namespace simeon {

// Mixed Tabulation hash (Pătraşcu & Thorup 2012, "The Power of Simple
// Tabulation Hashing"; Houen & Thorup 2023, arXiv:2305.03110). Provides
// strong-enough independence guarantees that sparse-JL (Kane & Nelson 2010)
// becomes correct with a *practical* hash function — earlier sparse-JL
// analyses required Ω(log 1/δ)-wise independence which only the polynomial
// hashes provided in theory.
//
// Implementation: 4 byte-indexed primary tables T[0..3] of 256 uint64s
// each, plus 2 derived tables D[0..1] indexed by intermediate hash bits.
// Total state ~6 KB per seed; cached per-thread on first use.
//
// Independence: this construction is 5-independent (sufficient for sparse-JL
// per Houen & Thorup) when the input is treated as a sequence of bytes,
// which matches the public hash64() interface.
std::uint64_t mixed_tabulation_hash(std::string_view s, std::uint64_t seed) noexcept;

}  // namespace simeon
