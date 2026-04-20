#pragma once

#include <cstdint>
#include <string_view>

#include "simeon/simeon.hpp"

namespace simeon {

std::uint64_t hash64(std::string_view s, std::uint64_t seed, HashFamily family) noexcept;

constexpr std::uint64_t splitmix64_mix(std::uint64_t z) noexcept {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace simeon
