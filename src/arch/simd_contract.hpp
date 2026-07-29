#pragma once

#include <cassert>
#include <cstdint>

namespace simeon::simd::detail {

inline void debug_assert_buffer(const void* pointer, std::uint32_t count) noexcept {
#if defined(NDEBUG)
    (void)pointer;
    (void)count;
#else
    assert(pointer != nullptr || count == 0);
#endif
}

inline void debug_assert_required(const void* pointer) noexcept {
#if defined(NDEBUG)
    (void)pointer;
#else
    assert(pointer != nullptr);
#endif
}

} // namespace simeon::simd::detail
