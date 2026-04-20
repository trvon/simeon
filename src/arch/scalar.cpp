#include "simeon/simd.hpp"

#include <cmath>

namespace simeon::simd {

float l2_normalize_scalar(float* v, std::uint32_t n) noexcept {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        acc += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    if (acc <= 0.0)
        return 0.0f;
    const float inv = static_cast<float>(1.0 / std::sqrt(acc));
    for (std::uint32_t i = 0; i < n; ++i) {
        v[i] *= inv;
    }
    return inv;
}

float dot_scalar(const float* a, const float* b, std::uint32_t n) noexcept {
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(s);
}

void add_vec_scalar(float* dst, const float* src, std::uint32_t n) noexcept {
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] += src[i];
}

void scale_vec_scalar(float* dst, const float* w, std::uint32_t n) noexcept {
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] *= w[i];
}

void saxpy_scalar(float* dst, const float* src, float alpha, std::uint32_t n) noexcept {
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] += alpha * src[i];
}

} // namespace simeon::simd
