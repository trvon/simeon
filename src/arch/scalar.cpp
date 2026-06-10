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

void dot4_scalar(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                 float* out4, std::uint32_t n) noexcept {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const double ai = static_cast<double>(a[i]);
        s0 += ai * static_cast<double>(b0[i]);
        s1 += ai * static_cast<double>(b1[i]);
        s2 += ai * static_cast<double>(b2[i]);
        s3 += ai * static_cast<double>(b3[i]);
    }
    out4[0] = static_cast<float>(s0);
    out4[1] = static_cast<float>(s1);
    out4[2] = static_cast<float>(s2);
    out4[3] = static_cast<float>(s3);
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
