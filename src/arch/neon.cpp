#include "simeon/simd.hpp"

#if defined(__aarch64__)

#include <arm_neon.h>
#include <cmath>

namespace simeon::simd {

float l2_normalize_neon(float* v, std::uint32_t n) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a = vld1q_f32(v + i);
        float32x4_t b = vld1q_f32(v + i + 4);
        acc0 = vfmaq_f32(acc0, a, a);
        acc1 = vfmaq_f32(acc1, b, b);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i)
        sum += v[i] * v[i];
    if (sum <= 0.0f)
        return 0.0f;

    const float inv = 1.0f / std::sqrt(sum);
    const float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(v + i, vmulq_f32(vld1q_f32(v + i), vinv));
    }
    for (; i < n; ++i)
        v[i] *= inv;
    return inv;
}

float dot_neon(const float* a, const float* b, std::uint32_t n) noexcept {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

void add_vec_neon(float* dst, const float* src, std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
        vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(dst + i + 4), vld1q_f32(src + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vld1q_f32(src + i)));
    }
    for (; i < n; ++i)
        dst[i] += src[i];
}

void scale_vec_neon(float* dst, const float* w, std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(dst + i), vld1q_f32(w + i)));
        vst1q_f32(dst + i + 4, vmulq_f32(vld1q_f32(dst + i + 4), vld1q_f32(w + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(dst + i), vld1q_f32(w + i)));
    }
    for (; i < n; ++i)
        dst[i] *= w[i];
}

void saxpy_neon(float* dst, const float* src, float alpha, std::uint32_t n) noexcept {
    const float32x4_t va = vdupq_n_f32(alpha);
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vld1q_f32(src + i), va));
        vst1q_f32(dst + i + 4, vfmaq_f32(vld1q_f32(dst + i + 4), vld1q_f32(src + i + 4), va));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vld1q_f32(src + i), va));
    }
    for (; i < n; ++i)
        dst[i] += alpha * src[i];
}

} // namespace simeon::simd

#endif
