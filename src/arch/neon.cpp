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

void dot4_neon(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
               float* out4, std::uint32_t n) noexcept {
    // Per-output accumulator structure mirrors dot_neon exactly (2 accumulators,
    // 8 floats per iteration, vaddvq reduction, scalar tail) so each lane is
    // bit-identical to an independent dot_neon call.
    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f);
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t a0 = vld1q_f32(a + i);
        const float32x4_t a1 = vld1q_f32(a + i + 4);
        c00 = vfmaq_f32(c00, a0, vld1q_f32(b0 + i));
        c01 = vfmaq_f32(c01, a1, vld1q_f32(b0 + i + 4));
        c10 = vfmaq_f32(c10, a0, vld1q_f32(b1 + i));
        c11 = vfmaq_f32(c11, a1, vld1q_f32(b1 + i + 4));
        c20 = vfmaq_f32(c20, a0, vld1q_f32(b2 + i));
        c21 = vfmaq_f32(c21, a1, vld1q_f32(b2 + i + 4));
        c30 = vfmaq_f32(c30, a0, vld1q_f32(b3 + i));
        c31 = vfmaq_f32(c31, a1, vld1q_f32(b3 + i + 4));
    }
    float s0 = vaddvq_f32(vaddq_f32(c00, c01));
    float s1 = vaddvq_f32(vaddq_f32(c10, c11));
    float s2 = vaddvq_f32(vaddq_f32(c20, c21));
    float s3 = vaddvq_f32(vaddq_f32(c30, c31));
    for (; i < n; ++i) {
        s0 += a[i] * b0[i];
        s1 += a[i] * b1[i];
        s2 += a[i] * b2[i];
        s3 += a[i] * b3[i];
    }
    out4[0] = s0;
    out4[1] = s1;
    out4[2] = s2;
    out4[3] = s3;
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
