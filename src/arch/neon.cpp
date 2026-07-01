#include "simeon/simd.hpp"

#if defined(__aarch64__)

#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstring>

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
        sum = std::fma(v[i], v[i], sum);
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
        s = std::fma(a[i], b[i], s);
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
        s0 = std::fma(a[i], b0[i], s0);
        s1 = std::fma(a[i], b1[i], s1);
        s2 = std::fma(a[i], b2[i], s2);
        s3 = std::fma(a[i], b3[i], s3);
    }
    out4[0] = s0;
    out4[1] = s1;
    out4[2] = s2;
    out4[3] = s3;
}

void dot2x4_neon(const float* a0, const float* a1, const float* b0, const float* b1,
                 const float* b2, const float* b3, float* out0, float* out1,
                 std::uint32_t n) noexcept {
    // 16 accumulators (2 per output, mirroring dot_neon) + 4 a-row registers;
    // each b load is shared by both a rows. Every output is bit-identical to
    // an independent dot_neon call.
    float32x4_t c000 = vdupq_n_f32(0.0f), c001 = vdupq_n_f32(0.0f);
    float32x4_t c010 = vdupq_n_f32(0.0f), c011 = vdupq_n_f32(0.0f);
    float32x4_t c020 = vdupq_n_f32(0.0f), c021 = vdupq_n_f32(0.0f);
    float32x4_t c030 = vdupq_n_f32(0.0f), c031 = vdupq_n_f32(0.0f);
    float32x4_t c100 = vdupq_n_f32(0.0f), c101 = vdupq_n_f32(0.0f);
    float32x4_t c110 = vdupq_n_f32(0.0f), c111 = vdupq_n_f32(0.0f);
    float32x4_t c120 = vdupq_n_f32(0.0f), c121 = vdupq_n_f32(0.0f);
    float32x4_t c130 = vdupq_n_f32(0.0f), c131 = vdupq_n_f32(0.0f);
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t x00 = vld1q_f32(a0 + i);
        const float32x4_t x01 = vld1q_f32(a0 + i + 4);
        const float32x4_t x10 = vld1q_f32(a1 + i);
        const float32x4_t x11 = vld1q_f32(a1 + i + 4);
        float32x4_t y0 = vld1q_f32(b0 + i);
        float32x4_t y1 = vld1q_f32(b0 + i + 4);
        c000 = vfmaq_f32(c000, x00, y0);
        c001 = vfmaq_f32(c001, x01, y1);
        c100 = vfmaq_f32(c100, x10, y0);
        c101 = vfmaq_f32(c101, x11, y1);
        y0 = vld1q_f32(b1 + i);
        y1 = vld1q_f32(b1 + i + 4);
        c010 = vfmaq_f32(c010, x00, y0);
        c011 = vfmaq_f32(c011, x01, y1);
        c110 = vfmaq_f32(c110, x10, y0);
        c111 = vfmaq_f32(c111, x11, y1);
        y0 = vld1q_f32(b2 + i);
        y1 = vld1q_f32(b2 + i + 4);
        c020 = vfmaq_f32(c020, x00, y0);
        c021 = vfmaq_f32(c021, x01, y1);
        c120 = vfmaq_f32(c120, x10, y0);
        c121 = vfmaq_f32(c121, x11, y1);
        y0 = vld1q_f32(b3 + i);
        y1 = vld1q_f32(b3 + i + 4);
        c030 = vfmaq_f32(c030, x00, y0);
        c031 = vfmaq_f32(c031, x01, y1);
        c130 = vfmaq_f32(c130, x10, y0);
        c131 = vfmaq_f32(c131, x11, y1);
    }
    float s00 = vaddvq_f32(vaddq_f32(c000, c001));
    float s01 = vaddvq_f32(vaddq_f32(c010, c011));
    float s02 = vaddvq_f32(vaddq_f32(c020, c021));
    float s03 = vaddvq_f32(vaddq_f32(c030, c031));
    float s10 = vaddvq_f32(vaddq_f32(c100, c101));
    float s11 = vaddvq_f32(vaddq_f32(c110, c111));
    float s12 = vaddvq_f32(vaddq_f32(c120, c121));
    float s13 = vaddvq_f32(vaddq_f32(c130, c131));
    for (; i < n; ++i) {
        s00 = std::fma(a0[i], b0[i], s00);
        s01 = std::fma(a0[i], b1[i], s01);
        s02 = std::fma(a0[i], b2[i], s02);
        s03 = std::fma(a0[i], b3[i], s03);
        s10 = std::fma(a1[i], b0[i], s10);
        s11 = std::fma(a1[i], b1[i], s11);
        s12 = std::fma(a1[i], b2[i], s12);
        s13 = std::fma(a1[i], b3[i], s13);
    }
    out0[0] = s00;
    out0[1] = s01;
    out0[2] = s02;
    out0[3] = s03;
    out1[0] = s10;
    out1[1] = s11;
    out1[2] = s12;
    out1[3] = s13;
}

void range_neon(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept {
    if (n == 0)
        return;
    if (n < 8) {
        float mn = v[0], mx = v[0];
        for (std::uint32_t i = 1; i < n; ++i) {
            mn = std::min(mn, v[i]);
            mx = std::max(mx, v[i]);
        }
        *out_min = mn;
        *out_max = mx;
        return;
    }
    float32x4_t mn0 = vld1q_f32(v), mn1 = vld1q_f32(v + 4);
    float32x4_t mx0 = mn0, mx1 = mn1;
    std::uint32_t i = 8;
    for (; i + 8 <= n; i += 8) {
        const float32x4_t a = vld1q_f32(v + i);
        const float32x4_t b = vld1q_f32(v + i + 4);
        mn0 = vminq_f32(mn0, a);
        mx0 = vmaxq_f32(mx0, a);
        mn1 = vminq_f32(mn1, b);
        mx1 = vmaxq_f32(mx1, b);
    }
    float mn = vminvq_f32(vminq_f32(mn0, mn1));
    float mx = vmaxvq_f32(vmaxq_f32(mx0, mx1));
    for (; i < n; ++i) {
        mn = std::min(mn, v[i]);
        mx = std::max(mx, v[i]);
    }
    *out_min = mn;
    *out_max = mx;
}

std::uint32_t scan_ge_neon(const float* v, std::uint32_t n, float threshold,
                           std::uint32_t* out) noexcept {
    const float32x4_t vt = vdupq_n_f32(threshold);
    std::uint32_t cnt = 0;
    std::uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const uint32x4_t m = vcgeq_f32(vld1q_f32(v + i), vt);
        if (vmaxvq_u32(m) == 0)
            continue;
        if (vgetq_lane_u32(m, 0))
            out[cnt++] = i;
        if (vgetq_lane_u32(m, 1))
            out[cnt++] = i + 1;
        if (vgetq_lane_u32(m, 2))
            out[cnt++] = i + 2;
        if (vgetq_lane_u32(m, 3))
            out[cnt++] = i + 3;
    }
    for (; i < n; ++i) {
        if (v[i] >= threshold)
            out[cnt++] = i;
    }
    return cnt;
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

void affine_norm_neon(const float* src, const float* mean, const float* std_dev, float* dst,
                      std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(dst + i, vdivq_f32(vsubq_f32(vld1q_f32(src + i), vld1q_f32(mean + i)),
                                     vld1q_f32(std_dev + i)));
        vst1q_f32(dst + i + 4, vdivq_f32(vsubq_f32(vld1q_f32(src + i + 4), vld1q_f32(mean + i + 4)),
                                         vld1q_f32(std_dev + i + 4)));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(dst + i, vdivq_f32(vsubq_f32(vld1q_f32(src + i), vld1q_f32(mean + i)),
                                     vld1q_f32(std_dev + i)));
    }
    for (; i < n; ++i)
        dst[i] = (src[i] - mean[i]) / std_dev[i];
}

void bf16_pack_neon(const float* src, std::uint16_t* dst, std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint32x4_t lo = vreinterpretq_u32_f32(vld1q_f32(src + i));
        const uint32x4_t hi = vreinterpretq_u32_f32(vld1q_f32(src + i + 4));
        vst1q_u16(dst + i, vcombine_u16(vshrn_n_u32(lo, 16), vshrn_n_u32(hi, 16)));
    }
    for (; i < n; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, src + i, sizeof(bits));
        dst[i] = static_cast<std::uint16_t>(bits >> 16);
    }
}

void bf16_unpack_neon(const std::uint16_t* src, float* dst, std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t v = vld1q_u16(src + i);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v), 16)));
    }
    for (; i < n; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
        std::memcpy(dst + i, &bits, sizeof(bits));
    }
}

} // namespace simeon::simd

#endif
