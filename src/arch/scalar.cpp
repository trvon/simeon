#include "simeon/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "simd_contract.hpp"

namespace simeon::simd {

float l2_normalize_scalar(float* v, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(v, n);
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
    detail::debug_assert_buffer(a, n);
    detail::debug_assert_buffer(b, n);
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(s);
}

void dot4_scalar(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                 float* out4, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(a, n);
    detail::debug_assert_buffer(b0, n);
    detail::debug_assert_buffer(b1, n);
    detail::debug_assert_buffer(b2, n);
    detail::debug_assert_buffer(b3, n);
    detail::debug_assert_required(out4);
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

void dot2x4_scalar(const float* a0, const float* a1, const float* b0, const float* b1,
                   const float* b2, const float* b3, float* out0, float* out1,
                   std::uint32_t n) noexcept {
    detail::debug_assert_buffer(a0, n);
    detail::debug_assert_buffer(a1, n);
    detail::debug_assert_buffer(b0, n);
    detail::debug_assert_buffer(b1, n);
    detail::debug_assert_buffer(b2, n);
    detail::debug_assert_buffer(b3, n);
    detail::debug_assert_required(out0);
    detail::debug_assert_required(out1);
    dot4_scalar(a0, b0, b1, b2, b3, out0, n);
    dot4_scalar(a1, b0, b1, b2, b3, out1, n);
}

void range_scalar(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept {
    detail::debug_assert_buffer(v, n);
    detail::debug_assert_buffer(out_min, n);
    detail::debug_assert_buffer(out_max, n);
    if (n == 0)
        return;
    // Four independent chains to break the sequential min/max dependency;
    // associativity makes the combined result identical to a single scan.
    float mn0 = v[0], mn1 = v[0], mn2 = v[0], mn3 = v[0];
    float mx0 = v[0], mx1 = v[0], mx2 = v[0], mx3 = v[0];
    std::uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        mn0 = std::min(mn0, v[i]);
        mx0 = std::max(mx0, v[i]);
        mn1 = std::min(mn1, v[i + 1]);
        mx1 = std::max(mx1, v[i + 1]);
        mn2 = std::min(mn2, v[i + 2]);
        mx2 = std::max(mx2, v[i + 2]);
        mn3 = std::min(mn3, v[i + 3]);
        mx3 = std::max(mx3, v[i + 3]);
    }
    for (; i < n; ++i) {
        mn0 = std::min(mn0, v[i]);
        mx0 = std::max(mx0, v[i]);
    }
    *out_min = std::min(std::min(mn0, mn1), std::min(mn2, mn3));
    *out_max = std::max(std::max(mx0, mx1), std::max(mx2, mx3));
}

std::uint32_t scan_ge_scalar(const float* v, std::uint32_t n, float threshold,
                             std::uint32_t* out) noexcept {
    detail::debug_assert_buffer(v, n);
    detail::debug_assert_buffer(out, n);
    std::uint32_t cnt = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (v[i] >= threshold)
            out[cnt++] = i;
    }
    return cnt;
}

void add_vec_scalar(float* dst, const float* src, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(dst, n);
    detail::debug_assert_buffer(src, n);
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] += src[i];
}

void scale_vec_scalar(float* dst, const float* w, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(dst, n);
    detail::debug_assert_buffer(w, n);
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] *= w[i];
}

void saxpy_scalar(float* dst, const float* src, float alpha, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(dst, n);
    detail::debug_assert_buffer(src, n);
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] += alpha * src[i];
}

void affine_norm_scalar(const float* src, const float* mean, const float* std_dev, float* dst,
                        std::uint32_t n) noexcept {
    detail::debug_assert_buffer(src, n);
    detail::debug_assert_buffer(mean, n);
    detail::debug_assert_buffer(std_dev, n);
    detail::debug_assert_buffer(dst, n);
    for (std::uint32_t i = 0; i < n; ++i)
        dst[i] = (src[i] - mean[i]) / std_dev[i];
}

void bf16_pack_scalar(const float* src, std::uint16_t* dst, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(src, n);
    detail::debug_assert_buffer(dst, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, src + i, sizeof(bits));
        dst[i] = static_cast<std::uint16_t>(bits >> 16);
    }
}

void bf16_unpack_scalar(const std::uint16_t* src, float* dst, std::uint32_t n) noexcept {
    detail::debug_assert_buffer(src, n);
    detail::debug_assert_buffer(dst, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
        std::memcpy(dst + i, &bits, sizeof(bits));
    }
}

} // namespace simeon::simd
