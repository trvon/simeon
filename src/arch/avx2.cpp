#include "simeon/simd.hpp"

#if defined(__AVX2__)

#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace simeon::simd {

float l2_normalize_avx2(float* v, std::uint32_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a = _mm256_loadu_ps(v + i);
        __m256 b = _mm256_loadu_ps(v + i + 8);
        acc0 = _mm256_fmadd_ps(a, a, acc0);
        acc1 = _mm256_fmadd_ps(b, b, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i)
        sum += v[i] * v[i];
    if (sum <= 0.0f)
        return 0.0f;

    const float inv = 1.0f / std::sqrt(sum);
    const __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), vinv));
    }
    for (; i < n; ++i)
        v[i] *= inv;
    return inv;
}

float dot_avx2(const float* a, const float* b, std::uint32_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
    }
    __m256 acc = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

void dot4_avx2(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
               float* out4, std::uint32_t n) noexcept {
    // Per-output accumulator structure mirrors dot_avx2 exactly (2 accumulators,
    // 16 floats per iteration, 8-lane scalar reduction, scalar tail) so each
    // output is bit-identical to an independent dot_avx2 call.
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 a0 = _mm256_loadu_ps(a + i);
        const __m256 a1 = _mm256_loadu_ps(a + i + 8);
        c00 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b0 + i), c00);
        c01 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b0 + i + 8), c01);
        c10 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b1 + i), c10);
        c11 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b1 + i + 8), c11);
        c20 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b2 + i), c20);
        c21 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b2 + i + 8), c21);
        c30 = _mm256_fmadd_ps(a0, _mm256_loadu_ps(b3 + i), c30);
        c31 = _mm256_fmadd_ps(a1, _mm256_loadu_ps(b3 + i + 8), c31);
    }
    const float* bs[4] = {b0, b1, b2, b3};
    const __m256 accs[4] = {_mm256_add_ps(c00, c01), _mm256_add_ps(c10, c11),
                            _mm256_add_ps(c20, c21), _mm256_add_ps(c30, c31)};
    for (int k = 0; k < 4; ++k) {
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, accs[k]);
        float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        for (std::uint32_t t = i; t < n; ++t)
            s += a[t] * bs[k][t];
        out4[k] = s;
    }
}

void range_avx2(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept {
    if (n == 0)
        return;
    if (n < 16) {
        float mn = v[0], mx = v[0];
        for (std::uint32_t i = 1; i < n; ++i) {
            mn = std::min(mn, v[i]);
            mx = std::max(mx, v[i]);
        }
        *out_min = mn;
        *out_max = mx;
        return;
    }
    __m256 mn0 = _mm256_loadu_ps(v), mn1 = _mm256_loadu_ps(v + 8);
    __m256 mx0 = mn0, mx1 = mn1;
    std::uint32_t i = 16;
    for (; i + 16 <= n; i += 16) {
        const __m256 a = _mm256_loadu_ps(v + i);
        const __m256 b = _mm256_loadu_ps(v + i + 8);
        mn0 = _mm256_min_ps(mn0, a);
        mx0 = _mm256_max_ps(mx0, a);
        mn1 = _mm256_min_ps(mn1, b);
        mx1 = _mm256_max_ps(mx1, b);
    }
    alignas(32) float tmn[8], tmx[8];
    _mm256_store_ps(tmn, _mm256_min_ps(mn0, mn1));
    _mm256_store_ps(tmx, _mm256_max_ps(mx0, mx1));
    float mn = tmn[0], mx = tmx[0];
    for (int k = 1; k < 8; ++k) {
        mn = std::min(mn, tmn[k]);
        mx = std::max(mx, tmx[k]);
    }
    for (; i < n; ++i) {
        mn = std::min(mn, v[i]);
        mx = std::max(mx, v[i]);
    }
    *out_min = mn;
    *out_max = mx;
}

std::uint32_t scan_ge_avx2(const float* v, std::uint32_t n, float threshold,
                           std::uint32_t* out) noexcept {
    const __m256 vt = _mm256_set1_ps(threshold);
    std::uint32_t cnt = 0;
    std::uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(v + i);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(x, vt, _CMP_GE_OQ));
        while (mask != 0) {
            const int lane = __builtin_ctz(static_cast<unsigned>(mask));
            out[cnt++] = i + static_cast<std::uint32_t>(lane);
            mask &= mask - 1;
        }
    }
    for (; i < n; ++i) {
        if (v[i] >= threshold)
            out[cnt++] = i;
    }
    return cnt;
}

void add_vec_avx2(float* dst, const float* src, std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(dst + i,
                         _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i + 8,
                         _mm256_add_ps(_mm256_loadu_ps(dst + i + 8), _mm256_loadu_ps(src + i + 8)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
    for (; i < n; ++i)
        dst[i] += src[i];
}

void scale_vec_avx2(float* dst, const float* w, std::uint32_t n) noexcept {
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(w + i)));
        _mm256_storeu_ps(dst + i + 8,
                         _mm256_mul_ps(_mm256_loadu_ps(dst + i + 8), _mm256_loadu_ps(w + i + 8)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(w + i)));
    }
    for (; i < n; ++i)
        dst[i] *= w[i];
}

void saxpy_avx2(float* dst, const float* src, float alpha, std::uint32_t n) noexcept {
    const __m256 va = _mm256_set1_ps(alpha);
    std::uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), va, _mm256_loadu_ps(dst + i)));
        _mm256_storeu_ps(dst + i + 8, _mm256_fmadd_ps(_mm256_loadu_ps(src + i + 8), va,
                                                      _mm256_loadu_ps(dst + i + 8)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst + i,
                         _mm256_fmadd_ps(_mm256_loadu_ps(src + i), va, _mm256_loadu_ps(dst + i)));
    }
    for (; i < n; ++i)
        dst[i] += alpha * src[i];
}

} // namespace simeon::simd

#endif
