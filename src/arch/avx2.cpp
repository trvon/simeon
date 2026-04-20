#include "simeon/simd.hpp"

#if defined(__AVX2__)

#include <immintrin.h>
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
