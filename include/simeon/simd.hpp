#pragma once

#include <cstdint>

#include "simeon/simeon.hpp"

namespace simeon::simd {

// L2-normalize in place. Returns the inverse norm actually applied, or 0 if the input was zero.
float l2_normalize_scalar(float* v, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
float l2_normalize_neon(float* v, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
float l2_normalize_avx2(float* v, std::uint32_t n) noexcept;
#endif

// Inner product. With L2-normalized inputs this is cosine similarity, which is
// the dominant per-query cost in dense rerank loops. Scalar accumulates in
// double for parity with the existing bench helper; SIMD variants accumulate
// in the native vector lane type.
float dot_scalar(const float* a, const float* b, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
float dot_neon(const float* a, const float* b, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
float dot_avx2(const float* a, const float* b, std::uint32_t n) noexcept;
#endif

// dst[i] += src[i]. Used on the PMI-sum encode path (once per in-vocab token
// per doc) and as an accumulate building block elsewhere.
void add_vec_scalar(float* dst, const float* src, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void add_vec_neon(float* dst, const float* src, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void add_vec_avx2(float* dst, const float* src, std::uint32_t n) noexcept;
#endif

// dst[i] *= w[i]. Used for matryoshka weighting (once per doc over output_dim).
void scale_vec_scalar(float* dst, const float* w, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void scale_vec_neon(float* dst, const float* w, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void scale_vec_avx2(float* dst, const float* w, std::uint32_t n) noexcept;
#endif

// dst[i] += alpha * src[i]. Generic building block used in the PMI-learn QR
// orthogonalization and available for future linear-combination needs.
void saxpy_scalar(float* dst, const float* src, float alpha, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void saxpy_neon(float* dst, const float* src, float alpha, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void saxpy_avx2(float* dst, const float* src, float alpha, std::uint32_t n) noexcept;
#endif

// Runtime-selected dispatcher.
inline float l2_normalize(float* v, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            return l2_normalize_neon(v, n);
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            return l2_normalize_avx2(v, n);
#endif
        default:
            return l2_normalize_scalar(v, n);
    }
}

inline float dot(const float* a, const float* b, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            return dot_neon(a, b, n);
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            return dot_avx2(a, b, n);
#endif
        default:
            return dot_scalar(a, b, n);
    }
}

inline void add_vec(float* dst, const float* src, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            add_vec_neon(dst, src, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            add_vec_avx2(dst, src, n);
            return;
#endif
        default:
            add_vec_scalar(dst, src, n);
            return;
    }
}

inline void scale_vec(float* dst, const float* w, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            scale_vec_neon(dst, w, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            scale_vec_avx2(dst, w, n);
            return;
#endif
        default:
            scale_vec_scalar(dst, w, n);
            return;
    }
}

inline void saxpy(float* dst, const float* src, float alpha, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            saxpy_neon(dst, src, alpha, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            saxpy_avx2(dst, src, alpha, n);
            return;
#endif
        default:
            saxpy_scalar(dst, src, alpha, n);
            return;
    }
}

} // namespace simeon::simd
