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

// Blocked inner product: one row `a` against four rows `b0..b3`, writing
// out4[0..3]. Each output keeps the accumulator structure of dot_*, so results
// are bit-identical to four independent dot() calls; the win is amortizing the
// `a` loads and instruction-level parallelism in pairwise similarity loops.
void dot4_scalar(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                 float* out4, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void dot4_neon(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
               float* out4, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void dot4_avx2(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
               float* out4, std::uint32_t n) noexcept;
#endif

// 2x4 blocked inner product: rows a0, a1 against rows b0..b3, writing
// out0[0..3] (a0·b) and out1[0..3] (a1·b). Same bit-identical-to-dot contract
// as dot4; reusing the b loads across both a rows roughly halves load traffic
// in pairwise loops. AVX2 dispatches to two dot4 calls (register budget).
void dot2x4_scalar(const float* a0, const float* a1, const float* b0, const float* b1,
                   const float* b2, const float* b3, float* out0, float* out1,
                   std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void dot2x4_neon(const float* a0, const float* a1, const float* b0, const float* b1,
                 const float* b2, const float* b3, float* out0, float* out1,
                 std::uint32_t n) noexcept;
#endif

// Min/max over v[0..n). min/max are associative and commutative, so the
// lane-parallel reduction returns the same values as a sequential scan
// (NaN-free inputs assumed, as elsewhere). No-op when n == 0.
void range_scalar(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept;

#if defined(SIMEON_HAS_NEON)
void range_neon(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void range_avx2(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept;
#endif

// Sparse threshold scan: write the indices of every v[i] >= threshold into
// out (caller provides capacity n) and return the count. Selection matches the
// scalar `>=` exactly on every tier; used by sparse graph construction where
// survivor density is low.
std::uint32_t scan_ge_scalar(const float* v, std::uint32_t n, float threshold,
                             std::uint32_t* out) noexcept;

#if defined(SIMEON_HAS_NEON)
std::uint32_t scan_ge_neon(const float* v, std::uint32_t n, float threshold,
                           std::uint32_t* out) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
std::uint32_t scan_ge_avx2(const float* v, std::uint32_t n, float threshold,
                           std::uint32_t* out) noexcept;
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

// dst[i] = (src[i] - mean[i]) / std[i]. Whitening apply step. Subtraction and
// IEEE division round identically lane-wise and scalar, so tiers are
// bit-identical.
void affine_norm_scalar(const float* src, const float* mean, const float* std_dev, float* dst,
                        std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void affine_norm_neon(const float* src, const float* mean, const float* std_dev, float* dst,
                      std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void affine_norm_avx2(const float* src, const float* mean, const float* std_dev, float* dst,
                      std::uint32_t n) noexcept;
#endif

// BF16 pack/unpack: truncating float32 -> bfloat16 (drop low 16 mantissa bits)
// and the exact inverse widening. Pure bit moves — every tier is bit-identical.
void bf16_pack_scalar(const float* src, std::uint16_t* dst, std::uint32_t n) noexcept;
void bf16_unpack_scalar(const std::uint16_t* src, float* dst, std::uint32_t n) noexcept;

#if defined(SIMEON_HAS_NEON)
void bf16_pack_neon(const float* src, std::uint16_t* dst, std::uint32_t n) noexcept;
void bf16_unpack_neon(const std::uint16_t* src, float* dst, std::uint32_t n) noexcept;
#endif

#if defined(SIMEON_HAS_AVX2)
void bf16_pack_avx2(const float* src, std::uint16_t* dst, std::uint32_t n) noexcept;
void bf16_unpack_avx2(const std::uint16_t* src, float* dst, std::uint32_t n) noexcept;
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

inline void dot4(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                 float* out4, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            dot4_neon(a, b0, b1, b2, b3, out4, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            dot4_avx2(a, b0, b1, b2, b3, out4, n);
            return;
#endif
        default:
            dot4_scalar(a, b0, b1, b2, b3, out4, n);
            return;
    }
}

inline void dot2x4(const float* a0, const float* a1, const float* b0, const float* b1,
                   const float* b2, const float* b3, float* out0, float* out1,
                   std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            dot2x4_neon(a0, a1, b0, b1, b2, b3, out0, out1, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            // 16 ymm registers can't hold the 16 accumulators a true 2x4 tile
            // needs; two dot4 passes are bit-identical and spill-free.
            dot4_avx2(a0, b0, b1, b2, b3, out0, n);
            dot4_avx2(a1, b0, b1, b2, b3, out1, n);
            return;
#endif
        default:
            dot2x4_scalar(a0, a1, b0, b1, b2, b3, out0, out1, n);
            return;
    }
}

inline void range(const float* v, std::uint32_t n, float* out_min, float* out_max) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            range_neon(v, n, out_min, out_max);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            range_avx2(v, n, out_min, out_max);
            return;
#endif
        default:
            range_scalar(v, n, out_min, out_max);
            return;
    }
}

inline std::uint32_t scan_ge(const float* v, std::uint32_t n, float threshold,
                             std::uint32_t* out) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            return scan_ge_neon(v, n, threshold, out);
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            return scan_ge_avx2(v, n, threshold, out);
#endif
        default:
            return scan_ge_scalar(v, n, threshold, out);
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

inline void affine_norm(const float* src, const float* mean, const float* std_dev, float* dst,
                        std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            affine_norm_neon(src, mean, std_dev, dst, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            affine_norm_avx2(src, mean, std_dev, dst, n);
            return;
#endif
        default:
            affine_norm_scalar(src, mean, std_dev, dst, n);
            return;
    }
}

inline void bf16_pack(const float* src, std::uint16_t* dst, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            bf16_pack_neon(src, dst, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            bf16_pack_avx2(src, dst, n);
            return;
#endif
        default:
            bf16_pack_scalar(src, dst, n);
            return;
    }
}

inline void bf16_unpack(const std::uint16_t* src, float* dst, std::uint32_t n) noexcept {
    SimdTier tier = active_simd_tier();
    switch (tier) {
#if defined(SIMEON_HAS_NEON)
        case SimdTier::Neon:
            bf16_unpack_neon(src, dst, n);
            return;
#endif
#if defined(SIMEON_HAS_AVX2)
        case SimdTier::Avx2:
            bf16_unpack_avx2(src, dst, n);
            return;
#endif
        default:
            bf16_unpack_scalar(src, dst, n);
            return;
    }
}

} // namespace simeon::simd
