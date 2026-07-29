#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

#include "simeon/simeon.hpp"

namespace simeon::simd {

// The tier-specific pointer functions below are unchecked hot kernels. For
// non-zero n, every input and output pointer must address the documented
// number of elements. Debug builds assert these contracts at kernel entry.
// Library consumers should prefer the sized dispatcher overloads at the end
// of this header; they validate once before entering a selected kernel.

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
            // needs; two dot4 passes are bit-identical and spill-free. A 1x8
            // tile (16 accs + a + b = 18 regs) spills too — under the
            // 2-accumulators-per-output parity contract, 1x4 is the widest
            // shared-b tile that fits this register file.
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

namespace detail {

inline std::uint32_t checked_count(std::size_t size) {
    if (size > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("SIMD span size exceeds uint32 capacity");
    return static_cast<std::uint32_t>(size);
}

inline void require_size(std::size_t actual, std::size_t expected) {
    if (actual != expected)
        throw std::invalid_argument("SIMD span sizes must match");
}

inline void require_output4(std::span<float> output) {
    if (output.size() != 4)
        throw std::invalid_argument("SIMD blocked-dot output span must contain four floats");
}

} // namespace detail

inline float l2_normalize(std::span<float> values) {
    return l2_normalize(values.data(), detail::checked_count(values.size()));
}

inline float dot(std::span<const float> a, std::span<const float> b) {
    detail::require_size(b.size(), a.size());
    return dot(a.data(), b.data(), detail::checked_count(a.size()));
}

inline void dot4(std::span<const float> a, std::span<const float> b0, std::span<const float> b1,
                 std::span<const float> b2, std::span<const float> b3, std::span<float> output) {
    detail::require_size(b0.size(), a.size());
    detail::require_size(b1.size(), a.size());
    detail::require_size(b2.size(), a.size());
    detail::require_size(b3.size(), a.size());
    detail::require_output4(output);
    dot4(a.data(), b0.data(), b1.data(), b2.data(), b3.data(), output.data(),
         detail::checked_count(a.size()));
}

inline void dot2x4(std::span<const float> a0, std::span<const float> a1, std::span<const float> b0,
                   std::span<const float> b1, std::span<const float> b2, std::span<const float> b3,
                   std::span<float> out0, std::span<float> out1) {
    detail::require_size(a1.size(), a0.size());
    detail::require_size(b0.size(), a0.size());
    detail::require_size(b1.size(), a0.size());
    detail::require_size(b2.size(), a0.size());
    detail::require_size(b3.size(), a0.size());
    detail::require_output4(out0);
    detail::require_output4(out1);
    dot2x4(a0.data(), a1.data(), b0.data(), b1.data(), b2.data(), b3.data(), out0.data(),
           out1.data(), detail::checked_count(a0.size()));
}

inline void range(std::span<const float> values, float& out_min, float& out_max) {
    range(values.data(), detail::checked_count(values.size()), &out_min, &out_max);
}

inline std::uint32_t scan_ge(std::span<const float> values, float threshold,
                             std::span<std::uint32_t> output) {
    if (output.size() < values.size())
        throw std::invalid_argument("SIMD threshold output span is smaller than input");
    return scan_ge(values.data(), detail::checked_count(values.size()), threshold, output.data());
}

inline void add_vec(std::span<float> destination, std::span<const float> source) {
    detail::require_size(source.size(), destination.size());
    add_vec(destination.data(), source.data(), detail::checked_count(destination.size()));
}

inline void scale_vec(std::span<float> destination, std::span<const float> weights) {
    detail::require_size(weights.size(), destination.size());
    scale_vec(destination.data(), weights.data(), detail::checked_count(destination.size()));
}

inline void saxpy(std::span<float> destination, std::span<const float> source, float alpha) {
    detail::require_size(source.size(), destination.size());
    saxpy(destination.data(), source.data(), alpha, detail::checked_count(destination.size()));
}

inline void affine_norm(std::span<const float> source, std::span<const float> mean,
                        std::span<const float> standard_deviation, std::span<float> destination) {
    detail::require_size(mean.size(), source.size());
    detail::require_size(standard_deviation.size(), source.size());
    detail::require_size(destination.size(), source.size());
    affine_norm(source.data(), mean.data(), standard_deviation.data(), destination.data(),
                detail::checked_count(source.size()));
}

inline void bf16_pack(std::span<const float> source, std::span<std::uint16_t> destination) {
    detail::require_size(destination.size(), source.size());
    bf16_pack(source.data(), destination.data(), detail::checked_count(source.size()));
}

inline void bf16_unpack(std::span<const std::uint16_t> source, std::span<float> destination) {
    detail::require_size(destination.size(), source.size());
    bf16_unpack(source.data(), destination.data(), detail::checked_count(source.size()));
}

} // namespace simeon::simd
