#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "simeon/simeon.hpp"

namespace simeon {

class Projection {
public:
    Projection(std::uint32_t sketch_dim, std::uint32_t output_dim, ProjectionMode mode,
               std::uint64_t seed, float sparse_jl_eps = 0.10f);
    ~Projection();

    Projection(const Projection&) = delete;
    Projection& operator=(const Projection&) = delete;
    Projection(Projection&&) noexcept = default;
    Projection& operator=(Projection&&) noexcept = default;

    std::uint32_t sketch_dim() const noexcept { return sketch_dim_; }
    std::uint32_t output_dim() const noexcept { return output_dim_; }
    ProjectionMode mode() const noexcept { return mode_; }

    // Compute out[0..output_dim_) = P * sketch.
    // sketch has sketch_dim_ int32 entries; out has output_dim_ float entries.
    // Throws std::invalid_argument when either pointer is null.
    void apply(const std::int32_t* sketch, float* out) const;

    // Sized overload. Both spans must exactly match the configured dimensions.
    void apply(std::span<const std::int32_t> sketch, std::span<float> out) const;

    // Densified view of the projection matrix for testing. Returns the entry at (row, col),
    // where row indexes the output dim and col indexes the sketch dim.
    float entry(std::uint32_t row, std::uint32_t col) const;

private:
    // Cached projection matrix. For AchlioptasSparse, weights are only
    // {-scale, 0, +scale}, so apply() accumulates exact int64 sums without a
    // multiply. It keeps two layouts of the same +/-1 entries: row-major
    // CSR (gather; fastest when most sketch buckets are non-zero, i.e. long
    // texts) and a column-major CSR (scatter; visits only non-zero buckets,
    // i.e. short texts). Both are flat arrays so traversal stays sequential. VerySparse falls back
    // to (col, weight) pairs because its nonzero magnitude depends on the sampled s value.
    // DenseGaussian uses a row-major float matrix.
    struct WeightedSparseRow {
        std::vector<std::uint32_t> cols;
        std::vector<float> weights;
    };

    std::uint32_t sketch_dim_;
    std::uint32_t output_dim_;
    ProjectionMode mode_;
    std::uint64_t seed_;
    float inv_scale_ = 1.0f;
    float achlioptas_scale_ = 1.0f; // cached sqrt(3) * inv_scale for Achlioptas
    std::vector<float> dense_;      // output_dim_ * sketch_dim_ (Gaussian path)
    // Achlioptas row r: columns [row_begin[r], row_neg[r]) are +1 and
    // [row_neg[r], row_begin[r + 1]) are -1 in achlioptas_cols_.
    std::vector<std::uint32_t> achlioptas_row_begin_; // output_dim_ + 1
    std::vector<std::uint32_t> achlioptas_row_neg_;   // output_dim_
    std::vector<std::uint32_t> achlioptas_cols_;
    // Achlioptas column c: rows [col_begin[c], col_neg[c]) are +1 and
    // [col_neg[c], col_begin[c + 1]) are -1 in achlioptas_rows_.
    std::vector<std::uint32_t> achlioptas_col_begin_; // sketch_dim_ + 1
    std::vector<std::uint32_t> achlioptas_col_neg_;   // sketch_dim_
    std::vector<std::uint32_t> achlioptas_rows_;
    std::vector<WeightedSparseRow> sparse_; // per-row nonzeros (VerySparse)

    // Fwht-only state. pad_n_ is the next power of 2 ≥ sketch_dim_; signs_
    // holds the random ±1 diagonal D over the padded space; sample_ holds
    // the output_dim_ row indices subsampled from [0, pad_n_) with scale
    // sqrt(pad_n_/output_dim_). dense_ is materialized via the closed-form
    // entry expression P[r, c] = D[c] * (-1)^popcount(sample_[r] & c) /
    // sqrt(output_dim_) for c in [0, sketch_dim_).
    std::uint32_t pad_n_ = 0;
    std::vector<float> signs_;
    std::vector<std::uint32_t> sample_;
    float fwht_scale_ = 1.0f; // 1/sqrt(output_dim_)

    void applyUnchecked(std::span<const std::int32_t> sketch, std::span<float> out) const;
};

} // namespace simeon
