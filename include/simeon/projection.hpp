#pragma once

#include <cstdint>
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
    void apply(const std::int32_t* sketch, float* out) const;

    // Densified view of the projection matrix for testing. Returns the entry at (row, col),
    // where row indexes the output dim and col indexes the sketch dim.
    float entry(std::uint32_t row, std::uint32_t col) const;

private:
    // Cached projection matrix. For AchlioptasSparse, weights are only
    // {-scale, 0, +scale} so we split per-row nonzeros into positive/negative
    // index lists and skip the multiply inside the hot loop (just int32
    // sum/sub over sketch indices). VerySparse falls back to (col, weight)
    // pairs because its nonzero magnitude depends on the sampled s value.
    // DenseGaussian uses a row-major float matrix.
    struct AchlioptasRow {
        std::vector<std::uint32_t> pos_cols;
        std::vector<std::uint32_t> neg_cols;
    };
    struct WeightedSparseRow {
        std::vector<std::uint32_t> cols;
        std::vector<float> weights;
    };

    std::uint32_t sketch_dim_;
    std::uint32_t output_dim_;
    ProjectionMode mode_;
    std::uint64_t seed_;
    float inv_scale_ = 1.0f;
    float achlioptas_scale_ = 1.0f;         // cached sqrt(3) * inv_scale for Achlioptas
    std::vector<float> dense_;              // output_dim_ * sketch_dim_ (Gaussian path)
    std::vector<AchlioptasRow> achlioptas_; // per-row pos/neg col lists (Achlioptas)
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
};

} // namespace simeon
