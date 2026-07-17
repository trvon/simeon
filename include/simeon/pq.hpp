#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace simeon {

// Product Quantization (Jégou, Douze, Schmid 2010). Splits a `dim`-dimensional
// vector into `m` subvectors of size `dim/m`, each independently quantized to
// one of `k` centroids. Each vector is stored as `m` bytes (assuming k<=256).
//
// Query flow uses asymmetric distance computation (ADC): the query stays in
// full precision, and a per-subspace lookup table of distances to every
// centroid is precomputed once. Database distance is then a sum of `m` table
// lookups — independent of dim — giving a large constant-factor speedup over
// brute-force dense distance.

struct PQConfig {
    std::uint32_t dim = 0; // input vector dimension; must be divisible by m
    std::uint32_t m = 0;   // number of subquantizers
    std::uint32_t k = 256; // centroids per subquantizer; must be <= 256 (fits in a byte)
    std::uint64_t seed = 0xC0FFEE5EED5EEDC0ULL;
};

class ProductQuantizer {
public:
    explicit ProductQuantizer(PQConfig cfg);
    ~ProductQuantizer();

    ProductQuantizer(const ProductQuantizer&) = delete;
    ProductQuantizer& operator=(const ProductQuantizer&) = delete;
    ProductQuantizer(ProductQuantizer&&) noexcept;
    ProductQuantizer& operator=(ProductQuantizer&&) noexcept;

    const PQConfig& config() const noexcept;
    std::uint32_t dim() const noexcept;
    std::uint32_t m() const noexcept;
    std::uint32_t k() const noexcept;
    std::uint32_t dsub() const noexcept; // dim / m

    // Training-free init: centroids are deterministic Gaussian samples per
    // subspace (seeded by `cfg.seed`). Recall is lower than a trained PQ but
    // substantially better than uniform random; use when no representative
    // training corpus is available.
    void init_random_gaussian();

    // Lloyd's k-means per subspace (deterministic, seeded k-means++ init,
    // empty-cluster reseeding). `training` is row-major [n_train * dim].
    // Overwrites any previously installed codebook. Safe to call multiple
    // times. `n_iters` is an upper bound; converges sooner in practice.
    void train(const float* training, std::uint32_t n_train, std::uint32_t n_iters = 25);

    // Encode a single vector from `vec[0..dim())` into `code[0..m())`.
    void encode(const float* vec, std::uint8_t* code) const noexcept;

    // Encode `n` row-major vectors from `vecs[0..n*dim())` into
    // `codes[0..n*m())`.
    void encode_batch(const float* vecs, std::uint32_t n, std::uint8_t* codes) const noexcept;

    // Reconstruct an approximation to the original vector from `code[0..m())`
    // into `vec[0..dim())`.
    void decode(const std::uint8_t* code, float* vec) const noexcept;

    // Raw centroid read-only accessor: returns pointer to `dsub` floats
    // representing centroid `ki` of subspace `mi`.
    const float* centroid(std::uint32_t mi, std::uint32_t ki) const noexcept;

    // Did training install non-default centroids? False if neither
    // init_random_gaussian() nor train() has been called.
    bool is_trained() const noexcept;

    // Flat view over all codebooks as [m * k * dsub] floats.
    std::span<const float> codebooks() const noexcept;

    // Install externally persisted codebooks. The input must contain exactly
    // m * k * dsub floats laid out identically to `codebooks()`.
    void import_codebooks(std::span<const float> codebooks, bool trained = true);

private:
    friend class PQQuery;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Asymmetric distance query. Construct once per query; amortizes a per-query
// lookup table over many database distance evaluations.
class PQQuery {
public:
    // `query` must point to `pq.dim()` floats. The LUT is precomputed in the
    // constructor; afterwards `distance_*` calls are O(m) per database code.
    PQQuery(const ProductQuantizer& pq, const float* query);
    ~PQQuery();

    PQQuery(const PQQuery&) = delete;
    PQQuery& operator=(const PQQuery&) = delete;
    PQQuery(PQQuery&&) noexcept;
    PQQuery& operator=(PQQuery&&) noexcept;

    // Squared L2 distance: sum over m subspaces of ||q_sub - centroid[code[mi]]||^2.
    float distance_l2_sq(const std::uint8_t* code) const noexcept;

    // Inner product: sum over m subspaces of q_sub · centroid[code[mi]].
    // Larger = more similar (for L2-normalized inputs, this is cosine).
    float inner_product(const std::uint8_t* code) const noexcept;

    // Direct access to the per-subspace LUTs (m * k floats each).
    std::span<const float> lut_l2_sq() const noexcept;
    std::span<const float> lut_ip() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Inner-product-only ADC query. Use when the caller never needs squared L2:
// it avoids building and storing the second m*k lookup table while preserving
// PQQuery::inner_product's score semantics.
class PQInnerProductQuery {
public:
    PQInnerProductQuery(const ProductQuantizer& pq, const float* query);
    ~PQInnerProductQuery();

    PQInnerProductQuery(const PQInnerProductQuery&) = delete;
    PQInnerProductQuery& operator=(const PQInnerProductQuery&) = delete;
    PQInnerProductQuery(PQInnerProductQuery&&) noexcept;
    PQInnerProductQuery& operator=(PQInnerProductQuery&&) noexcept;

    float inner_product(const std::uint8_t* code) const noexcept;
    std::span<const float> lut_ip() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace simeon
