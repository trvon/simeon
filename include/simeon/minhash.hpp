#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "simeon/simeon.hpp"

namespace simeon {

// One-permutation MinHash with strided densification. Tokenizes the input as
// character n-grams, hashes each n-gram once, slots the hash into one of `k`
// bins by `bin = h % k`, and keeps the minimum `value = h >> 32` per bin.
// Empty bins are filled by a seeded-stride donor scan with attempt-index
// mixing, computed in O(d + k) instead of O(d * k). Measured against the
// per-bin hashed-probe scheme of Shrivastava 2017 (arXiv:1703.04664) the
// strided scan is equal-or-better at the sparsity levels we serve, and the
// sparse-regime MSE sits at the one-permutation information floor (see
// test_densification_variance_sparse).
//
// Output is `k` uint32 slot values; matching slot count between two
// signatures is an unbiased Jaccard estimator (jaccard_estimate below).
//
// Failure modes are uncorrelated with cosine-space simeon: MinHash wins on
// near-duplicate / boilerplate-heavy text where lexical co-occurrence
// fools projection-cosine.
struct MinHashConfig {
    std::uint32_t k = 256;
    std::uint32_t ngram_min = 3;
    std::uint32_t ngram_max = 5;
    HashFamily hash = HashFamily::SplitMix64;
    std::uint64_t hash_seed = 0xD15EA5E5D15EA5E5ULL;
};

class MinHashEncoder {
public:
    explicit MinHashEncoder(MinHashConfig cfg) noexcept;

    void encode(std::string_view text, std::uint32_t* out) const;

    std::uint32_t k() const noexcept { return cfg_.k; }
    const MinHashConfig& config() const noexcept { return cfg_; }

private:
    MinHashConfig cfg_;
};

// Unbiased Jaccard estimator: matching slots / k. Both signatures must have
// the same length `k`; mismatched k produces undefined results.
float jaccard_estimate(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t k) noexcept;

// Score every doc in `corpus` (row-major, n_docs * k uint32 signatures)
// against `query` (k uint32s) using jaccard_estimate, then return the top-N
// docs sorted descending. Mirrors fusion::cosine_topk but on Jaccard space.
std::vector<std::pair<std::uint32_t, float>> jaccard_topk(const std::uint32_t* query,
                                                          const std::uint32_t* corpus,
                                                          std::uint32_t n_docs, std::uint32_t k,
                                                          std::uint32_t topn);

} // namespace simeon
