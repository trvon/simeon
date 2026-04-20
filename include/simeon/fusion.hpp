#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace simeon {

// One ranked input list: (doc_id, score) pairs in any order. RRF only uses
// rank order derived from `score`; the magnitude is not consulted otherwise.
using Ranking = std::span<const std::pair<std::uint32_t, float>>;

// Reciprocal Rank Fusion (Cormack et al. 2009). For each input ranking, sort
// descending by score and assign rank r in [1, |ranking|]. Each doc accumulates
// `1 / (k + r)` across rankings. Returns the merged ranking sorted by fused
// score descending. Tie-break by doc_id ascending for determinism.
//
// Default `k = 60` follows Cormack et al. and is the standard BEIR default.
std::vector<std::pair<std::uint32_t, float>> rrf_fuse(std::span<const Ranking> rankings,
                                                      float k = 60.0f);

// Convenience: convert a per-doc score array into a top-K (doc_id, score)
// ranking sorted descending by score. Tie-break by doc_id ascending.
std::vector<std::pair<std::uint32_t, float>> top_k(std::span<const float> scores, std::uint32_t k);

// Score every doc in `corpus` (row-major, n_docs × dim) against `query` (dim
// floats) using the SIMD-dispatched dot-product kernel, then return the top-K
// docs sorted descending. With L2-normalized inputs the dot is cosine
// similarity. This is the canonical kernel for the dense-rerank step in a
// BM25 → simeon cascade; callers should prefer it over hand-rolled scalar
// loops.
std::vector<std::pair<std::uint32_t, float>> cosine_topk(const float* query, const float* corpus,
                                                         std::uint32_t n_docs, std::uint32_t dim,
                                                         std::uint32_t k);

// As cosine_topk, but writes to a caller-allocated score buffer of length
// n_docs and skips the partial sort. Useful when the caller wants raw per-doc
// scores (e.g. to z-score them inside a BM25 pool before fusing).
void cosine_scores(const float* query, const float* corpus, std::uint32_t n_docs, std::uint32_t dim,
                   float* out_scores) noexcept;

} // namespace simeon
