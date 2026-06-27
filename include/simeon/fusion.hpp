#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace simeon {

class Bm25Index;

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

// Score `query` against each BM25 variant in `variants`, then RRF-fuse the
// per-variant rankings using the canonical k=60 (Cormack et al. 2009).
// Writes fused RRF scores into `out_scores` (size = nd). Higher score =
// better rank; docs not present in any variant get 0.
//
// Each Bm25Index in `variants` must be finalized and built over the same
// doc set (same nd = out_scores.size()).
//
// Validated cross-fold on nfcorpus vs the production frontier
// (phssapprox_k100_t8_richcov_gap): +0.0089/+0.0045 dev/test nDCG@10,
// +0.0109/+0.0013 R@100. Mixed cross-fold on scifact and fiqa —
// corpus-specific lever, not a universal default. See
// docs/research.md.
void score_bm25_variants_rrf(std::span<const Bm25Index* const> variants, std::string_view query,
                             std::span<float> out_scores, float k_rrf = 60.0f);

// Fixed-weight convex combination of z-normalized score legs (Bruch-Gai 2022).
// Each leg is z-scored over its full span (callers pass pool-restricted
// vectors), then out[i] = Σ_l weights[l] · z(legs[l])[i]. Validated dev→test
// on the WSDM(SAB)+WSDM(Atire) pair; see docs/research.md fusion pass.
void convex_fuse_z(std::span<const std::span<const float>> legs, std::span<const float> weights,
                   std::span<float> out_scores) noexcept;

} // namespace simeon
