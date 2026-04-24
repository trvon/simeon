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
// docs/research/rrf_variants_results.md.
void score_bm25_variants_rrf(std::span<const Bm25Index* const> variants, std::string_view query,
                             std::span<float> out_scores, float k_rrf = 60.0f);

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

// Hyperparameters for per-query entropy-weighted linear-α fusion (Step 1m,
// simeon novel contribution). See docs/fusion_entropy_alpha.md.
struct EntropyAlphaConfig {
    // Top-K slice used to estimate per-leg score-distribution entropy. Larger
    // K gives a smoother entropy estimate but is more sensitive to tail noise;
    // 50 matches the top_k_score_entropy predictor in QueryRouter.
    std::uint32_t top_k = 50;
    // When pool_overlap_jaccard >= this threshold the two legs already agree;
    // entropy weighting collapses to an equal-weight blend (α = 0.5). 0.8 is
    // the "high agreement" regime observed on scifact/NFCorpus.
    float agreement_threshold = 0.8f;
};

// Runtime α selection for linear-α fusion. Inputs are the top-K score slices
// of each leg (z-scoring optional — the softmax is shift-invariant) and the
// pool-overlap Jaccard between the two legs on this query.
//
// Shannon entropy H_i is computed over softmax(top-K scores of leg i);
// confidence_i = 1 - H_i / log(K). α is the normalized-confidence of leg A:
//   α = confidence_A / (confidence_A + confidence_B).
// When pool_jaccard >= cfg.agreement_threshold the legs already agree and we
// return 0.5 exactly (strictly reduces to equal-weight linear combination).
// When both confidences are zero (uniform top-K on both legs) we also return
// 0.5 rather than 0/0.
//
// The return value is clamped to [0, 1]. Deterministic given the same inputs.
float entropy_alpha(std::span<const float> a_top_k, std::span<const float> b_top_k,
                    float pool_jaccard, const EntropyAlphaConfig& cfg) noexcept;

// Convenience: z-score both score spans within their top-K tail, compute α
// via entropy_alpha(), then write α · z(a_scores) + (1-α) · z(b_scores) into
// `out_scores`. Inputs `a_scores` / `b_scores` are full per-doc score vectors
// restricted to the shared pool (set other docs to -inf before fusing). The
// `a_top_k` / `b_top_k` spans are the top-K raw scores used for the entropy
// estimate; pass the same pool tail from top_k(). `out_scores.size()` must
// equal `a_scores.size() == b_scores.size()`.
void linear_alpha_entropy_fuse(std::span<const float> a_scores, std::span<const float> b_scores,
                               std::span<const float> a_top_k, std::span<const float> b_top_k,
                               float pool_jaccard, const EntropyAlphaConfig& cfg,
                               std::span<float> out_scores) noexcept;

} // namespace simeon
