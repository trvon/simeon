#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include "simeon/persistent_homology.hpp"

namespace simeon {

class Bm25Index;
class Encoder;

// ---------------------------------------------------------------------------
// Sparse signature: weighted hash-term set for fast overlap/containment.
// ---------------------------------------------------------------------------

struct WeightedHashTerm {
    std::uint64_t hash = 0;
    float weight = 0.0f;
};

struct SparseSignature {
    std::vector<WeightedHashTerm> terms;
    float weight_sum = 0.0f;
};

// ---------------------------------------------------------------------------
// Semantic fragment: dense vector + sparse lexical signature.
// ---------------------------------------------------------------------------

struct SemanticFragment {
    std::vector<float> vec;
    std::vector<std::uint16_t> vec_bf16;
    SparseSignature signature;
};

// ---------------------------------------------------------------------------
// Configuration for fragment-geometry scoring.
// ---------------------------------------------------------------------------

struct FragmentGeometryConfig {
    // BM25 pool size.
    std::uint32_t pool_size = 100;

    // Blend weight: alpha * z(BM25_pool) + (1-alpha) * z(geometry_pool).
    float alpha = 0.8f;

    // How many top fragments per document to bring into the pool.
    std::uint32_t top_fragments_per_doc = 4;

    // Softmax temperature for query-to-fragment attention.
    float attention_scale = 8.0f;

    // Fixed kNN graph: number of neighbors per fragment.
    std::uint32_t knn = 8;

    // Number of diffusion propagation steps.
    std::uint32_t steps = 2;

    // BM25-side adaptive gating (legacy; kept for parity with research).
    bool adaptive = false;
    float adaptive_idf_lo = 2.0f;
    float adaptive_idf_hi = 5.0f;
    float adaptive_decay_lo = 0.25f;
    float adaptive_decay_hi = 0.95f;
    float adaptive_alpha_lo = 0.70f;
    float adaptive_alpha_hi = 0.98f;
    float adaptive_scale_lo = 4.0f;
    float adaptive_scale_hi = 10.0f;
    std::uint32_t adaptive_knn_lo = 4;
    std::uint32_t adaptive_knn_hi = 16;
    std::uint32_t adaptive_steps_lo = 1;
    std::uint32_t adaptive_steps_hi = 3;

    // Query-side geometry gating (legacy; kept for parity with research).
    bool geometry_signal_adaptive = false;
    float geometry_alpha_lo = 0.65f;
    float geometry_alpha_hi = 0.98f;
    float geometry_scale_lo = 3.0f;
    float geometry_scale_hi = 10.0f;
    std::uint32_t geometry_knn_lo = 4;
    std::uint32_t geometry_knn_hi = 16;
    std::uint32_t geometry_steps_lo = 1;
    std::uint32_t geometry_steps_hi = 3;

    // Persistent Homology Scale Selection (PHSS).
    // When true, compute 0D persistent homology on the fragment similarity
    // graph and use the selected scale to determine which edges to keep,
    // replacing the fixed top-k selection. Default true (data-driven).
    bool use_phss = true;
    PhssConfig phss_config{};

    // Query-adaptive PHSS: if enabled, use PHSS only when the geometry-side
    // query confidence clears the threshold; otherwise fall back to fixed-kNN.
    bool phss_adaptive = false;
    float phss_confidence_threshold = 0.55f;

    // PHSS-1D Phase A — per-fragment triangle-count importance probe.
    // After the kNN graph is built (PHSS-selected or fixed), count triangles
    // each fragment participates in and apply weight w_i = 1 +
    // triangle_alpha * log1p(tri_count[i]). The weight multiplies either the
    // post-attention seed mass (QueryAttention placement) or the
    // post-diffusion mass before doc aggregation (Diffusion placement).
    enum class TrianglePlacement : std::uint8_t {
        None,
        QueryAttention,
        Diffusion,
    };
    bool use_triangle_weight = false;
    float triangle_alpha = 0.0f;
    TrianglePlacement triangle_placement = TrianglePlacement::None;

    // MaxSim aggregation probe (training-free ColBERTv2 analog).
    // Sum (default): geom_pool[doc] += mass[fragment] — averages per-fragment
    //   signal across the t fragments per doc, smoothing variance.
    // Max: geom_pool[doc] = max(geom_pool[doc], mass[fragment]) — preserves
    //   the strongest per-fragment match per doc, addressing the multi-
    //   fragment averaging bottleneck identified by phss_1d_triangle_results.md.
    enum class DocAggregator : std::uint8_t {
        Sum,
        Max,
    };
    DocAggregator doc_aggregator = DocAggregator::Sum;

    // Plan 2 self-KB candidate-set expansion (Phase A probe).
    // When non-empty, after the BM25 top-K pool is built, union it with the
    // precomputed top-N doc-doc neighbors of each pool member. Re-rank the
    // expanded pool with the existing geometry pipeline. Addresses the
    // BM25-pool R@100 ceiling (Bottleneck 2) at the offline-precomputation
    // layer instead of the per-query rerank layer.
    //
    // doc_doc_neighbors[d] = top-N neighbor IDs of doc d (precomputed).
    // Empty span = no expansion (default; pure first-pass top-K).
    std::span<const std::vector<std::uint32_t>> doc_doc_neighbors{};

    // Plan 4 — Single-fragment-per-doc builder. At query time, keep only
    // the argmax query-similar fragment per doc (suppress all others to
    // zero mass via a -inf qsim). Addresses the multi-fragment averaging
    // bottleneck (phss_1d_triangle_results.md mechanism) at a different
    // attack point than MaxSim: MaxSim aggregates across fragments; this
    // filter selects a single fragment before aggregation.
    bool single_fragment_per_doc = false;

    // Phase B tunables for self-KB expansion.
    // Cap neighbors used per pool member at query time (0 = use all available
    // in the precomputed graph). Smaller cap reduces expansion cost while
    // keeping R@100 lift if the top neighbors carry most of the signal.
    std::uint32_t selfkb_neighbors_per_pool_doc = 0;
    // BM25-relevance filter: only add neighbor `n` if bm25_scores[n] > this
    // threshold. Removes zero-query-overlap neighbors that cause top-10
    // ranking noise. Default -inf = no filter (Phase A behavior).
    float selfkb_min_bm25_score = -std::numeric_limits<float>::infinity();
    // Topology confidence gate (Component C per plan): skip expansion when
    // BM25 pool score decay is below this threshold (flat / diffuse pools).
    // 0 = always expand (Phase A behavior). Sweep on dev fold to pick.
    float selfkb_gate_score_decay_min = 0.0f;
};

struct FragmentGeometryProfile {
    double total_us = 0.0;
    double bm25_us = 0.0;
    double query_encode_us = 0.0;
    double gather_us = 0.0;
    double whiten_us = 0.0;
    double phss_pairwise_us = 0.0;
    double phss_select_us = 0.0;
    // Sub-phase breakdown of phss_select_us (populated by phss_select_scale).
    // These should sum to ~phss_select_us within timer-overhead noise.
    double phss_select_edge_gather_us = 0.0;
    double phss_select_edge_sort_us = 0.0;
    double phss_select_uf_us = 0.0;
    double phss_select_survivor_us = 0.0;
    double phss_select_death_sort_us = 0.0;
    double phss_select_criterion_us = 0.0;
    // PHSS-1D triangle counter (Phase A).
    double triangle_count_us = 0.0;
    std::uint64_t triangle_count_total = 0;
    double query_attention_us = 0.0;
    double adjacency_us = 0.0;
    double diffuse_us = 0.0;
    double blend_us = 0.0;
    std::uint32_t pool_docs = 0;
    std::uint32_t pool_fragments = 0;
    std::uint64_t graph_edges = 0;
    bool phss_enabled = false;
    bool phss_used = false;
    float phss_selected_scale = 0.0f;
    float query_confidence = 0.0f;
};

// ---------------------------------------------------------------------------
// Fragment builders (one per selector strategy).
// ---------------------------------------------------------------------------

// Basic: top TextRank sentences encoded with PMI.
std::vector<SemanticFragment> build_doc_semantic_fragments(const Encoder& enc, std::string_view doc,
                                                           const Bm25Index& idx,
                                                           std::uint32_t top_sentence_fragments,
                                                           std::uint32_t fragment_signature_terms,
                                                           float position_weight = 0.0f);

// Rich: basic sentences + centroid-of-sentences + whole-document vector.
std::vector<SemanticFragment>
build_doc_semantic_fragments_rich(const Encoder& enc, std::string_view doc, const Bm25Index& idx,
                                  std::uint32_t top_sentence_fragments,
                                  std::uint32_t fragment_signature_terms);

// Rich + hard overlap caps (richcov safety).
std::vector<SemanticFragment> build_doc_semantic_fragments_rich_covered(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap);

// Rich + MMR soft novelty selection.
std::vector<SemanticFragment> build_doc_semantic_fragments_rich_mmr(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float redundancy_lambda,
    float sentence_min_score, float anchor_min_score);

// Rich + two-stage asymmetric (hard sentence caps + MMR anchors).
std::vector<SemanticFragment> build_doc_semantic_fragments_rich_asymmetric(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float anchor_redundancy_lambda,
    float anchor_min_score);

// Rich + hard budget caps (max sentences + max anchors).
std::vector<SemanticFragment> build_doc_semantic_fragments_rich_budgeted(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, std::uint32_t max_sentence_keep, float anchor_overlap_cap,
    float anchor_novelty_floor, std::uint32_t max_anchor_keep);

// ---------------------------------------------------------------------------
// Query scoring.
// ---------------------------------------------------------------------------

// Score a query using fragment-geometry reranking inside a BM25 pool.
// Returns one score per document (size = nd). Documents outside the BM25 pool
// are set to `-inf`; callers that need candidate-subset fallback should project
// these scores onto their candidate set and backfill from a lexical baseline.
// The geometry leg is z-scored and blended with z-scored BM25 pool scores
// using cfg.alpha.
std::vector<float> score_fragment_geometry(std::string_view query, const Bm25Index& idx,
                                           const Encoder& enc,
                                           std::span<const std::vector<SemanticFragment>> doc_frags,
                                           const FragmentGeometryConfig& cfg);

std::vector<float>
score_fragment_geometry_profiled(std::string_view query, const Bm25Index& idx, const Encoder& enc,
                                 std::span<const std::vector<SemanticFragment>> doc_frags,
                                 const FragmentGeometryConfig& cfg,
                                 FragmentGeometryProfile* profile);

// Compress all fragment vectors from float32 to bfloat16 in-place.
// After calling, vec_bf16 holds the compressed data and vec is cleared.
// The geometry scorer decompresses on the fly via read_frag_vec().
// Halves peak memory for fragment vectors (2× per fragment dim).
void compress_fragments_to_bf16(std::span<std::vector<SemanticFragment>> doc_frags,
                                std::uint32_t dim);

} // namespace simeon
