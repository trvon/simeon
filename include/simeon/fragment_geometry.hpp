#pragma once

#include <cstdint>
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
};

struct FragmentGeometryProfile {
    double total_us = 0.0;
    double bm25_us = 0.0;
    double query_encode_us = 0.0;
    double gather_us = 0.0;
    double whiten_us = 0.0;
    double phss_pairwise_us = 0.0;
    double phss_select_us = 0.0;
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
                                                           std::uint32_t fragment_signature_terms);

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

} // namespace simeon
