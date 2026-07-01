#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "simeon/persistent_homology.hpp"
#include "simeon/text_rank.hpp"

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
// Per-document shared prep — TextRank ranking + per-sentence/anchor signatures.
// Built once per doc, consumed by multiple fragment builders so the corpus pass
// is O(N) total instead of O(N × builders). Sentence text is a string_view into
// the caller's `doc` buffer; the buffer must outlive the prep.
// ---------------------------------------------------------------------------

struct DocPrep {
    std::vector<RankedSentence> ranked;         // top-K, sorted by score
    std::vector<SparseSignature> sentence_sigs; // parallel to `ranked` (1× signature_terms)
    SparseSignature anchor_sig_1x;              // 1× signature_terms (SIF/BSIF/Poincaré anchor)
    SparseSignature anchor_sig_2x;              // 2× signature_terms (rich-family anchor)
    std::uint32_t fragment_signature_terms = 0;
    std::uint32_t top_sentence_fragments = 0;
};

// Build a DocPrep for `doc`. Reusable by all rich-family and richcov-family
// fragment builders. `position_weight` matches the existing per-builder default
// (0.0 = pure TextRank score, >0 = blend with positional prior).
DocPrep prepare_doc(std::string_view doc, const Bm25Index& idx,
                    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
                    float position_weight = 0.0f);

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

    // SPLATE-style outer MaxSim (Formal et al. 2024). When true, bypass
    // PHSS+diffusion and compute the geometry score as max cosine similarity
    // over the doc's whitened fragments. MaxSim operates at the outermost
    // layer (before the alpha blend), not inside the diffusion stack.
    bool outer_maxsim = false;

    // Per-query fragment "whitening": center+scale query and fragment vectors by
    // the mean/variance of the gathered pool fragments before cosine. Audited as
    // a suspected signal inversion (the query-clustered pool centroid ≈ the query
    // direction, so subtracting it could remove the shared relevance axis), but
    // the per-arm retrieval bench showed it neutral-to-slightly-positive vs plain
    // L2-normalized cosine (0.224 vs 0.206 nDCG@10 on the code corpus). Kept on by
    // default; set false for plain cosine. Worth a prose-corpus re-check.
    bool whiten = true;

    // Per-doc aggregation over fragment qsims when outer_maxsim is enabled.
    enum class DocScorerKind : std::uint8_t {
        MaxSim = 0,     // max qsim per doc
        MeanSim = 1,    // mean qsim per doc
        TopKMean = 2,   // mean of top-K qsims, K = doc_scorer_top_k
        SoftMaxSum = 3, // softmax-weighted sum, β = doc_scorer_softmax_beta
        GeoMean = 4,    // geometric mean — penalizes weak fragments
    };
    DocScorerKind doc_scorer_kind = DocScorerKind::MaxSim;
    std::uint32_t doc_scorer_top_k = 3;
    float doc_scorer_softmax_beta = 4.0f;

    // Dual-stage candidate generation. When dense_pool_size > 0 AND
    // doc_dense_vecs != nullptr, the BM25 top-pool_size pool is augmented
    // with the top-dense_pool_size docs by cosine(query_vec, doc_dense_vec).
    // The unified pool (deduplicated, BM25 score retained for the dense-only
    // arrivals) is then reranked by the existing geometry pipeline. Each
    // entry in `*doc_dense_vecs` must be either empty (skip) or `enc.output_dim()`
    // floats; doc_dense_vecs->size() must equal the corpus doc count.
    std::uint32_t dense_pool_size = 0;
    const std::vector<std::vector<float>>* doc_dense_vecs = nullptr;

    // CSLS-style hubness correction (Conneau et al. 2018; QB-Norm family):
    // subtract csls_beta * mean(top-csls_k pairwise sims) from each fragment's
    // query similarity before attention. 0 = off (default). Research knob.
    std::uint32_t csls_k = 0;
    float csls_beta = 1.0f;

    // Prefix dimensionality for the fragment-graph pairwise similarities
    // (PHSS scale selection + adjacency edges). 0 = full encoder dim
    // (default). When set, pairwise cosines use the leading dims of the
    // whitened fragments, renormalized to unit length; PMI rows come from a
    // truncated SVD so leading dims carry most of the energy (matryoshka
    // principle). Query attention always uses full dim. Quality-gate per
    // corpus before enabling (see docs/research.md).
    std::uint32_t graph_prefix_dim = 0;
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

// SIF-weighted PMI fragments + richcov overlap caps (Arora et al. 2017).
// IDF-weighted token accumulation replaces uniform sum: content nouns upweighted,
// function words downweighted. Falls back to standard richcov on non-PMI encoders.
std::vector<SemanticFragment>
build_doc_semantic_fragments_richcov_sif(const Encoder& enc, std::string_view doc,
                                         const Bm25Index& idx, std::uint32_t top_sentence_fragments,
                                         std::uint32_t fragment_signature_terms,
                                         float sentence_overlap_cap, float anchor_overlap_cap);

// Bigram Hadamard SIF fragments (Mitchell & Lapata 2010, multiplicative composition).
// Adjacent token pairs contribute pmi(a) ⊙ pmi(b) * bigram_weight * sqrt(idf(a)*idf(b)).
// Falls back to standard richcov on non-PMI encoders.
std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_bsif(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float bigram_weight = 0.5f);

// ---------------------------------------------------------------------------
// Shared-prep fragment builders. Each takes a `DocPrep` produced by
// `prepare_doc()` and reuses its TextRank ranking + per-sentence/anchor
// signatures across multiple builders. The encoder-specific work (per-fragment
// vector encoding + centroid normalization) is the only per-builder cost.
// ---------------------------------------------------------------------------

std::vector<SemanticFragment> build_doc_semantic_fragments_from_prep(const Encoder& enc,
                                                                     std::string_view doc,
                                                                     const DocPrep& prep);

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_from_prep(const Encoder& enc,
                                                                          std::string_view doc,
                                                                          const DocPrep& prep);

std::vector<SemanticFragment>
build_doc_semantic_fragments_rich_covered_from_prep(const Encoder& enc, std::string_view doc,
                                                    const DocPrep& prep, float sentence_overlap_cap,
                                                    float anchor_overlap_cap);

std::vector<SemanticFragment>
build_doc_semantic_fragments_rich_mmr_from_prep(const Encoder& enc, std::string_view doc,
                                                const DocPrep& prep, float sentence_overlap_cap,
                                                float anchor_overlap_cap, float redundancy_lambda,
                                                float sentence_min_score, float anchor_min_score);

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_budgeted_from_prep(
    const Encoder& enc, std::string_view doc, const DocPrep& prep, float sentence_overlap_cap,
    std::uint32_t max_sentence_keep, float anchor_overlap_cap, float anchor_novelty_floor,
    std::uint32_t max_anchor_keep);

std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_sif_from_prep(
    const Encoder& enc, const Bm25Index& idx, std::string_view doc, const DocPrep& prep,
    float sentence_overlap_cap, float anchor_overlap_cap);

std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_bsif_from_prep(
    const Encoder& enc, const Bm25Index& idx, std::string_view doc, const DocPrep& prep,
    float sentence_overlap_cap, float anchor_overlap_cap, float bigram_weight = 0.5f);

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

// Read a fragment's vector into `dst`, decompressing from BF16 if `frag.vec`
// is empty and `frag.vec_bf16` is populated. `dst` must have space for `dim`
// floats. If the fragment is uninitialized, `dst` is zero-filled.
void read_frag_vec(const SemanticFragment& frag, std::uint32_t dim, float* dst);

} // namespace simeon
