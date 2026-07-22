#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "simeon/fragment_geometry.hpp"
#include "simeon/pmi.hpp"
#include "experiment_support.hpp"

namespace simeon::experiment {

enum class RerankerEncoderProfile : std::uint8_t {
    FixedHash,
    PmiWord,
};

enum class FragmentBuilderKind : std::uint8_t {
    Basic,
    Rich,
    RichCovered,
};

enum class FragmentStorageKind : std::uint8_t {
    Float32,
    BFloat16,
};

struct FragmentRerankerConfig {
    RerankerEncoderProfile encoder_profile = RerankerEncoderProfile::FixedHash;
    // The fixed profile defaults to the exact simeon-v1-384 FFI identity.
    EncoderConfig fixed_encoder;
    PmiConfig pmi;
    FragmentBuilderKind builder = FragmentBuilderKind::Rich;
    FragmentStorageKind storage = FragmentStorageKind::Float32;
    std::uint32_t top_sentence_fragments = 4;
    std::uint32_t fragment_signature_terms = 8;
    float sentence_overlap_cap = 0.60f;
    float anchor_overlap_cap = 0.80f;
    FragmentGeometryConfig geometry;
};

// Strictly resolves every scorer, builder, storage, and encoder field. Unknown
// keys fail instead of silently creating a different experiment.
FragmentRerankerConfig resolve_fragment_reranker_config(const VariantSpec& variant);

struct RerankerWorkspaceStats {
    double setup_us = 0.0;
    double bm25_add_us = 0.0;
    double bm25_finalize_us = 0.0;
    double encoder_artifact_build_us = 0.0;
    double encoder_init_us = 0.0;
    double fragment_build_us = 0.0;
    double fragment_compress_us = 0.0;
    std::uint32_t encoder_dim = 0;
    std::uint32_t pmi_vocab_size = 0;
    std::uint64_t fragment_count = 0;
    std::uint64_t fragment_vector_bytes = 0;
    std::uint64_t encoder_artifact_bytes_lower_bound = 0;
};

struct RerankerProfileSummary {
    double total_mean_us = 0.0;
    double total_p50_us = 0.0;
    double total_p95_us = 0.0;
    double bm25_mean_us = 0.0;
    double query_encode_mean_us = 0.0;
    double gather_mean_us = 0.0;
    double whiten_mean_us = 0.0;
    double pairwise_mean_us = 0.0;
    double phss_select_mean_us = 0.0;
    double query_attention_mean_us = 0.0;
    double adjacency_mean_us = 0.0;
    double diffuse_mean_us = 0.0;
    double blend_mean_us = 0.0;
    double pool_docs_mean = 0.0;
    double pool_fragments_mean = 0.0;
    double graph_edges_mean = 0.0;
    double phss_used_rate = 0.0;
};

struct RerankerRetrievalRun {
    std::vector<Ranking> rankings;
    std::vector<Ranking> baseline_rankings;
    // Full finite rerank pool, retained so candidate recall can be evaluated
    // after scoring without exposing qrels to the workspace.
    std::vector<Ranking> candidate_rankings;
    RerankerProfileSummary profile;
    double baseline_score_us = 0.0;
    double candidate_recall = 0.0;
    std::size_t query_repeats = 1;
};

double evaluate_candidate_recall(const std::vector<Ranking>& candidate_rankings,
                                 const Fixture& fixture);

// Corpus artifacts and document fragments are built once and may be reused by
// scorer-only variants. Builder, storage, BM25, and encoder identity changes
// deliberately require a separate workspace and therefore a separately
// reported setup cost.
class FragmentRerankerWorkspace {
public:
    FragmentRerankerWorkspace(const Fixture& fixture, const FragmentRerankerConfig& config);
    ~FragmentRerankerWorkspace();

    FragmentRerankerWorkspace(FragmentRerankerWorkspace&&) noexcept;
    FragmentRerankerWorkspace& operator=(FragmentRerankerWorkspace&&) noexcept;
    FragmentRerankerWorkspace(const FragmentRerankerWorkspace&) = delete;
    FragmentRerankerWorkspace& operator=(const FragmentRerankerWorkspace&) = delete;

    bool compatible(const FragmentRerankerConfig& config) const;
    RerankerRetrievalRun run(const FragmentRerankerConfig& config, std::size_t query_repeats,
                             std::size_t retrieval_depth = 100) const;
    const RerankerWorkspaceStats& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string reranker_result_json(const ResultContext& context, const FragmentRerankerConfig& config,
                                 const RerankerWorkspaceStats& workspace,
                                 const RerankerRetrievalRun& run, const Metrics& metrics,
                                 const Metrics& baseline_metrics, double evaluation_us,
                                 std::size_t query_count, std::size_t document_count,
                                 std::size_t qrel_count);

} // namespace simeon::experiment
