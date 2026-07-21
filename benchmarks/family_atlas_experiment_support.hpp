#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "experiment_support.hpp"

namespace simeon::experiment {

struct FeatureFamilyAtlasWorkspaceStats {
    double setup_us = 0.0;
    double char_idf_artifact_build_us = 0.0;
    double word_idf_artifact_build_us = 0.0;
    double char_document_encode_us = 0.0;
    double word_document_encode_us = 0.0;
    double char_query_encode_us = 0.0;
    double word_query_encode_us = 0.0;
    double base_idf_artifact_build_us = 0.0;
    double base_document_encode_us = 0.0;
    double base_query_encode_us = 0.0;
    double base_score_us = 0.0;
    std::uint64_t char_idf_artifact_bytes = 0;
    std::uint64_t word_idf_artifact_bytes = 0;
    std::uint64_t base_idf_artifact_bytes = 0;
    std::uint64_t base_score_evidence_bytes = 0;
    std::uint64_t raw_document_vector_bytes = 0;
    std::uint64_t raw_query_vector_bytes = 0;
    std::string char_idf_artifact_fingerprint;
    std::string word_idf_artifact_fingerprint;
    std::string base_idf_artifact_fingerprint;
};

struct FeatureFamilyAtlasRun {
    std::vector<Ranking> rankings;
    // Evaluation-only views. They expose family complementarity to the common
    // evaluator but are never inputs to the fusion policy.
    std::vector<Ranking> char_rankings;
    std::vector<Ranking> word_rankings;
    std::vector<Ranking> family_rankings;
    std::vector<Ranking> base_rankings;
    double char_score_us = 0.0;
    double word_score_us = 0.0;
    double policy_replay_us = 0.0;
    std::uint64_t cached_score_evidence_bytes = 0;
    std::uint64_t deployable_document_vector_bytes = 0;
    bool evidence_reused = false;
    bool base_evidence_available = false;
    bool base_evidence_reused = false;
    double mean_char_prefix_energy_fraction = 0.0;
    double mean_word_prefix_energy_fraction = 0.0;
    double mean_joint_char_query_energy_fraction = 0.0;
    double mean_joint_char_document_energy_fraction = 0.0;
    double char_document_prefix_rms = 0.0;
    double word_document_prefix_rms = 0.0;
    double mean_top100_overlap = 0.0;
    double mean_top100_jaccard = 0.0;
    double mean_similarity_distortion = 0.0;
    double mean_calibrated_word_query_energy = 0.0;
    double mean_base_family_top100_overlap = 0.0;
    std::size_t admitted_queries = 0;
    std::size_t rejected_queries = 0;
};

// Builds independent character-only and word-only hashed-IDF/FWHT charts at
// the manifest's maximum width. Dimension allocations are nested prefixes;
// weights replay over cached exact scores. Raw-cosine composition has an exact
// concatenated-block interpretation. Query z-score and rank RRF are explicitly
// reported as multi-score policies rather than single-vector embeddings.
class FeatureFamilyAtlasWorkspace {
public:
    // The fixture owns text referenced by lazy base-chart construction and
    // must outlive this workspace. Reject temporaries to prevent dangling
    // string views after the constructor returns.
    FeatureFamilyAtlasWorkspace(const Fixture& fixture, const FeatureFamilyAtlasConfig& config);
    FeatureFamilyAtlasWorkspace(Fixture&& fixture, const FeatureFamilyAtlasConfig& config) = delete;
    FeatureFamilyAtlasWorkspace(const Fixture&& fixture,
                                const FeatureFamilyAtlasConfig& config) = delete;
    ~FeatureFamilyAtlasWorkspace();

    FeatureFamilyAtlasWorkspace(FeatureFamilyAtlasWorkspace&&) noexcept;
    FeatureFamilyAtlasWorkspace& operator=(FeatureFamilyAtlasWorkspace&&) noexcept;
    FeatureFamilyAtlasWorkspace(const FeatureFamilyAtlasWorkspace&) = delete;
    FeatureFamilyAtlasWorkspace& operator=(const FeatureFamilyAtlasWorkspace&) = delete;

    bool compatible(const FeatureFamilyAtlasConfig& config) const;
    FeatureFamilyAtlasRun run(const FeatureFamilyAtlasConfig& config,
                              std::size_t retrieval_depth = 100) const;
    const FeatureFamilyAtlasWorkspaceStats& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Evaluation-only upper-bound diagnostic: relevant-document recall in the
// union of the two top-100 family rankings. It does not affect fusion.
double family_union_recall_at_100(const std::vector<Ranking>& char_rankings,
                                  const std::vector<Ranking>& word_rankings,
                                  const Fixture& fixture);

std::string feature_family_atlas_result_json(
    const ResultContext& context, const FeatureFamilyAtlasConfig& config,
    const FeatureFamilyAtlasWorkspaceStats& workspace, const FeatureFamilyAtlasRun& run,
    const Metrics& metrics, const Metrics& base_metrics, const Metrics& family_metrics,
    const Metrics& char_metrics, const Metrics& word_metrics, double union_recall_at_100,
    double evaluation_us, std::size_t query_count, std::size_t document_count,
    std::size_t qrel_count);

} // namespace simeon::experiment
