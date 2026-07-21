#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "experiment_support.hpp"

namespace simeon::experiment {

struct FusionWorkspaceStats {
    double setup_us = 0.0;
    double base_query_score_us = 0.0;
    double idf_artifact_build_us = 0.0;
    double idf_document_encode_us = 0.0;
    double idf_query_encode_us = 0.0;
    double idf_score_us = 0.0;
    std::uint64_t artifact_bytes = 0;
    std::uint64_t idf_document_vector_bytes = 0;
    std::uint64_t cached_evidence_bytes = 0;
    std::string artifact_fingerprint;
};

struct FusionRetrievalRun {
    std::vector<Ranking> rankings;
    double fusion_us = 0.0;
    double mean_pool_size = 0.0;
    double candidate_recall = 0.0;
};

// Expensive corpus/query evidence is constructed once and replayed across a
// manifest's weight/candidate variants. The six-leg base pool exactly matches
// the promoted recipe workbench: Atire, WSDM(Atire), SAB, WSDM(SAB), PMI
// fragment geometry, and five-BM25 RRF.
class WsdmIdfFusionWorkspace {
public:
    WsdmIdfFusionWorkspace(const Fixture& fixture, const WsdmIdfFusionConfig& config);
    ~WsdmIdfFusionWorkspace();

    WsdmIdfFusionWorkspace(WsdmIdfFusionWorkspace&&) noexcept;
    WsdmIdfFusionWorkspace& operator=(WsdmIdfFusionWorkspace&&) noexcept;
    WsdmIdfFusionWorkspace(const WsdmIdfFusionWorkspace&) = delete;
    WsdmIdfFusionWorkspace& operator=(const WsdmIdfFusionWorkspace&) = delete;

    bool compatible(const WsdmIdfFusionConfig& config) const;
    FusionRetrievalRun run(const WsdmIdfFusionConfig& config) const;
    const FusionWorkspaceStats& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string fusion_result_json(const ResultContext& context, const WsdmIdfFusionConfig& config,
                               const FusionWorkspaceStats& workspace, const FusionRetrievalRun& run,
                               const Metrics& metrics, double evaluation_us,
                               std::size_t query_count, std::size_t document_count,
                               std::size_t qrel_count);

} // namespace simeon::experiment
