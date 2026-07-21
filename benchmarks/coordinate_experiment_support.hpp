#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "experiment_support.hpp"

namespace simeon::experiment {

struct CoordinateWorkspaceStats {
    double setup_us = 0.0;
    double idf_artifact_build_us = 0.0;
    double document_encode_us = 0.0;
    double query_encode_us = 0.0;
    double coordinate_calibration_us = 0.0;
    std::uint64_t idf_artifact_bytes = 0;
    std::uint64_t coordinate_artifact_bytes = 0;
    std::uint64_t raw_document_vector_bytes = 0;
    std::string idf_artifact_fingerprint;
    std::string coordinate_artifact_fingerprint;
    double mean_vector_l2 = 0.0;
    double rms_document_l2 = 0.0;
    double anisotropy_ratio = 0.0;
    double mean_coordinate_variance = 0.0;
    double coordinate_stddev_cv = 0.0;
    double variance_effective_dimensions = 0.0;
    double variance_effective_fraction = 0.0;
    double min_coordinate_stddev = 0.0;
    double max_coordinate_stddev = 0.0;
};

struct CoordinateRetrievalRun {
    std::vector<Ranking> rankings;
    double document_transform_us = 0.0;
    double query_transform_us = 0.0;
    double retrieval_score_us = 0.0;
    double reference_score_us = 0.0;
    double policy_replay_us = 0.0;
    std::uint64_t transformed_document_vector_bytes = 0;
    std::uint64_t cached_score_evidence_bytes = 0;
    bool evidence_reused = false;
    double mean_prefix_energy_fraction = 0.0;
    double mean_top100_overlap = 0.0;
    double mean_top100_jaccard = 0.0;
    double mean_chart_distortion = 0.0;
    std::size_t narrowed_queries = 0;
    std::size_t augmented_queries = 0;
};

// Builds one maximum-width raw FWHT workspace and one corpus mean/variance
// artifact. Nested prefix and calibration variants replay over those assets.
// The full-width scores are used only as observed chart evidence; they are
// reported separately from the retrieval scorer's deployable work.
class CoordinateCalibrationWorkspace {
public:
    CoordinateCalibrationWorkspace(const Fixture& fixture,
                                   const CoordinateCalibrationConfig& config);
    ~CoordinateCalibrationWorkspace();

    CoordinateCalibrationWorkspace(CoordinateCalibrationWorkspace&&) noexcept;
    CoordinateCalibrationWorkspace& operator=(CoordinateCalibrationWorkspace&&) noexcept;
    CoordinateCalibrationWorkspace(const CoordinateCalibrationWorkspace&) = delete;
    CoordinateCalibrationWorkspace& operator=(const CoordinateCalibrationWorkspace&) = delete;

    bool compatible(const CoordinateCalibrationConfig& config) const;
    CoordinateRetrievalRun run(const CoordinateCalibrationConfig& config,
                               std::size_t retrieval_depth = 100) const;
    const CoordinateWorkspaceStats& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string coordinate_result_json(const ResultContext& context,
                                   const CoordinateCalibrationConfig& config,
                                   const CoordinateWorkspaceStats& workspace,
                                   const CoordinateRetrievalRun& run, const Metrics& metrics,
                                   double evaluation_us, std::size_t query_count,
                                   std::size_t document_count, std::size_t qrel_count);

} // namespace simeon::experiment
