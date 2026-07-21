#include "coordinate_experiment_support.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "simeon/fusion.hpp"
#include "simeon/simd.hpp"

namespace simeon::experiment {
namespace {

using Clock = std::chrono::steady_clock;

const char* transform_name(CoordinateTransform transform) noexcept {
    switch (transform) {
        case CoordinateTransform::None:
            return "none";
        case CoordinateTransform::Center:
            return "center";
        case CoordinateTransform::Standardize:
            return "standardize";
    }
    return "unknown";
}

const char* routing_policy_name(CoordinateRoutingPolicy policy) noexcept {
    switch (policy) {
        case CoordinateRoutingPolicy::Fixed:
            return "fixed";
        case CoordinateRoutingPolicy::Blend:
            return "blend";
        case CoordinateRoutingPolicy::Selective:
            return "selective";
        case CoordinateRoutingPolicy::SelectiveEnergy:
            return "selective_energy";
    }
    return "unknown";
}

bool same_identity(const CoordinateCalibrationConfig& lhs, const CoordinateCalibrationConfig& rhs) {
    return lhs.idf_encoder.idf.hash_dim == rhs.idf_encoder.idf.hash_dim &&
           lhs.idf_encoder.idf.scope == rhs.idf_encoder.idf.scope &&
           encoder_config_json(lhs.idf_encoder.encoder) ==
               encoder_config_json(rhs.idf_encoder.encoder);
}

std::string coordinate_fingerprint(std::span<const float> mean, std::span<const float> variance) {
    constexpr std::uint64_t offset = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    const auto add_u32 = [&](std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= prime;
        }
    };
    add_u32(static_cast<std::uint32_t>(mean.size()));
    for (const float value : mean)
        add_u32(std::bit_cast<std::uint32_t>(value));
    for (const float value : variance)
        add_u32(std::bit_cast<std::uint32_t>(value));
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

void compute_norms(std::span<const float> vectors, std::size_t rows, std::uint32_t maximum_dim,
                   std::uint32_t prefix_dim, std::vector<float>& prefix_norms,
                   std::vector<float>& full_norms) {
    prefix_norms.resize(rows);
    full_norms.resize(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* vector = vectors.data() + row * maximum_dim;
        double prefix_squared = 0.0;
        double full_squared = 0.0;
        for (std::uint32_t coordinate = 0; coordinate < maximum_dim; ++coordinate) {
            const double value = vector[coordinate];
            full_squared += value * value;
            if (coordinate < prefix_dim)
                prefix_squared += value * value;
        }
        prefix_norms[row] = static_cast<float>(std::sqrt(prefix_squared));
        full_norms[row] = static_cast<float>(std::sqrt(full_squared));
    }
}

float normalized_score(float dot, float lhs_norm, float rhs_norm) noexcept {
    const float denominator = lhs_norm * rhs_norm;
    return denominator > 0.0f && std::isfinite(denominator) ? dot / denominator : 0.0f;
}

} // namespace

struct CoordinateCalibrationWorkspace::Impl {
    struct QueryScoreEvidence {
        std::vector<float> prefix_scores;
        std::vector<float> full_scores;
        double prefix_energy = 0.0;
        double top_overlap = 0.0;
        double top_jaccard = 0.0;
        double similarity_distortion = 0.0;
    };

    struct ScoreEvidence {
        CoordinateTransform transform = CoordinateTransform::None;
        float variance_shrinkage = 0.0f;
        float min_variance_ratio = 0.0f;
        std::uint32_t prefix_dim = 0;
        std::size_t retrieval_depth = 0;
        double document_transform_us = 0.0;
        double query_transform_us = 0.0;
        double prefix_score_us = 0.0;
        double full_score_us = 0.0;
        std::uint64_t transformed_document_vector_bytes = 0;
        std::vector<QueryScoreEvidence> queries;
    };

    CoordinateCalibrationConfig identity;
    CoordinateWorkspaceStats stats;
    std::uint32_t dimension = 0;
    std::size_t document_count = 0;
    std::size_t query_count = 0;
    std::vector<float> documents;
    std::vector<float> queries;
    std::vector<float> mean;
    std::vector<float> variance;
    mutable std::unique_ptr<ScoreEvidence> cached_evidence;

    Impl(const Fixture& fixture, const CoordinateCalibrationConfig& config) : identity(config) {
        const auto setup_start = Clock::now();
        if (fixture.query_ids.size() != fixture.query_texts.size() ||
            fixture.doc_ids.size() != fixture.doc_texts.size())
            throw std::runtime_error("coordinate fixture ids and texts have different row counts");
        if (fixture.query_texts.empty() || fixture.doc_texts.empty())
            throw std::runtime_error("coordinate fixture requires at least one query and document");

        std::vector<std::string_view> document_views;
        document_views.reserve(fixture.doc_texts.size());
        for (const auto& document : fixture.doc_texts)
            document_views.emplace_back(document);
        std::vector<std::string_view> query_views;
        query_views.reserve(fixture.query_texts.size());
        for (const auto& query : fixture.query_texts)
            query_views.emplace_back(query);

        const auto artifact_start = Clock::now();
        const HashedIdf idf_artifact =
            HashedIdf::learn(document_views, config.idf_encoder.encoder, config.idf_encoder.idf);
        stats.idf_artifact_build_us =
            std::chrono::duration<double, std::micro>(Clock::now() - artifact_start).count();
        stats.idf_artifact_bytes = idf_artifact.storage_bytes();
        stats.idf_artifact_fingerprint = idf_artifact.fingerprint();

        EncoderConfig raw_config = config.idf_encoder.encoder;
        raw_config.hashed_idf = &idf_artifact;
        raw_config.l2_normalize = false;
        const Encoder encoder(raw_config);
        dimension = encoder.output_dim();
        document_count = fixture.doc_texts.size();
        query_count = fixture.query_texts.size();
        documents.resize(document_count * static_cast<std::size_t>(dimension));
        queries.resize(query_count * static_cast<std::size_t>(dimension));
        stats.raw_document_vector_bytes = documents.size() * sizeof(float);

        const auto document_start = Clock::now();
        encoder.encode_batch(document_views, documents.data());
        stats.document_encode_us =
            std::chrono::duration<double, std::micro>(Clock::now() - document_start).count();
        const auto query_start = Clock::now();
        encoder.encode_batch(query_views, queries.data());
        stats.query_encode_us =
            std::chrono::duration<double, std::micro>(Clock::now() - query_start).count();

        // Coordinate calibration belongs to the cosine sphere. Raw hashed-IDF
        // magnitudes track document length, so subtracting their corpus mean
        // from a much shorter query would conflate length with direction. A
        // single full-width normalization is harmless to every nested prefix:
        // prefix cosine divides out that row-wise scalar again.
        for (std::size_t document = 0; document < document_count; ++document)
            simd::l2_normalize(documents.data() + document * dimension, dimension);
        for (std::size_t query = 0; query < query_count; ++query)
            simd::l2_normalize(queries.data() + query * dimension, dimension);

        const auto calibration_start = Clock::now();
        std::vector<double> sum(dimension, 0.0);
        std::vector<double> square_sum(dimension, 0.0);
        for (std::size_t document = 0; document < document_count; ++document) {
            const float* vector = documents.data() + document * dimension;
            for (std::uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
                const double value = vector[coordinate];
                sum[coordinate] += value;
                square_sum[coordinate] += value * value;
            }
        }
        mean.resize(dimension);
        variance.resize(dimension);
        const double count = static_cast<double>(document_count);
        double mean_norm_squared = 0.0;
        double total_second_moment = 0.0;
        double variance_sum = 0.0;
        double variance_square_sum = 0.0;
        double min_stddev = std::numeric_limits<double>::infinity();
        double max_stddev = 0.0;
        for (std::uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
            const double coordinate_mean = sum[coordinate] / count;
            const double second_moment = square_sum[coordinate] / count;
            const double coordinate_variance =
                std::max(0.0, second_moment - coordinate_mean * coordinate_mean);
            mean[coordinate] = static_cast<float>(coordinate_mean);
            variance[coordinate] = static_cast<float>(coordinate_variance);
            mean_norm_squared += coordinate_mean * coordinate_mean;
            total_second_moment += second_moment;
            variance_sum += coordinate_variance;
            variance_square_sum += coordinate_variance * coordinate_variance;
            const double stddev = std::sqrt(coordinate_variance);
            min_stddev = std::min(min_stddev, stddev);
            max_stddev = std::max(max_stddev, stddev);
        }
        stats.mean_vector_l2 = std::sqrt(mean_norm_squared);
        stats.rms_document_l2 = std::sqrt(total_second_moment);
        stats.anisotropy_ratio =
            stats.rms_document_l2 > 0.0 ? stats.mean_vector_l2 / stats.rms_document_l2 : 0.0;
        stats.mean_coordinate_variance = variance_sum / static_cast<double>(dimension);
        if (variance_sum > 0.0 && variance_square_sum > 0.0) {
            stats.variance_effective_dimensions = variance_sum * variance_sum / variance_square_sum;
            stats.variance_effective_fraction =
                stats.variance_effective_dimensions / static_cast<double>(dimension);
            double squared_deviation_sum = 0.0;
            const double mean_stddev = [&] {
                double stddev_sum = 0.0;
                for (const float value : variance)
                    stddev_sum += std::sqrt(static_cast<double>(value));
                return stddev_sum / static_cast<double>(dimension);
            }();
            for (const float value : variance) {
                const double difference = std::sqrt(static_cast<double>(value)) - mean_stddev;
                squared_deviation_sum += difference * difference;
            }
            stats.coordinate_stddev_cv =
                mean_stddev > 0.0
                    ? std::sqrt(squared_deviation_sum / static_cast<double>(dimension)) /
                          mean_stddev
                    : 0.0;
        }
        stats.min_coordinate_stddev = std::isfinite(min_stddev) ? min_stddev : 0.0;
        stats.max_coordinate_stddev = max_stddev;
        stats.coordinate_artifact_bytes =
            (mean.size() + variance.size()) * static_cast<std::uint64_t>(sizeof(float));
        stats.coordinate_artifact_fingerprint = coordinate_fingerprint(mean, variance);
        stats.coordinate_calibration_us =
            std::chrono::duration<double, std::micro>(Clock::now() - calibration_start).count();
        stats.setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
    }

    std::vector<float> transform(std::span<const float> input,
                                 const CoordinateCalibrationConfig& config) const {
        std::vector<float> output(input.size());
        const double mean_variance = stats.mean_coordinate_variance;
        const double floor = mean_variance * static_cast<double>(config.min_variance_ratio);
        for (std::size_t row = 0; row < input.size() / dimension; ++row) {
            for (std::uint32_t coordinate = 0; coordinate < dimension; ++coordinate) {
                const std::size_t index = row * dimension + coordinate;
                if (config.transform == CoordinateTransform::None) {
                    output[index] = input[index];
                    continue;
                }
                double value = static_cast<double>(input[index]) - mean[coordinate];
                if (config.transform == CoordinateTransform::Standardize) {
                    const double shrunk =
                        (1.0 - static_cast<double>(config.variance_shrinkage)) *
                            variance[coordinate] +
                        static_cast<double>(config.variance_shrinkage) * mean_variance;
                    value /= std::sqrt(std::max(shrunk, floor));
                }
                output[index] = static_cast<float>(value);
            }
        }
        return output;
    }
};

CoordinateCalibrationWorkspace::CoordinateCalibrationWorkspace(
    const Fixture& fixture, const CoordinateCalibrationConfig& config)
    : impl_(std::make_unique<Impl>(fixture, config)) {}

CoordinateCalibrationWorkspace::~CoordinateCalibrationWorkspace() = default;
CoordinateCalibrationWorkspace::CoordinateCalibrationWorkspace(
    CoordinateCalibrationWorkspace&&) noexcept = default;
CoordinateCalibrationWorkspace&
CoordinateCalibrationWorkspace::operator=(CoordinateCalibrationWorkspace&&) noexcept = default;

bool CoordinateCalibrationWorkspace::compatible(const CoordinateCalibrationConfig& config) const {
    return same_identity(impl_->identity, config);
}

CoordinateRetrievalRun
CoordinateCalibrationWorkspace::run(const CoordinateCalibrationConfig& config,
                                    std::size_t retrieval_depth) const {
    if (!compatible(config))
        throw std::runtime_error(
            "coordinate_calibrated_idf variants must share maximum-width IDF encoder identity");
    if (retrieval_depth == 0)
        throw std::runtime_error("coordinate retrieval depth must be greater than zero");

    const bool reusable = impl_->cached_evidence != nullptr &&
                          impl_->cached_evidence->transform == config.transform &&
                          impl_->cached_evidence->variance_shrinkage == config.variance_shrinkage &&
                          impl_->cached_evidence->min_variance_ratio == config.min_variance_ratio &&
                          impl_->cached_evidence->retrieval_depth == retrieval_depth &&
                          (config.retrieval_dim == impl_->dimension ||
                           impl_->cached_evidence->prefix_dim == config.retrieval_dim);
    if (!reusable) {
        auto evidence = std::make_unique<Impl::ScoreEvidence>();
        evidence->transform = config.transform;
        evidence->variance_shrinkage = config.variance_shrinkage;
        evidence->min_variance_ratio = config.min_variance_ratio;
        evidence->prefix_dim = config.retrieval_dim;
        evidence->retrieval_depth = retrieval_depth;

        const auto document_start = Clock::now();
        auto documents = impl_->transform(impl_->documents, config);
        evidence->document_transform_us =
            std::chrono::duration<double, std::micro>(Clock::now() - document_start).count();
        evidence->transformed_document_vector_bytes = documents.size() * sizeof(float);
        const auto query_start = Clock::now();
        auto queries = impl_->transform(impl_->queries, config);
        evidence->query_transform_us =
            std::chrono::duration<double, std::micro>(Clock::now() - query_start).count();

        std::vector<float> document_prefix_norms;
        std::vector<float> document_full_norms;
        std::vector<float> query_prefix_norms;
        std::vector<float> query_full_norms;
        compute_norms(documents, impl_->document_count, impl_->dimension, config.retrieval_dim,
                      document_prefix_norms, document_full_norms);
        compute_norms(queries, impl_->query_count, impl_->dimension, config.retrieval_dim,
                      query_prefix_norms, query_full_norms);

        evidence->queries.resize(impl_->query_count);
        for (std::size_t query = 0; query < impl_->query_count; ++query) {
            auto& query_evidence = evidence->queries[query];
            query_evidence.prefix_scores.resize(impl_->document_count);
            query_evidence.full_scores.resize(impl_->document_count);
            const float* query_vector = queries.data() + query * impl_->dimension;

            const auto prefix_start = Clock::now();
            std::size_t document = 0;
            for (; document + 4 <= impl_->document_count; document += 4) {
                float scores[4];
                const float* base = documents.data() + document * impl_->dimension;
                simd::dot4(query_vector, base, base + impl_->dimension, base + 2 * impl_->dimension,
                           base + 3 * impl_->dimension, scores, config.retrieval_dim);
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    query_evidence.prefix_scores[document + lane] =
                        normalized_score(scores[lane], query_prefix_norms[query],
                                         document_prefix_norms[document + lane]);
                }
            }
            for (; document < impl_->document_count; ++document) {
                const float dot =
                    simd::dot(query_vector, documents.data() + document * impl_->dimension,
                              config.retrieval_dim);
                query_evidence.prefix_scores[document] = normalized_score(
                    dot, query_prefix_norms[query], document_prefix_norms[document]);
            }
            evidence->prefix_score_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - prefix_start).count();

            if (config.retrieval_dim == impl_->dimension) {
                query_evidence.full_scores = query_evidence.prefix_scores;
            } else {
                const auto full_start = Clock::now();
                document = 0;
                for (; document + 4 <= impl_->document_count; document += 4) {
                    float scores[4];
                    const float* base = documents.data() + document * impl_->dimension;
                    simd::dot4(query_vector, base, base + impl_->dimension,
                               base + 2 * impl_->dimension, base + 3 * impl_->dimension, scores,
                               impl_->dimension);
                    for (std::size_t lane = 0; lane < 4; ++lane) {
                        query_evidence.full_scores[document + lane] =
                            normalized_score(scores[lane], query_full_norms[query],
                                             document_full_norms[document + lane]);
                    }
                }
                for (; document < impl_->document_count; ++document) {
                    const float dot =
                        simd::dot(query_vector, documents.data() + document * impl_->dimension,
                                  impl_->dimension);
                    query_evidence.full_scores[document] = normalized_score(
                        dot, query_full_norms[query], document_full_norms[document]);
                }
                evidence->full_score_us +=
                    std::chrono::duration<double, std::micro>(Clock::now() - full_start).count();
            }

            const auto prefix_top =
                top_k(query_evidence.prefix_scores, static_cast<std::uint32_t>(retrieval_depth));
            const auto full_top =
                top_k(query_evidence.full_scores, static_cast<std::uint32_t>(retrieval_depth));
            std::unordered_set<std::uint32_t> prefix_documents;
            prefix_documents.reserve(prefix_top.size());
            for (const auto& [document_id, _] : prefix_top)
                prefix_documents.insert(document_id);
            std::unordered_set<std::uint32_t> union_documents = prefix_documents;
            std::size_t overlap = 0;
            for (const auto& [document_id, _] : full_top) {
                overlap += prefix_documents.contains(document_id) ? 1 : 0;
                union_documents.insert(document_id);
            }
            const std::size_t denominator = std::max(prefix_top.size(), full_top.size());
            query_evidence.top_overlap =
                denominator == 0 ? 1.0
                                 : static_cast<double>(overlap) / static_cast<double>(denominator);
            query_evidence.top_jaccard =
                union_documents.empty()
                    ? 1.0
                    : static_cast<double>(overlap) / static_cast<double>(union_documents.size());
            for (const auto document_id : union_documents) {
                query_evidence.similarity_distortion +=
                    std::fabs(static_cast<double>(query_evidence.prefix_scores[document_id]) -
                              static_cast<double>(query_evidence.full_scores[document_id]));
            }
            if (!union_documents.empty())
                query_evidence.similarity_distortion /= static_cast<double>(union_documents.size());
            const double prefix_norm = query_prefix_norms[query];
            const double full_norm = query_full_norms[query];
            query_evidence.prefix_energy =
                full_norm > 0.0 ? (prefix_norm * prefix_norm) / (full_norm * full_norm) : 0.0;
        }
        impl_->cached_evidence = std::move(evidence);
    }

    const auto& evidence = *impl_->cached_evidence;
    CoordinateRetrievalRun run;
    run.evidence_reused = reusable;
    run.document_transform_us = evidence.document_transform_us;
    run.query_transform_us = evidence.query_transform_us;
    run.retrieval_score_us = evidence.prefix_score_us;
    run.reference_score_us = evidence.full_score_us;
    run.transformed_document_vector_bytes = evidence.transformed_document_vector_bytes;
    run.cached_score_evidence_bytes = static_cast<std::uint64_t>(impl_->query_count) *
                                      impl_->document_count *
                                      static_cast<std::uint64_t>(2 * sizeof(float));
    run.rankings.resize(impl_->query_count);
    std::vector<float> final_scores(impl_->document_count);
    double prefix_energy_sum = 0.0;
    double overlap_sum = 0.0;
    double jaccard_sum = 0.0;
    double distortion_sum = 0.0;
    const auto replay_start = Clock::now();

    for (std::size_t query = 0; query < impl_->query_count; ++query) {
        const auto& query_evidence = evidence.queries[query];
        const bool full_width = config.retrieval_dim == impl_->dimension;
        const auto& prefix_scores =
            full_width ? query_evidence.full_scores : query_evidence.prefix_scores;
        const double prefix_energy = full_width ? 1.0 : query_evidence.prefix_energy;
        const double overlap = full_width ? 1.0 : query_evidence.top_overlap;
        const double jaccard = full_width ? 1.0 : query_evidence.top_jaccard;
        const double distortion = full_width ? 0.0 : query_evidence.similarity_distortion;
        prefix_energy_sum += prefix_energy;
        overlap_sum += overlap;
        jaccard_sum += jaccard;
        distortion_sum += distortion;

        const bool chart_admissible =
            overlap >= static_cast<double>(config.min_chart_overlap) &&
            distortion <= static_cast<double>(config.max_chart_distortion);
        const double expected_prefix_energy =
            static_cast<double>(config.retrieval_dim) / static_cast<double>(impl_->dimension);
        const bool energy_admissible = std::fabs(prefix_energy - expected_prefix_energy) <=
                                       static_cast<double>(config.max_energy_deviation);
        bool use_prefix = config.routing_policy == CoordinateRoutingPolicy::Fixed;
        if (config.routing_policy == CoordinateRoutingPolicy::Selective) {
            use_prefix = chart_admissible;
            if (use_prefix)
                ++run.narrowed_queries;
            else
                ++run.augmented_queries;
        } else if (config.routing_policy == CoordinateRoutingPolicy::Blend) {
            ++run.augmented_queries;
        } else if (config.routing_policy == CoordinateRoutingPolicy::SelectiveEnergy) {
            use_prefix = energy_admissible;
            if (use_prefix)
                ++run.narrowed_queries;
            else
                ++run.augmented_queries;
        }

        if (use_prefix) {
            final_scores = prefix_scores;
        } else {
            const float prefix_weight = 1.0f - config.full_weight;
            for (std::size_t index = 0; index < final_scores.size(); ++index) {
                final_scores[index] = prefix_weight * prefix_scores[index] +
                                      config.full_weight * query_evidence.full_scores[index];
            }
        }
        const auto final = top_k(final_scores, static_cast<std::uint32_t>(retrieval_depth));
        auto& ranking = run.rankings[query];
        ranking.reserve(final.size());
        for (const auto& [document_id, score] : final)
            ranking.emplace_back(score, document_id);
    }

    run.policy_replay_us =
        std::chrono::duration<double, std::micro>(Clock::now() - replay_start).count();
    const double query_count = static_cast<double>(impl_->query_count);
    run.mean_prefix_energy_fraction = prefix_energy_sum / query_count;
    run.mean_top100_overlap = overlap_sum / query_count;
    run.mean_top100_jaccard = jaccard_sum / query_count;
    run.mean_chart_distortion = distortion_sum / query_count;
    return run;
}

const CoordinateWorkspaceStats& CoordinateCalibrationWorkspace::stats() const noexcept {
    return impl_->stats;
}

std::string coordinate_result_json(const ResultContext& context,
                                   const CoordinateCalibrationConfig& config,
                                   const CoordinateWorkspaceStats& workspace,
                                   const CoordinateRetrievalRun& run, const Metrics& metrics,
                                   double evaluation_us, std::size_t query_count,
                                   std::size_t document_count, std::size_t qrel_count) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"schema\":\"simeon.experiment.result.v1\""
        << ",\"experiment\":\"" << json_escape(context.experiment) << "\""
        << ",\"variant\":\"" << json_escape(context.variant) << "\""
        << ",\"kind\":\"" << json_escape(context.kind) << "\""
        << ",\"metric_profile\":\"" << json_escape(context.metric_profile) << "\""
        << ",\"provenance\":{\"manifest\":\"" << json_escape(context.manifest_fingerprint)
        << "\",\"fixture\":\"" << json_escape(context.fixture_fingerprint)
        << "\",\"fixture_name\":\"" << json_escape(context.fixture) << "\",\"split\":\""
        << json_escape(context.split) << "\",\"code_revision\":\""
        << json_escape(context.code_revision) << "\",\"simd_tier\":\""
        << json_escape(simd_tier_name(active_simd_tier())) << "\"}"
        << ",\"metadata\":{";
    bool first_metadata = true;
    for (const auto& [key, value] : context.metadata) {
        if (!first_metadata)
            out << ',';
        first_metadata = false;
        out << '"' << json_escape(key) << "\":\"" << json_escape(value) << '"';
    }
    out << "},\"config\":{\"retrieval_dim\":" << config.retrieval_dim
        << ",\"coordinate_transform\":\"" << transform_name(config.transform) << "\""
        << ",\"coordinate_policy\":\"" << routing_policy_name(config.routing_policy) << "\""
        << ",\"full_weight\":" << config.full_weight
        << ",\"min_chart_overlap\":" << config.min_chart_overlap
        << ",\"max_chart_distortion\":" << config.max_chart_distortion
        << ",\"max_energy_deviation\":" << config.max_energy_deviation
        << ",\"variance_shrinkage\":" << config.variance_shrinkage
        << ",\"min_variance_ratio\":" << config.min_variance_ratio
        << ",\"idf_encoder\":" << encoder_config_json(config.idf_encoder.encoder)
        << ",\"idf_artifact\":{\"fingerprint\":\""
        << json_escape(workspace.idf_artifact_fingerprint)
        << "\",\"hash_dim\":" << config.idf_encoder.idf.hash_dim
        << ",\"storage_bytes\":" << workspace.idf_artifact_bytes << "}"
        << ",\"coordinate_artifact\":{\"fingerprint\":\""
        << json_escape(workspace.coordinate_artifact_fingerprint)
        << "\",\"storage_bytes\":" << workspace.coordinate_artifact_bytes << "}}"
        << ",\"counts\":{\"queries\":" << query_count << ",\"documents\":" << document_count
        << ",\"qrels\":" << qrel_count << ",\"evaluated_queries\":" << metrics.evaluated_queries
        << "}"
        << ",\"metrics\":{\"ndcg_at_10\":" << metrics.ndcg_at_10
        << ",\"precision_at_10\":" << metrics.precision_at_10
        << ",\"recall_at_10\":" << metrics.recall_at_10
        << ",\"recall_at_100\":" << metrics.recall_at_100 << ",\"mrr_at_10\":" << metrics.mrr_at_10
        << "}"
        << ",\"chart_evidence\":{\"mean_prefix_energy_fraction\":"
        << run.mean_prefix_energy_fraction << ",\"mean_top100_overlap\":" << run.mean_top100_overlap
        << ",\"mean_top100_jaccard\":" << run.mean_top100_jaccard
        << ",\"mean_similarity_distortion\":" << run.mean_chart_distortion
        << ",\"narrowed_queries\":" << run.narrowed_queries
        << ",\"augmented_queries\":" << run.augmented_queries
        << ",\"evidence_reused\":" << (run.evidence_reused ? "true" : "false") << "}"
        << ",\"coordinate_diagnostics\":{\"mean_vector_l2\":" << workspace.mean_vector_l2
        << ",\"rms_document_l2\":" << workspace.rms_document_l2
        << ",\"anisotropy_ratio\":" << workspace.anisotropy_ratio
        << ",\"mean_coordinate_variance\":" << workspace.mean_coordinate_variance
        << ",\"coordinate_stddev_cv\":" << workspace.coordinate_stddev_cv
        << ",\"variance_effective_dimensions\":" << workspace.variance_effective_dimensions
        << ",\"variance_effective_fraction\":" << workspace.variance_effective_fraction
        << ",\"min_coordinate_stddev\":" << workspace.min_coordinate_stddev
        << ",\"max_coordinate_stddev\":" << workspace.max_coordinate_stddev << "}"
        << ",\"timing_us\":{\"workspace_setup_total\":" << workspace.setup_us
        << ",\"idf_artifact_build_total\":" << workspace.idf_artifact_build_us
        << ",\"document_encode_total\":" << workspace.document_encode_us
        << ",\"query_encode_total\":" << workspace.query_encode_us
        << ",\"coordinate_calibration_total\":" << workspace.coordinate_calibration_us
        << ",\"document_transform_total\":" << run.document_transform_us
        << ",\"query_transform_total\":" << run.query_transform_us
        << ",\"retrieval_score_total\":" << run.retrieval_score_us
        << ",\"reference_score_total\":" << run.reference_score_us
        << ",\"policy_replay_total\":" << run.policy_replay_us
        << ",\"evaluation_total\":" << evaluation_us << "}"
        << ",\"resources\":{\"idf_artifact_bytes\":" << workspace.idf_artifact_bytes
        << ",\"coordinate_artifact_bytes\":" << workspace.coordinate_artifact_bytes
        << ",\"raw_document_vector_bytes\":" << workspace.raw_document_vector_bytes
        << ",\"transformed_document_vector_bytes\":" << run.transformed_document_vector_bytes
        << ",\"cached_score_evidence_bytes\":" << run.cached_score_evidence_bytes << "}}";
    return out.str();
}

} // namespace simeon::experiment
