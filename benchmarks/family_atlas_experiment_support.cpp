#include "family_atlas_experiment_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "simeon/fusion.hpp"
#include "simeon/simd.hpp"

namespace simeon::experiment {
namespace {

using Clock = std::chrono::steady_clock;

const char* normalization_name(FeatureFamilyNormalization normalization) noexcept {
    switch (normalization) {
        case FeatureFamilyNormalization::Independent:
            return "independent";
        case FeatureFamilyNormalization::Joint:
            return "joint";
        case FeatureFamilyNormalization::JointRms:
            return "joint_rms";
    }
    return "unknown";
}

const char* policy_name(FeatureFamilyPolicy policy) noexcept {
    switch (policy) {
        case FeatureFamilyPolicy::FamilyOnly:
            return "family_only";
        case FeatureFamilyPolicy::BaseOnly:
            return "base_only";
        case FeatureFamilyPolicy::ResidualBlend:
            return "residual_blend";
        case FeatureFamilyPolicy::Selective:
            return "selective";
    }
    return "unknown";
}

const char*
residual_score_normalization_name(FeatureResidualScoreNormalization normalization) noexcept {
    switch (normalization) {
        case FeatureResidualScoreNormalization::RawCosine:
            return "raw_cosine";
        case FeatureResidualScoreNormalization::QueryZScore:
            return "query_zscore";
        case FeatureResidualScoreNormalization::RankRrf:
            return "rank_rrf";
    }
    return "unknown";
}

std::pair<double, double> mean_and_stddev(std::span<const float> values) {
    double mean = 0.0;
    double square_sum = 0.0;
    for (const float value : values) {
        mean += value;
        square_sum += static_cast<double>(value) * value;
    }
    const double count = static_cast<double>(values.size());
    mean = count > 0.0 ? mean / count : 0.0;
    const double variance = count > 0.0 ? std::max(0.0, square_sum / count - mean * mean) : 0.0;
    return {mean, std::sqrt(variance)};
}

EncoderConfig family_config(const EncoderConfig& base, NGramMode mode) {
    EncoderConfig config = base;
    config.ngram_mode = mode;
    config.hashed_idf = nullptr;
    return config;
}

bool same_identity(const FeatureFamilyAtlasConfig& lhs, const FeatureFamilyAtlasConfig& rhs) {
    return lhs.idf_encoder.idf.hash_dim == rhs.idf_encoder.idf.hash_dim &&
           lhs.idf_encoder.idf.scope == rhs.idf_encoder.idf.scope &&
           encoder_config_json(lhs.idf_encoder.encoder) ==
               encoder_config_json(rhs.idf_encoder.encoder);
}

float normalized_score(float dot, float lhs_norm, float rhs_norm) noexcept {
    const float denominator = lhs_norm * rhs_norm;
    return denominator > 0.0f && std::isfinite(denominator) ? dot / denominator : 0.0f;
}

void compute_prefix_norms(std::span<const float> vectors, std::size_t rows,
                          std::uint32_t maximum_dim, std::uint32_t prefix_dim,
                          std::vector<float>& prefix_norms, double& mean_energy_fraction) {
    prefix_norms.resize(rows);
    double energy_sum = 0.0;
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
        energy_sum += full_squared > 0.0 ? prefix_squared / full_squared : 0.0;
    }
    mean_energy_fraction = rows == 0 ? 0.0 : energy_sum / static_cast<double>(rows);
}

Ranking ranking_from_scores(std::span<const float> scores, std::size_t retrieval_depth) {
    const auto best = top_k(scores, static_cast<std::uint32_t>(retrieval_depth));
    Ranking ranking;
    ranking.reserve(best.size());
    for (const auto& [document, score] : best)
        ranking.emplace_back(score, document);
    return ranking;
}

struct FamilyScores {
    std::vector<std::vector<float>> values;
    std::vector<Ranking> rankings;
    std::vector<float> document_norms;
    std::vector<float> query_norms;
    double score_us = 0.0;
    double mean_query_prefix_energy_fraction = 0.0;
};

FamilyScores score_family(std::span<const float> documents, std::span<const float> queries,
                          std::size_t document_count, std::size_t query_count,
                          std::uint32_t maximum_dim, std::uint32_t prefix_dim,
                          std::size_t retrieval_depth) {
    FamilyScores result;
    double ignored_document_energy = 0.0;
    compute_prefix_norms(documents, document_count, maximum_dim, prefix_dim, result.document_norms,
                         ignored_document_energy);
    compute_prefix_norms(queries, query_count, maximum_dim, prefix_dim, result.query_norms,
                         result.mean_query_prefix_energy_fraction);

    result.values.resize(query_count);
    result.rankings.resize(query_count);
    for (std::size_t query = 0; query < query_count; ++query) {
        auto& scores = result.values[query];
        scores.resize(document_count);
        const float* query_vector = queries.data() + query * maximum_dim;
        const auto score_start = Clock::now();
        std::size_t document = 0;
        for (; document + 4 <= document_count; document += 4) {
            float dots[4];
            const float* base = documents.data() + document * maximum_dim;
            simd::dot4(query_vector, base, base + maximum_dim, base + 2 * maximum_dim,
                       base + 3 * maximum_dim, dots, prefix_dim);
            for (std::size_t lane = 0; lane < 4; ++lane) {
                scores[document + lane] = normalized_score(dots[lane], result.query_norms[query],
                                                           result.document_norms[document + lane]);
            }
        }
        for (; document < document_count; ++document) {
            const float dot =
                simd::dot(query_vector, documents.data() + document * maximum_dim, prefix_dim);
            scores[document] =
                normalized_score(dot, result.query_norms[query], result.document_norms[document]);
        }
        result.score_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - score_start).count();
        result.rankings[query] = ranking_from_scores(scores, retrieval_depth);
    }
    return result;
}

void append_metrics(std::ostringstream& out, const Metrics& metrics) {
    out << "{\"ndcg_at_10\":" << metrics.ndcg_at_10
        << ",\"precision_at_10\":" << metrics.precision_at_10
        << ",\"recall_at_10\":" << metrics.recall_at_10
        << ",\"recall_at_100\":" << metrics.recall_at_100 << ",\"mrr_at_10\":" << metrics.mrr_at_10
        << '}';
}

} // namespace

struct FeatureFamilyAtlasWorkspace::Impl {
    struct ScoreEvidence {
        std::uint32_t char_dim = 0;
        std::uint32_t word_dim = 0;
        std::size_t retrieval_depth = 0;
        FamilyScores character;
        FamilyScores word;
        double mean_top_overlap = 0.0;
        double mean_top_jaccard = 0.0;
        double mean_similarity_distortion = 0.0;
        double mean_joint_char_query_energy_fraction = 0.0;
        double mean_joint_char_document_energy_fraction = 0.0;
        double char_document_prefix_rms = 0.0;
        double word_document_prefix_rms = 0.0;
        std::vector<double> top_overlaps;
    };

    struct BaseEvidence {
        std::size_t retrieval_depth = 0;
        std::vector<std::vector<float>> scores;
        std::vector<Ranking> rankings;
    };

    FeatureFamilyAtlasConfig identity;
    mutable FeatureFamilyAtlasWorkspaceStats stats;
    std::uint32_t dimension = 0;
    std::size_t document_count = 0;
    std::size_t query_count = 0;
    std::vector<float> char_documents;
    std::vector<float> char_queries;
    std::vector<float> word_documents;
    std::vector<float> word_queries;
    std::vector<std::string_view> document_views;
    std::vector<std::string_view> query_views;
    mutable std::unique_ptr<ScoreEvidence> cached_evidence;
    mutable std::unique_ptr<BaseEvidence> cached_base;

    Impl(const Fixture& fixture, const FeatureFamilyAtlasConfig& config) : identity(config) {
        const auto setup_start = Clock::now();
        if (fixture.query_ids.size() != fixture.query_texts.size() ||
            fixture.doc_ids.size() != fixture.doc_texts.size())
            throw std::runtime_error(
                "feature-family fixture ids and texts have different row counts");
        if (fixture.query_texts.empty() || fixture.doc_texts.empty())
            throw std::runtime_error(
                "feature-family fixture requires at least one query and document");

        document_views.reserve(fixture.doc_texts.size());
        for (const auto& document : fixture.doc_texts)
            document_views.emplace_back(document);
        query_views.reserve(fixture.query_texts.size());
        for (const auto& query : fixture.query_texts)
            query_views.emplace_back(query);

        dimension = config.idf_encoder.encoder.output_dim;
        document_count = fixture.doc_texts.size();
        query_count = fixture.query_texts.size();
        char_documents.resize(document_count * static_cast<std::size_t>(dimension));
        char_queries.resize(query_count * static_cast<std::size_t>(dimension));
        word_documents.resize(document_count * static_cast<std::size_t>(dimension));
        word_queries.resize(query_count * static_cast<std::size_t>(dimension));

        const auto encode_family = [&](NGramMode mode, std::vector<float>& documents,
                                       std::vector<float>& queries, double& artifact_us,
                                       double& document_us, double& query_us,
                                       std::uint64_t& artifact_bytes,
                                       std::string& artifact_fingerprint) {
            EncoderConfig family = family_config(config.idf_encoder.encoder, mode);
            const auto artifact_start = Clock::now();
            const HashedIdf artifact =
                HashedIdf::learn(document_views, family, config.idf_encoder.idf);
            artifact_us =
                std::chrono::duration<double, std::micro>(Clock::now() - artifact_start).count();
            artifact_bytes = artifact.storage_bytes();
            artifact_fingerprint = artifact.fingerprint();

            family.hashed_idf = &artifact;
            family.l2_normalize = false;
            const Encoder encoder(std::move(family));
            const auto document_start = Clock::now();
            encoder.encode_batch(document_views, documents.data());
            document_us =
                std::chrono::duration<double, std::micro>(Clock::now() - document_start).count();
            const auto query_start = Clock::now();
            encoder.encode_batch(query_views, queries.data());
            query_us =
                std::chrono::duration<double, std::micro>(Clock::now() - query_start).count();
        };

        encode_family(NGramMode::CharOnly, char_documents, char_queries,
                      stats.char_idf_artifact_build_us, stats.char_document_encode_us,
                      stats.char_query_encode_us, stats.char_idf_artifact_bytes,
                      stats.char_idf_artifact_fingerprint);
        encode_family(NGramMode::WordOnly, word_documents, word_queries,
                      stats.word_idf_artifact_build_us, stats.word_document_encode_us,
                      stats.word_query_encode_us, stats.word_idf_artifact_bytes,
                      stats.word_idf_artifact_fingerprint);

        stats.raw_document_vector_bytes =
            (char_documents.size() + word_documents.size()) * sizeof(float);
        stats.raw_query_vector_bytes = (char_queries.size() + word_queries.size()) * sizeof(float);
        stats.setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
    }

    bool ensure_base(std::size_t retrieval_depth) const {
        if (cached_base != nullptr && cached_base->retrieval_depth == retrieval_depth)
            return true;

        stats.base_idf_artifact_build_us = 0.0;
        stats.base_document_encode_us = 0.0;
        stats.base_query_encode_us = 0.0;
        stats.base_score_us = 0.0;
        stats.base_idf_artifact_bytes = 0;
        stats.base_score_evidence_bytes = 0;
        stats.base_idf_artifact_fingerprint.clear();
        auto base = std::make_unique<BaseEvidence>();
        base->retrieval_depth = retrieval_depth;
        EncoderConfig config = identity.idf_encoder.encoder;
        const auto artifact_start = Clock::now();
        const HashedIdf artifact =
            HashedIdf::learn(document_views, config, identity.idf_encoder.idf);
        stats.base_idf_artifact_build_us =
            std::chrono::duration<double, std::micro>(Clock::now() - artifact_start).count();
        stats.base_idf_artifact_bytes = artifact.storage_bytes();
        stats.base_idf_artifact_fingerprint = artifact.fingerprint();
        config.hashed_idf = &artifact;
        const Encoder encoder(std::move(config));

        std::vector<float> query_vectors(query_count * static_cast<std::size_t>(dimension));
        const auto query_start = Clock::now();
        encoder.encode_batch(query_views, query_vectors.data());
        stats.base_query_encode_us =
            std::chrono::duration<double, std::micro>(Clock::now() - query_start).count();

        base->scores.resize(query_count);
        for (auto& scores : base->scores)
            scores.resize(document_count);
        constexpr std::size_t kDocumentBlockSize = 256;
        const std::size_t block_capacity = std::min(kDocumentBlockSize, document_count);
        std::vector<std::string_view> block_views(block_capacity);
        std::vector<float> document_vectors(block_capacity * static_cast<std::size_t>(dimension));
        for (std::size_t block_start = 0; block_start < document_count;
             block_start += block_capacity) {
            const std::size_t block_size = std::min(block_capacity, document_count - block_start);
            for (std::size_t row = 0; row < block_size; ++row)
                block_views[row] = document_views[block_start + row];
            const auto document_start = Clock::now();
            encoder.encode_batch(std::span<const std::string_view>(block_views.data(), block_size),
                                 document_vectors.data());
            stats.base_document_encode_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - document_start).count();

            const auto score_start = Clock::now();
            for (std::size_t query = 0; query < query_count; ++query) {
                const float* query_vector = query_vectors.data() + query * dimension;
                std::size_t row = 0;
                for (; row + 4 <= block_size; row += 4) {
                    float dots[4];
                    const float* documents = document_vectors.data() + row * dimension;
                    simd::dot4(query_vector, documents, documents + dimension,
                               documents + 2 * dimension, documents + 3 * dimension, dots,
                               dimension);
                    for (std::size_t lane = 0; lane < 4; ++lane)
                        base->scores[query][block_start + row + lane] = dots[lane];
                }
                for (; row < block_size; ++row) {
                    base->scores[query][block_start + row] = simd::dot(
                        query_vector, document_vectors.data() + row * dimension, dimension);
                }
            }
            stats.base_score_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - score_start).count();
        }
        base->rankings.resize(query_count);
        for (std::size_t query = 0; query < query_count; ++query)
            base->rankings[query] = ranking_from_scores(base->scores[query], retrieval_depth);
        stats.base_score_evidence_bytes =
            static_cast<std::uint64_t>(query_count) * document_count * sizeof(float);
        cached_base = std::move(base);
        return false;
    }
};

FeatureFamilyAtlasWorkspace::FeatureFamilyAtlasWorkspace(const Fixture& fixture,
                                                         const FeatureFamilyAtlasConfig& config)
    : impl_(std::make_unique<Impl>(fixture, config)) {}

FeatureFamilyAtlasWorkspace::~FeatureFamilyAtlasWorkspace() = default;
FeatureFamilyAtlasWorkspace::FeatureFamilyAtlasWorkspace(FeatureFamilyAtlasWorkspace&&) noexcept =
    default;
FeatureFamilyAtlasWorkspace&
FeatureFamilyAtlasWorkspace::operator=(FeatureFamilyAtlasWorkspace&&) noexcept = default;

bool FeatureFamilyAtlasWorkspace::compatible(const FeatureFamilyAtlasConfig& config) const {
    return same_identity(impl_->identity, config);
}

FeatureFamilyAtlasRun FeatureFamilyAtlasWorkspace::run(const FeatureFamilyAtlasConfig& config,
                                                       std::size_t retrieval_depth) const {
    if (!compatible(config))
        throw std::runtime_error(
            "feature_family_atlas_idf variants must share maximum-width encoder/IDF identity");
    if (retrieval_depth == 0)
        throw std::runtime_error("feature-family retrieval depth must be greater than zero");

    const bool reusable = impl_->cached_evidence != nullptr &&
                          impl_->cached_evidence->char_dim == config.char_dim &&
                          impl_->cached_evidence->word_dim == config.word_dim &&
                          impl_->cached_evidence->retrieval_depth == retrieval_depth;
    if (!reusable) {
        auto evidence = std::make_unique<Impl::ScoreEvidence>();
        evidence->char_dim = config.char_dim;
        evidence->word_dim = config.word_dim;
        evidence->retrieval_depth = retrieval_depth;
        evidence->character =
            score_family(impl_->char_documents, impl_->char_queries, impl_->document_count,
                         impl_->query_count, impl_->dimension, config.char_dim, retrieval_depth);
        evidence->word =
            score_family(impl_->word_documents, impl_->word_queries, impl_->document_count,
                         impl_->query_count, impl_->dimension, config.word_dim, retrieval_depth);

        double overlap_sum = 0.0;
        double jaccard_sum = 0.0;
        double distortion_sum = 0.0;
        evidence->top_overlaps.resize(impl_->query_count);
        for (std::size_t query = 0; query < impl_->query_count; ++query) {
            std::unordered_set<std::uint32_t> character_documents;
            character_documents.reserve(evidence->character.rankings[query].size());
            for (const auto& [_, document] : evidence->character.rankings[query])
                character_documents.insert(document);
            std::unordered_set<std::uint32_t> union_documents = character_documents;
            std::size_t overlap = 0;
            for (const auto& [_, document] : evidence->word.rankings[query]) {
                overlap += character_documents.contains(document) ? 1 : 0;
                union_documents.insert(document);
            }
            const std::size_t denominator = std::max(evidence->character.rankings[query].size(),
                                                     evidence->word.rankings[query].size());
            const double top_overlap =
                denominator == 0 ? 1.0
                                 : static_cast<double>(overlap) / static_cast<double>(denominator);
            evidence->top_overlaps[query] = top_overlap;
            overlap_sum += top_overlap;
            jaccard_sum +=
                union_documents.empty()
                    ? 1.0
                    : static_cast<double>(overlap) / static_cast<double>(union_documents.size());
            double query_distortion = 0.0;
            for (const auto document : union_documents) {
                query_distortion +=
                    std::fabs(static_cast<double>(evidence->character.values[query][document]) -
                              static_cast<double>(evidence->word.values[query][document]));
            }
            distortion_sum += union_documents.empty()
                                  ? 0.0
                                  : query_distortion / static_cast<double>(union_documents.size());
        }
        const double count = static_cast<double>(impl_->query_count);
        evidence->mean_top_overlap = overlap_sum / count;
        evidence->mean_top_jaccard = jaccard_sum / count;
        evidence->mean_similarity_distortion = distortion_sum / count;
        double query_char_energy_sum = 0.0;
        for (std::size_t query = 0; query < impl_->query_count; ++query) {
            const double char_norm = evidence->character.query_norms[query];
            const double word_norm = evidence->word.query_norms[query];
            const double total = char_norm * char_norm + word_norm * word_norm;
            query_char_energy_sum += total > 0.0 ? char_norm * char_norm / total : 0.0;
        }
        double document_char_energy_sum = 0.0;
        double char_document_squared_sum = 0.0;
        double word_document_squared_sum = 0.0;
        for (std::size_t document = 0; document < impl_->document_count; ++document) {
            const double char_norm = evidence->character.document_norms[document];
            const double word_norm = evidence->word.document_norms[document];
            char_document_squared_sum += char_norm * char_norm;
            word_document_squared_sum += word_norm * word_norm;
            const double total = char_norm * char_norm + word_norm * word_norm;
            document_char_energy_sum += total > 0.0 ? char_norm * char_norm / total : 0.0;
        }
        evidence->mean_joint_char_query_energy_fraction = query_char_energy_sum / count;
        evidence->mean_joint_char_document_energy_fraction =
            document_char_energy_sum / static_cast<double>(impl_->document_count);
        evidence->char_document_prefix_rms =
            std::sqrt(char_document_squared_sum / static_cast<double>(impl_->document_count));
        evidence->word_document_prefix_rms =
            std::sqrt(word_document_squared_sum / static_cast<double>(impl_->document_count));
        impl_->cached_evidence = std::move(evidence);
    }

    const auto& evidence = *impl_->cached_evidence;
    bool base_reused = false;
    const bool needs_base = config.policy != FeatureFamilyPolicy::FamilyOnly;
    if (needs_base)
        base_reused = impl_->ensure_base(retrieval_depth);
    FeatureFamilyAtlasRun run;
    run.evidence_reused = reusable;
    run.base_evidence_available = needs_base;
    run.base_evidence_reused = base_reused;
    run.char_score_us = evidence.character.score_us;
    run.word_score_us = evidence.word.score_us;
    run.cached_score_evidence_bytes =
        static_cast<std::uint64_t>(impl_->query_count) * impl_->document_count * 2 * sizeof(float);
    if (needs_base)
        run.cached_score_evidence_bytes += impl_->stats.base_score_evidence_bytes;
    std::uint64_t deployable_dimensions = config.storage_budget_dim;
    if (config.policy == FeatureFamilyPolicy::BaseOnly)
        deployable_dimensions = impl_->dimension;
    else if (config.policy == FeatureFamilyPolicy::ResidualBlend ||
             config.policy == FeatureFamilyPolicy::Selective)
        deployable_dimensions += impl_->dimension;
    run.deployable_document_vector_bytes =
        static_cast<std::uint64_t>(impl_->document_count) * deployable_dimensions * sizeof(float);
    run.mean_char_prefix_energy_fraction = evidence.character.mean_query_prefix_energy_fraction;
    run.mean_word_prefix_energy_fraction = evidence.word.mean_query_prefix_energy_fraction;
    run.mean_joint_char_query_energy_fraction = evidence.mean_joint_char_query_energy_fraction;
    run.mean_joint_char_document_energy_fraction =
        evidence.mean_joint_char_document_energy_fraction;
    run.char_document_prefix_rms = evidence.char_document_prefix_rms;
    run.word_document_prefix_rms = evidence.word_document_prefix_rms;
    run.mean_top100_overlap = evidence.mean_top_overlap;
    run.mean_top100_jaccard = evidence.mean_top_jaccard;
    run.mean_similarity_distortion = evidence.mean_similarity_distortion;
    run.char_rankings = evidence.character.rankings;
    run.word_rankings = evidence.word.rankings;
    if (needs_base)
        run.base_rankings = impl_->cached_base->rankings;
    run.family_rankings.resize(impl_->query_count);
    run.rankings.resize(impl_->query_count);

    std::vector<float> family_scores(impl_->document_count);
    std::vector<float> final_scores(impl_->document_count);
    const float word_weight = 1.0f - config.char_weight;
    double calibrated_word_energy_sum = 0.0;
    double base_family_overlap_sum = 0.0;
    const auto replay_start = Clock::now();
    for (std::size_t query = 0; query < impl_->query_count; ++query) {
        for (std::size_t document = 0; document < impl_->document_count; ++document) {
            const float char_score = evidence.character.values[query][document];
            const float word_score = evidence.word.values[query][document];
            if (config.normalization == FeatureFamilyNormalization::Independent) {
                family_scores[document] =
                    config.char_weight * char_score + word_weight * word_score;
                continue;
            }
            const double char_query_norm = evidence.character.query_norms[query];
            const double word_query_norm = evidence.word.query_norms[query];
            const double char_document_norm = evidence.character.document_norms[document];
            const double word_document_norm = evidence.word.document_norms[document];
            double char_coefficient = static_cast<double>(config.char_weight);
            double word_coefficient = static_cast<double>(word_weight);
            if (config.normalization == FeatureFamilyNormalization::JointRms) {
                const double char_rms = evidence.char_document_prefix_rms;
                const double word_rms = evidence.word_document_prefix_rms;
                char_coefficient = char_rms > 0.0 ? char_coefficient / (char_rms * char_rms) : 0.0;
                word_coefficient = word_rms > 0.0 ? word_coefficient / (word_rms * word_rms) : 0.0;
            }
            const double numerator =
                char_coefficient * char_score * char_query_norm * char_document_norm +
                word_coefficient * word_score * word_query_norm * word_document_norm;
            const double query_squared = char_coefficient * char_query_norm * char_query_norm +
                                         word_coefficient * word_query_norm * word_query_norm;
            const double document_squared =
                char_coefficient * char_document_norm * char_document_norm +
                word_coefficient * word_document_norm * word_document_norm;
            const double denominator = std::sqrt(query_squared * document_squared);
            family_scores[document] = denominator > 0.0 && std::isfinite(denominator)
                                          ? static_cast<float>(numerator / denominator)
                                          : 0.0f;
        }
        run.family_rankings[query] = ranking_from_scores(family_scores, retrieval_depth);

        const double char_query_norm = evidence.character.query_norms[query];
        const double word_query_norm = evidence.word.query_norms[query];
        const double char_rms = evidence.char_document_prefix_rms;
        const double word_rms = evidence.word_document_prefix_rms;
        const double calibrated_char_squared =
            char_rms > 0.0 ? static_cast<double>(config.char_weight) * char_query_norm *
                                 char_query_norm / (char_rms * char_rms)
                           : 0.0;
        const double calibrated_word_squared =
            word_rms > 0.0 ? static_cast<double>(word_weight) * word_query_norm * word_query_norm /
                                 (word_rms * word_rms)
                           : 0.0;
        const double calibrated_total = calibrated_char_squared + calibrated_word_squared;
        const double calibrated_word_energy =
            calibrated_total > 0.0 ? calibrated_word_squared / calibrated_total : 0.0;
        calibrated_word_energy_sum += calibrated_word_energy;

        double base_family_overlap = 0.0;
        if (needs_base) {
            std::unordered_set<std::uint32_t> base_documents;
            base_documents.reserve(impl_->cached_base->rankings[query].size());
            for (const auto& [_, document] : impl_->cached_base->rankings[query])
                base_documents.insert(document);
            std::size_t overlap = 0;
            for (const auto& [_, document] : run.family_rankings[query])
                overlap += base_documents.contains(document) ? 1 : 0;
            const std::size_t denominator =
                std::max(base_documents.size(), run.family_rankings[query].size());
            base_family_overlap =
                denominator == 0 ? 1.0
                                 : static_cast<double>(overlap) / static_cast<double>(denominator);
            base_family_overlap_sum += base_family_overlap;
        }

        bool admit_residual = config.policy == FeatureFamilyPolicy::ResidualBlend;
        if (config.policy == FeatureFamilyPolicy::Selective) {
            admit_residual =
                calibrated_word_energy >= static_cast<double>(config.min_word_energy) &&
                calibrated_word_energy <= static_cast<double>(config.max_word_energy) &&
                evidence.top_overlaps[query] >= static_cast<double>(config.min_family_overlap) &&
                evidence.top_overlaps[query] <= static_cast<double>(config.max_family_overlap) &&
                base_family_overlap >= static_cast<double>(config.min_base_family_overlap) &&
                base_family_overlap <= static_cast<double>(config.max_base_family_overlap);
        }

        if (config.policy == FeatureFamilyPolicy::FamilyOnly) {
            run.rankings[query] = run.family_rankings[query];
        } else if (config.policy == FeatureFamilyPolicy::BaseOnly || !admit_residual) {
            run.rankings[query] = impl_->cached_base->rankings[query];
            if (config.policy == FeatureFamilyPolicy::Selective)
                ++run.rejected_queries;
        } else {
            const float base_weight = 1.0f - config.residual_weight;
            if (config.residual_score_normalization == FeatureResidualScoreNormalization::RankRrf) {
                std::fill(final_scores.begin(), final_scores.end(), 0.0f);
                const auto add_ranks = [&](const Ranking& ranking, float weight) {
                    for (std::size_t rank = 0; rank < ranking.size(); ++rank) {
                        final_scores[ranking[rank].second] +=
                            weight / (config.residual_rrf_k + static_cast<float>(rank + 1));
                    }
                };
                add_ranks(impl_->cached_base->rankings[query], base_weight);
                add_ranks(run.family_rankings[query], config.residual_weight);
                run.rankings[query] = ranking_from_scores(final_scores, retrieval_depth);
                ++run.admitted_queries;
                continue;
            }
            double base_mean = 0.0;
            double base_stddev = 1.0;
            double family_mean = 0.0;
            double family_stddev = 1.0;
            if (config.residual_score_normalization ==
                FeatureResidualScoreNormalization::QueryZScore) {
                std::tie(base_mean, base_stddev) =
                    mean_and_stddev(impl_->cached_base->scores[query]);
                std::tie(family_mean, family_stddev) = mean_and_stddev(family_scores);
            }
            for (std::size_t document = 0; document < impl_->document_count; ++document) {
                double base_score = impl_->cached_base->scores[query][document];
                double family_score = family_scores[document];
                if (config.residual_score_normalization ==
                    FeatureResidualScoreNormalization::QueryZScore) {
                    base_score = base_stddev > 0.0 ? (base_score - base_mean) / base_stddev : 0.0;
                    family_score =
                        family_stddev > 0.0 ? (family_score - family_mean) / family_stddev : 0.0;
                }
                final_scores[document] =
                    static_cast<float>(static_cast<double>(base_weight) * base_score +
                                       static_cast<double>(config.residual_weight) * family_score);
            }
            run.rankings[query] = ranking_from_scores(final_scores, retrieval_depth);
            ++run.admitted_queries;
        }
    }
    run.policy_replay_us =
        std::chrono::duration<double, std::micro>(Clock::now() - replay_start).count();
    run.mean_calibrated_word_query_energy =
        calibrated_word_energy_sum / static_cast<double>(impl_->query_count);
    if (needs_base)
        run.mean_base_family_top100_overlap =
            base_family_overlap_sum / static_cast<double>(impl_->query_count);
    return run;
}

const FeatureFamilyAtlasWorkspaceStats& FeatureFamilyAtlasWorkspace::stats() const noexcept {
    return impl_->stats;
}

double family_union_recall_at_100(const std::vector<Ranking>& char_rankings,
                                  const std::vector<Ranking>& word_rankings,
                                  const Fixture& fixture) {
    if (char_rankings.size() != fixture.query_ids.size() ||
        word_rankings.size() != fixture.query_ids.size())
        throw std::runtime_error("family ranking count must match fixture query count");
    std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> relevance(
        fixture.query_ids.size());
    for (const auto& qrel : fixture.qrels) {
        if (qrel.query >= relevance.size() || qrel.document >= fixture.doc_ids.size())
            throw std::runtime_error("qrel index is outside the feature-family fixture");
        relevance[qrel.query][qrel.document] =
            std::max(relevance[qrel.query][qrel.document], qrel.relevance);
    }
    double recall_sum = 0.0;
    std::size_t evaluated = 0;
    for (std::size_t query = 0; query < relevance.size(); ++query) {
        if (relevance[query].empty())
            continue;
        ++evaluated;
        std::unordered_set<std::uint32_t> candidates;
        for (std::size_t rank = 0; rank < char_rankings[query].size() && rank < 100; ++rank)
            candidates.insert(char_rankings[query][rank].second);
        for (std::size_t rank = 0; rank < word_rankings[query].size() && rank < 100; ++rank)
            candidates.insert(word_rankings[query][rank].second);
        std::size_t hits = 0;
        for (const auto& [document, grade] : relevance[query])
            hits += grade > 0 && candidates.contains(document) ? 1 : 0;
        recall_sum += static_cast<double>(hits) / static_cast<double>(relevance[query].size());
    }
    return evaluated == 0 ? 0.0 : recall_sum / static_cast<double>(evaluated);
}

std::string feature_family_atlas_result_json(
    const ResultContext& context, const FeatureFamilyAtlasConfig& config,
    const FeatureFamilyAtlasWorkspaceStats& workspace, const FeatureFamilyAtlasRun& run,
    const Metrics& metrics, const Metrics& base_metrics, const Metrics& family_metrics,
    const Metrics& char_metrics, const Metrics& word_metrics, double union_recall_at_100,
    double evaluation_us, std::size_t query_count, std::size_t document_count,
    std::size_t qrel_count) {
    EncoderConfig char_config = family_config(config.idf_encoder.encoder, NGramMode::CharOnly);
    EncoderConfig word_config = family_config(config.idf_encoder.encoder, NGramMode::WordOnly);
    std::uint64_t deployable_dimensions = config.storage_budget_dim;
    if (config.policy == FeatureFamilyPolicy::BaseOnly)
        deployable_dimensions = config.idf_encoder.encoder.output_dim;
    else if (config.policy == FeatureFamilyPolicy::ResidualBlend ||
             config.policy == FeatureFamilyPolicy::Selective)
        deployable_dimensions += config.idf_encoder.encoder.output_dim;
    const bool exact_single_vector =
        config.residual_score_normalization == FeatureResidualScoreNormalization::RawCosine ||
        config.policy == FeatureFamilyPolicy::FamilyOnly ||
        config.policy == FeatureFamilyPolicy::BaseOnly;
    const bool requires_query_corpus_score_moments =
        (config.policy == FeatureFamilyPolicy::ResidualBlend ||
         config.policy == FeatureFamilyPolicy::Selective) &&
        config.residual_score_normalization == FeatureResidualScoreNormalization::QueryZScore;
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
        out << '\"' << json_escape(key) << "\":\"" << json_escape(value) << '\"';
    }
    out << "},\"config\":{\"char_dim\":" << config.char_dim << ",\"word_dim\":" << config.word_dim
        << ",\"storage_budget_dim\":" << config.storage_budget_dim << ",\"family_normalization\":\""
        << normalization_name(config.normalization) << "\""
        << ",\"char_weight\":" << config.char_weight
        << ",\"word_weight\":" << (1.0f - config.char_weight) << ",\"family_policy\":\""
        << policy_name(config.policy) << "\""
        << ",\"residual_score_normalization\":\""
        << residual_score_normalization_name(config.residual_score_normalization) << "\""
        << ",\"residual_weight\":" << config.residual_weight
        << ",\"residual_rrf_k\":" << config.residual_rrf_k
        << ",\"min_word_energy\":" << config.min_word_energy
        << ",\"max_word_energy\":" << config.max_word_energy
        << ",\"min_family_overlap\":" << config.min_family_overlap
        << ",\"max_family_overlap\":" << config.max_family_overlap
        << ",\"min_base_family_overlap\":" << config.min_base_family_overlap
        << ",\"max_base_family_overlap\":" << config.max_base_family_overlap
        << ",\"score_composition\":{\"base_dim\":"
        << (run.base_evidence_available ? config.idf_encoder.encoder.output_dim : 0)
        << ",\"deployable_total_dim\":" << deployable_dimensions
        << ",\"exact_single_vector\":" << (exact_single_vector ? "true" : "false")
        << ",\"requires_query_corpus_score_moments\":"
        << (requires_query_corpus_score_moments ? "true" : "false") << '}'
        << ",\"char_encoder\":" << encoder_config_json(char_config)
        << ",\"word_encoder\":" << encoder_config_json(word_config)
        << ",\"family_artifacts\":{\"char\":{\"fingerprint\":\""
        << json_escape(workspace.char_idf_artifact_fingerprint)
        << "\",\"hash_dim\":" << config.idf_encoder.idf.hash_dim
        << ",\"storage_bytes\":" << workspace.char_idf_artifact_bytes
        << "},\"word\":{\"fingerprint\":\"" << json_escape(workspace.word_idf_artifact_fingerprint)
        << "\",\"hash_dim\":" << config.idf_encoder.idf.hash_dim
        << ",\"storage_bytes\":" << workspace.word_idf_artifact_bytes << "}}"
        << ",\"base_artifact\":";
    if (run.base_evidence_available) {
        out << "{\"fingerprint\":\"" << json_escape(workspace.base_idf_artifact_fingerprint)
            << "\",\"hash_dim\":" << config.idf_encoder.idf.hash_dim
            << ",\"storage_bytes\":" << workspace.base_idf_artifact_bytes << '}';
    } else {
        out << "null";
    }
    out << '}' << ",\"counts\":{\"queries\":" << query_count << ",\"documents\":" << document_count
        << ",\"qrels\":" << qrel_count << ",\"evaluated_queries\":" << metrics.evaluated_queries
        << "},\"metrics\":";
    append_metrics(out, metrics);
    out << ",\"family_metrics\":{\"base\":";
    if (run.base_evidence_available)
        append_metrics(out, base_metrics);
    else
        out << "null";
    out << ",\"fused\":";
    append_metrics(out, family_metrics);
    out << ",\"char\":";
    append_metrics(out, char_metrics);
    out << ",\"word\":";
    append_metrics(out, word_metrics);
    out << ",\"evaluation_only_union_recall_at_100\":" << union_recall_at_100 << '}'
        << ",\"chart_evidence\":{\"mean_char_prefix_energy_fraction\":"
        << run.mean_char_prefix_energy_fraction
        << ",\"mean_word_prefix_energy_fraction\":" << run.mean_word_prefix_energy_fraction
        << ",\"mean_joint_char_query_energy_fraction\":"
        << run.mean_joint_char_query_energy_fraction
        << ",\"mean_joint_char_document_energy_fraction\":"
        << run.mean_joint_char_document_energy_fraction
        << ",\"char_document_prefix_rms\":" << run.char_document_prefix_rms
        << ",\"word_document_prefix_rms\":" << run.word_document_prefix_rms
        << ",\"mean_top100_overlap\":" << run.mean_top100_overlap
        << ",\"mean_top100_jaccard\":" << run.mean_top100_jaccard
        << ",\"mean_similarity_distortion\":" << run.mean_similarity_distortion
        << ",\"mean_calibrated_word_query_energy\":" << run.mean_calibrated_word_query_energy
        << ",\"mean_base_family_top100_overlap\":" << run.mean_base_family_top100_overlap
        << ",\"admitted_queries\":" << run.admitted_queries
        << ",\"rejected_queries\":" << run.rejected_queries
        << ",\"evidence_reused\":" << (run.evidence_reused ? "true" : "false")
        << ",\"base_evidence_available\":" << (run.base_evidence_available ? "true" : "false")
        << ",\"base_evidence_reused\":" << (run.base_evidence_reused ? "true" : "false") << '}'
        << ",\"timing_us\":{\"workspace_setup_total\":" << workspace.setup_us
        << ",\"char_idf_artifact_build_total\":" << workspace.char_idf_artifact_build_us
        << ",\"word_idf_artifact_build_total\":" << workspace.word_idf_artifact_build_us
        << ",\"char_document_encode_total\":" << workspace.char_document_encode_us
        << ",\"word_document_encode_total\":" << workspace.word_document_encode_us
        << ",\"char_query_encode_total\":" << workspace.char_query_encode_us
        << ",\"word_query_encode_total\":" << workspace.word_query_encode_us
        << ",\"base_idf_artifact_build_total\":" << workspace.base_idf_artifact_build_us
        << ",\"base_document_encode_total\":" << workspace.base_document_encode_us
        << ",\"base_query_encode_total\":" << workspace.base_query_encode_us
        << ",\"base_score_total\":" << workspace.base_score_us
        << ",\"char_score_total\":" << run.char_score_us
        << ",\"word_score_total\":" << run.word_score_us
        << ",\"policy_replay_total\":" << run.policy_replay_us
        << ",\"evaluation_total\":" << evaluation_us << '}'
        << ",\"resources\":{\"idf_artifact_bytes\":"
        << (workspace.char_idf_artifact_bytes + workspace.word_idf_artifact_bytes +
            (run.base_evidence_available ? workspace.base_idf_artifact_bytes : 0))
        << ",\"raw_document_vector_bytes\":" << workspace.raw_document_vector_bytes
        << ",\"raw_query_vector_bytes\":" << workspace.raw_query_vector_bytes
        << ",\"deployable_document_vector_bytes\":" << run.deployable_document_vector_bytes
        << ",\"cached_score_evidence_bytes\":" << run.cached_score_evidence_bytes << "}}";
    return out.str();
}

} // namespace simeon::experiment
