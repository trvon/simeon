#include "reranker_experiment_support.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "simeon/bm25.hpp"
#include "simeon/simd.hpp"

namespace simeon::experiment {
namespace {

using Clock = std::chrono::steady_clock;

std::uint32_t parse_u32(std::string_view value, std::string_view key) {
    std::uint64_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
    if (error != std::errc{} || end != value.data() + value.size() ||
        parsed > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("invalid uint32 for '" + std::string(key) +
                                 "': " + std::string(value));
    return static_cast<std::uint32_t>(parsed);
}

float parse_float(std::string_view value, std::string_view key) {
    std::istringstream input{std::string(value)};
    input.imbue(std::locale::classic());
    float parsed = 0.0f;
    if (!(input >> parsed) || !std::isfinite(parsed))
        throw std::runtime_error("invalid finite float for '" + std::string(key) + "'");
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error("invalid finite float for '" + std::string(key) + "'");
    return parsed;
}

bool parse_bool(std::string_view value, std::string_view key) {
    if (value == "true" || value == "1" || value == "yes" || value == "on")
        return true;
    if (value == "false" || value == "0" || value == "no" || value == "off")
        return false;
    throw std::runtime_error("invalid boolean for '" + std::string(key) +
                             "': " + std::string(value));
}

std::size_t checked_size_product(std::size_t lhs, std::size_t rhs, const char* description) {
    if (lhs == 0 || rhs == 0)
        return 0;
    if (rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::length_error(description);
    return lhs * rhs;
}

const char* encoder_profile_name(RerankerEncoderProfile profile) {
    return profile == RerankerEncoderProfile::FixedHash ? "fixed_hash" : "pmi_word";
}

const char* builder_name(FragmentBuilderKind builder) {
    switch (builder) {
        case FragmentBuilderKind::Basic:
            return "basic";
        case FragmentBuilderKind::Rich:
            return "rich";
        case FragmentBuilderKind::RichCovered:
            return "rich_covered";
    }
    return "unknown";
}

const char* storage_name(FragmentStorageKind storage) {
    return storage == FragmentStorageKind::Float32 ? "f32" : "bf16";
}

const char* doc_scorer_name(FragmentGeometryConfig::DocScorerKind scorer) {
    using Kind = FragmentGeometryConfig::DocScorerKind;
    switch (scorer) {
        case Kind::MaxSim:
            return "maxsim";
        case Kind::MeanSim:
            return "mean";
        case Kind::TopKMean:
            return "topk_mean";
        case Kind::SoftMaxSum:
            return "softmax_sum";
        case Kind::GeoMean:
            return "geomean";
    }
    return "unknown";
}

const char* phss_criterion_name(PhssConfig::Criterion criterion) {
    switch (criterion) {
        case PhssConfig::Criterion::LargestGap:
            return "largest_gap";
        case PhssConfig::Criterion::LargestGapApprox:
            return "largest_gap_approx";
        case PhssConfig::Criterion::MaxPersistence:
            return "max_persistence";
        case PhssConfig::Criterion::Elbow:
            return "elbow";
    }
    return "unknown";
}

bool same_pmi_config(const PmiConfig& lhs, const PmiConfig& rhs) {
    return lhs.window_size == rhs.window_size && lhs.target_rank == rhs.target_rank &&
           lhs.min_token_count == rhs.min_token_count && lhs.max_vocab_size == rhs.max_vocab_size &&
           lhs.shift_log_k == rhs.shift_log_k && lhs.svd_iters == rhs.svd_iters &&
           lhs.svd_oversample == rhs.svd_oversample && lhs.svd_seed == rhs.svd_seed;
}

bool same_workspace_identity(const FragmentRerankerConfig& lhs, const FragmentRerankerConfig& rhs) {
    if (lhs.encoder_profile != rhs.encoder_profile || lhs.builder != rhs.builder ||
        lhs.storage != rhs.storage || lhs.top_sentence_fragments != rhs.top_sentence_fragments ||
        lhs.fragment_signature_terms != rhs.fragment_signature_terms ||
        lhs.sentence_overlap_cap != rhs.sentence_overlap_cap ||
        lhs.anchor_overlap_cap != rhs.anchor_overlap_cap)
        return false;
    if (lhs.encoder_profile == RerankerEncoderProfile::PmiWord)
        return same_pmi_config(lhs.pmi, rhs.pmi);
    return encoder_config_json(lhs.fixed_encoder) == encoder_config_json(rhs.fixed_encoder);
}

Ranking ranking_from_scores(std::span<const float> scores, std::size_t depth) {
    Ranking ranking;
    ranking.reserve(std::min(depth, scores.size()));
    for (std::size_t document = 0; document < scores.size(); ++document) {
        if (std::isfinite(scores[document]))
            ranking.emplace_back(scores[document], static_cast<std::uint32_t>(document));
    }
    std::sort(ranking.begin(), ranking.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first)
            return lhs.first > rhs.first;
        return lhs.second < rhs.second;
    });
    if (ranking.size() > depth)
        ranking.resize(depth);
    return ranking;
}

double candidate_recall(const std::vector<Ranking>& candidate_rankings, const Fixture& fixture) {
    if (candidate_rankings.size() != fixture.query_ids.size())
        throw std::runtime_error("candidate ranking count must match fixture query count");
    std::vector<std::vector<std::uint32_t>> relevant(fixture.query_ids.size());
    for (const auto& qrel : fixture.qrels)
        relevant.at(qrel.query).push_back(qrel.document);
    double sum = 0.0;
    std::size_t queries = 0;
    for (std::size_t query = 0; query < relevant.size(); ++query) {
        const std::size_t relevant_count = relevant[query].size();
        if (relevant_count == 0)
            continue;
        std::unordered_set<std::uint32_t> candidates;
        candidates.reserve(candidate_rankings[query].size());
        for (const auto& [_, document] : candidate_rankings[query])
            candidates.insert(document);
        std::size_t found = 0;
        for (const auto document : relevant[query])
            found += candidates.contains(document) ? 1 : 0;
        sum += static_cast<double>(found) / static_cast<double>(relevant_count);
        ++queries;
    }
    return queries == 0 ? 0.0 : sum / static_cast<double>(queries);
}

template <class Field>
double profile_mean(const std::vector<FragmentGeometryProfile>& profiles, Field&& field) {
    double sum = 0.0;
    for (const auto& profile : profiles)
        sum += static_cast<double>(field(profile));
    return profiles.empty() ? 0.0 : sum / static_cast<double>(profiles.size());
}

double quantile(std::vector<double> values, double probability) {
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const auto rank =
        static_cast<std::size_t>(std::ceil(probability * static_cast<double>(values.size())));
    const std::size_t index = rank == 0 ? 0 : rank - 1;
    return values[index];
}

RerankerProfileSummary summarize(const std::vector<FragmentGeometryProfile>& profiles) {
    RerankerProfileSummary summary;
    std::vector<double> totals;
    totals.reserve(profiles.size());
    for (const auto& profile : profiles)
        totals.push_back(profile.total_us);
    summary.total_mean_us = profile_mean(profiles, [](const auto& p) { return p.total_us; });
    summary.total_p50_us = quantile(totals, 0.50);
    summary.total_p95_us = quantile(std::move(totals), 0.95);
    summary.bm25_mean_us = profile_mean(profiles, [](const auto& p) { return p.bm25_us; });
    summary.query_encode_mean_us =
        profile_mean(profiles, [](const auto& p) { return p.query_encode_us; });
    summary.gather_mean_us = profile_mean(profiles, [](const auto& p) { return p.gather_us; });
    summary.whiten_mean_us = profile_mean(profiles, [](const auto& p) { return p.whiten_us; });
    summary.pairwise_mean_us =
        profile_mean(profiles, [](const auto& p) { return p.phss_pairwise_us; });
    summary.phss_select_mean_us =
        profile_mean(profiles, [](const auto& p) { return p.phss_select_us; });
    summary.query_attention_mean_us =
        profile_mean(profiles, [](const auto& p) { return p.query_attention_us; });
    summary.adjacency_mean_us =
        profile_mean(profiles, [](const auto& p) { return p.adjacency_us; });
    summary.diffuse_mean_us = profile_mean(profiles, [](const auto& p) { return p.diffuse_us; });
    summary.blend_mean_us = profile_mean(profiles, [](const auto& p) { return p.blend_us; });
    summary.pool_docs_mean = profile_mean(profiles, [](const auto& p) { return p.pool_docs; });
    summary.pool_fragments_mean =
        profile_mean(profiles, [](const auto& p) { return p.pool_fragments; });
    summary.graph_edges_mean = profile_mean(profiles, [](const auto& p) { return p.graph_edges; });
    summary.phss_used_rate =
        profile_mean(profiles, [](const auto& p) { return p.phss_used ? 1.0 : 0.0; });
    return summary;
}

void append_metadata(std::ostringstream& out, const std::map<std::string, std::string>& metadata) {
    bool first = true;
    for (const auto& [key, value] : metadata) {
        if (!first)
            out << ',';
        first = false;
        out << '"' << json_escape(key) << "\":\"" << json_escape(value) << '"';
    }
}

void append_metrics(std::ostringstream& out, const Metrics& metrics) {
    out << "{\"ndcg_at_10\":" << metrics.ndcg_at_10
        << ",\"precision_at_10\":" << metrics.precision_at_10
        << ",\"recall_at_10\":" << metrics.recall_at_10
        << ",\"recall_at_100\":" << metrics.recall_at_100 << ",\"mrr_at_10\":" << metrics.mrr_at_10
        << '}';
}

} // namespace

FragmentRerankerConfig resolve_fragment_reranker_config(const VariantSpec& variant) {
    if (variant.kind != "fragment_geometry_reranker")
        throw std::runtime_error("variant '" + variant.name + "' has kind '" + variant.kind +
                                 "'; expected 'fragment_geometry_reranker'");

    FragmentRerankerConfig result;
    result.fixed_encoder = simeon_v1_384_config();
    VariantSpec encoder_variant = variant;
    encoder_variant.kind = "encoder";
    const bool pmi_parameters_supplied =
        std::any_of(encoder_variant.parameters.begin(), encoder_variant.parameters.end(),
                    [](const auto& parameter) { return parameter.first.starts_with("pmi_"); });
    const bool overlap_caps_supplied =
        encoder_variant.parameters.contains("sentence_overlap_cap") ||
        encoder_variant.parameters.contains("anchor_overlap_cap");
    const auto take = [&](std::string_view key) -> const std::string* {
        const auto found = encoder_variant.parameters.find(std::string(key));
        if (found == encoder_variant.parameters.end())
            return nullptr;
        return &found->second;
    };
    const auto erase = [&](std::string_view key) {
        encoder_variant.parameters.erase(std::string(key));
    };

    if (const auto* value = take("encoder_profile")) {
        if (*value == "fixed_hash")
            result.encoder_profile = RerankerEncoderProfile::FixedHash;
        else if (*value == "pmi_word")
            result.encoder_profile = RerankerEncoderProfile::PmiWord;
        else
            throw std::runtime_error("unsupported encoder_profile '" + *value + "'");
        erase("encoder_profile");
    }
    if (const auto* value = take("builder")) {
        if (*value == "basic")
            result.builder = FragmentBuilderKind::Basic;
        else if (*value == "rich")
            result.builder = FragmentBuilderKind::Rich;
        else if (*value == "rich_covered")
            result.builder = FragmentBuilderKind::RichCovered;
        else
            throw std::runtime_error("unsupported fragment builder '" + *value + "'");
        erase("builder");
    }
    if (const auto* value = take("fragment_storage")) {
        if (*value == "f32")
            result.storage = FragmentStorageKind::Float32;
        else if (*value == "bf16")
            result.storage = FragmentStorageKind::BFloat16;
        else
            throw std::runtime_error("unsupported fragment_storage '" + *value + "'");
        erase("fragment_storage");
    }

    const auto take_u32 = [&](const char* key, std::uint32_t& destination) {
        if (const auto* value = take(key)) {
            destination = parse_u32(*value, key);
            erase(key);
        }
    };
    const auto take_float = [&](const char* key, float& destination) {
        if (const auto* value = take(key)) {
            destination = parse_float(*value, key);
            erase(key);
        }
    };
    const auto take_bool = [&](const char* key, bool& destination) {
        if (const auto* value = take(key)) {
            destination = parse_bool(*value, key);
            erase(key);
        }
    };
    take_u32("top_sentence_fragments", result.top_sentence_fragments);
    take_u32("fragment_signature_terms", result.fragment_signature_terms);
    take_float("sentence_overlap_cap", result.sentence_overlap_cap);
    take_float("anchor_overlap_cap", result.anchor_overlap_cap);
    take_u32("pool_size", result.geometry.pool_size);
    take_float("alpha", result.geometry.alpha);
    take_u32("top_fragments_per_doc", result.geometry.top_fragments_per_doc);
    take_float("attention_scale", result.geometry.attention_scale);
    take_u32("knn", result.geometry.knn);
    take_u32("steps", result.geometry.steps);
    take_bool("use_phss", result.geometry.use_phss);
    take_bool("phss_adaptive", result.geometry.phss_adaptive);
    take_float("phss_confidence_threshold", result.geometry.phss_confidence_threshold);
    take_bool("outer_maxsim", result.geometry.outer_maxsim);
    take_bool("whiten", result.geometry.whiten);
    take_u32("doc_scorer_top_k", result.geometry.doc_scorer_top_k);
    take_float("doc_scorer_softmax_beta", result.geometry.doc_scorer_softmax_beta);
    take_u32("csls_k", result.geometry.csls_k);
    take_float("csls_beta", result.geometry.csls_beta);
    take_u32("graph_prefix_dim", result.geometry.graph_prefix_dim);
    if (const auto* value = take("doc_scorer")) {
        using Kind = FragmentGeometryConfig::DocScorerKind;
        if (*value == "maxsim")
            result.geometry.doc_scorer_kind = Kind::MaxSim;
        else if (*value == "mean")
            result.geometry.doc_scorer_kind = Kind::MeanSim;
        else if (*value == "topk_mean")
            result.geometry.doc_scorer_kind = Kind::TopKMean;
        else if (*value == "softmax_sum")
            result.geometry.doc_scorer_kind = Kind::SoftMaxSum;
        else if (*value == "geomean")
            result.geometry.doc_scorer_kind = Kind::GeoMean;
        else
            throw std::runtime_error("unsupported doc_scorer '" + *value + "'");
        erase("doc_scorer");
    }
    if (const auto* value = take("phss_criterion")) {
        if (*value == "largest_gap")
            result.geometry.phss_config.criterion = PhssConfig::Criterion::LargestGap;
        else if (*value == "largest_gap_approx")
            result.geometry.phss_config.criterion = PhssConfig::Criterion::LargestGapApprox;
        else if (*value == "max_persistence")
            result.geometry.phss_config.criterion = PhssConfig::Criterion::MaxPersistence;
        else if (*value == "elbow")
            result.geometry.phss_config.criterion = PhssConfig::Criterion::Elbow;
        else
            throw std::runtime_error("unsupported phss_criterion '" + *value + "'");
        erase("phss_criterion");
    }

    take_u32("pmi_window_size", result.pmi.window_size);
    take_u32("pmi_target_rank", result.pmi.target_rank);
    take_u32("pmi_min_token_count", result.pmi.min_token_count);
    take_u32("pmi_max_vocab_size", result.pmi.max_vocab_size);
    take_float("pmi_shift_log_k", result.pmi.shift_log_k);
    take_u32("pmi_svd_iters", result.pmi.svd_iters);
    take_u32("pmi_svd_oversample", result.pmi.svd_oversample);

    if (result.encoder_profile == RerankerEncoderProfile::PmiWord) {
        if (!encoder_variant.parameters.empty())
            throw std::runtime_error("PMI reranker does not accept fixed encoder parameter '" +
                                     encoder_variant.parameters.begin()->first + "'");
    } else {
        if (pmi_parameters_supplied)
            throw std::runtime_error("fixed-hash reranker does not accept PMI parameters");
        encoder_variant.parameters.try_emplace("sketch_dim", "4096");
        encoder_variant.parameters.try_emplace("output_dim", "384");
        encoder_variant.parameters.try_emplace("projection", "achlioptas");
        result.fixed_encoder = resolve_encoder_config(encoder_variant);
    }
    if (result.top_sentence_fragments == 0 || result.fragment_signature_terms == 0)
        throw std::runtime_error("fragment builder counts must be greater than zero");
    if (overlap_caps_supplied && result.builder != FragmentBuilderKind::RichCovered)
        throw std::runtime_error("fragment overlap caps require builder=rich_covered");
    if (!(result.sentence_overlap_cap >= 0.0f && result.sentence_overlap_cap <= 1.0f) ||
        !(result.anchor_overlap_cap >= 0.0f && result.anchor_overlap_cap <= 1.0f))
        throw std::runtime_error("fragment overlap caps must be within [0, 1]");
    if (result.geometry.pool_size == 0 || result.geometry.top_fragments_per_doc == 0)
        throw std::runtime_error("reranker pool and per-document fragment counts must be positive");
    if (!(result.geometry.alpha >= 0.0f && result.geometry.alpha <= 1.0f))
        throw std::runtime_error("reranker alpha must be within [0, 1]");
    if (!(result.geometry.attention_scale > 0.0f))
        throw std::runtime_error("reranker attention_scale must be positive");
    if (!(result.geometry.phss_confidence_threshold >= 0.0f &&
          result.geometry.phss_confidence_threshold <= 1.0f))
        throw std::runtime_error("reranker PHSS confidence threshold must be within [0, 1]");
    if (result.geometry.csls_k > 0 && result.geometry.csls_beta < 0.0f)
        throw std::runtime_error("reranker CSLS beta must be non-negative");
    if (result.geometry.outer_maxsim &&
        result.geometry.doc_scorer_kind == FragmentGeometryConfig::DocScorerKind::SoftMaxSum &&
        result.geometry.doc_scorer_softmax_beta <= 0.0f)
        throw std::runtime_error("reranker softmax beta must be positive");
    if (result.encoder_profile == RerankerEncoderProfile::PmiWord &&
        (result.pmi.target_rank == 0 || result.pmi.min_token_count == 0 ||
         result.pmi.max_vocab_size == 0))
        throw std::runtime_error("PMI rank and vocabulary limits must be positive");
    return result;
}

struct FragmentRerankerWorkspace::Impl {
    FragmentRerankerConfig identity;
    RerankerWorkspaceStats stats;
    Bm25Index index;
    std::unique_ptr<PmiEmbeddings> pmi;
    std::unique_ptr<Encoder> encoder;
    std::vector<std::vector<SemanticFragment>> fragments;
    std::vector<std::string> query_texts;

    Impl(const Fixture& source, const FragmentRerankerConfig& config) : identity(config) {
        const auto setup_start = Clock::now();
        if (source.query_ids.size() != source.query_texts.size() ||
            source.doc_ids.size() != source.doc_texts.size())
            throw std::runtime_error("reranker fixture ids and texts have different row counts");
        if (source.query_texts.empty() || source.doc_texts.empty())
            throw std::runtime_error("reranker fixture requires queries and documents");
        if (source.doc_texts.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("reranker fixture exceeds uint32 document capacity");
        query_texts = source.query_texts;

        index.reserve_docs(source.doc_texts.size());
        const auto add_start = Clock::now();
        for (const auto& document : source.doc_texts)
            index.add_doc(document);
        const auto add_end = Clock::now();
        index.finalize();
        const auto finalize_end = Clock::now();
        stats.bm25_add_us = std::chrono::duration<double, std::micro>(add_end - add_start).count();
        stats.bm25_finalize_us =
            std::chrono::duration<double, std::micro>(finalize_end - add_end).count();

        EncoderConfig encoder_config = config.fixed_encoder;
        if (config.encoder_profile == RerankerEncoderProfile::PmiWord) {
            std::vector<std::string_view> views;
            views.reserve(source.doc_texts.size());
            for (const auto& document : source.doc_texts)
                views.emplace_back(document);
            const auto artifact_start = Clock::now();
            pmi = std::make_unique<PmiEmbeddings>(PmiEmbeddings::learn(views, config.pmi));
            stats.encoder_artifact_build_us =
                std::chrono::duration<double, std::micro>(Clock::now() - artifact_start).count();
            stats.pmi_vocab_size = pmi->vocab_size();
            const std::uint64_t row_count =
                static_cast<std::uint64_t>(pmi->vocab_size()) * pmi->dim();
            if (row_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float))
                throw std::length_error("PMI artifact byte count is too large");
            stats.encoder_artifact_bytes_lower_bound = row_count * sizeof(float);
            encoder_config = EncoderConfig{};
            encoder_config.ngram_mode = NGramMode::WordOnly;
            encoder_config.ngram_min = 1;
            encoder_config.ngram_max = 1;
            encoder_config.sketch_dim = 0;
            encoder_config.output_dim = pmi->dim();
            encoder_config.projection = ProjectionMode::None;
            encoder_config.l2_normalize = true;
            encoder_config.pmi_rows = pmi.get();
        }
        const auto encoder_start = Clock::now();
        encoder = std::make_unique<Encoder>(encoder_config);
        stats.encoder_init_us =
            std::chrono::duration<double, std::micro>(Clock::now() - encoder_start).count();
        stats.encoder_dim = encoder->output_dim();

        const auto fragment_start = Clock::now();
        fragments.resize(source.doc_texts.size());
        for (std::size_t document = 0; document < source.doc_texts.size(); ++document) {
            const auto prep =
                prepare_doc(source.doc_texts[document], index, config.top_sentence_fragments,
                            config.fragment_signature_terms);
            switch (config.builder) {
                case FragmentBuilderKind::Basic:
                    fragments[document] = build_doc_semantic_fragments_from_prep(
                        *encoder, source.doc_texts[document], prep);
                    break;
                case FragmentBuilderKind::Rich:
                    fragments[document] = build_doc_semantic_fragments_rich_from_prep(
                        *encoder, source.doc_texts[document], prep);
                    break;
                case FragmentBuilderKind::RichCovered:
                    fragments[document] = build_doc_semantic_fragments_rich_covered_from_prep(
                        *encoder, source.doc_texts[document], prep, config.sentence_overlap_cap,
                        config.anchor_overlap_cap);
                    break;
            }
        }
        stats.fragment_build_us =
            std::chrono::duration<double, std::micro>(Clock::now() - fragment_start).count();
        if (config.storage == FragmentStorageKind::BFloat16) {
            const auto compress_start = Clock::now();
            compress_fragments_to_bf16(fragments, encoder->output_dim());
            stats.fragment_compress_us =
                std::chrono::duration<double, std::micro>(Clock::now() - compress_start).count();
        }
        for (const auto& document : fragments) {
            stats.fragment_count += document.size();
            for (const auto& fragment : document) {
                stats.fragment_vector_bytes += fragment.vec.size() * sizeof(float);
                stats.fragment_vector_bytes += fragment.vec_bf16.size() * sizeof(std::uint16_t);
            }
        }
        stats.setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
    }
};

FragmentRerankerWorkspace::FragmentRerankerWorkspace(const Fixture& fixture,
                                                     const FragmentRerankerConfig& config)
    : impl_(std::make_unique<Impl>(fixture, config)) {}

FragmentRerankerWorkspace::~FragmentRerankerWorkspace() = default;
FragmentRerankerWorkspace::FragmentRerankerWorkspace(FragmentRerankerWorkspace&&) noexcept =
    default;
FragmentRerankerWorkspace&
FragmentRerankerWorkspace::operator=(FragmentRerankerWorkspace&&) noexcept = default;

bool FragmentRerankerWorkspace::compatible(const FragmentRerankerConfig& config) const {
    return same_workspace_identity(impl_->identity, config);
}

RerankerRetrievalRun FragmentRerankerWorkspace::run(const FragmentRerankerConfig& config,
                                                    std::size_t query_repeats,
                                                    std::size_t retrieval_depth) const {
    if (!compatible(config))
        throw std::runtime_error("reranker configuration does not match workspace artifacts");
    if (query_repeats == 0 || retrieval_depth == 0)
        throw std::runtime_error("reranker repeats and retrieval depth must be positive");

    RerankerRetrievalRun run;
    run.query_repeats = query_repeats;
    run.rankings.resize(impl_->query_texts.size());
    run.baseline_rankings.resize(impl_->query_texts.size());
    std::vector<float> baseline_scores(impl_->index.doc_count());
    for (std::size_t query = 0; query < impl_->query_texts.size(); ++query) {
        const auto score_start = Clock::now();
        impl_->index.score(impl_->query_texts[query], baseline_scores);
        run.baseline_score_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - score_start).count();
        run.baseline_rankings[query] = ranking_from_scores(baseline_scores, retrieval_depth);
    }

    std::vector<Ranking> full_candidate_rankings(impl_->query_texts.size());
    std::vector<FragmentGeometryProfile> profiles;
    profiles.reserve(checked_size_product(query_repeats, impl_->query_texts.size(),
                                          "reranker profile count is too large"));
    for (std::size_t repeat = 0; repeat < query_repeats; ++repeat) {
        for (std::size_t query = 0; query < impl_->query_texts.size(); ++query) {
            FragmentGeometryProfile profile;
            const auto scores = score_fragment_geometry_profiled(
                impl_->query_texts[query], impl_->index, *impl_->encoder, impl_->fragments,
                config.geometry, &profile);
            profiles.push_back(profile);
            if (repeat != 0)
                continue;
            full_candidate_rankings[query] =
                ranking_from_scores(scores, std::numeric_limits<std::size_t>::max());
            run.rankings[query] = full_candidate_rankings[query];
            if (run.rankings[query].size() > retrieval_depth)
                run.rankings[query].resize(retrieval_depth);
        }
    }
    run.candidate_rankings = std::move(full_candidate_rankings);
    run.profile = summarize(profiles);
    return run;
}

double evaluate_candidate_recall(const std::vector<Ranking>& candidate_rankings,
                                 const Fixture& fixture) {
    return candidate_recall(candidate_rankings, fixture);
}

const RerankerWorkspaceStats& FragmentRerankerWorkspace::stats() const noexcept {
    return impl_->stats;
}

std::string reranker_result_json(const ResultContext& context, const FragmentRerankerConfig& config,
                                 const RerankerWorkspaceStats& workspace,
                                 const RerankerRetrievalRun& run, const Metrics& metrics,
                                 const Metrics& baseline_metrics, double evaluation_us,
                                 std::size_t query_count, std::size_t document_count,
                                 std::size_t qrel_count) {
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
    append_metadata(out, context.metadata);
    out << "},\"config\":{\"encoder_profile\":\"" << encoder_profile_name(config.encoder_profile)
        << "\",\"encoder\":";
    if (config.encoder_profile == RerankerEncoderProfile::FixedHash) {
        out << encoder_config_json(config.fixed_encoder);
    } else {
        out << "{\"window_size\":" << config.pmi.window_size
            << ",\"target_rank\":" << config.pmi.target_rank
            << ",\"min_token_count\":" << config.pmi.min_token_count
            << ",\"max_vocab_size\":" << config.pmi.max_vocab_size
            << ",\"shift_log_k\":" << config.pmi.shift_log_k
            << ",\"svd_iters\":" << config.pmi.svd_iters
            << ",\"svd_oversample\":" << config.pmi.svd_oversample << '}';
    }
    const auto& geometry = config.geometry;
    out << ",\"builder\":\"" << builder_name(config.builder) << "\",\"fragment_storage\":\""
        << storage_name(config.storage)
        << "\",\"top_sentence_fragments\":" << config.top_sentence_fragments
        << ",\"fragment_signature_terms\":" << config.fragment_signature_terms
        << ",\"sentence_overlap_cap\":" << config.sentence_overlap_cap
        << ",\"anchor_overlap_cap\":" << config.anchor_overlap_cap
        << ",\"geometry\":{\"pool_size\":" << geometry.pool_size << ",\"alpha\":" << geometry.alpha
        << ",\"top_fragments_per_doc\":" << geometry.top_fragments_per_doc
        << ",\"attention_scale\":" << geometry.attention_scale << ",\"knn\":" << geometry.knn
        << ",\"steps\":" << geometry.steps
        << ",\"use_phss\":" << (geometry.use_phss ? "true" : "false") << ",\"phss_criterion\":\""
        << phss_criterion_name(geometry.phss_config.criterion)
        << "\",\"phss_adaptive\":" << (geometry.phss_adaptive ? "true" : "false")
        << ",\"phss_confidence_threshold\":" << geometry.phss_confidence_threshold
        << ",\"outer_maxsim\":" << (geometry.outer_maxsim ? "true" : "false")
        << ",\"doc_scorer\":\"" << doc_scorer_name(geometry.doc_scorer_kind)
        << "\",\"doc_scorer_top_k\":" << geometry.doc_scorer_top_k
        << ",\"doc_scorer_softmax_beta\":" << geometry.doc_scorer_softmax_beta
        << ",\"whiten\":" << (geometry.whiten ? "true" : "false")
        << ",\"csls_k\":" << geometry.csls_k << ",\"csls_beta\":" << geometry.csls_beta
        << ",\"graph_prefix_dim\":" << geometry.graph_prefix_dim << "}}"
        << ",\"execution\":{\"retrieval_depth\":" << context.retrieval_depth
        << ",\"query_repeats\":" << run.query_repeats
        << ",\"effective_output_dim\":" << workspace.encoder_dim
        << ",\"mean_candidate_pool_size\":" << run.profile.pool_docs_mean
        << ",\"mean_pool_fragments\":" << run.profile.pool_fragments_mean
        << ",\"mean_graph_edges\":" << run.profile.graph_edges_mean << "}"
        << ",\"counts\":{\"queries\":" << query_count << ",\"documents\":" << document_count
        << ",\"qrels\":" << qrel_count << ",\"evaluated_queries\":" << metrics.evaluated_queries
        << ",\"fragments\":" << workspace.fragment_count << "}"
        << ",\"metrics\":{\"reranker\":";
    append_metrics(out, metrics);
    out << ",\"bm25_baseline\":";
    append_metrics(out, baseline_metrics);
    out << ",\"delta_ndcg_at_10\":" << metrics.ndcg_at_10 - baseline_metrics.ndcg_at_10
        << ",\"delta_mrr_at_10\":" << metrics.mrr_at_10 - baseline_metrics.mrr_at_10
        << ",\"candidate_recall\":" << run.candidate_recall << "}"
        << ",\"timing_us\":{\"workspace_setup_total\":" << workspace.setup_us
        << ",\"bm25_add_total\":" << workspace.bm25_add_us
        << ",\"bm25_finalize_total\":" << workspace.bm25_finalize_us
        << ",\"encoder_artifact_build_total\":" << workspace.encoder_artifact_build_us
        << ",\"encoder_init_total\":" << workspace.encoder_init_us
        << ",\"fragment_build_total\":" << workspace.fragment_build_us
        << ",\"fragment_compress_total\":" << workspace.fragment_compress_us
        << ",\"baseline_score_total\":" << run.baseline_score_us
        << ",\"reranker_query_mean\":" << run.profile.total_mean_us
        << ",\"reranker_query_p50\":" << run.profile.total_p50_us
        << ",\"reranker_query_p95\":" << run.profile.total_p95_us
        << ",\"bm25_query_mean\":" << run.profile.bm25_mean_us
        << ",\"query_encode_mean\":" << run.profile.query_encode_mean_us
        << ",\"gather_mean\":" << run.profile.gather_mean_us
        << ",\"whiten_mean\":" << run.profile.whiten_mean_us
        << ",\"pairwise_mean\":" << run.profile.pairwise_mean_us
        << ",\"phss_select_mean\":" << run.profile.phss_select_mean_us
        << ",\"query_attention_mean\":" << run.profile.query_attention_mean_us
        << ",\"adjacency_mean\":" << run.profile.adjacency_mean_us
        << ",\"diffuse_mean\":" << run.profile.diffuse_mean_us
        << ",\"blend_mean\":" << run.profile.blend_mean_us
        << ",\"evaluation_total\":" << evaluation_us << "}"
        << ",\"rates\":{\"reranker_qps\":"
        << (run.profile.total_mean_us > 0.0 ? 1'000'000.0 / run.profile.total_mean_us : 0.0)
        << ",\"phss_used\":" << run.profile.phss_used_rate << "}"
        << ",\"resources\":{\"fragment_vector_bytes\":" << workspace.fragment_vector_bytes
        << ",\"encoder_artifact_bytes_lower_bound\":"
        << workspace.encoder_artifact_bytes_lower_bound
        << ",\"pmi_vocab_size\":" << workspace.pmi_vocab_size << "}}";
    return out.str();
}

} // namespace simeon::experiment
