#include "fusion_experiment_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "simeon/bm25.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/fusion.hpp"
#include "simeon/pmi.hpp"
#include "simeon/simd.hpp"

namespace simeon::experiment {
namespace {

using Clock = std::chrono::steady_clock;

Bm25Index build_bm25(const Fixture& fixture, Bm25Config config) {
    Bm25Index index(config);
    index.reserve_docs(fixture.doc_texts.size());
    for (const auto& document : fixture.doc_texts)
        index.add_doc(document);
    index.finalize();
    return index;
}

bool same_idf_identity(const WsdmIdfFusionConfig& lhs, const WsdmIdfFusionConfig& rhs) {
    return lhs.pool_per_leg == rhs.pool_per_leg &&
           lhs.idf_encoder.idf.hash_dim == rhs.idf_encoder.idf.hash_dim &&
           lhs.idf_encoder.idf.scope == rhs.idf_encoder.idf.scope &&
           encoder_config_json(lhs.idf_encoder.encoder) ==
               encoder_config_json(rhs.idf_encoder.encoder);
}

std::uint64_t evidence_storage_bytes(const std::vector<std::uint32_t>& documents,
                                     const std::vector<float>& wsdm_sab,
                                     const std::vector<float>& wsdm_atire,
                                     const std::vector<float>& idf) {
    return documents.capacity() * sizeof(std::uint32_t) +
           (wsdm_sab.capacity() + wsdm_atire.capacity() + idf.capacity()) * sizeof(float);
}

} // namespace

struct WsdmIdfFusionWorkspace::Impl {
    struct QueryEvidence {
        std::vector<std::uint32_t> documents;
        std::vector<float> wsdm_sab;
        std::vector<float> wsdm_atire;
        std::vector<float> idf;
        std::size_t base_pool_size = 0;
    };

    WsdmIdfFusionConfig identity;
    FusionWorkspaceStats stats;
    std::vector<QueryEvidence> evidence;
    std::vector<std::vector<std::uint32_t>> relevant_documents;

    Impl(const Fixture& fixture, const WsdmIdfFusionConfig& config) : identity(config) {
        const auto setup_start = Clock::now();
        if (fixture.query_ids.size() != fixture.query_texts.size() ||
            fixture.doc_ids.size() != fixture.doc_texts.size())
            throw std::runtime_error("fusion fixture ids and texts have different row counts");
        if (fixture.query_texts.empty() || fixture.doc_texts.empty())
            throw std::runtime_error("fusion fixture requires at least one query and document");
        if (fixture.doc_texts.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("fusion fixture exceeds uint32 document capacity");

        Bm25Config atire_config;
        atire_config.build_word_bigrams = true;
        const auto atire = build_bm25(fixture, atire_config);

        Bm25Config sab_config;
        sab_config.variant = Bm25Variant::SubwordAwareBackoff;
        sab_config.delta = 1.0f;
        sab_config.subword_gamma = 5.0f;
        sab_config.build_word_bigrams = true;
        const auto sab = build_bm25(fixture, sab_config);

        Bm25Config plus_config;
        plus_config.variant = Bm25Variant::BM25Plus;
        const auto plus = build_bm25(fixture, plus_config);
        Bm25Config l_config;
        l_config.variant = Bm25Variant::BM25L;
        const auto bm25_l = build_bm25(fixture, l_config);
        Bm25Config dlh_config;
        dlh_config.variant = Bm25Variant::DLH13;
        const auto dlh = build_bm25(fixture, dlh_config);
        const std::array<const Bm25Index*, 5> rrf_indexes{&atire, &plus, &bm25_l, &dlh, &sab};
        const WeightedSdmConfig wsdm_config;

        std::vector<std::string_view> document_views;
        document_views.reserve(fixture.doc_texts.size());
        for (const auto& document : fixture.doc_texts)
            document_views.emplace_back(document);

        PmiConfig pmi_config;
        pmi_config.target_rank = 128;
        pmi_config.min_token_count = 5;
        pmi_config.max_vocab_size = 20'000;
        const auto pmi = PmiEmbeddings::learn(document_views, pmi_config);
        EncoderConfig geometry_encoder_config;
        geometry_encoder_config.ngram_mode = NGramMode::WordOnly;
        geometry_encoder_config.ngram_min = 1;
        geometry_encoder_config.ngram_max = 1;
        geometry_encoder_config.sketch_dim = 0;
        geometry_encoder_config.output_dim = pmi.dim();
        geometry_encoder_config.projection = ProjectionMode::None;
        geometry_encoder_config.l2_normalize = true;
        geometry_encoder_config.pmi_rows = &pmi;
        const Encoder geometry_encoder(geometry_encoder_config);

        std::vector<std::vector<SemanticFragment>> fragments(fixture.doc_texts.size());
        for (std::size_t document = 0; document < fixture.doc_texts.size(); ++document) {
            const auto prep = prepare_doc(fixture.doc_texts[document], atire, 6, 8, 0.0f);
            fragments[document] = build_doc_semantic_fragments_rich_covered_from_prep(
                geometry_encoder, fixture.doc_texts[document], prep, 0.60f, 0.80f);
        }
        compress_fragments_to_bf16(fragments, geometry_encoder.output_dim());
        FragmentGeometryConfig geometry_config;
        geometry_config.pool_size = 100;
        geometry_config.alpha = 0.0f;
        geometry_config.top_fragments_per_doc = 8;
        geometry_config.attention_scale = 8.0f;
        geometry_config.knn = 8;
        geometry_config.steps = 2;
        geometry_config.use_phss = true;
        geometry_config.phss_config.criterion = PhssConfig::Criterion::LargestGapApprox;

        const auto artifact_start = Clock::now();
        const HashedIdf idf_artifact =
            HashedIdf::learn(document_views, config.idf_encoder.encoder, config.idf_encoder.idf);
        stats.idf_artifact_build_us =
            std::chrono::duration<double, std::micro>(Clock::now() - artifact_start).count();
        stats.artifact_bytes = idf_artifact.storage_bytes();
        stats.artifact_fingerprint = idf_artifact.fingerprint();

        EncoderConfig idf_encoder_config = config.idf_encoder.encoder;
        idf_encoder_config.hashed_idf = &idf_artifact;
        const Encoder idf_encoder(idf_encoder_config);
        const std::uint32_t idf_dim = idf_encoder.output_dim();
        stats.idf_document_vector_bytes =
            fixture.doc_texts.size() * static_cast<std::uint64_t>(idf_dim) * sizeof(float);
        std::vector<float> idf_document_vectors(fixture.doc_texts.size() *
                                                static_cast<std::size_t>(idf_dim));
        const auto document_encode_start = Clock::now();
        idf_encoder.encode_batch(document_views, idf_document_vectors.data());
        stats.idf_document_encode_us =
            std::chrono::duration<double, std::micro>(Clock::now() - document_encode_start).count();

        std::vector<std::string_view> query_views;
        query_views.reserve(fixture.query_texts.size());
        for (const auto& query : fixture.query_texts)
            query_views.emplace_back(query);
        std::vector<float> idf_query_vectors(fixture.query_texts.size() *
                                             static_cast<std::size_t>(idf_dim));
        const auto query_encode_start = Clock::now();
        idf_encoder.encode_batch(query_views, idf_query_vectors.data());
        stats.idf_query_encode_us =
            std::chrono::duration<double, std::micro>(Clock::now() - query_encode_start).count();

        evidence.resize(fixture.query_texts.size());
        relevant_documents.resize(fixture.query_texts.size());
        for (const auto& qrel : fixture.qrels)
            relevant_documents.at(qrel.query).push_back(qrel.document);

        enum Leg : std::size_t {
            Atire = 0,
            WsdmAtire,
            Sab,
            WsdmSab,
            Geometry,
            Rrf5,
            LegCount,
        };
        std::array<std::vector<float>, LegCount> leg_scores;
        for (auto& scores : leg_scores)
            scores.resize(fixture.doc_texts.size());
        std::vector<float> idf_scores(fixture.doc_texts.size());

        for (std::size_t query = 0; query < fixture.query_texts.size(); ++query) {
            const auto base_start = Clock::now();
            atire.score(fixture.query_texts[query], leg_scores[Atire]);
            atire.score_wsdm(fixture.query_texts[query], leg_scores[WsdmAtire], wsdm_config);
            sab.score(fixture.query_texts[query], leg_scores[Sab]);
            sab.score_wsdm(fixture.query_texts[query], leg_scores[WsdmSab], wsdm_config);
            leg_scores[Geometry] = score_fragment_geometry(
                fixture.query_texts[query], atire, geometry_encoder, fragments, geometry_config);
            std::fill(leg_scores[Rrf5].begin(), leg_scores[Rrf5].end(), 0.0f);
            score_bm25_variants_rrf(rrf_indexes, fixture.query_texts[query], leg_scores[Rrf5]);
            stats.base_query_score_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - base_start).count();

            const auto idf_score_start = Clock::now();
            const float* query_vector = idf_query_vectors.data() + query * idf_dim;
            std::size_t document = 0;
            for (; document + 4 <= fixture.doc_texts.size(); document += 4) {
                float scores[4];
                const float* base = idf_document_vectors.data() + document * idf_dim;
                simd::dot4(query_vector, base, base + idf_dim, base + 2 * idf_dim,
                           base + 3 * idf_dim, scores, idf_dim);
                for (std::size_t lane = 0; lane < 4; ++lane)
                    idf_scores[document + lane] = scores[lane];
            }
            for (; document < fixture.doc_texts.size(); ++document) {
                idf_scores[document] = simd::dot(
                    query_vector, idf_document_vectors.data() + document * idf_dim, idf_dim);
            }
            stats.idf_score_us +=
                std::chrono::duration<double, std::micro>(Clock::now() - idf_score_start).count();

            auto& query_evidence = evidence[query];
            std::unordered_set<std::uint32_t> seen;
            seen.reserve(static_cast<std::size_t>(config.pool_per_leg) * (LegCount + 1));
            const auto append_top = [&](std::span<const float> scores) {
                for (const auto& [document_id, _] : top_k(scores, config.pool_per_leg)) {
                    if (seen.insert(document_id).second)
                        query_evidence.documents.push_back(document_id);
                }
            };
            for (const auto& scores : leg_scores)
                append_top(scores);
            query_evidence.base_pool_size = query_evidence.documents.size();
            append_top(idf_scores);

            const auto pool_size = query_evidence.documents.size();
            query_evidence.wsdm_sab.resize(pool_size);
            query_evidence.wsdm_atire.resize(pool_size);
            query_evidence.idf.resize(pool_size);
            for (std::size_t position = 0; position < pool_size; ++position) {
                const auto document_id = query_evidence.documents[position];
                query_evidence.wsdm_sab[position] = leg_scores[WsdmSab][document_id];
                query_evidence.wsdm_atire[position] = leg_scores[WsdmAtire][document_id];
                query_evidence.idf[position] = idf_scores[document_id];
            }
            stats.cached_evidence_bytes +=
                evidence_storage_bytes(query_evidence.documents, query_evidence.wsdm_sab,
                                       query_evidence.wsdm_atire, query_evidence.idf);
        }
        stats.setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - setup_start).count();
    }
};

WsdmIdfFusionWorkspace::WsdmIdfFusionWorkspace(const Fixture& fixture,
                                               const WsdmIdfFusionConfig& config)
    : impl_(std::make_unique<Impl>(fixture, config)) {}

WsdmIdfFusionWorkspace::~WsdmIdfFusionWorkspace() = default;
WsdmIdfFusionWorkspace::WsdmIdfFusionWorkspace(WsdmIdfFusionWorkspace&&) noexcept = default;
WsdmIdfFusionWorkspace&
WsdmIdfFusionWorkspace::operator=(WsdmIdfFusionWorkspace&&) noexcept = default;

bool WsdmIdfFusionWorkspace::compatible(const WsdmIdfFusionConfig& config) const {
    return same_idf_identity(impl_->identity, config);
}

FusionRetrievalRun WsdmIdfFusionWorkspace::run(const WsdmIdfFusionConfig& config) const {
    if (!compatible(config))
        throw std::runtime_error(
            "wsdm_idf_fusion variants in one manifest must share pool and IDF identity");
    FusionRetrievalRun run;
    run.rankings.resize(impl_->evidence.size());
    const auto fusion_start = Clock::now();
    double pool_size_sum = 0.0;
    double candidate_recall_sum = 0.0;
    std::size_t recall_queries = 0;
    for (std::size_t query = 0; query < impl_->evidence.size(); ++query) {
        const auto& evidence = impl_->evidence[query];
        const std::size_t pool_size =
            config.idf_candidates ? evidence.documents.size() : evidence.base_pool_size;
        pool_size_sum += static_cast<double>(pool_size);

        std::vector<float> fused(pool_size, 0.0f);
        const std::span<const float> wsdm_sab(evidence.wsdm_sab.data(), pool_size);
        const std::span<const float> wsdm_atire(evidence.wsdm_atire.data(), pool_size);
        if (config.idf_weight == 0.0f) {
            const std::array<std::span<const float>, 2> legs{wsdm_sab, wsdm_atire};
            const std::array<float, 2> weights{0.6f, 0.4f};
            convex_fuse_z(legs, weights, fused);
        } else {
            const std::span<const float> idf(evidence.idf.data(), pool_size);
            const float remaining = 1.0f - config.idf_weight;
            const std::array<std::span<const float>, 3> legs{idf, wsdm_sab, wsdm_atire};
            const std::array<float, 3> weights{config.idf_weight, remaining * 0.6f,
                                               remaining * 0.4f};
            convex_fuse_z(legs, weights, fused);
        }

        auto& ranking = run.rankings[query];
        ranking.reserve(pool_size);
        for (std::size_t position = 0; position < pool_size; ++position)
            ranking.emplace_back(fused[position], evidence.documents[position]);

        const auto& relevant = impl_->relevant_documents[query];
        if (!relevant.empty()) {
            std::unordered_set<std::uint32_t> candidates;
            candidates.reserve(pool_size);
            candidates.insert(evidence.documents.begin(), evidence.documents.begin() + pool_size);
            std::size_t found = 0;
            for (const auto document : relevant)
                found += candidates.contains(document) ? 1 : 0;
            candidate_recall_sum +=
                static_cast<double>(found) / static_cast<double>(relevant.size());
            ++recall_queries;
        }
    }
    run.fusion_us = std::chrono::duration<double, std::micro>(Clock::now() - fusion_start).count();
    run.mean_pool_size =
        impl_->evidence.empty() ? 0.0 : pool_size_sum / static_cast<double>(impl_->evidence.size());
    run.candidate_recall =
        recall_queries == 0 ? 0.0 : candidate_recall_sum / static_cast<double>(recall_queries);
    return run;
}

const FusionWorkspaceStats& WsdmIdfFusionWorkspace::stats() const noexcept {
    return impl_->stats;
}

std::string fusion_result_json(const ResultContext& context, const WsdmIdfFusionConfig& config,
                               const FusionWorkspaceStats& workspace, const FusionRetrievalRun& run,
                               const Metrics& metrics, double evaluation_us,
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
    bool first_metadata = true;
    for (const auto& [key, value] : context.metadata) {
        if (!first_metadata)
            out << ',';
        first_metadata = false;
        out << '"' << json_escape(key) << "\":\"" << json_escape(value) << '"';
    }
    out << "},\"config\":{\"base_fusion\":\"six_leg_union_wsdm_pair_v1\""
        << ",\"pool_per_leg\":" << config.pool_per_leg
        << ",\"idf_candidates\":" << (config.idf_candidates ? "true" : "false")
        << ",\"idf_weight\":" << config.idf_weight
        << ",\"idf_encoder\":" << encoder_config_json(config.idf_encoder.encoder)
        << ",\"idf_artifact\":{\"fingerprint\":\"" << json_escape(workspace.artifact_fingerprint)
        << "\",\"hash_dim\":" << config.idf_encoder.idf.hash_dim << ",\"scope\":\""
        << (config.idf_encoder.idf.scope == HashedIdfScope::All         ? "all"
            : config.idf_encoder.idf.scope == HashedIdfScope::Character ? "char"
                                                                        : "word")
        << "\",\"storage_bytes\":" << workspace.artifact_bytes << "}}"
        << ",\"execution\":{\"retrieval_depth\":100,\"mean_candidate_pool_size\":"
        << run.mean_pool_size << "}"
        << ",\"counts\":{\"queries\":" << query_count << ",\"documents\":" << document_count
        << ",\"qrels\":" << qrel_count << ",\"evaluated_queries\":" << metrics.evaluated_queries
        << "}"
        << ",\"metrics\":{\"ndcg_at_10\":" << metrics.ndcg_at_10
        << ",\"precision_at_10\":" << metrics.precision_at_10
        << ",\"recall_at_10\":" << metrics.recall_at_10
        << ",\"recall_at_100\":" << metrics.recall_at_100 << ",\"mrr_at_10\":" << metrics.mrr_at_10
        << ",\"candidate_recall\":" << run.candidate_recall << "}"
        << ",\"timing_us\":{\"workspace_setup_total\":" << workspace.setup_us
        << ",\"base_query_score_total\":" << workspace.base_query_score_us
        << ",\"idf_artifact_build_total\":" << workspace.idf_artifact_build_us
        << ",\"idf_document_encode_total\":" << workspace.idf_document_encode_us
        << ",\"idf_query_encode_total\":" << workspace.idf_query_encode_us
        << ",\"idf_score_total\":" << workspace.idf_score_us
        << ",\"fusion_total\":" << run.fusion_us << ",\"evaluation_total\":" << evaluation_us << "}"
        << ",\"resources\":{\"artifact_bytes\":" << workspace.artifact_bytes
        << ",\"idf_document_vector_bytes\":" << workspace.idf_document_vector_bytes
        << ",\"cached_evidence_bytes\":" << workspace.cached_evidence_bytes << "}}";
    return out.str();
}

} // namespace simeon::experiment
