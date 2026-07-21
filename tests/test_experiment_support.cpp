#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coordinate_experiment_support.hpp"
#include "experiment_support.hpp"
#include "family_atlas_experiment_support.hpp"
#include "fusion_experiment_support.hpp"

namespace experiment = simeon::experiment;

namespace {

experiment::Manifest parse(std::string_view text) {
    std::istringstream in(std::string{text});
    return experiment::parse_manifest(in);
}

void test_manifest_parses_named_variants() {
    const auto manifest = parse(R"(
        schema = 1
        name = "embedding-foundation"
        metric_profile = trec-v1

        [variant.hash-32k]
        kind = encoder
        ngram_mode = char
        sketch_dim = 32768
        projection = none

        [variant.hash-4k-384]
        kind = encoder
        sketch_dim = 4096
        output_dim = 384
        projection = achlioptas
    )");

    assert(manifest.schema_version == 1);
    assert(manifest.name == "embedding-foundation");
    assert(manifest.metadata.at("metric_profile") == "trec-v1");
    assert(manifest.variants.size() == 2);
    assert(manifest.variants[0].name == "hash-32k");
    assert(manifest.variants[0].kind == "encoder");
    assert(manifest.variants[0].parameters.at("sketch_dim") == "32768");
    assert(manifest.variants[1].name == "hash-4k-384");
}

void test_manifest_rejects_duplicate_keys_with_a_line_number() {
    try {
        (void)parse("schema=1\nname=x\nname=y\n[variant.a]\nkind=encoder\n");
        assert(false && "duplicate manifest key should throw");
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        assert(message.find("line 3") != std::string::npos);
        assert(message.find("duplicate") != std::string::npos);
    }
}

void test_manifest_fingerprint_is_semantic() {
    const auto a = parse(R"(
        schema=1
        name=stable
        [variant.hash]
        kind=encoder
        sketch_dim=1024
        projection=none
    )");
    const auto b = parse(R"(
        name = stable
        schema = 1
        [variant.hash]
        projection = none
        sketch_dim = 1024
        kind = encoder
    )");
    assert(experiment::manifest_fingerprint(a) == experiment::manifest_fingerprint(b));
}

void test_encoder_config_is_resolved_and_unknown_parameters_fail() {
    const auto manifest = parse(R"(
        schema=1
        name=config
        [variant.hash]
        kind=encoder
        ngram_mode=char_word
        ngram_min=2
        ngram_max=4
        hash=xxhash64
        hash_seed=42
        sketch_dim=1024
        output_dim=128
        projection=achlioptas
        projection_seed=17
        text_normalization=ascii_lower
        char_ngram_scope=word_bounded
        feature_weighting=sqrt_tf
        sketch_weighting=signed_sqrt
        l2_normalize=false
        matryoshka=true
        matryoshka_decay=24.5
    )");
    const auto cfg = experiment::resolve_encoder_config(manifest.variants.front());
    assert(cfg.ngram_mode == simeon::NGramMode::CharAndWord);
    assert(cfg.ngram_min == 2);
    assert(cfg.ngram_max == 4);
    assert(cfg.hash == simeon::HashFamily::XxHash64);
    assert(cfg.hash_seed == 42);
    assert(cfg.sketch_dim == 1024);
    assert(cfg.output_dim == 128);
    assert(cfg.projection == simeon::ProjectionMode::AchlioptasSparse);
    assert(cfg.projection_seed == 17);
    assert(cfg.text_normalization == simeon::TextNormalization::AsciiLower);
    assert(cfg.char_ngram_scope == simeon::CharNGramScope::WordBounded);
    assert(cfg.feature_weighting == simeon::FeatureWeighting::SqrtTf);
    assert(cfg.sketch_weighting == simeon::SketchWeighting::SignedSqrt);
    assert(!cfg.l2_normalize);
    assert(cfg.matryoshka);
    assert(std::fabs(cfg.matryoshka_decay - 24.5f) < 1e-6f);

    auto invalid = manifest.variants.front();
    invalid.parameters["sketch_dims"] = "2048";
    try {
        (void)experiment::resolve_encoder_config(invalid);
        assert(false && "unknown encoder parameter should throw");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("sketch_dims") != std::string::npos);
    }
}

void test_corpus_adaptive_encoder_config_and_regime_are_strict() {
    const auto manifest = parse(R"(
        schema=1
        name=idf
        training_regime=corpus-adaptive
        [variant.baseline]
        kind=encoder
        [variant.idf]
        kind=hashed_idf_encoder
        idf_hash_dim=65536
        idf_scope=word
        ngram_mode=char_word
        char_ngram_scope=word_bounded
    )");
    experiment::validate_training_regime(manifest);
    const auto resolved = experiment::resolve_hashed_idf_encoder_config(manifest.variants[1]);
    assert(resolved.idf.hash_dim == 65'536);
    assert(resolved.idf.scope == simeon::HashedIdfScope::Word);
    assert(resolved.encoder.ngram_mode == simeon::NGramMode::CharAndWord);
    assert(resolved.encoder.char_ngram_scope == simeon::CharNGramScope::WordBounded);

    auto forbidden = manifest;
    forbidden.metadata["training_regime"] = "artifact-free";
    try {
        experiment::validate_training_regime(forbidden);
        assert(false && "artifact-free manifests must reject corpus-adaptive variants");
    } catch (const std::runtime_error&) {
    }

    auto unknown = manifest.variants[1];
    unknown.parameters["idf_hash_dims"] = "1024";
    try {
        (void)experiment::resolve_hashed_idf_encoder_config(unknown);
        assert(false && "unknown corpus-adaptive parameter must throw");
    } catch (const std::runtime_error&) {
    }
}

void test_wsdm_idf_fusion_config_and_regime_are_strict() {
    const auto manifest = parse(R"(
        schema=1
        name=fusion
        training_regime=corpus-adaptive
        [variant.blend]
        kind=wsdm_idf_fusion
        idf_weight=0.2
        idf_candidates=true
        pool_per_leg=75
        idf_hash_dim=65536
        idf_scope=all
        ngram_mode=char_word
        char_ngram_scope=word_bounded
        sketch_dim=8192
        output_dim=768
        projection=fwht
    )");
    experiment::validate_training_regime(manifest);
    const auto config = experiment::resolve_wsdm_idf_fusion_config(manifest.variants.front());
    assert(std::fabs(config.idf_weight - 0.2f) < 1e-6f);
    assert(config.idf_candidates);
    assert(config.pool_per_leg == 75);
    assert(config.idf_encoder.idf.hash_dim == 65'536);
    assert(config.idf_encoder.encoder.output_dim == 768);

    auto forbidden = manifest;
    forbidden.metadata["training_regime"] = "artifact-free";
    try {
        experiment::validate_training_regime(forbidden);
        assert(false && "artifact-free must reject corpus-adaptive fusion");
    } catch (const std::runtime_error&) {
    }

    auto invalid_weight = manifest.variants.front();
    invalid_weight.parameters["idf_weight"] = "1.01";
    try {
        (void)experiment::resolve_wsdm_idf_fusion_config(invalid_weight);
        assert(false && "fusion weight outside [0,1] must fail");
    } catch (const std::runtime_error&) {
    }

    auto unknown = manifest.variants.front();
    unknown.parameters["candidate_weight"] = "0.2";
    try {
        (void)experiment::resolve_wsdm_idf_fusion_config(unknown);
        assert(false && "unknown fusion parameter must fail");
    } catch (const std::runtime_error&) {
    }
}

void test_wsdm_idf_fusion_workspace_replays_variants() {
    experiment::Fixture fixture;
    fixture.name = "tiny-fusion";
    fixture.split = "dev";
    fixture.query_ids = {"q0", "q1"};
    fixture.query_texts = {"alpha beta", "gamma delta"};
    fixture.doc_ids = {"d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"};
    fixture.doc_texts = {
        "alpha alpha alpha beta beta beta. shared context terms appear here.",
        "gamma gamma gamma delta delta delta. shared context terms appear here.",
        "alpha beta alpha beta alpha beta. shared context terms appear here.",
        "gamma delta gamma delta gamma delta. shared context terms appear here.",
        "alpha alpha gamma gamma shared shared. context terms appear here.",
        "beta beta delta delta shared shared. context terms appear here.",
        "alpha delta alpha delta shared context. terms appear here.",
        "beta gamma beta gamma shared context. terms appear here.",
    };
    fixture.qrels = {{0, 0, 1}, {0, 2, 1}, {1, 1, 1}, {1, 3, 1}};

    experiment::WsdmIdfFusionConfig config;
    config.pool_per_leg = 2;
    config.idf_encoder.idf.hash_dim = 1024;
    config.idf_encoder.encoder = simeon::compact_retrieval_config();
    config.idf_encoder.encoder.sketch_dim = 256;
    config.idf_encoder.encoder.output_dim = 64;
    experiment::WsdmIdfFusionWorkspace workspace(fixture, config);

    const auto baseline = workspace.run(config);
    assert(baseline.rankings.size() == fixture.query_ids.size());
    assert(baseline.mean_pool_size > 0.0);
    assert(baseline.candidate_recall >= 0.0 && baseline.candidate_recall <= 1.0);

    config.idf_candidates = true;
    config.idf_weight = 0.2f;
    const auto blended = workspace.run(config);
    assert(blended.rankings.size() == baseline.rankings.size());
    assert(blended.mean_pool_size >= baseline.mean_pool_size);
    assert(workspace.stats().artifact_bytes == 2048);
    assert(!workspace.stats().artifact_fingerprint.empty());

    experiment::ResultContext context;
    context.experiment = "fusion";
    context.manifest_fingerprint = "manifest-id";
    context.variant = "blend";
    context.kind = "wsdm_idf_fusion";
    context.fixture = fixture.name;
    context.fixture_fingerprint = experiment::fixture_fingerprint(fixture);
    context.split = fixture.split;
    const auto metrics = experiment::evaluate_rankings(blended.rankings, fixture);
    const auto json = experiment::fusion_result_json(context, config, workspace.stats(), blended,
                                                     metrics, 1.0, fixture.query_ids.size(),
                                                     fixture.doc_ids.size(), fixture.qrels.size());
    assert(json.find("\"kind\":\"wsdm_idf_fusion\"") != std::string::npos);
    assert(json.find("\"idf_candidates\":true") != std::string::npos);
    assert(json.find("\"candidate_recall\":") != std::string::npos);
    assert(json.find("\"idf_document_vector_bytes\":") != std::string::npos);

    auto incompatible = config;
    incompatible.idf_encoder.encoder.output_dim = 128;
    assert(!workspace.compatible(incompatible));
    try {
        (void)workspace.run(incompatible);
        assert(false && "workspace must reject a different encoder identity");
    } catch (const std::runtime_error&) {
    }
}

void test_coordinate_calibration_config_and_regime_are_strict() {
    const auto manifest = parse(R"(
        schema=1
        name=coordinates
        training_regime=corpus-adaptive
        [variant.centered]
        kind=coordinate_calibrated_idf
        retrieval_dim=48
        coordinate_transform=standardize
        coordinate_policy=selective
        full_weight=0.5
        min_chart_overlap=0.6
        max_chart_distortion=0.1
        variance_shrinkage=0.25
        min_variance_ratio=0.001
        idf_hash_dim=1024
        idf_scope=all
        ngram_mode=char_word
        char_ngram_scope=word_bounded
        sketch_dim=128
        output_dim=64
        projection=fwht
        l2_normalize=true
    )");
    experiment::validate_training_regime(manifest);
    const auto config =
        experiment::resolve_coordinate_calibration_config(manifest.variants.front());
    assert(config.retrieval_dim == 48);
    assert(config.transform == experiment::CoordinateTransform::Standardize);
    assert(config.routing_policy == experiment::CoordinateRoutingPolicy::Selective);
    assert(std::fabs(config.full_weight - 0.5f) < 1e-6f);
    assert(std::fabs(config.variance_shrinkage - 0.25f) < 1e-6f);
    assert(config.idf_encoder.encoder.output_dim == 64);

    auto forbidden = manifest;
    forbidden.metadata["training_regime"] = "artifact-free";
    try {
        experiment::validate_training_regime(forbidden);
        assert(false && "artifact-free must reject coordinate calibration");
    } catch (const std::runtime_error&) {
    }

    auto invalid_dimension = manifest.variants.front();
    invalid_dimension.parameters["retrieval_dim"] = "65";
    try {
        (void)experiment::resolve_coordinate_calibration_config(invalid_dimension);
        assert(false && "prefix wider than the maximum chart must fail");
    } catch (const std::runtime_error&) {
    }

    auto unknown = manifest.variants.front();
    unknown.parameters["coordinate_weight"] = "0.2";
    try {
        (void)experiment::resolve_coordinate_calibration_config(unknown);
        assert(false && "unknown coordinate parameter must fail");
    } catch (const std::runtime_error&) {
    }
}

void test_coordinate_calibration_workspace_replays_nested_charts() {
    experiment::Fixture fixture;
    fixture.name = "tiny-coordinates";
    fixture.split = "dev";
    fixture.query_ids = {"q0", "q1"};
    fixture.query_texts = {"alpha beta", "gamma delta"};
    fixture.doc_ids = {"d0", "d1", "d2", "d3", "d4"};
    fixture.doc_texts = {
        "alpha alpha beta shared context",   "gamma gamma delta shared context",
        "alpha beta topic retrieval",        "gamma delta topic retrieval",
        "unrelated epsilon zeta background",
    };
    fixture.qrels = {{0, 0, 1}, {0, 2, 1}, {1, 1, 1}, {1, 3, 1}};

    experiment::CoordinateCalibrationConfig config;
    config.idf_encoder.idf.hash_dim = 1024;
    config.idf_encoder.encoder = simeon::compact_retrieval_config();
    config.idf_encoder.encoder.sketch_dim = 128;
    config.idf_encoder.encoder.output_dim = 64;
    config.retrieval_dim = 32;
    experiment::CoordinateCalibrationWorkspace workspace(fixture, config);

    const auto baseline = workspace.run(config, 5);
    assert(baseline.rankings.size() == fixture.query_ids.size());
    assert(baseline.mean_prefix_energy_fraction > 0.0);
    assert(baseline.mean_prefix_energy_fraction < 1.0);
    assert(baseline.mean_top100_overlap >= 0.0 && baseline.mean_top100_overlap <= 1.0);
    assert(workspace.stats().variance_effective_dimensions > 0.0);
    assert(workspace.stats().coordinate_artifact_bytes == 64 * 2 * sizeof(float));

    config.transform = experiment::CoordinateTransform::Center;
    const auto centered = workspace.run(config, 5);
    assert(centered.rankings.size() == baseline.rankings.size());
    config.transform = experiment::CoordinateTransform::Standardize;
    config.variance_shrinkage = 0.5f;
    config.routing_policy = experiment::CoordinateRoutingPolicy::Selective;
    config.min_chart_overlap = 0.0f;
    config.max_chart_distortion = 2.0f;
    const auto standardized = workspace.run(config, 5);
    assert(standardized.rankings.size() == baseline.rankings.size());
    assert(standardized.narrowed_queries == fixture.query_ids.size());
    assert(standardized.augmented_queries == 0);

    config.routing_policy = experiment::CoordinateRoutingPolicy::Blend;
    const auto blended = workspace.run(config, 5);
    assert(blended.augmented_queries == fixture.query_ids.size());
    assert(blended.evidence_reused);
    assert(blended.cached_score_evidence_bytes ==
           fixture.query_ids.size() * fixture.doc_ids.size() * 2 * sizeof(float));

    experiment::ResultContext context;
    context.experiment = "coordinates";
    context.manifest_fingerprint = "manifest-id";
    context.variant = "standardized";
    context.kind = "coordinate_calibrated_idf";
    context.fixture = fixture.name;
    context.fixture_fingerprint = experiment::fixture_fingerprint(fixture);
    context.split = fixture.split;
    const auto metrics = experiment::evaluate_rankings(blended.rankings, fixture);
    const auto json = experiment::coordinate_result_json(
        context, config, workspace.stats(), blended, metrics, 1.0, fixture.query_ids.size(),
        fixture.doc_ids.size(), fixture.qrels.size());
    assert(json.find("\"kind\":\"coordinate_calibrated_idf\"") != std::string::npos);
    assert(json.find("\"coordinate_transform\":\"standardize\"") != std::string::npos);
    assert(json.find("\"coordinate_policy\":\"blend\"") != std::string::npos);
    assert(json.find("\"coordinate_diagnostics\":") != std::string::npos);
    assert(json.find("\"chart_evidence\":") != std::string::npos);
    assert(json.find("\"cached_score_evidence_bytes\":") != std::string::npos);

    auto compatible = config;
    compatible.retrieval_dim = 64;
    assert(workspace.compatible(compatible));
    auto incompatible = config;
    incompatible.idf_encoder.encoder.output_dim = 96;
    assert(!workspace.compatible(incompatible));
}

void test_feature_family_atlas_config_and_regime_are_strict() {
    const auto manifest = parse(R"(
        schema=1
        name=family-atlas
        training_regime=corpus-adaptive
        [variant.split]
        kind=feature_family_atlas_idf
        char_dim=48
        word_dim=16
        storage_budget_dim=64
        family_normalization=joint_rms
        char_weight=0.75
        family_policy=selective
        residual_score_normalization=rank_rrf
        residual_rrf_k=50
        residual_weight=0.20
        min_word_energy=0.10
        max_word_energy=0.90
        min_family_overlap=0.05
        max_family_overlap=0.80
        min_base_family_overlap=0.20
        max_base_family_overlap=0.70
        idf_hash_dim=1024
        idf_scope=all
        ngram_mode=char_word
        char_ngram_scope=word_bounded
        sketch_dim=128
        output_dim=64
        projection=fwht
        l2_normalize=true
    )");
    experiment::validate_training_regime(manifest);
    const auto config = experiment::resolve_feature_family_atlas_config(manifest.variants.front());
    assert(config.char_dim == 48);
    assert(config.word_dim == 16);
    assert(config.storage_budget_dim == 64);
    assert(config.normalization == experiment::FeatureFamilyNormalization::JointRms);
    assert(std::fabs(config.char_weight - 0.75f) < 1e-6f);
    assert(config.policy == experiment::FeatureFamilyPolicy::Selective);
    assert(config.residual_score_normalization ==
           experiment::FeatureResidualScoreNormalization::RankRrf);
    assert(std::fabs(config.residual_rrf_k - 50.0f) < 1e-6f);
    assert(std::fabs(config.residual_weight - 0.20f) < 1e-6f);
    assert(std::fabs(config.min_word_energy - 0.10f) < 1e-6f);
    assert(std::fabs(config.max_family_overlap - 0.80f) < 1e-6f);
    assert(std::fabs(config.min_base_family_overlap - 0.20f) < 1e-6f);
    assert(std::fabs(config.max_base_family_overlap - 0.70f) < 1e-6f);

    auto forbidden = manifest;
    forbidden.metadata["training_regime"] = "artifact-free";
    try {
        experiment::validate_training_regime(forbidden);
        assert(false && "artifact-free must reject the corpus-adaptive family atlas");
    } catch (const std::runtime_error&) {
    }

    auto invalid_residual = manifest.variants.front();
    invalid_residual.parameters["residual_weight"] = "0";
    try {
        (void)experiment::resolve_feature_family_atlas_config(invalid_residual);
        assert(false && "residual policies require a positive residual weight");
    } catch (const std::runtime_error&) {
    }

    auto invalid_budget = manifest.variants.front();
    invalid_budget.parameters["storage_budget_dim"] = "63";
    try {
        (void)experiment::resolve_feature_family_atlas_config(invalid_budget);
        assert(false && "family dimensions must exactly consume the declared storage budget");
    } catch (const std::runtime_error&) {
    }

    auto invalid_mode = manifest.variants.front();
    invalid_mode.parameters["ngram_mode"] = "char";
    try {
        (void)experiment::resolve_feature_family_atlas_config(invalid_mode);
        assert(false && "the family atlas base identity must declare both feature families");
    } catch (const std::runtime_error&) {
    }

    auto unknown = manifest.variants.front();
    unknown.parameters["word_weight"] = "0.25";
    try {
        (void)experiment::resolve_feature_family_atlas_config(unknown);
        assert(false && "unknown family-atlas parameters must fail");
    } catch (const std::runtime_error&) {
    }

    auto invalid_interval = manifest.variants.front();
    invalid_interval.parameters["min_word_energy"] = "0.9";
    invalid_interval.parameters["max_word_energy"] = "0.1";
    try {
        (void)experiment::resolve_feature_family_atlas_config(invalid_interval);
        assert(false && "family admission intervals must be ordered");
    } catch (const std::runtime_error&) {
    }
}

void test_feature_family_atlas_replays_weights_and_exposes_evaluation_views() {
    experiment::Fixture fixture;
    fixture.name = "tiny-family-atlas";
    fixture.split = "dev";
    fixture.query_ids = {"q0", "q1"};
    fixture.query_texts = {"alpha beta", "gamma delta"};
    fixture.doc_ids = {"d0", "d1", "d2", "d3", "d4"};
    fixture.doc_texts = {
        "alpha alpha beta shared context",   "gamma gamma delta shared context",
        "alpha beta topic retrieval",        "gamma delta topic retrieval",
        "unrelated epsilon zeta background",
    };
    fixture.qrels = {{0, 0, 1}, {0, 2, 1}, {1, 1, 1}, {1, 3, 1}};

    experiment::FeatureFamilyAtlasConfig config;
    config.idf_encoder.idf.hash_dim = 1024;
    config.idf_encoder.encoder = simeon::compact_retrieval_config();
    config.idf_encoder.encoder.sketch_dim = 128;
    config.idf_encoder.encoder.output_dim = 64;
    config.char_dim = 48;
    config.word_dim = 16;
    config.storage_budget_dim = 64;
    config.char_weight = 0.75f;
    experiment::FeatureFamilyAtlasWorkspace workspace(fixture, config);

    const auto baseline = workspace.run(config, 5);
    assert(baseline.rankings.size() == fixture.query_ids.size());
    assert(baseline.char_rankings.size() == baseline.rankings.size());
    assert(baseline.word_rankings.size() == baseline.rankings.size());
    assert(!baseline.evidence_reused);
    assert(baseline.cached_score_evidence_bytes ==
           fixture.query_ids.size() * fixture.doc_ids.size() * 2 * sizeof(float));
    assert(baseline.deployable_document_vector_bytes ==
           fixture.doc_ids.size() * config.storage_budget_dim * sizeof(float));
    assert(baseline.mean_char_prefix_energy_fraction > 0.0);
    assert(baseline.mean_word_prefix_energy_fraction > 0.0);
    assert(baseline.mean_top100_overlap >= 0.0 && baseline.mean_top100_overlap <= 1.0);

    config.char_weight = 0.5f;
    config.normalization = experiment::FeatureFamilyNormalization::JointRms;
    const auto replayed = workspace.run(config, 5);
    assert(replayed.evidence_reused);
    const auto char_metrics = experiment::evaluate_rankings(replayed.char_rankings, fixture);
    const auto word_metrics = experiment::evaluate_rankings(replayed.word_rankings, fixture);
    const double union_recall = experiment::family_union_recall_at_100(
        replayed.char_rankings, replayed.word_rankings, fixture);
    assert(union_recall >= char_metrics.recall_at_100);
    assert(union_recall >= word_metrics.recall_at_100);

    config.policy = experiment::FeatureFamilyPolicy::BaseOnly;
    const auto base = workspace.run(config, 5);
    assert(base.base_evidence_available);
    assert(!base.base_evidence_reused);
    assert(base.base_rankings.size() == fixture.query_ids.size());
    assert(base.cached_score_evidence_bytes ==
           fixture.query_ids.size() * fixture.doc_ids.size() * 3 * sizeof(float));
    assert(base.deployable_document_vector_bytes ==
           fixture.doc_ids.size() * config.idf_encoder.encoder.output_dim * sizeof(float));

    config.policy = experiment::FeatureFamilyPolicy::ResidualBlend;
    config.residual_weight = 0.2f;
    config.residual_score_normalization =
        experiment::FeatureResidualScoreNormalization::QueryZScore;
    const auto residual = workspace.run(config, 5);
    assert(residual.base_evidence_reused);
    assert(residual.admitted_queries == fixture.query_ids.size());
    assert(residual.deployable_document_vector_bytes ==
           fixture.doc_ids.size() *
               (config.idf_encoder.encoder.output_dim + config.storage_budget_dim) * sizeof(float));

    config.residual_score_normalization = experiment::FeatureResidualScoreNormalization::RankRrf;
    const auto rank_residual = workspace.run(config, 5);
    assert(rank_residual.base_evidence_reused);
    assert(rank_residual.admitted_queries == fixture.query_ids.size());
    assert(rank_residual.rankings.size() == fixture.query_ids.size());

    config.policy = experiment::FeatureFamilyPolicy::Selective;
    config.residual_score_normalization =
        experiment::FeatureResidualScoreNormalization::QueryZScore;
    config.min_word_energy = 1.0f;
    config.max_word_energy = 1.0f;
    const auto selective = workspace.run(config, 5);
    assert(selective.base_evidence_reused);
    assert(selective.rejected_queries == fixture.query_ids.size());
    assert(selective.rankings == selective.base_rankings);

    experiment::ResultContext context;
    context.experiment = "family-atlas";
    context.manifest_fingerprint = "manifest-id";
    context.variant = "split";
    context.kind = "feature_family_atlas_idf";
    context.fixture = fixture.name;
    context.fixture_fingerprint = experiment::fixture_fingerprint(fixture);
    context.split = fixture.split;
    const auto base_metrics = experiment::evaluate_rankings(selective.base_rankings, fixture);
    const auto family_metrics = experiment::evaluate_rankings(selective.family_rankings, fixture);
    const auto selected_metrics = experiment::evaluate_rankings(selective.rankings, fixture);
    const auto json = experiment::feature_family_atlas_result_json(
        context, config, workspace.stats(), selective, selected_metrics, base_metrics,
        family_metrics, char_metrics, word_metrics, union_recall, 1.0, fixture.query_ids.size(),
        fixture.doc_ids.size(), fixture.qrels.size());
    assert(json.find("\"kind\":\"feature_family_atlas_idf\"") != std::string::npos);
    assert(json.find("\"family_metrics\":") != std::string::npos);
    assert(json.find("\"family_normalization\":\"joint_rms\"") != std::string::npos);
    assert(json.find("\"family_policy\":\"selective\"") != std::string::npos);
    assert(json.find("\"residual_score_normalization\":\"query_zscore\"") != std::string::npos);
    assert(json.find("\"deployable_total_dim\":128") != std::string::npos);
    assert(json.find("\"exact_single_vector\":false") != std::string::npos);
    assert(json.find("\"requires_query_corpus_score_moments\":true") != std::string::npos);
    assert(json.find("\"base_evidence_available\":true") != std::string::npos);
    assert(json.find("\"admitted_queries\":0") != std::string::npos);
    assert(json.find("\"evaluation_only_union_recall_at_100\":") != std::string::npos);
    assert(json.find("\"deployable_document_vector_bytes\":") != std::string::npos);

    auto compatible = config;
    compatible.char_dim = 32;
    compatible.word_dim = 32;
    assert(workspace.compatible(compatible));
    auto incompatible = config;
    incompatible.idf_encoder.encoder.output_dim = 96;
    assert(!workspace.compatible(incompatible));
}

void test_standard_metrics_and_deterministic_ties() {
    experiment::Fixture fixture;
    fixture.split = "test";
    fixture.query_ids = {"q0", "q1"};
    fixture.query_texts = {"alpha", "beta"};
    fixture.doc_ids = {"d0", "d1", "d2"};
    fixture.doc_texts = {"alpha", "also alpha", "beta"};
    fixture.qrels = {{0, 0, 3}, {0, 1, 1}, {1, 2, 1}};

    // q0 intentionally supplies tied docs in reverse order. The evaluator's
    // doc-index tie break must still put d0 first and produce perfect nDCG.
    const std::vector<experiment::Ranking> rankings = {
        {{1.0f, 1}, {1.0f, 0}, {0.0f, 2}},
        {{2.0f, 2}, {0.0f, 0}, {0.0f, 1}},
    };
    const auto metrics = experiment::evaluate_rankings(rankings, fixture);
    assert(metrics.evaluated_queries == 2);
    assert(std::fabs(metrics.ndcg_at_10 - 1.0) < 1e-12);
    assert(std::fabs(metrics.precision_at_10 - 0.15) < 1e-12);
    assert(std::fabs(metrics.recall_at_10 - 1.0) < 1e-12);
    assert(std::fabs(metrics.recall_at_100 - 1.0) < 1e-12);
    assert(std::fabs(metrics.mrr_at_10 - 1.0) < 1e-12);

    const auto fingerprint = experiment::fixture_fingerprint(fixture);
    fixture.name = "same-data-different-directory";
    assert(experiment::fixture_fingerprint(fixture) == fingerprint);
    fixture.doc_texts[0] = "changed";
    assert(experiment::fixture_fingerprint(fixture) != fingerprint);

    auto non_finite = rankings;
    non_finite[0][0].first = std::numeric_limits<float>::quiet_NaN();
    try {
        (void)experiment::evaluate_rankings(non_finite, fixture);
        assert(false && "non-finite ranking score must fail");
    } catch (const std::runtime_error&) {
    }
}

void test_fixture_split_rejects_path_characters() {
    try {
        (void)experiment::load_fixture(".", "../test");
        assert(false && "fixture split path traversal must fail");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("unsupported character") != std::string::npos);
    }
}

void test_trec_ndcg_uses_qrel_values_as_linear_gains() {
    experiment::Fixture fixture;
    fixture.query_ids = {"q0"};
    fixture.query_texts = {"query"};
    fixture.doc_ids = {"high", "low"};
    fixture.doc_texts = {"high", "low"};
    fixture.qrels = {{0, 0, 3}, {0, 1, 1}};
    const std::vector<experiment::Ranking> ranking = {{{2.0f, 1}, {1.0f, 0}}};

    const auto metrics = experiment::evaluate_rankings(ranking, fixture);
    const double discount = std::log2(3.0);
    const double expected = (1.0 + 3.0 / discount) / (3.0 + 1.0 / discount);
    assert(std::fabs(metrics.ndcg_at_10 - expected) < 1e-12);
}

void test_json_escape_handles_control_characters() {
    assert(experiment::json_escape("quote\" slash\\ newline\n") ==
           "quote\\\" slash\\\\ newline\\n");
}

void test_encoder_retrieval_is_block_size_invariant() {
    experiment::Fixture fixture;
    fixture.name = "tiny";
    fixture.split = "test";
    fixture.query_ids = {"q0", "q1"};
    fixture.query_texts = {"alpha alpha", "beta beta"};
    fixture.doc_ids = {"d0", "d1", "d2", "d3"};
    fixture.doc_texts = {"alpha alpha", "beta beta", "unrelated gamma", "other delta"};
    fixture.qrels = {{0, 0, 1}, {1, 1, 1}};

    simeon::EncoderConfig config;
    config.sketch_dim = 256;
    config.text_normalization = simeon::TextNormalization::AsciiLower;
    config.char_ngram_scope = simeon::CharNGramScope::WordBounded;
    config.sketch_weighting = simeon::SketchWeighting::SignedSqrt;
    config.projection = simeon::ProjectionMode::None;

    const auto one_at_a_time = experiment::run_encoder_retrieval(config, fixture, 1, 4);
    const auto one_block = experiment::run_encoder_retrieval(config, fixture, 4, 4);
    assert(one_at_a_time.output_dim == 256);
    assert(one_at_a_time.rankings == one_block.rankings);
    assert(one_at_a_time.rankings[0][0].second == 0);
    assert(one_at_a_time.rankings[1][0].second == 1);
    assert(one_at_a_time.peak_working_bytes < one_block.peak_working_bytes);

    const auto top_two = experiment::run_encoder_retrieval(config, fixture, 4, 2);
    assert(top_two.rankings.size() == 2);
    assert(top_two.rankings[0].size() == 2);
    assert(top_two.rankings[1].size() == 2);
    assert(top_two.rankings[0][0] == one_block.rankings[0][0]);
    assert(top_two.rankings[0][1] == one_block.rankings[0][1]);
    assert(top_two.rankings[1][0] == one_block.rankings[1][0]);
    assert(top_two.rankings[1][1] == one_block.rankings[1][1]);
    assert(top_two.peak_working_bytes < one_block.peak_working_bytes);

    try {
        (void)experiment::run_encoder_retrieval(config, fixture, 0, 4);
        assert(false && "zero block size should throw");
    } catch (const std::runtime_error&) {
    }
    try {
        (void)experiment::run_encoder_retrieval(config, fixture, 1, 0);
        assert(false && "zero retrieval depth should throw");
    } catch (const std::runtime_error&) {
    }
}

void test_result_record_has_a_versioned_self_describing_schema() {
    experiment::ResultContext context;
    context.experiment = "foundation";
    context.manifest_fingerprint = "manifest-id";
    context.variant = "hash";
    context.fixture = "tiny";
    context.fixture_fingerprint = "fixture-id";
    context.split = "test";
    context.code_revision = "abc123";
    context.metadata["phase"] = "frozen";
    context.document_block_size = 2;
    context.retrieval_depth = 100;

    simeon::EncoderConfig config;
    config.sketch_dim = 256;
    config.text_normalization = simeon::TextNormalization::AsciiLower;
    config.char_ngram_scope = simeon::CharNGramScope::WordBounded;
    config.feature_weighting = simeon::FeatureWeighting::SqrtTf;
    config.sketch_weighting = simeon::SketchWeighting::SignedSqrt;
    experiment::EncoderRetrievalRun run;
    run.output_dim = 256;
    run.query_encode_us = 1.0;
    run.document_encode_us = 2.0;
    run.score_us = 3.0;
    run.peak_working_bytes = 4096;
    experiment::Metrics metrics;
    metrics.ndcg_at_10 = 0.75;
    metrics.evaluated_queries = 2;

    const std::string json =
        experiment::encoder_result_json(context, config, run, metrics, 4.0, 2, 3, 2);
    assert(json.find("\"schema\":\"simeon.experiment.result.v1\"") != std::string::npos);
    assert(json.find("\"metric_profile\":\"trec-v1\"") != std::string::npos);
    assert(json.find("\"manifest\":\"manifest-id\"") != std::string::npos);
    assert(json.find("\"fixture\":\"fixture-id\"") != std::string::npos);
    assert(json.find("\"code_revision\":\"abc123\"") != std::string::npos);
    assert(json.find("\"metadata\":{\"phase\":\"frozen\"}") != std::string::npos);
    assert(json.find("\"effective_output_dim\":256") != std::string::npos);
    assert(json.find("\"retrieval_depth\":100") != std::string::npos);
    assert(json.find("\"text_normalization\":\"ascii_lower\"") != std::string::npos);
    assert(json.find("\"char_ngram_scope\":\"word_bounded\"") != std::string::npos);
    assert(json.find("\"feature_weighting\":\"sqrt_tf\"") != std::string::npos);
    assert(json.find("\"sketch_weighting\":\"signed_sqrt\"") != std::string::npos);
    assert(json.find("\"hashed_idf\":null") != std::string::npos);
    assert(json.find("\"ndcg_at_10\":0.75") != std::string::npos);
}

void test_run_policy_blocks_holdout_peeking_and_requires_lineage() {
    auto exploration = parse(R"(
        schema=1
        name=selection
        phase=exploration
        selection_split=dev
        holdout_split=test
        [variant.a]
        kind=encoder
    )");
    experiment::validate_run_policy(exploration, "dev");
    try {
        experiment::validate_run_policy(exploration, "test");
        assert(false && "exploration manifest must not run on holdout");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("holdout") != std::string::npos);
    }

    exploration.metadata["phase"] = "frozen";
    try {
        experiment::validate_run_policy(exploration, "test");
        assert(false && "frozen holdout must identify its selection manifest");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("selection_manifest") != std::string::npos);
    }
    exploration.metadata["selection_manifest"] = "fnv1a64:selection";
    experiment::validate_run_policy(exploration, "test");
}

} // namespace

int main() {
    test_manifest_parses_named_variants();
    test_manifest_rejects_duplicate_keys_with_a_line_number();
    test_manifest_fingerprint_is_semantic();
    test_encoder_config_is_resolved_and_unknown_parameters_fail();
    test_corpus_adaptive_encoder_config_and_regime_are_strict();
    test_wsdm_idf_fusion_config_and_regime_are_strict();
    test_wsdm_idf_fusion_workspace_replays_variants();
    test_coordinate_calibration_config_and_regime_are_strict();
    test_coordinate_calibration_workspace_replays_nested_charts();
    test_feature_family_atlas_config_and_regime_are_strict();
    test_feature_family_atlas_replays_weights_and_exposes_evaluation_views();
    test_standard_metrics_and_deterministic_ties();
    test_fixture_split_rejects_path_characters();
    test_trec_ndcg_uses_qrel_values_as_linear_gains();
    test_json_escape_handles_control_characters();
    test_encoder_retrieval_is_block_size_invariant();
    test_result_record_has_a_versioned_self_describing_schema();
    test_run_policy_blocks_holdout_peeking_and_requires_lineage();
    return 0;
}
