#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "experiment_support.hpp"
#include "reranker_experiment_support.hpp"

namespace experiment = simeon::experiment;

namespace {

experiment::Manifest parse(std::string_view text) {
    std::istringstream input(std::string{text});
    return experiment::parse_manifest(input);
}

void test_fixed_defaults_match_the_ffi_identity() {
    const auto manifest = parse(R"(
        schema=1
        name=reranker
        training_regime=corpus-adaptive
        [variant.ffi]
        kind=fragment_geometry_reranker
    )");
    experiment::validate_training_regime(manifest);
    const auto config = experiment::resolve_fragment_reranker_config(manifest.variants.front());
    assert(config.encoder_profile == experiment::RerankerEncoderProfile::FixedHash);
    assert(config.fixed_encoder.sketch_dim == 4096);
    assert(config.fixed_encoder.output_dim == 384);
    assert(config.fixed_encoder.projection == simeon::ProjectionMode::AchlioptasSparse);
    assert(config.builder == experiment::FragmentBuilderKind::Rich);
    assert(config.storage == experiment::FragmentStorageKind::Float32);
    assert(config.geometry.pool_size == 100);
}

void test_regime_and_parameter_validation_are_strict() {
    const auto artifact_free = parse(R"(
        schema=1
        name=invalid
        training_regime=artifact-free
        [variant.reranker]
        kind=fragment_geometry_reranker
    )");
    try {
        experiment::validate_training_regime(artifact_free);
        assert(false && "artifact-free reranker must be rejected");
    } catch (const std::runtime_error&) {
    }

    auto invalid = parse(R"(
        schema=1
        name=invalid
        training_regime=corpus-adaptive
        [variant.reranker]
        kind=fragment_geometry_reranker
        pool_sizes=10
    )");
    try {
        (void)experiment::resolve_fragment_reranker_config(invalid.variants.front());
        assert(false && "unknown reranker parameter must be rejected");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("pool_sizes") != std::string::npos);
    }

    experiment::VariantSpec incompatible;
    incompatible.name = "incompatible";
    incompatible.kind = "fragment_geometry_reranker";
    incompatible.parameters["pmi_target_rank"] = "64";
    try {
        (void)experiment::resolve_fragment_reranker_config(incompatible);
        assert(false && "fixed profile must reject ignored PMI parameters");
    } catch (const std::runtime_error&) {
    }
    incompatible.parameters.clear();
    incompatible.parameters["sentence_overlap_cap"] = "0.5";
    try {
        (void)experiment::resolve_fragment_reranker_config(incompatible);
        assert(false && "basic/rich builder must reject ignored overlap caps");
    } catch (const std::runtime_error&) {
    }
}

void test_workspace_reports_quality_and_profile_evidence() {
    experiment::Fixture fixture;
    fixture.name = "memory";
    fixture.split = "dev";
    fixture.query_ids = {"q1", "q2"};
    fixture.query_texts = {"neural training", "bread recipe"};
    fixture.doc_ids = {"d1", "d2", "d3"};
    fixture.doc_texts = {
        "neural network training gradients and optimization",
        "bread recipe flour water salt and sourdough starter",
        "mountain trail weather and hiking equipment",
    };
    fixture.qrels = {{0, 0, 1}, {1, 1, 1}};

    experiment::VariantSpec variant;
    variant.name = "smoke";
    variant.kind = "fragment_geometry_reranker";
    variant.parameters = {{"builder", "basic"},
                          {"fragment_storage", "bf16"},
                          {"pool_size", "3"},
                          {"outer_maxsim", "true"}};
    const auto config = experiment::resolve_fragment_reranker_config(variant);
    experiment::FragmentRerankerWorkspace workspace(fixture, config);
    auto run = workspace.run(config, 2, 3);
    run.candidate_recall = experiment::evaluate_candidate_recall(run.candidate_rankings, fixture);
    assert(run.rankings.size() == fixture.query_ids.size());
    assert(run.baseline_rankings.size() == fixture.query_ids.size());
    assert(run.profile.total_mean_us > 0.0);
    assert(run.profile.total_p95_us >= run.profile.total_p50_us);
    assert(run.candidate_recall >= 0.0 && run.candidate_recall <= 1.0);
    assert(workspace.stats().encoder_dim == 384);
    assert(workspace.stats().fragment_count > 0);
    assert(workspace.stats().fragment_vector_bytes ==
           workspace.stats().fragment_count * 384 * sizeof(std::uint16_t));

    const auto metrics = experiment::evaluate_rankings(run.rankings, fixture);
    const auto baseline = experiment::evaluate_rankings(run.baseline_rankings, fixture);
    experiment::ResultContext context;
    context.experiment = "smoke";
    context.variant = "smoke";
    context.kind = variant.kind;
    context.fixture = fixture.name;
    context.split = fixture.split;
    context.retrieval_depth = 3;
    const auto json = experiment::reranker_result_json(
        context, config, workspace.stats(), run, metrics, baseline, 1.0, fixture.query_ids.size(),
        fixture.doc_ids.size(), fixture.qrels.size());
    assert(json.find("\"candidate_recall\"") != std::string::npos);
    assert(json.find("\"reranker_query_p95\"") != std::string::npos);
    assert(json.find("\"bm25_baseline\"") != std::string::npos);
}

} // namespace

int main() {
    test_fixed_defaults_match_the_ffi_identity();
    test_regime_and_parameter_validation_are_strict();
    test_workspace_reports_quality_and_profile_evidence();
    return 0;
}
