// Manifest-driven exact retrieval experiments for the training-free encoder.
//
// This binary is intentionally thin: the stable experiment contract, fixture
// loading, scoring, metrics, provenance, and JSONL serialization live in
// experiment_support.* where they are exercised independently by unit tests.

#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "coordinate_experiment_support.hpp"
#include "experiment_support.hpp"
#include "family_atlas_experiment_support.hpp"
#include "fusion_experiment_support.hpp"

namespace {

struct Options {
    std::string manifest_path;
    std::string fixture_path;
    std::string split = "test";
    std::string variant;
    std::string code_revision = "unrecorded";
    std::size_t document_block_size = 256;
    bool block_size_overridden = false;
};

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
    if (!error.empty())
        std::fprintf(stderr, "error: %s\n\n", error.c_str());
    std::fprintf(stderr,
                 "usage: %s <manifest> --fixture <dir> [--split <name>]\n"
                 "       [--variant <name>] [--document-block-size <count>]\n"
                 "       [--code-revision <id>]\n",
                 program);
    std::exit(error.empty() ? 0 : 2);
}

std::size_t parse_size(std::string_view value, std::string_view option) {
    std::uint64_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed, 10);
    if (error != std::errc{} || end != value.data() + value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max())
        throw std::runtime_error(std::string(option) + " requires a positive integer");
    return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help")
        usage(argv[0]);
    if (argc < 2)
        usage(argv[0], "manifest path is required");

    Options options;
    options.manifest_path = argv[1];
    if (const char* revision = std::getenv("SIMEON_CODE_REVISION"))
        options.code_revision = revision;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--help")
            usage(argv[0]);
        if (i + 1 >= argc)
            usage(argv[0], std::string(option) + " requires a value");
        const std::string value = argv[++i];
        if (option == "--fixture") {
            options.fixture_path = value;
        } else if (option == "--split") {
            options.split = value;
        } else if (option == "--variant") {
            options.variant = value;
        } else if (option == "--document-block-size") {
            options.document_block_size = parse_size(value, option);
            options.block_size_overridden = true;
        } else if (option == "--code-revision") {
            options.code_revision = value;
        } else {
            usage(argv[0], "unknown option " + std::string(option));
        }
    }
    if (options.fixture_path.empty())
        usage(argv[0], "--fixture is required");
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        constexpr std::size_t kRetrievalDepth = 100;
        Options options = parse_options(argc, argv);
        const auto manifest = simeon::experiment::load_manifest(options.manifest_path);
        simeon::experiment::validate_run_policy(manifest, options.split);
        simeon::experiment::validate_training_regime(manifest);

        std::string metric_profile = "trec-v1";
        if (const auto found = manifest.metadata.find("metric_profile");
            found != manifest.metadata.end())
            metric_profile = found->second;
        if (metric_profile != "trec-v1")
            throw std::runtime_error("unsupported metric_profile '" + metric_profile + "'");

        if (!options.block_size_overridden) {
            if (const auto found = manifest.metadata.find("document_block_size");
                found != manifest.metadata.end())
                options.document_block_size = parse_size(found->second, "document_block_size");
        }

        const auto fixture = simeon::experiment::load_fixture(options.fixture_path, options.split);
        const std::string manifest_id = simeon::experiment::manifest_fingerprint(manifest);
        const std::string fixture_id = simeon::experiment::fixture_fingerprint(fixture);
        bool ran_variant = false;
        std::unique_ptr<simeon::experiment::WsdmIdfFusionWorkspace> fusion_workspace;
        std::unique_ptr<simeon::experiment::CoordinateCalibrationWorkspace> coordinate_workspace;
        std::unique_ptr<simeon::experiment::FeatureFamilyAtlasWorkspace> family_atlas_workspace;

        for (const auto& variant : manifest.variants) {
            if (!options.variant.empty() && variant.name != options.variant)
                continue;
            ran_variant = true;
            std::fprintf(stderr, "[experiment] %s/%s fixture=%s split=%s\n", manifest.name.c_str(),
                         variant.name.c_str(), fixture.name.c_str(), fixture.split.c_str());

            simeon::experiment::ResultContext context;
            context.experiment = manifest.name;
            context.manifest_fingerprint = manifest_id;
            context.variant = variant.name;
            context.kind = variant.kind;
            context.fixture = fixture.name;
            context.fixture_fingerprint = fixture_id;
            context.split = fixture.split;
            context.code_revision = options.code_revision;
            context.metric_profile = metric_profile;
            context.metadata = manifest.metadata;
            context.document_block_size = options.document_block_size;
            context.retrieval_depth = kRetrievalDepth;

            if (variant.kind == "wsdm_idf_fusion") {
                const auto resolved = simeon::experiment::resolve_wsdm_idf_fusion_config(variant);
                if (fusion_workspace == nullptr) {
                    fusion_workspace = std::make_unique<simeon::experiment::WsdmIdfFusionWorkspace>(
                        fixture, resolved);
                } else if (!fusion_workspace->compatible(resolved)) {
                    throw std::runtime_error(
                        "wsdm_idf_fusion variants must share pool and IDF identity");
                }
                const auto run = fusion_workspace->run(resolved);
                const auto evaluation_start = std::chrono::steady_clock::now();
                const auto metrics = simeon::experiment::evaluate_rankings(run.rankings, fixture);
                const double evaluation_us =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                              evaluation_start)
                        .count();
                std::cout << simeon::experiment::fusion_result_json(
                                 context, resolved, fusion_workspace->stats(), run, metrics,
                                 evaluation_us, fixture.query_ids.size(), fixture.doc_ids.size(),
                                 fixture.qrels.size())
                          << std::endl;
                continue;
            }

            if (variant.kind == "coordinate_calibrated_idf") {
                const auto resolved =
                    simeon::experiment::resolve_coordinate_calibration_config(variant);
                if (coordinate_workspace == nullptr) {
                    coordinate_workspace =
                        std::make_unique<simeon::experiment::CoordinateCalibrationWorkspace>(
                            fixture, resolved);
                } else if (!coordinate_workspace->compatible(resolved)) {
                    throw std::runtime_error("coordinate_calibrated_idf variants must share "
                                             "maximum-width IDF encoder identity");
                }
                const auto run = coordinate_workspace->run(resolved, kRetrievalDepth);
                const auto evaluation_start = std::chrono::steady_clock::now();
                const auto metrics = simeon::experiment::evaluate_rankings(run.rankings, fixture);
                const double evaluation_us =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                              evaluation_start)
                        .count();
                std::cout << simeon::experiment::coordinate_result_json(
                                 context, resolved, coordinate_workspace->stats(), run, metrics,
                                 evaluation_us, fixture.query_ids.size(), fixture.doc_ids.size(),
                                 fixture.qrels.size())
                          << std::endl;
                continue;
            }

            if (variant.kind == "feature_family_atlas_idf") {
                const auto resolved =
                    simeon::experiment::resolve_feature_family_atlas_config(variant);
                if (family_atlas_workspace == nullptr) {
                    family_atlas_workspace =
                        std::make_unique<simeon::experiment::FeatureFamilyAtlasWorkspace>(fixture,
                                                                                          resolved);
                } else if (!family_atlas_workspace->compatible(resolved)) {
                    throw std::runtime_error("feature_family_atlas_idf variants must share "
                                             "maximum-width encoder/IDF identity");
                }
                const auto run = family_atlas_workspace->run(resolved, kRetrievalDepth);
                const auto evaluation_start = std::chrono::steady_clock::now();
                const auto metrics = simeon::experiment::evaluate_rankings(run.rankings, fixture);
                simeon::experiment::Metrics base_metrics;
                if (run.base_evidence_available)
                    base_metrics =
                        simeon::experiment::evaluate_rankings(run.base_rankings, fixture);
                const auto family_metrics =
                    simeon::experiment::evaluate_rankings(run.family_rankings, fixture);
                const auto char_metrics =
                    simeon::experiment::evaluate_rankings(run.char_rankings, fixture);
                const auto word_metrics =
                    simeon::experiment::evaluate_rankings(run.word_rankings, fixture);
                const double union_recall = simeon::experiment::family_union_recall_at_100(
                    run.char_rankings, run.word_rankings, fixture);
                const double evaluation_us =
                    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                              evaluation_start)
                        .count();
                std::cout << simeon::experiment::feature_family_atlas_result_json(
                                 context, resolved, family_atlas_workspace->stats(), run, metrics,
                                 base_metrics, family_metrics, char_metrics, word_metrics,
                                 union_recall, evaluation_us, fixture.query_ids.size(),
                                 fixture.doc_ids.size(), fixture.qrels.size())
                          << std::endl;
                continue;
            }

            simeon::EncoderConfig config;
            std::optional<simeon::HashedIdf> idf;
            double artifact_build_us = 0.0;
            if (variant.kind == "hashed_idf_encoder") {
                const auto resolved =
                    simeon::experiment::resolve_hashed_idf_encoder_config(variant);
                config = resolved.encoder;
                std::vector<std::string_view> documents;
                documents.reserve(fixture.doc_texts.size());
                for (const auto& text : fixture.doc_texts)
                    documents.emplace_back(text);
                const auto build_start = std::chrono::steady_clock::now();
                idf.emplace(simeon::HashedIdf::learn(documents, config, resolved.idf));
                artifact_build_us = std::chrono::duration<double, std::micro>(
                                        std::chrono::steady_clock::now() - build_start)
                                        .count();
                config.hashed_idf = &*idf;
            } else {
                config = simeon::experiment::resolve_encoder_config(variant);
            }
            auto run = simeon::experiment::run_encoder_retrieval(
                config, fixture, options.document_block_size, kRetrievalDepth);
            run.artifact_build_us = artifact_build_us;
            run.artifact_bytes = idf.has_value() ? idf->storage_bytes() : 0;

            const auto evaluation_start = std::chrono::steady_clock::now();
            const auto metrics = simeon::experiment::evaluate_rankings(run.rankings, fixture);
            const double evaluation_us = std::chrono::duration<double, std::micro>(
                                             std::chrono::steady_clock::now() - evaluation_start)
                                             .count();

            std::cout << simeon::experiment::encoder_result_json(
                             context, config, run, metrics, evaluation_us, fixture.query_ids.size(),
                             fixture.doc_ids.size(), fixture.qrels.size())
                      << std::endl;
        }
        if (!ran_variant)
            throw std::runtime_error("manifest has no variant named '" + options.variant + "'");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "simeon_embedding_experiment: %s\n", error.what());
        return 2;
    }
}
