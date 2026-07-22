// Manifest-driven fragment-geometry reranker experiments. Quality labels are
// consumed only by the common evaluator; BM25, document preparation, encoder
// artifacts, fragments, and query scores are built without qrel access.

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "experiment_support.hpp"
#include "reranker_experiment_support.hpp"

namespace {

struct Options {
    std::string manifest_path;
    std::string fixture_path;
    std::string split = "test";
    std::string variant;
    std::string code_revision = "unrecorded";
    std::size_t query_repeats = 1;
    bool repeats_overridden = false;
};

[[noreturn]] void usage(const char* program, const std::string& error = {}) {
    if (!error.empty())
        std::fprintf(stderr, "error: %s\n\n", error.c_str());
    std::fprintf(stderr,
                 "usage: %s <manifest> --fixture <dir> [--split <name>]\n"
                 "       [--variant <name>] [--query-repeats <count>]\n"
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
        } else if (option == "--query-repeats") {
            options.query_repeats = parse_size(value, option);
            options.repeats_overridden = true;
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
        auto options = parse_options(argc, argv);
        const auto manifest = simeon::experiment::load_manifest(options.manifest_path);
        simeon::experiment::validate_run_policy(manifest, options.split);
        simeon::experiment::validate_training_regime(manifest);
        if (manifest.metadata.at("training_regime") != "corpus-adaptive")
            throw std::runtime_error(
                "fragment reranking is corpus-adaptive because BM25 consumes corpus statistics");

        std::string metric_profile = "trec-v1";
        if (const auto found = manifest.metadata.find("metric_profile");
            found != manifest.metadata.end())
            metric_profile = found->second;
        if (metric_profile != "trec-v1")
            throw std::runtime_error("unsupported metric_profile '" + metric_profile + "'");
        if (!options.repeats_overridden) {
            if (const auto found = manifest.metadata.find("query_repeats");
                found != manifest.metadata.end())
                options.query_repeats = parse_size(found->second, "query_repeats");
        }

        const auto fixture = simeon::experiment::load_fixture(options.fixture_path, options.split);
        const std::string manifest_id = simeon::experiment::manifest_fingerprint(manifest);
        const std::string fixture_id = simeon::experiment::fixture_fingerprint(fixture);
        std::vector<std::unique_ptr<simeon::experiment::FragmentRerankerWorkspace>> workspaces;
        bool ran_variant = false;

        for (const auto& variant : manifest.variants) {
            if (!options.variant.empty() && variant.name != options.variant)
                continue;
            ran_variant = true;
            const auto config = simeon::experiment::resolve_fragment_reranker_config(variant);
            auto workspace =
                std::find_if(workspaces.begin(), workspaces.end(),
                             [&](const auto& item) { return item->compatible(config); });
            if (workspace == workspaces.end()) {
                workspaces.push_back(
                    std::make_unique<simeon::experiment::FragmentRerankerWorkspace>(fixture,
                                                                                    config));
                workspace = std::prev(workspaces.end());
            }
            std::fprintf(stderr, "[reranker] %s/%s fixture=%s split=%s repeats=%zu\n",
                         manifest.name.c_str(), variant.name.c_str(), fixture.name.c_str(),
                         fixture.split.c_str(), options.query_repeats);

            auto run = (*workspace)->run(config, options.query_repeats, kRetrievalDepth);
            const auto evaluation_start = std::chrono::steady_clock::now();
            run.candidate_recall =
                simeon::experiment::evaluate_candidate_recall(run.candidate_rankings, fixture);
            const auto metrics = simeon::experiment::evaluate_rankings(run.rankings, fixture);
            const auto baseline_metrics =
                simeon::experiment::evaluate_rankings(run.baseline_rankings, fixture);
            const double evaluation_us = std::chrono::duration<double, std::micro>(
                                             std::chrono::steady_clock::now() - evaluation_start)
                                             .count();

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
            context.retrieval_depth = kRetrievalDepth;
            std::cout << simeon::experiment::reranker_result_json(
                             context, config, (*workspace)->stats(), run, metrics, baseline_metrics,
                             evaluation_us, fixture.query_ids.size(), fixture.doc_ids.size(),
                             fixture.qrels.size())
                      << std::endl;
        }
        if (!ran_variant)
            throw std::runtime_error("manifest has no variant named '" + options.variant + "'");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "simeon_reranker_experiment: %s\n", error.what());
        return 2;
    }
}
