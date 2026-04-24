// Isolated fragment-geometry profile harness.
//
// Loads a reference fixture (corpus.tsv + queries.tsv), builds one BM25 index,
// learns PMI rows, constructs document fragments, and runs N repeats of the
// fragment-geometry scorer while collecting phase timings. Designed for
// `sample` / `xctrace record --launch`.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/pmi.hpp"
#include "simeon/simeon.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::pair<std::string, std::string>> read_tsv2(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::pair<std::string, std::string>> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos)
            throw std::runtime_error("malformed TSV (missing tab) in " + path);
        out.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    return out;
}

double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct RunConfig {
    const char* builder = "basic";
    const char* mode = "phss";
};

simeon::FragmentGeometryConfig make_geom_cfg(const RunConfig& run) {
    simeon::FragmentGeometryConfig cfg;
    cfg.pool_size = 100;
    cfg.alpha = 0.8f;
    cfg.attention_scale = 8.0f;
    cfg.knn = 8;
    cfg.steps = 2;
    cfg.use_phss = std::string_view(run.mode) != "fixed";
    cfg.phss_config.criterion = std::string_view(run.mode) == "approx"
                                    ? simeon::PhssConfig::Criterion::LargestGapApprox
                                    : simeon::PhssConfig::Criterion::LargestGap;
    cfg.phss_adaptive = std::string_view(run.mode) == "adaptive";
    cfg.phss_confidence_threshold = 0.55f;
    const std::string_view builder = run.builder;
    cfg.top_fragments_per_doc = (builder == "basic" || builder == "basicpos") ? 4u : 8u;
    return cfg;
}

std::vector<std::vector<simeon::SemanticFragment>>
build_doc_frags(const RunConfig& run, const simeon::Encoder& enc, const simeon::Bm25Index& idx,
                std::span<const std::string> docs) {
    std::vector<std::vector<simeon::SemanticFragment>> out;
    out.reserve(docs.size());
    for (const auto& doc : docs) {
        if (std::string_view(run.builder) == "basic") {
            out.push_back(simeon::build_doc_semantic_fragments(enc, doc, idx, 6, 8));
        } else if (std::string_view(run.builder) == "basicpos") {
            out.push_back(simeon::build_doc_semantic_fragments(enc, doc, idx, 6, 8, 0.20f));
        } else if (std::string_view(run.builder) == "richcov") {
            out.push_back(simeon::build_doc_semantic_fragments_rich_covered(enc, doc, idx, 6, 8,
                                                                            0.60f, 0.80f));
        } else if (std::string_view(run.builder) == "richmmr") {
            out.push_back(simeon::build_doc_semantic_fragments_rich_mmr(
                enc, doc, idx, 6, 8, 0.60f, 0.80f, 0.35f, 0.30f, 0.15f));
        } else {
            throw std::runtime_error("unknown builder: " + std::string(run.builder));
        }
    }
    simeon::compress_fragments_to_bf16(out, enc.output_dim());
    return out;
}

template <class F>
double mean_of(const std::vector<simeon::FragmentGeometryProfile>& rows, F&& field) {
    if (rows.empty())
        return 0.0;
    double sum = 0.0;
    for (const auto& row : rows)
        sum += field(row);
    return sum / static_cast<double>(rows.size());
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: profile_fragment_geometry <fixture_dir> "
                             "<builder:basic|basicpos|richcov|richmmr> "
                             "<mode:fixed|phss|adaptive|approx> [iters=5]\n");
        return 2;
    }

    const std::string dir = argv[1];
    const RunConfig run{argv[2], argv[3]};
    const int iters = argc >= 5 ? std::atoi(argv[4]) : 5;

    namespace fs = std::filesystem;
    auto corpus = read_tsv2((fs::path(dir) / "corpus.tsv").string());
    auto queries = read_tsv2((fs::path(dir) / "queries.tsv").string());

    std::vector<std::string> doc_texts;
    doc_texts.reserve(corpus.size());
    for (auto& [id, text] : corpus)
        doc_texts.push_back(std::move(text));

    std::vector<std::string> query_texts;
    query_texts.reserve(queries.size());
    for (auto& [id, text] : queries)
        query_texts.push_back(std::move(text));

    std::fprintf(stderr, "[profile] dir=%s docs=%zu queries=%zu builder=%s mode=%s iters=%d\n",
                 dir.c_str(), doc_texts.size(), query_texts.size(), run.builder, run.mode, iters);

    simeon::Bm25Config bcfg;
    bcfg.variant = simeon::Bm25Variant::SubwordAwareBackoff;
    bcfg.subword_gamma = 5.0f;

    const auto t_add0 = Clock::now();
    simeon::Bm25Index idx(bcfg);
    for (const auto& doc : doc_texts)
        idx.add_doc(doc);
    const auto t_add1 = Clock::now();
    idx.finalize();
    const auto t_bm251 = Clock::now();

    std::vector<std::string_view> doc_views;
    doc_views.reserve(doc_texts.size());
    for (const auto& doc : doc_texts)
        doc_views.emplace_back(doc);

    simeon::PmiConfig pcfg;
    pcfg.target_rank = 128;
    pcfg.min_token_count = 5;
    pcfg.max_vocab_size = 20'000;
    const auto t_pmi0 = Clock::now();
    auto pmi = simeon::PmiEmbeddings::learn(std::span<const std::string_view>(doc_views), pcfg);
    const auto t_pmi1 = Clock::now();

    simeon::EncoderConfig ecfg;
    ecfg.ngram_mode = simeon::NGramMode::WordOnly;
    ecfg.ngram_min = 1;
    ecfg.ngram_max = 1;
    ecfg.sketch_dim = 0;
    ecfg.output_dim = pmi.dim();
    ecfg.projection = simeon::ProjectionMode::None;
    ecfg.l2_normalize = true;
    ecfg.pmi_rows = &pmi;
    simeon::Encoder enc(ecfg);

    const auto t_frag0 = Clock::now();
    auto doc_frags = build_doc_frags(run, enc, idx, doc_texts);
    const auto t_frag1 = Clock::now();

    std::size_t total_fragments = 0;
    for (const auto& doc : doc_frags)
        total_fragments += doc.size();
    const std::size_t f32_fragment_bytes =
        total_fragments * static_cast<std::size_t>(enc.output_dim()) * sizeof(float);
    const std::size_t bf16_fragment_bytes =
        total_fragments * static_cast<std::size_t>(enc.output_dim()) * sizeof(std::uint16_t);

    const auto cfg = make_geom_cfg(run);
    std::vector<simeon::FragmentGeometryProfile> profiles;
    profiles.reserve(static_cast<std::size_t>(iters) * query_texts.size());

    const auto t_query0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
        for (const auto& q : query_texts) {
            simeon::FragmentGeometryProfile profile;
            auto scores =
                simeon::score_fragment_geometry_profiled(q, idx, enc, doc_frags, cfg, &profile);
            (void)scores;
            profiles.push_back(profile);
        }
    }
    const auto t_query1 = Clock::now();

    std::printf("metric\tvalue\n");
    std::printf("builder\t%s\n", run.builder);
    std::printf("fragment_storage\tbf16\n");
    std::printf("mode\t%s\n", run.mode);
    std::printf("docs\t%zu\n", doc_texts.size());
    std::printf("queries\t%zu\n", query_texts.size());
    std::printf("iters\t%d\n", iters);
    std::printf("fragment_dim\t%u\n", enc.output_dim());
    std::printf("fragment_count\t%zu\n", total_fragments);
    std::printf("fragment_bytes_f32\t%zu\n", f32_fragment_bytes);
    std::printf("fragment_bytes_bf16\t%zu\n", bf16_fragment_bytes);
    std::printf("bm25_add_docs_us\t%.3f\n", us(t_add0, t_add1));
    std::printf("bm25_finalize_us\t%.3f\n", us(t_add1, t_bm251));
    std::printf("pmi_learn_us\t%.3f\n", us(t_pmi0, t_pmi1));
    std::printf("fragment_build_us\t%.3f\n", us(t_frag0, t_frag1));
    std::printf("query_total_us\t%.3f\n", us(t_query0, t_query1));
    std::printf("query_total_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.total_us; }));
    std::printf("bm25_mean_us\t%.3f\n", mean_of(profiles, [](const auto& p) { return p.bm25_us; }));
    std::printf("query_encode_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.query_encode_us; }));
    std::printf("gather_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.gather_us; }));
    std::printf("whiten_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.whiten_us; }));
    std::printf("phss_pairwise_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_pairwise_us; }));
    std::printf("phss_select_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_us; }));
    std::printf("phss_select_edge_gather_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_edge_gather_us; }));
    std::printf("phss_select_edge_sort_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_edge_sort_us; }));
    std::printf("phss_select_uf_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_uf_us; }));
    std::printf("phss_select_survivor_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_survivor_us; }));
    std::printf("phss_select_death_sort_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_death_sort_us; }));
    std::printf("phss_select_criterion_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_select_criterion_us; }));
    std::printf("triangle_count_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.triangle_count_us; }));
    std::printf("triangle_count_total_mean\t%.3f\n", mean_of(profiles, [](const auto& p) {
                    return static_cast<double>(p.triangle_count_total);
                }));
    std::printf("query_attention_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.query_attention_us; }));
    std::printf("adjacency_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.adjacency_us; }));
    std::printf("diffuse_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.diffuse_us; }));
    std::printf("blend_mean_us\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.blend_us; }));
    std::printf("pool_docs_mean\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.pool_docs; }));
    std::printf("pool_fragments_mean\t%.3f\n",
                mean_of(profiles, [](const auto& p) { return p.pool_fragments; }));
    std::printf("graph_edges_mean\t%.3f\n", mean_of(profiles, [](const auto& p) {
                    return static_cast<double>(p.graph_edges);
                }));
    std::printf("query_confidence_mean\t%.6f\n",
                mean_of(profiles, [](const auto& p) { return p.query_confidence; }));
    std::printf("phss_scale_mean\t%.6f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_selected_scale; }));
    std::printf("phss_used_rate\t%.6f\n",
                mean_of(profiles, [](const auto& p) { return p.phss_used ? 1.0 : 0.0; }));
    return 0;
}
