// Isolated SAB-smooth profile harness.
//
// Loads a reference-fixture corpus (corpus.tsv + queries.tsv — qrels and
// reference.bin not required), builds one Bm25Index with
// SubwordAwareBackoff variant, and times build + N-repeat query loop.
// Output is TSV on stdout; all timing goes through steady_clock so
// `/usr/bin/sample` and `xctrace record --launch` see a single busy phase.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/bm25.hpp"

namespace {

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

double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: profile_sab_smooth <fixture_dir> [iters=10]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int iters = (argc >= 3) ? std::atoi(argv[2]) : 10;

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

    // Match bench_vs_reference bm25_sab_smooth_gamma5 exactly.
    simeon::Bm25Config cfg;
    cfg.variant = simeon::Bm25Variant::SubwordAwareBackoff;
    cfg.subword_gamma = 5.0f;

    std::fprintf(stderr, "[profile] dir=%s docs=%zu queries=%zu iters=%d\n", dir.c_str(),
                 doc_texts.size(), query_texts.size(), iters);

    // ---------------- Build phase ----------------
    simeon::Bm25Index idx(cfg);
    const auto t_add_begin = std::chrono::steady_clock::now();
    for (const auto& text : doc_texts)
        idx.add_doc(text);
    const auto t_add_end = std::chrono::steady_clock::now();
    idx.finalize();
    const auto t_build_end = std::chrono::steady_clock::now();

    const double add_us = us(t_add_begin, t_add_end);
    const double finalize_us = us(t_add_end, t_build_end);
    const double build_us = us(t_add_begin, t_build_end);

    std::fprintf(stderr, "[profile] add=%.3f ms finalize=%.3f ms build=%.3f ms\n", add_us / 1000.0,
                 finalize_us / 1000.0, build_us / 1000.0);

    // ---------------- Query phase ----------------
    std::vector<float> scores(doc_texts.size(), 0.0f);
    std::vector<double> per_q;
    per_q.reserve(static_cast<std::size_t>(iters) * query_texts.size());

    const auto t_query_begin = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (const auto& q : query_texts) {
            std::fill(scores.begin(), scores.end(), 0.0f);
            const auto tq0 = std::chrono::steady_clock::now();
            idx.score(q, std::span<float>{scores});
            const auto tq1 = std::chrono::steady_clock::now();
            per_q.push_back(us(tq0, tq1));
        }
    }
    const auto t_query_end = std::chrono::steady_clock::now();
    const double query_total_us = us(t_query_begin, t_query_end);

    std::sort(per_q.begin(), per_q.end());
    const double q_min = per_q.front();
    const double q_p50 = per_q[per_q.size() / 2];
    const double q_p95 = per_q[static_cast<std::size_t>(per_q.size() * 0.95)];
    double sum = 0.0;
    for (double v : per_q)
        sum += v;
    const double q_mean = sum / static_cast<double>(per_q.size());

    // TSV to stdout for easy diff across runs.
    std::printf("metric\tvalue_us\n");
    std::printf("add_docs\t%.3f\n", add_us);
    std::printf("finalize\t%.3f\n", finalize_us);
    std::printf("build_total\t%.3f\n", build_us);
    std::printf("query_total\t%.3f\n", query_total_us);
    std::printf("query_per_q_min\t%.3f\n", q_min);
    std::printf("query_per_q_mean\t%.3f\n", q_mean);
    std::printf("query_per_q_p50\t%.3f\n", q_p50);
    std::printf("query_per_q_p95\t%.3f\n", q_p95);
    std::printf("qps\t%.2f\n", 1.0e6 / q_mean);
    return 0;
}
