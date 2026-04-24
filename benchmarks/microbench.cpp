// simeon microbenchmark — single-doc latency and batch throughput across
// projection dims and sketch sizes. Emits one JSONL record per row to
// stdout; redirect into a file for analysis.
//
//   ./microbench > simeon_microbench.jsonl
//
// Output fields:
//   backend, sketch_dim, output_dim, ngram_min, ngram_max, projection,
//   docs, doc_len_bytes, elapsed_us, us_per_doc, docs_per_sec, simd_tier.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"

namespace {

std::string make_doc(std::mt19937& rng, std::size_t bytes) {
    static constexpr char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?\n";
    std::uniform_int_distribution<int> d(0, static_cast<int>(sizeof(alphabet) - 2));
    std::string s;
    s.reserve(bytes);
    for (std::size_t i = 0; i < bytes; ++i)
        s.push_back(alphabet[d(rng)]);
    return s;
}

struct Row {
    std::uint32_t sketch_dim;
    std::uint32_t output_dim;
    simeon::ProjectionMode projection;
    const char* projection_name;
    std::uint32_t ngram_min;
    std::uint32_t ngram_max;
};

double run_single(const simeon::EncoderConfig& cfg, const std::vector<std::string>& docs) {
    simeon::Encoder e(cfg);
    std::vector<float> out(e.output_dim(), 0.0f);
    const auto t0 = std::chrono::steady_clock::now();
    for (const auto& d : docs)
        e.encode(d, out.data());
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

void emit(const Row& r, std::size_t docs, std::size_t doc_len, double us) {
    const double us_per_doc = us / static_cast<double>(docs);
    const double docs_per_sec = 1.0e6 / us_per_doc;
    std::printf(
        "{\"backend\":\"simeon\",\"sketch_dim\":%u,\"output_dim\":%u,\"ngram_min\":%u,"
        "\"ngram_max\":%u,\"projection\":\"%s\",\"docs\":%zu,\"doc_len_bytes\":%zu,"
        "\"elapsed_us\":%.2f,\"us_per_doc\":%.3f,\"docs_per_sec\":%.1f,\"simd_tier\":\"%s\"}\n",
        r.sketch_dim, r.output_dim, r.ngram_min, r.ngram_max, r.projection_name, docs, doc_len, us,
        us_per_doc, docs_per_sec, simeon::simd_tier_name(simeon::active_simd_tier()));
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t docs = argc > 1 ? static_cast<std::size_t>(std::atoll(argv[1])) : 2000;
    const std::size_t doc_len = argc > 2 ? static_cast<std::size_t>(std::atoll(argv[2])) : 512;

    std::mt19937 rng(0xC0FFEE);
    std::vector<std::string> corpus;
    corpus.reserve(docs);
    for (std::size_t i = 0; i < docs; ++i)
        corpus.push_back(make_doc(rng, doc_len));

    using PM = simeon::ProjectionMode;
    const Row rows[] = {
        {1024, 0, PM::None, "none", 3, 5},
        {4096, 0, PM::None, "none", 3, 5},

        {2048, 256, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {2048, 384, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {4096, 128, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {4096, 256, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {4096, 384, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {4096, 512, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {4096, 768, PM::AchlioptasSparse, "achlioptas", 3, 5},
        {8192, 384, PM::AchlioptasSparse, "achlioptas", 3, 5},

        {4096, 384, PM::AchlioptasSparse, "achlioptas", 3, 3},
        {4096, 384, PM::AchlioptasSparse, "achlioptas", 3, 7},
        {4096, 384, PM::AchlioptasSparse, "achlioptas", 4, 6},

        {4096, 256, PM::DenseGaussian, "gaussian", 3, 5},
        {4096, 384, PM::DenseGaussian, "gaussian", 3, 5},
        {4096, 768, PM::DenseGaussian, "gaussian", 3, 5},
        {4096, 256, PM::Fwht, "fwht", 3, 5},
        {4096, 384, PM::Fwht, "fwht", 3, 5},
        {4096, 768, PM::Fwht, "fwht", 3, 5},
        {4096, 256, PM::VerySparse, "very_sparse", 3, 5},
        {4096, 384, PM::VerySparse, "very_sparse", 3, 5},
        {4096, 768, PM::VerySparse, "very_sparse", 3, 5},
    };

    for (const auto& r : rows) {
        simeon::EncoderConfig cfg;
        cfg.ngram_mode = simeon::NGramMode::CharOnly;
        cfg.ngram_min = r.ngram_min;
        cfg.ngram_max = r.ngram_max;
        cfg.sketch_dim = r.sketch_dim;
        cfg.output_dim = r.output_dim;
        cfg.projection = r.projection;
        cfg.l2_normalize = true;

        // Warmup (fill thread_local sketch buffer, populate projection matrix).
        run_single(cfg, std::vector<std::string>(corpus.begin(),
                                                 corpus.begin() + std::min<std::size_t>(16, docs)));
        const double us = run_single(cfg, corpus);
        emit(r, docs, doc_len, us);
    }

    // SIMD kernel throughput. Reports ops/sec at the active tier so the Step
    // 1d Pareto table can cite the per-kernel number, not just the end-to-end.
    {
        const std::uint32_t n = 4096;
        const std::size_t iters = 2'000'000; // ~8 GFLOPs to dwarf wall-clock noise
        std::vector<float> a(n), b(n), w(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>((i * 31 + 7) % 97) / 97.0f;
            b[i] = static_cast<float>((i * 53 + 3) % 89) / 89.0f;
            w[i] = 1.0f - static_cast<float>(i) / static_cast<float>(n);
        }

        const char* tier = simeon::simd_tier_name(simeon::active_simd_tier());

        auto run_kernel = [&](const char* name, auto&& kernel_body) {
            // Warmup.
            for (int k = 0; k < 4; ++k)
                kernel_body();
            const auto t0 = std::chrono::steady_clock::now();
            for (std::size_t k = 0; k < iters; ++k)
                kernel_body();
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            const double calls_per_sec = static_cast<double>(iters) * 1.0e6 / us;
            const double elems_per_sec = calls_per_sec * static_cast<double>(n);
            std::printf("{\"backend\":\"simeon_kernel\",\"kernel\":\"%s\",\"n\":%u,"
                        "\"iters\":%zu,\"elapsed_us\":%.2f,\"calls_per_sec\":%.1f,"
                        "\"elems_per_sec\":%.1f,\"simd_tier\":\"%s\"}\n",
                        name, n, iters, us, calls_per_sec, elems_per_sec, tier);
            std::fflush(stdout);
        };

        run_kernel("add_vec_4096", [&]() { simeon::simd::add_vec(a.data(), b.data(), n); });
        // Reset a to avoid monotone growth biasing cache behavior.
        for (std::uint32_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>((i * 31 + 7) % 97) / 97.0f;
        }
        run_kernel("scale_vec_4096", [&]() { simeon::simd::scale_vec(a.data(), w.data(), n); });
        for (std::uint32_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>((i * 31 + 7) % 97) / 97.0f;
        }
        run_kernel("saxpy_4096", [&]() { simeon::simd::saxpy(a.data(), b.data(), 0.5f, n); });
    }
    return 0;
}
