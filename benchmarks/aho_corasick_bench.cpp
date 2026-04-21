// simeon Aho-Corasick microbench.
//
// Sweeps dictionary size × input type (hit-heavy vs noise) and reports
// build latency, automaton node count, match throughput, and match
// count. Fixed seed → deterministic JSONL output.
//
// Emits one record per cell:
//   {
//     "bench": "aho_corasick",
//     "pattern_count": 100000,
//     "node_count": 412345,
//     "build_ms": 120.4,
//     "input_type": "hits",
//     "input_bytes": 1048576,
//     "match_ms": 18.9,
//     "match_throughput_mb_s": 55.4,
//     "match_count": 10234
//   }

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/aho_corasick.hpp"

namespace {

std::string random_pattern(std::mt19937& rng, std::uniform_int_distribution<int>& len) {
    std::uniform_int_distribution<int> ch('a', 'z');
    const int k = len(rng);
    std::string s;
    s.resize(static_cast<std::size_t>(k));
    for (int j = 0; j < k; ++j)
        s[j] = static_cast<char>(ch(rng));
    return s;
}

std::string build_noise_text(std::size_t target_bytes) {
    // Space-delimited random tokens — extremely low hit rate so we
    // measure the no-match transition cost.
    std::mt19937 rng(0xBEEFF00D);
    std::uniform_int_distribution<int> ch('a', 'z');
    std::uniform_int_distribution<int> len(4, 10);
    std::string out;
    out.reserve(target_bytes + 16);
    while (out.size() < target_bytes) {
        const int k = len(rng);
        for (int i = 0; i < k; ++i)
            out.push_back(static_cast<char>(ch(rng)));
        out.push_back(' ');
    }
    out.resize(target_bytes);
    return out;
}

void bench_row(std::size_t n_patterns, const std::string& hits_text,
               const std::string& noise_text) {
    std::mt19937 rng(static_cast<std::uint32_t>(0xC0DEC0DE ^ n_patterns));
    std::uniform_int_distribution<int> len(5, 12);

    std::vector<std::string> owned(n_patterns);
    std::vector<std::string_view> views(n_patterns);
    std::vector<std::uint16_t> types(n_patterns, 1);
    for (std::size_t i = 0; i < n_patterns; ++i) {
        owned[i] = random_pattern(rng, len);
        views[i] = owned[i];
    }

    simeon::AhoCorasick ac;
    const auto t0 = std::chrono::steady_clock::now();
    auto err = ac.build(views, types);
    const auto t1 = std::chrono::steady_clock::now();
    if (err.has_value()) {
        std::fprintf(stderr, "build failed: %s\n", err->message.c_str());
        return;
    }
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Rebuild hits_text against *this* automaton's patterns so hit-heavy
    // input is actually hit-heavy. The caller's hits_text was generated
    // from a prior pattern set of (possibly) different size — but for a
    // proper hit-heavy measurement we want patterns that exist in the
    // automaton, so regenerate here using a sample of `owned`.
    const std::size_t target_bytes = hits_text.size();
    std::string local_hits;
    local_hits.reserve(target_bytes + 32);
    std::mt19937 prng(0xAA55AA55);
    std::uniform_int_distribution<std::size_t> pick(0, owned.size() - 1);
    while (local_hits.size() < target_bytes) {
        local_hits += owned[pick(prng)];
        local_hits += ' ';
    }
    local_hits.resize(target_bytes);

    auto run = [&](const char* label, const std::string& text) {
        const auto m0 = std::chrono::steady_clock::now();
        const auto hits = ac.match(text);
        const auto m1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(m1 - m0).count();
        const double mbps =
            (ms > 0.0) ? (static_cast<double>(text.size()) / (1024.0 * 1024.0)) / (ms / 1000.0)
                       : 0.0;
        std::printf("{\"bench\":\"aho_corasick\",\"pattern_count\":%zu,"
                    "\"node_count\":%zu,\"build_ms\":%.3f,"
                    "\"input_type\":\"%s\",\"input_bytes\":%zu,"
                    "\"match_ms\":%.3f,\"match_throughput_mb_s\":%.2f,"
                    "\"match_count\":%zu}\n",
                    n_patterns, ac.node_count(), build_ms, label, text.size(), ms, mbps,
                    hits.size());
        std::fflush(stdout);
    };
    run("hits", local_hits);
    run("noise", noise_text);
}

// Sharded AC: split the pattern list into N disjoint automatons by
// hash(pattern_id), match each against the full input sequentially,
// and sum per-shard throughput as total wall time / input size. The
// goal is to see whether shrinking the monolithic automaton recovers
// throughput at UMLS-scale dictionaries.
void bench_sharded(std::size_t n_patterns, std::size_t n_shards, const std::string& noise_text) {
    std::mt19937 rng(static_cast<std::uint32_t>(0xC0DEC0DE ^ n_patterns));
    std::uniform_int_distribution<int> len(5, 12);

    std::vector<std::string> owned(n_patterns);
    for (std::size_t i = 0; i < n_patterns; ++i)
        owned[i] = random_pattern(rng, len);

    // Partition by hash(pattern) % n_shards.
    std::vector<std::vector<std::string_view>> views(n_shards);
    std::vector<std::vector<std::uint16_t>> types(n_shards);
    std::hash<std::string> h;
    for (std::size_t i = 0; i < n_patterns; ++i) {
        const auto s = h(owned[i]) % n_shards;
        views[s].push_back(owned[i]);
        types[s].push_back(1);
    }

    std::vector<simeon::AhoCorasick> shards(n_shards);
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t s = 0; s < n_shards; ++s) {
        auto err = shards[s].build(views[s], types[s]);
        if (err.has_value()) {
            std::fprintf(stderr, "shard build failed: %s\n", err->message.c_str());
            return;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Hits text: concatenation of sampled patterns.
    const std::size_t target_bytes = noise_text.size();
    std::string local_hits;
    local_hits.reserve(target_bytes + 32);
    std::mt19937 prng(0xAA55AA55);
    std::uniform_int_distribution<std::size_t> pick(0, owned.size() - 1);
    while (local_hits.size() < target_bytes) {
        local_hits += owned[pick(prng)];
        local_hits += ' ';
    }
    local_hits.resize(target_bytes);

    auto run = [&](const char* label, const std::string& text) {
        const auto m0 = std::chrono::steady_clock::now();
        std::size_t total_hits = 0;
        for (const auto& ac : shards) {
            const auto hits = ac.match(text);
            total_hits += hits.size();
        }
        const auto m1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(m1 - m0).count();
        const double mbps =
            (ms > 0.0) ? (static_cast<double>(text.size()) / (1024.0 * 1024.0)) / (ms / 1000.0)
                       : 0.0;
        std::printf("{\"bench\":\"aho_corasick_sharded\",\"pattern_count\":%zu,"
                    "\"n_shards\":%zu,\"build_ms\":%.3f,"
                    "\"input_type\":\"%s\",\"input_bytes\":%zu,"
                    "\"match_ms\":%.3f,\"match_throughput_mb_s\":%.2f,"
                    "\"match_count\":%zu}\n",
                    n_patterns, n_shards, build_ms, label, text.size(), ms, mbps, total_hits);
        std::fflush(stdout);
    };
    run("hits", local_hits);
    run("noise", noise_text);
}

} // namespace

int main(int argc, char** argv) {
    // Default 1 MB input, smoke mode accepts smaller via argv[1].
    std::size_t input_bytes = 1 * 1024 * 1024;
    if (argc > 1) {
        input_bytes = static_cast<std::size_t>(std::atoll(argv[1]));
    }

    const std::string hits_placeholder(input_bytes, ' '); // size only; regenerated per row
    const std::string noise = build_noise_text(input_bytes);

    const std::size_t sizes[] = {1'000, 10'000, 100'000, 500'000};
    for (std::size_t n : sizes) {
        // Smoke mode (input_bytes small) skips the largest dictionary so
        // meson test finishes under its timeout.
        if (input_bytes < 256 * 1024 && n > 10'000)
            continue;
        bench_row(n, hits_placeholder, noise);
    }

    // Sharding sweep at 500k (only in full-size runs).
    if (input_bytes >= 256 * 1024) {
        for (std::size_t n_shards : {2u, 4u, 8u, 16u}) {
            bench_sharded(500'000, n_shards, noise);
        }
    }
    return 0;
}
