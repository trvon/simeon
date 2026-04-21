// Prototype DA-trie AC bench. Mirrors aho_corasick_bench but uses the
// double-array backed variant. Branch-only; not wired into meson test.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/aho_corasick_da.hpp"

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

void bench_row(std::size_t n_patterns, const std::string& noise_text) {
    std::mt19937 rng(static_cast<std::uint32_t>(0xC0DEC0DE ^ n_patterns));
    std::uniform_int_distribution<int> len(5, 12);

    std::vector<std::string> owned(n_patterns);
    std::vector<std::string_view> views(n_patterns);
    std::vector<std::uint16_t> types(n_patterns, 1);
    for (std::size_t i = 0; i < n_patterns; ++i) {
        owned[i] = random_pattern(rng, len);
        views[i] = owned[i];
    }

    simeon::AhoCorasickDa ac;
    const auto t0 = std::chrono::steady_clock::now();
    auto err = ac.build(views, types);
    const auto t1 = std::chrono::steady_clock::now();
    if (err.has_value()) {
        std::fprintf(stderr, "build failed: %s\n", err->message.c_str());
        return;
    }
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

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
        const auto hits = ac.match(text);
        const auto m1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(m1 - m0).count();
        const double mbps =
            (ms > 0.0) ? (static_cast<double>(text.size()) / (1024.0 * 1024.0)) / (ms / 1000.0)
                       : 0.0;
        std::printf("{\"bench\":\"aho_corasick_da\",\"pattern_count\":%zu,"
                    "\"node_count\":%zu,\"da_slots\":%zu,\"build_ms\":%.3f,"
                    "\"input_type\":\"%s\",\"input_bytes\":%zu,"
                    "\"match_ms\":%.3f,\"match_throughput_mb_s\":%.2f,"
                    "\"match_count\":%zu}\n",
                    n_patterns, ac.node_count(), ac.da_slot_count(), build_ms, label, text.size(),
                    ms, mbps, hits.size());
        std::fflush(stdout);
    };
    run("hits", local_hits);
    run("noise", noise_text);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t input_bytes = 1 * 1024 * 1024;
    if (argc > 1) {
        input_bytes = static_cast<std::size_t>(std::atoll(argv[1]));
    }
    const std::string noise = build_noise_text(input_bytes);

    // DA build is O(n²) worst-case with naive linear scan; cap the prototype
    // at 100k to keep build under a minute.
    const std::size_t sizes[] = {1'000, 10'000, 100'000};
    for (std::size_t n : sizes) {
        if (input_bytes < 256 * 1024 && n > 10'000)
            continue;
        bench_row(n, noise);
    }
    return 0;
}
