// PRF / RM1-build microbench — validates the forward-index build_relevance_model
// against the reference O(total postings) inverted scan, at several corpus
// scales, so the O(corpus) → O(feedback) win is visible as a flat-vs-rising
// per-call curve. Emits one JSONL record per scale to stdout.
//
//   ./simeon_bench_prf [n1 n2 n3 ...]   (doc counts; default 2k 10k 50k 250k)
//
// Output fields per row:
//   n_docs, total_postings, feedback_k, rm_terms, repeats,
//   inverted_us_per_call, forward_first_us, forward_us_per_call, speedup,
//   bit_identical, simd_tier.
//
// bit_identical compares the two RM term sets (sorted by hash, then weight)
// for exact float equality — a hard correctness gate before any speed claim.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/simd.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// Deterministic Zipfian-ish corpus: V-word vocabulary, each doc ~L tokens drawn
// from a skewed distribution so posting-list lengths look like real text.
std::vector<std::string> make_corpus(std::size_t n_docs, std::uint32_t seed) {
    constexpr std::size_t kVocab = 40000;
    constexpr std::size_t kDocLen = 110;
    std::mt19937 rng(seed);
    // Zipf weights ~ 1/(rank+1); sample via a precomputed CDF over a capped head
    // plus a uniform tail to keep generation cheap.
    std::uniform_int_distribution<std::size_t> head(0, 2000);       // frequent words
    std::uniform_int_distribution<std::size_t> tail(0, kVocab - 1); // rare words
    std::bernoulli_distribution pick_head(0.6);
    std::vector<std::string> docs;
    docs.reserve(n_docs);
    std::string doc;
    for (std::size_t d = 0; d < n_docs; ++d) {
        doc.clear();
        for (std::size_t t = 0; t < kDocLen; ++t) {
            const std::size_t w = pick_head(rng) ? head(rng) : tail(rng);
            doc += 'w';
            doc += std::to_string(w);
            doc += ' ';
        }
        docs.push_back(doc);
    }
    return docs;
}

std::vector<std::pair<std::uint64_t, float>>
sorted_rm(std::vector<std::pair<std::uint64_t, float>> rm) {
    std::sort(rm.begin(), rm.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });
    return rm;
}

void run_scale(std::size_t n_docs) {
    auto docs = make_corpus(n_docs, /*seed=*/1234u);
    simeon::Bm25Index idx;
    idx.reserve_docs(n_docs);
    for (const auto& d : docs)
        idx.add_doc(d);
    idx.finalize();

    // First-pass BM25 on a fixed query; feedback = top-10 by score.
    std::vector<float> scores(idx.doc_count(), 0.0f);
    idx.score("w3 w17 w42 w108", scores);
    constexpr std::uint32_t kFeedback = 10;
    std::vector<std::uint32_t> order(scores.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
        order[i] = i;
    const std::uint32_t k = std::min<std::uint32_t>(kFeedback, idx.doc_count());
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return scores[a] > scores[b]; });
    std::vector<std::uint32_t> fb_ids(order.begin(), order.begin() + k);
    std::vector<float> fb_w(k, 0.0f);
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < k; ++i)
        sum += scores[fb_ids[i]] > 0.0f ? scores[fb_ids[i]] : 0.0f;
    for (std::uint32_t i = 0; i < k; ++i)
        fb_w[i] = sum > 0.0f ? (scores[fb_ids[i]] > 0.0f ? scores[fb_ids[i]] / sum : 0.0f)
                             : 1.0f / static_cast<float>(k);

    std::vector<std::pair<std::uint64_t, float>> rm_inv, rm_fwd;

    // Correctness gate: the two builders must agree bit-for-bit (as a set).
    idx.build_relevance_model_inverted(fb_ids, fb_w, rm_inv);
    idx.build_relevance_model(fb_ids, fb_w, rm_fwd); // first call builds forward index
    const bool identical = sorted_rm(rm_inv) == sorted_rm(rm_fwd);

    // x-axis label: corpus token budget (≈ total postings up to per-doc term
    // repeats). The inverted builder cost scales with this; the forward builder
    // does not.
    const std::uint64_t corpus_tokens = static_cast<std::uint64_t>(n_docs) * 110ull;

    constexpr int kRepeats = 200;
    // Inverted: O(corpus) each call.
    auto t0 = Clock::now();
    for (int r = 0; r < kRepeats; ++r)
        idx.build_relevance_model_inverted(fb_ids, fb_w, rm_inv);
    const double inv_us = us_since(t0) / kRepeats;

    // Forward steady-state: index already built, so this is O(feedback terms).
    auto t1 = Clock::now();
    for (int r = 0; r < kRepeats; ++r)
        idx.build_relevance_model(fb_ids, fb_w, rm_fwd);
    const double fwd_us = us_since(t1) / kRepeats;

    // One-time forward build cost: rebuild a fresh index and time the first call.
    simeon::Bm25Index idx2;
    idx2.reserve_docs(n_docs);
    for (const auto& d : docs)
        idx2.add_doc(d);
    idx2.finalize();
    std::vector<std::pair<std::uint64_t, float>> rm_first;
    auto t2 = Clock::now();
    idx2.build_relevance_model(fb_ids, fb_w, rm_first);
    const double first_us = us_since(t2);

    std::printf("{\"n_docs\":%zu,\"corpus_tokens\":%llu,\"feedback_k\":%u,\"rm_terms\":%zu,"
                "\"repeats\":%d,\"inverted_us_per_call\":%.3f,\"forward_first_us\":%.3f,"
                "\"forward_us_per_call\":%.4f,\"speedup\":%.1f,\"bit_identical\":%s,"
                "\"simd_tier\":\"%s\"}\n",
                n_docs, static_cast<unsigned long long>(corpus_tokens), k, rm_fwd.size(), kRepeats,
                inv_us, first_us, fwd_us, fwd_us > 0.0 ? inv_us / fwd_us : 0.0,
                identical ? "true" : "false", simeon::simd_tier_name(simeon::active_simd_tier()));

    if (!identical) {
        std::fprintf(stderr,
                     "FAIL: RM mismatch at n_docs=%zu (inverted=%zu terms, forward=%zu terms)\n",
                     n_docs, rm_inv.size(), rm_fwd.size());
        std::exit(1);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::size_t> scales;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i)
            scales.push_back(static_cast<std::size_t>(std::strtoull(argv[i], nullptr, 10)));
    } else {
        scales = {2000, 10000, 50000, 250000};
    }
    for (std::size_t n : scales)
        run_scale(n);
    return 0;
}
