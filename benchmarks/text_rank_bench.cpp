// simeon TextRank microbench.
//
// Synthetic sweep: generates documents with varying sentence counts,
// measures per-doc latency percentiles, and reports the agreement rate
// between TextRank top-1 and lead-1 (first sentence) baseline.
//
// Ground-truth-free: this bench is a scaling sanity check, not a
// quality regression gate. The yams-side end-to-end bench is the
// quality signal (Phase 2 plan).
//
// Emits one JSONL row per configuration:
//   {
//     "bench": "text_rank",
//     "doc_count": 200,
//     "sentences_per_doc": 16,
//     "tokens_per_sentence": 12,
//     "p50_ms": 0.184,
//     "p99_ms": 0.431,
//     "mean_iters": 6.8,
//     "lead1_agreement": 0.24
//   }

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "simeon/text_rank.hpp"

namespace {

std::vector<std::string> topic_pools() {
    return {
        "database index query table row btree commit lock transaction deadlock",
        "compiler parser lexer token ast codegen optimize inline loop vector",
        "kernel scheduler thread mutex semaphore process page fault trap syscall",
        "neural network layer gradient backprop optimizer loss batch epoch",
        "tls certificate cipher handshake session ticket resume key exchange",
    };
}

std::string make_sentence(std::mt19937& rng, const std::string& pool,
                          std::size_t tokens_per_sentence) {
    // Split the pool into words, sample with replacement.
    std::vector<std::string> words;
    std::string cur;
    for (char c : pool) {
        if (c == ' ') {
            if (!cur.empty()) {
                words.push_back(cur);
                cur.clear();
            }
        } else
            cur.push_back(c);
    }
    if (!cur.empty())
        words.push_back(cur);

    std::uniform_int_distribution<std::size_t> pick(0, words.size() - 1);
    std::string s;
    s.reserve(tokens_per_sentence * 8);
    for (std::size_t i = 0; i < tokens_per_sentence; ++i) {
        if (i > 0)
            s += ' ';
        s += words[pick(rng)];
    }
    s += '.';
    return s;
}

std::string make_doc(std::mt19937& rng, std::size_t n_sentences, std::size_t tokens_per_sentence,
                     const std::vector<std::string>& pools, std::size_t topic_idx) {
    std::string doc;
    doc.reserve(n_sentences * tokens_per_sentence * 8);
    for (std::size_t i = 0; i < n_sentences; ++i) {
        doc += make_sentence(rng, pools[topic_idx], tokens_per_sentence);
        doc += ' ';
    }
    return doc;
}

double percentile(std::vector<double>& v, double p) {
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    const auto idx = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
    return v[idx];
}

void bench_row(std::size_t doc_count, std::size_t n_sentences, std::size_t tokens_per_sentence) {
    std::mt19937 rng(static_cast<std::uint32_t>(0xABCD0001 ^ n_sentences));
    const auto pools = topic_pools();
    std::uniform_int_distribution<std::size_t> topic(0, pools.size() - 1);

    std::vector<std::string> docs(doc_count);
    for (std::size_t i = 0; i < doc_count; ++i) {
        docs[i] = make_doc(rng, n_sentences, tokens_per_sentence, pools, topic(rng));
    }

    simeon::TextRank tr;
    std::vector<double> latencies(doc_count, 0.0);
    std::size_t lead1_agree = 0;

    for (std::size_t i = 0; i < doc_count; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto r = tr.rank(docs[i], 1);
        const auto t1 = std::chrono::steady_clock::now();
        latencies[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!r.empty() && r[0].index == 0)
            ++lead1_agree;
    }

    std::printf("{\"bench\":\"text_rank\",\"doc_count\":%zu,"
                "\"sentences_per_doc\":%zu,\"tokens_per_sentence\":%zu,"
                "\"p50_ms\":%.3f,\"p99_ms\":%.3f,"
                "\"lead1_agreement\":%.3f}\n",
                doc_count, n_sentences, tokens_per_sentence, percentile(latencies, 0.50),
                percentile(latencies, 0.99),
                static_cast<double>(lead1_agree) / static_cast<double>(doc_count));
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t doc_count = 200;
    if (argc > 1)
        doc_count = static_cast<std::size_t>(std::atoll(argv[1]));
    const std::size_t tokens_per_sentence = 12;

    const std::array<std::size_t, 5> sentences_sweep = {4, 8, 16, 32, 64};
    for (const auto ns : sentences_sweep) {
        // Smoke mode: small doc_count × the two smallest sentence counts.
        if (doc_count < 50 && ns > 16)
            continue;
        bench_row(doc_count, ns, tokens_per_sentence);
    }
    return 0;
}
