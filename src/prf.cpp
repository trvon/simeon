#include "simeon/prf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

struct HashingSink final : NGramEmitter {
    std::vector<std::uint64_t>* hashes;
    const Bm25Index* idx;
    void on_token(std::string_view tok, float) override { hashes->push_back(idx->hash_term(tok)); }
};

// Mirrors Bm25Index's word-only tokenizer config so the query is tokenized
// identically between PRF and the underlying BM25 postings.
constexpr TokenizerConfig word_only_cfg() noexcept {
    return TokenizerConfig{0, 0, false, true};
}

// Partial sort the doc-id space by descending first_pass score; returns the
// top-K doc ids and their corresponding first-pass scores.
void top_k_docs(std::span<const float> scores, std::uint32_t k, std::vector<std::uint32_t>& out_ids,
                std::vector<float>& out_scores) {
    out_ids.clear();
    out_scores.clear();
    const std::size_t n = scores.size();
    if (k == 0 || n == 0)
        return;
    const std::uint32_t kk = static_cast<std::uint32_t>(std::min<std::size_t>(k, n));

    // Small-K case: partial_sort over an index vector. For scifact-class
    // corpora with K ≤ 50 this is cheaper than building a heap.
    std::vector<std::uint32_t> idx(n);
    for (std::size_t i = 0; i < n; ++i)
        idx[i] = static_cast<std::uint32_t>(i);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](std::uint32_t a, std::uint32_t b) { return scores[a] > scores[b]; });

    out_ids.reserve(kk);
    out_scores.reserve(kk);
    for (std::uint32_t i = 0; i < kk; ++i) {
        const std::uint32_t did = idx[i];
        const float s = scores[did];
        if (s <= 0.0f)
            break; // Negative/zero first-pass scores can't contribute to RM1.
        out_ids.push_back(did);
        out_scores.push_back(s);
    }
}

} // namespace

void score_with_prf(const Bm25Index& idx, std::string_view query, std::span<float> out_scores,
                    const PrfConfig& cfg) {
    const std::size_t n_docs = idx.doc_count();
    if (out_scores.size() != n_docs)
        throw std::runtime_error("score_with_prf: out_scores size mismatch");

    // 1. First-pass BM25 (full variant dispatch, including SAB n-gram backoff).
    std::vector<float> first_pass(n_docs, 0.0f);
    idx.score(query, first_pass);

    // 2. Top-K feedback docs (skipping any with non-positive score).
    std::vector<std::uint32_t> top_ids;
    std::vector<float> top_scores;
    top_k_docs(first_pass, cfg.k, top_ids, top_scores);

    // 3. Hash original query terms using idx's hash function.
    std::vector<std::uint64_t> q_hashes;
    q_hashes.reserve(16);
    HashingSink sink{};
    sink.hashes = &q_hashes;
    sink.idx = &idx;
    const auto tcfg = word_only_cfg();
    tokenize(query, tcfg, sink);

    // Degenerate path: if there is no feedback to build an RM from, or the
    // user disabled the expansion (alpha=0 / n_terms=0), fall back to plain
    // weighted BM25 on the original query so PRF(alpha=0) = BM25 exactly.
    const bool no_feedback = top_ids.empty() || cfg.n_terms == 0 || cfg.alpha <= 0.0f;
    if (no_feedback) {
        // alpha=0 must reproduce BM25 bit-for-bit on non-SAB variants; just
        // copy the first-pass scores (score() already ran the correct path).
        std::copy(first_pass.begin(), first_pass.end(), out_scores.begin());
        return;
    }

    // 4. Normalize feedback-doc weights to sum to 1 so p(d | Q) is a proper
    // distribution over the feedback set.
    double sum_fp = 0.0;
    for (float s : top_scores)
        sum_fp += s;
    std::vector<float> doc_weights(top_scores.size(), 0.0f);
    if (sum_fp > 0.0) {
        for (std::size_t i = 0; i < top_scores.size(); ++i)
            doc_weights[i] = static_cast<float>(top_scores[i] / sum_fp);
    } else {
        // Shouldn't reach here — top_k_docs filtered non-positive scores.
        std::copy(first_pass.begin(), first_pass.end(), out_scores.begin());
        return;
    }

    // 5. Build RM1 relevance model: weight(w) = Σ_d (tf(w,d)/|d|) * p(d|Q).
    std::vector<std::pair<std::uint64_t, float>> rm;
    idx.build_relevance_model(top_ids, doc_weights, rm);

    // 6. Keep top-N expansion terms and renormalize to a probability dist.
    const std::size_t keep = std::min<std::size_t>(cfg.n_terms, rm.size());
    if (keep == 0) {
        std::copy(first_pass.begin(), first_pass.end(), out_scores.begin());
        return;
    }
    // Sort the full RM by (weight desc, hash asc) so the top-N keep is
    // deterministic on weight ties across different Bm25Index instances —
    // partial_sort with a weight-only comparator leaves tied elements in
    // arbitrary order (driven by postings_ hash-bucket layout), which would
    // make score_with_prf non-deterministic across builds.
    std::sort(rm.begin(), rm.end(), [](const auto& a, const auto& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    if (rm.size() > keep)
        rm.resize(keep);
    double rm_sum = 0.0;
    for (const auto& [h, w] : rm)
        rm_sum += w;
    if (rm_sum > 0.0) {
        for (auto& [h, w] : rm)
            w = static_cast<float>(w / rm_sum);
    }

    // 7. Build combined weighted query:
    //       θ'(w) = (1-α) * p_Q(w) + α * p_R(w)
    // where p_Q is the MLE over the original query tokens.
    std::unordered_map<std::uint64_t, float> combined;
    combined.reserve(q_hashes.size() + rm.size());
    if (!q_hashes.empty()) {
        const float q_total = static_cast<float>(q_hashes.size());
        const float one_minus_alpha = 1.0f - cfg.alpha;
        for (std::uint64_t h : q_hashes) {
            combined[h] += one_minus_alpha * (1.0f / q_total);
        }
    }
    for (const auto& [h, w] : rm) {
        combined[h] += cfg.alpha * w;
    }

    // 8. Re-score under the combined weighted query. Sort by hash so the
    // summation order in score_weighted_hashes is fixed across Bm25Index
    // instances — combined is an unordered_map whose iteration order varies
    // with bucket layout, which otherwise makes FP accumulation non-
    // deterministic across builds.
    std::vector<std::pair<std::uint64_t, float>> weighted;
    weighted.reserve(combined.size());
    for (const auto& [h, w] : combined)
        weighted.emplace_back(h, w);
    std::sort(weighted.begin(), weighted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    idx.score_weighted_hashes(weighted, out_scores);
}

std::uint32_t n_terms_for_clarity(float clarity, std::uint32_t n_min, std::uint32_t n_max,
                                  float clarity_lo, float clarity_hi) noexcept {
    if (n_max <= n_min)
        return n_min;
    if (clarity_hi <= clarity_lo)
        return n_min;
    if (clarity <= clarity_lo)
        return n_min;
    if (clarity >= clarity_hi)
        return n_max;
    const float t = (clarity - clarity_lo) / (clarity_hi - clarity_lo);
    const float k = static_cast<float>(n_min) + t * static_cast<float>(n_max - n_min);
    const auto kr = static_cast<std::uint32_t>(k + 0.5f);
    if (kr < n_min)
        return n_min;
    if (kr > n_max)
        return n_max;
    return kr;
}

} // namespace simeon
