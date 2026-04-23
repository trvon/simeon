#include "simeon/query_router.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simeon/fusion.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

// Word-only tokenization, mirroring Bm25Index::add_doc / score so the
// router's term lookups match the index's keys exactly.
struct WordSink final : NGramEmitter {
    std::vector<std::string>* words;
    void on_token(std::string_view tok, float) override { words->emplace_back(tok); }
};

} // namespace

QueryRouter::QueryRouter(const Bm25Index& idx, RouterConfig cfg) noexcept : idx_(idx), cfg_(cfg) {}

QueryFeatures QueryRouter::features(std::string_view query) const {
    std::vector<std::string> words;
    WordSink sink{};
    sink.words = &words;
    const TokenizerConfig tcfg{0, 0, false, true};
    tokenize(query, tcfg, sink);

    QueryFeatures f;
    f.n_terms = static_cast<std::uint32_t>(words.size());
    if (f.n_terms == 0)
        return f;

    std::uint32_t oov = 0;
    double idf_sum = 0.0;
    double idf_sq_sum = 0.0;
    std::uint32_t present = 0;
    std::uint64_t char_sum = 0;
    float min_idf_seen = std::numeric_limits<float>::infinity();
    // Step 1k B2: collect per-distinct-term query counts so simplified
    // clarity uses the true MLE p(t|Q) (duplicate query terms contribute
    // once to the KL sum, weighted by their query-side count).
    std::unordered_map<std::uint64_t, std::uint32_t> q_count;
    q_count.reserve(words.size());
    double scq_sum_acc = 0.0;
    for (const auto& w : words) {
        char_sum += w.size();
        const std::uint32_t df = idx_.df(w);
        if (df == 0) {
            ++oov;
            continue;
        }
        const float i = idx_.idf(w);
        idf_sum += static_cast<double>(i);
        idf_sq_sum += static_cast<double>(i) * static_cast<double>(i);
        if (i > f.max_idf)
            f.max_idf = i;
        if (i < min_idf_seen)
            min_idf_seen = i;
        ++present;
        // SCQ: (1 + log(tf_C(t))) * idf(t). Per-occurrence accumulation
        // matches Zhao 2008's sum-over-query-terms definition (duplicate
        // terms contribute twice).
        const std::uint64_t ttf = idx_.total_tf(w);
        if (ttf > 0) {
            scq_sum_acc += (1.0 + std::log(static_cast<double>(ttf))) * static_cast<double>(i);
        }
        ++q_count[idx_.hash_term(w)];
    }
    f.oov_rate = static_cast<float>(oov) / static_cast<float>(f.n_terms);
    if (present > 0) {
        f.avg_idf = static_cast<float>(idf_sum / present);
        f.min_idf = min_idf_seen;
        if (present > 1) {
            const double mean = idf_sum / present;
            const double var = (idf_sq_sum / present) - (mean * mean);
            f.idf_stddev = static_cast<float>(var > 0.0 ? std::sqrt(var) : 0.0);
        }
    }
    f.avg_term_chars =
        static_cast<float>(static_cast<double>(char_sum) / static_cast<double>(f.n_terms));
    f.scq_sum = static_cast<float>(scq_sum_acc);
    // Simplified clarity: Σ_t p(t|Q) log(p(t|Q) / p(t|C)) over distinct
    // present query terms. q_count was filled above (only present terms).
    const std::uint64_t total_tokens = idx_.total_tokens();
    if (!q_count.empty() && total_tokens > 0) {
        const double q_norm = static_cast<double>(f.n_terms);
        const double c_norm = static_cast<double>(total_tokens);
        double clarity_acc = 0.0;
        std::unordered_map<std::uint64_t, bool> visited;
        visited.reserve(q_count.size());
        for (const auto& w : words) {
            const std::uint64_t h = idx_.hash_term(w);
            if (visited[h])
                continue;
            visited[h] = true;
            const std::uint64_t ttf = idx_.total_tf(w);
            if (ttf == 0)
                continue;
            const auto it = q_count.find(h);
            if (it == q_count.end() || it->second == 0)
                continue;
            const double p_q = static_cast<double>(it->second) / q_norm;
            const double p_c = static_cast<double>(ttf) / c_norm;
            clarity_acc += p_q * std::log(p_q / p_c);
        }
        f.simplified_clarity = static_cast<float>(clarity_acc);
    }
    return f;
}

QueryFeatures QueryRouter::features_with_pool(std::string_view query,
                                              std::span<const Bm25Index* const> pools,
                                              std::uint32_t k) const {
    QueryFeatures f = features(query);
    if (pools.empty() || k == 0 || f.n_terms == 0)
        return f;

    // Pool 0: decay/var/entropy from top-K BM25 scores. Pool 1 (if present):
    // top-K Jaccard against pool 0. Cost is one BM25 score() per pool plus a
    // partial sort — both bounded; on scifact (~5K docs, K=50) ≤ a few ms.
    const Bm25Index* idx0 = pools[0];
    if (idx0 == nullptr || idx0->doc_count() == 0)
        return f;
    const std::uint32_t nd0 = idx0->doc_count();
    const std::uint32_t k0 = std::min(k, nd0);
    std::vector<float> scores0(nd0, 0.0f);
    idx0->score(query, scores0);
    auto pool0 = top_k(scores0, k0);

    if (!pool0.empty()) {
        const float s0 = pool0.front().second;
        const std::size_t i9 = pool0.size() >= 10 ? std::size_t{9} : pool0.size() - 1;
        if (s0 > 0.0f) {
            const float decay = (s0 - pool0[i9].second) / s0;
            f.score_decay_rate = std::clamp(decay, 0.0f, 1.0f);
        }
        double mean = 0.0;
        for (const auto& p : pool0)
            mean += p.second;
        mean /= static_cast<double>(pool0.size());
        if (mean > 0.0) {
            double var = 0.0;
            for (const auto& p : pool0) {
                const double d = p.second - mean;
                var += d * d;
            }
            var /= static_cast<double>(pool0.size());
            f.score_normalized_var = static_cast<float>(var / mean);
        }
        // T1 NQC (Shtok-Kurland-Carmel 2012) + T2 full WIG (Zhou-Croft 2007).
        // Both need corpus-level mean BM25 score. Reuse the dense scores0
        // vector rather than re-scoring — O(nd) one pass, already hot.
        double corpus_sum = 0.0;
        for (float s : scores0)
            corpus_sum += s;
        const double corpus_mean = corpus_sum / static_cast<double>(nd0);
        if (corpus_mean > 0.0) {
            double pool_var = 0.0;
            for (const auto& p : pool0) {
                const double d = p.second - mean;
                pool_var += d * d;
            }
            pool_var /= static_cast<double>(pool0.size());
            f.nqc = static_cast<float>(std::sqrt(pool_var) / corpus_mean);
        }
        if (f.n_terms > 0) {
            f.wig_full = static_cast<float>((mean - corpus_mean) /
                                            std::sqrt(static_cast<double>(f.n_terms)));
        }
        // Numerically-stable softmax entropy: subtract max before exp.
        double smax = pool0.front().second;
        double zsum = 0.0;
        for (const auto& p : pool0) {
            zsum += std::exp(static_cast<double>(p.second) - smax);
        }
        if (zsum > 0.0) {
            double ent = 0.0;
            for (const auto& p : pool0) {
                const double w = std::exp(static_cast<double>(p.second) - smax) / zsum;
                if (w > 0.0)
                    ent -= w * std::log(w);
            }
            f.top_k_score_entropy = static_cast<float>(ent);
        }
    }

    if (pools.size() >= 2 && pools[1] != nullptr && pools[1]->doc_count() > 0) {
        const Bm25Index* idx1 = pools[1];
        const std::uint32_t nd1 = idx1->doc_count();
        const std::uint32_t k1 = std::min(k, nd1);
        std::vector<float> scores1(nd1, 0.0f);
        idx1->score(query, scores1);
        auto pool1 = top_k(scores1, k1);
        std::unordered_set<std::uint32_t> set0;
        set0.reserve(pool0.size());
        for (const auto& p : pool0)
            set0.insert(p.first);
        std::uint32_t inter = 0;
        for (const auto& p : pool1) {
            if (set0.count(p.first))
                ++inter;
        }
        const std::uint32_t uni = static_cast<std::uint32_t>(pool0.size() + pool1.size()) - inter;
        f.pool_overlap_jaccard =
            uni == 0 ? 1.0f : static_cast<float>(inter) / static_cast<float>(uni);
    }
    return f;
}

Recipe QueryRouter::choose(std::string_view query) const {
    return choose(features(query));
}

Recipe QueryRouter::choose(const QueryFeatures& f) const noexcept {
    if (f.n_terms == 0)
        return Recipe::Bm25SabSmooth;
    if (f.oov_rate > cfg_.oov_threshold)
        return Recipe::Bm25SabSmooth;
    if (f.avg_idf > cfg_.high_idf_threshold && f.n_terms >= cfg_.atire_min_terms &&
        f.min_idf >= cfg_.atire_min_idf_floor &&
        f.pool_overlap_jaccard <= cfg_.atire_max_pool_jaccard &&
        f.score_decay_rate >= cfg_.atire_min_score_decay && f.scq_sum >= cfg_.atire_min_scq &&
        f.simplified_clarity <= cfg_.atire_max_clarity) {
        return Recipe::Bm25Atire;
    }
    if (f.n_terms >= cfg_.cascade_min_terms && f.avg_idf <= cfg_.cascade_max_idf) {
        return Recipe::CascadeLinearAlpha;
    }
    return Recipe::Bm25SabSmooth;
}

const char* recipe_name(Recipe r) noexcept {
    switch (r) {
        case Recipe::Bm25Atire:
            return "Bm25Atire";
        case Recipe::Bm25SabSmooth:
            return "Bm25SabSmooth";
        case Recipe::CascadeLinearAlpha:
            return "CascadeLinearAlpha";
    }
    return "unknown";
}

} // namespace simeon
