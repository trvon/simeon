#include "simeon/fusion.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "simeon/bm25.hpp"
#include "simeon/simd.hpp"

namespace simeon {

namespace {

void sort_desc_by_score(std::vector<std::pair<std::uint32_t, float>>& v) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
}

} // namespace

std::vector<std::pair<std::uint32_t, float>> rrf_fuse(std::span<const Ranking> rankings, float k) {
    std::unordered_map<std::uint32_t, float> acc;
    for (const auto& ranking : rankings) {
        std::vector<std::pair<std::uint32_t, float>> sorted(ranking.begin(), ranking.end());
        sort_desc_by_score(sorted);
        for (std::size_t r = 0; r < sorted.size(); ++r) {
            acc[sorted[r].first] += 1.0f / (k + static_cast<float>(r + 1));
        }
    }
    std::vector<std::pair<std::uint32_t, float>> out(acc.begin(), acc.end());
    sort_desc_by_score(out);
    return out;
}

void score_bm25_variants_rrf(std::span<const Bm25Index* const> variants, std::string_view query,
                             std::span<float> out_scores, float k_rrf) {
    std::fill(out_scores.begin(), out_scores.end(), 0.0f);
    if (variants.empty() || out_scores.empty())
        return;
    const std::size_t nd = out_scores.size();
    std::vector<std::vector<std::pair<std::uint32_t, float>>> per_variant(variants.size());
    std::vector<float> scratch(nd, 0.0f);
    for (std::size_t v = 0; v < variants.size(); ++v) {
        std::fill(scratch.begin(), scratch.end(), 0.0f);
        variants[v]->score(query, scratch);
        per_variant[v].resize(nd);
        for (std::uint32_t d = 0; d < nd; ++d)
            per_variant[v][d] = {d, scratch[d]};
    }
    std::vector<Ranking> ins;
    ins.reserve(variants.size());
    for (const auto& r : per_variant)
        ins.emplace_back(r);
    auto fused = rrf_fuse(ins, k_rrf);
    for (const auto& [did, score] : fused) {
        if (did < nd)
            out_scores[did] = score;
    }
}

std::vector<std::pair<std::uint32_t, float>> top_k(std::span<const float> scores, std::uint32_t k) {
    const std::uint32_t n = static_cast<std::uint32_t>(scores.size());
    if (k > n)
        k = n;
    std::vector<std::pair<std::uint32_t, float>> all(n);
    for (std::uint32_t i = 0; i < n; ++i)
        all[i] = {i, scores[i]};
    std::partial_sort(all.begin(), all.begin() + k, all.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
    all.resize(k);
    return all;
}

void cosine_scores(const float* query, const float* corpus, std::uint32_t n_docs, std::uint32_t dim,
                   float* out_scores) noexcept {
    for (std::uint32_t i = 0; i < n_docs; ++i) {
        out_scores[i] = simd::dot(query, corpus + static_cast<std::size_t>(i) * dim, dim);
    }
}

std::vector<std::pair<std::uint32_t, float>> cosine_topk(const float* query, const float* corpus,
                                                         std::uint32_t n_docs, std::uint32_t dim,
                                                         std::uint32_t k) {
    std::vector<float> scores(n_docs);
    cosine_scores(query, corpus, n_docs, dim, scores.data());
    return top_k(scores, k);
}

namespace {

// Softmax-normalized Shannon entropy of a score slice. Numerically stable
// (subtract max before exp). Returns entropy in nats in [0, log(n)]. For the
// edge cases: empty or single-element slices return 0 (degenerate distribution
// with H = 0).
double softmax_entropy_nats(std::span<const float> scores) noexcept {
    if (scores.size() < 2)
        return 0.0;
    double smax = scores[0];
    for (float s : scores) {
        if (static_cast<double>(s) > smax)
            smax = static_cast<double>(s);
    }
    double zsum = 0.0;
    for (float s : scores) {
        zsum += std::exp(static_cast<double>(s) - smax);
    }
    if (zsum <= 0.0)
        return 0.0;
    double ent = 0.0;
    for (float s : scores) {
        const double w = std::exp(static_cast<double>(s) - smax) / zsum;
        if (w > 0.0)
            ent -= w * std::log(w);
    }
    if (ent < 0.0)
        ent = 0.0;
    return ent;
}

void zscore_inplace(std::vector<float>& v) noexcept {
    if (v.empty())
        return;
    double mean = 0.0;
    for (float x : v)
        mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v) {
        const double d = x - mean;
        var += d * d;
    }
    const float sd = static_cast<float>(std::sqrt(var / static_cast<double>(v.size())) + 1e-12);
    for (float& x : v) {
        x = static_cast<float>((x - mean) / sd);
    }
}

} // namespace

float entropy_alpha(std::span<const float> a_top_k, std::span<const float> b_top_k,
                    float pool_jaccard, const EntropyAlphaConfig& cfg) noexcept {
    if (pool_jaccard >= cfg.agreement_threshold)
        return 0.5f;
    const std::size_t ka =
        std::min<std::size_t>(a_top_k.size(), cfg.top_k == 0 ? a_top_k.size() : cfg.top_k);
    const std::size_t kb =
        std::min<std::size_t>(b_top_k.size(), cfg.top_k == 0 ? b_top_k.size() : cfg.top_k);
    auto slice_a = a_top_k.subspan(0, ka);
    auto slice_b = b_top_k.subspan(0, kb);
    const double ha = softmax_entropy_nats(slice_a);
    const double hb = softmax_entropy_nats(slice_b);
    const double log_ka = ka > 1 ? std::log(static_cast<double>(ka)) : 0.0;
    const double log_kb = kb > 1 ? std::log(static_cast<double>(kb)) : 0.0;
    double conf_a = log_ka > 0.0 ? 1.0 - ha / log_ka : 0.0;
    double conf_b = log_kb > 0.0 ? 1.0 - hb / log_kb : 0.0;
    if (conf_a < 0.0)
        conf_a = 0.0;
    if (conf_b < 0.0)
        conf_b = 0.0;
    const double denom = conf_a + conf_b;
    if (denom <= 0.0)
        return 0.5f;
    double alpha = conf_a / denom;
    if (alpha < 0.0)
        alpha = 0.0;
    if (alpha > 1.0)
        alpha = 1.0;
    return static_cast<float>(alpha);
}

void linear_alpha_entropy_fuse(std::span<const float> a_scores, std::span<const float> b_scores,
                               std::span<const float> a_top_k, std::span<const float> b_top_k,
                               float pool_jaccard, const EntropyAlphaConfig& cfg,
                               std::span<float> out_scores) noexcept {
    const std::size_t n = a_scores.size();
    if (n == 0 || b_scores.size() != n || out_scores.size() != n)
        return;
    const float alpha = entropy_alpha(a_top_k, b_top_k, pool_jaccard, cfg);
    std::vector<float> za(a_scores.begin(), a_scores.end());
    std::vector<float> zb(b_scores.begin(), b_scores.end());
    zscore_inplace(za);
    zscore_inplace(zb);
    for (std::size_t i = 0; i < n; ++i) {
        out_scores[i] = alpha * za[i] + (1.0f - alpha) * zb[i];
    }
}

} // namespace simeon
