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
        per_variant[v].reserve(nd);
        for (std::uint32_t d = 0; d < nd; ++d) {
            if (std::isfinite(scratch[d]) && scratch[d] > 0.0f) {
                per_variant[v].push_back({d, scratch[d]});
            }
        }
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

namespace {

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

void convex_fuse_z(std::span<const std::span<const float>> legs, std::span<const float> weights,
                   std::span<float> out_scores) noexcept {
    const std::size_t n = out_scores.size();
    if (legs.empty() || weights.size() != legs.size())
        return;
    std::fill(out_scores.begin(), out_scores.end(), 0.0f);
    std::vector<float> z;
    for (std::size_t l = 0; l < legs.size(); ++l) {
        if (legs[l].size() != n)
            return;
        z.assign(legs[l].begin(), legs[l].end());
        zscore_inplace(z);
        const float w = weights[l];
        for (std::size_t i = 0; i < n; ++i)
            out_scores[i] += w * z[i];
    }
}

} // namespace simeon
