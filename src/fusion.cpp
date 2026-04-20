#include "simeon/fusion.hpp"

#include <algorithm>
#include <unordered_map>

#include "simeon/simd.hpp"

namespace simeon {

namespace {

void sort_desc_by_score(std::vector<std::pair<std::uint32_t, float>>& v) {
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
}

}  // namespace

std::vector<std::pair<std::uint32_t, float>>
rrf_fuse(std::span<const Ranking> rankings, float k) {
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

std::vector<std::pair<std::uint32_t, float>>
top_k(std::span<const float> scores, std::uint32_t k) {
    const std::uint32_t n = static_cast<std::uint32_t>(scores.size());
    if (k > n) k = n;
    std::vector<std::pair<std::uint32_t, float>> all(n);
    for (std::uint32_t i = 0; i < n; ++i) all[i] = {i, scores[i]};
    std::partial_sort(all.begin(), all.begin() + k, all.end(),
                      [](const auto& a, const auto& b) {
                          if (a.second != b.second) return a.second > b.second;
                          return a.first < b.first;
                      });
    all.resize(k);
    return all;
}

void cosine_scores(const float* query, const float* corpus, std::uint32_t n_docs,
                   std::uint32_t dim, float* out_scores) noexcept {
    for (std::uint32_t i = 0; i < n_docs; ++i) {
        out_scores[i] = simd::dot(query, corpus + static_cast<std::size_t>(i) * dim, dim);
    }
}

std::vector<std::pair<std::uint32_t, float>>
cosine_topk(const float* query, const float* corpus, std::uint32_t n_docs,
            std::uint32_t dim, std::uint32_t k) {
    std::vector<float> scores(n_docs);
    cosine_scores(query, corpus, n_docs, dim, scores.data());
    return top_k(scores, k);
}

}  // namespace simeon
