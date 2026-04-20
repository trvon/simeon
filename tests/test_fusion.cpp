#include <array>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include "simeon/fusion.hpp"

using simeon::Ranking;
using simeon::rrf_fuse;
using simeon::top_k;

namespace {

bool approx(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

void test_rrf_two_lists_hand_computed() {
    // Two rankings:
    //   A: doc 0 first, doc 1 second
    //   B: doc 1 first, doc 0 second
    std::vector<std::pair<std::uint32_t, float>> a = {{0, 0.9f}, {1, 0.5f}};
    std::vector<std::pair<std::uint32_t, float>> b = {{1, 0.8f}, {0, 0.4f}};
    std::array<Ranking, 2> ins = {Ranking(a), Ranking(b)};
    auto fused = rrf_fuse(ins, 60.0f);

    assert(fused.size() == 2);
    // Both docs appear at rank 1 once and rank 2 once -> identical RRF score.
    const float expected = 1.0f / (60.0f + 1.0f) + 1.0f / (60.0f + 2.0f);
    assert(approx(fused[0].second, expected));
    assert(approx(fused[1].second, expected));
    // Tie-break by doc_id ascending.
    assert(fused[0].first == 0);
    assert(fused[1].first == 1);
}

void test_rrf_consensus_winner() {
    // Doc 5 ranks first in both lists => clearly highest.
    std::vector<std::pair<std::uint32_t, float>> a = {{5, 0.9f}, {3, 0.5f}, {1, 0.1f}};
    std::vector<std::pair<std::uint32_t, float>> b = {{5, 0.8f}, {1, 0.4f}, {3, 0.2f}};
    std::array<Ranking, 2> ins = {Ranking(a), Ranking(b)};
    auto fused = rrf_fuse(ins, 60.0f);
    assert(fused.size() == 3);
    assert(fused[0].first == 5);
}

void test_rrf_single_list_passthrough() {
    std::vector<std::pair<std::uint32_t, float>> a = {{2, 9.9f}, {0, 0.5f}, {7, 1.2f}};
    std::array<Ranking, 1> ins = {Ranking(a)};
    auto fused = rrf_fuse(ins, 60.0f);
    assert(fused.size() == 3);
    // Single-list fusion preserves the score-induced ranking.
    assert(fused[0].first == 2);
    assert(fused[1].first == 7);
    assert(fused[2].first == 0);
}

void test_rrf_empty_inputs() {
    std::vector<Ranking> ins;
    auto fused = rrf_fuse(ins, 60.0f);
    assert(fused.empty());
}

void test_top_k_basic() {
    std::vector<float> scores = {0.1f, 0.9f, 0.5f, 0.3f, 0.8f};
    auto t = top_k(scores, 3);
    assert(t.size() == 3);
    assert(t[0].first == 1);  // 0.9
    assert(t[1].first == 4);  // 0.8
    assert(t[2].first == 2);  // 0.5
}

void test_top_k_clamps_to_size() {
    std::vector<float> scores = {0.1f, 0.9f};
    auto t = top_k(scores, 99);
    assert(t.size() == 2);
}

void test_top_k_tie_break_by_id() {
    std::vector<float> scores = {0.5f, 0.5f, 0.5f};
    auto t = top_k(scores, 3);
    assert(t.size() == 3);
    assert(t[0].first == 0);
    assert(t[1].first == 1);
    assert(t[2].first == 2);
}

void test_top_k_zero_returns_empty() {
    std::vector<float> scores = {0.1f, 0.2f};
    auto t = top_k(scores, 0);
    assert(t.empty());
}

void test_cosine_topk_self_recovery() {
    // Each row of the corpus is a unit basis vector; querying with row i must
    // recover doc i with score 1 and the other docs with score 0.
    constexpr std::uint32_t dim = 8;
    constexpr std::uint32_t n = 8;
    std::vector<float> corpus(n * dim, 0.0f);
    for (std::uint32_t i = 0; i < n; ++i) {
        corpus[i * dim + i] = 1.0f;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        auto t = simeon::cosine_topk(corpus.data() + i * dim, corpus.data(), n, dim, 3);
        assert(t.size() == 3);
        assert(t.front().first == i);
        assert(approx(t.front().second, 1.0f));
        assert(approx(t[1].second, 0.0f));
        assert(approx(t[2].second, 0.0f));
    }
}

void test_cosine_scores_matches_naive() {
    constexpr std::uint32_t dim = 32;
    constexpr std::uint32_t n = 9;
    std::vector<float> corpus(n * dim);
    std::vector<float> q(dim);
    for (std::uint32_t i = 0; i < q.size(); ++i) {
        q[i] = static_cast<float>(i) * 0.01f - 0.1f;
    }
    for (std::uint32_t i = 0; i < corpus.size(); ++i) {
        corpus[i] = static_cast<float>((i * 17) % 23) * 0.013f - 0.15f;
    }
    std::vector<float> got(n);
    simeon::cosine_scores(q.data(), corpus.data(), n, dim, got.data());
    for (std::uint32_t d = 0; d < n; ++d) {
        double s = 0.0;
        for (std::uint32_t j = 0; j < dim; ++j) {
            s += static_cast<double>(q[j]) * static_cast<double>(corpus[d * dim + j]);
        }
        assert(std::fabs(got[d] - static_cast<float>(s)) < 1e-4f);
    }
}

}  // namespace

int main() {
    test_rrf_two_lists_hand_computed();
    test_rrf_consensus_winner();
    test_rrf_single_list_passthrough();
    test_rrf_empty_inputs();
    test_top_k_basic();
    test_top_k_clamps_to_size();
    test_top_k_tie_break_by_id();
    test_top_k_zero_returns_empty();
    test_cosine_topk_self_recovery();
    test_cosine_scores_matches_naive();
    return 0;
}
