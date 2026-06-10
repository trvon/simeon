#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
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
    assert(t[0].first == 1); // 0.9
    assert(t[1].first == 4); // 0.8
    assert(t[2].first == 2); // 0.5
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

void test_entropy_alpha_equal_when_top_k_identical() {
    // Identical top-K slices ⇒ identical entropy ⇒ identical confidence ⇒ α = 0.5.
    std::vector<float> tk = {5.0f, 3.0f, 2.5f, 1.0f, 0.5f, 0.1f, 0.0f, -0.1f};
    simeon::EntropyAlphaConfig cfg{};
    const float alpha = simeon::entropy_alpha(tk, tk, /*pool_jaccard=*/0.0f, cfg);
    assert(approx(alpha, 0.5f, 1e-5f));
}

void test_entropy_alpha_collapses_when_jaccard_above_threshold() {
    // Even with wildly different entropies, pool_jaccard above the agreement
    // threshold forces α = 0.5 (legs already agree, no weighting needed).
    std::vector<float> peaky = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> flat = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    simeon::EntropyAlphaConfig cfg{};
    const float alpha = simeon::entropy_alpha(peaky, flat, /*pool_jaccard=*/0.95f, cfg);
    assert(approx(alpha, 0.5f, 1e-6f));
}

void test_entropy_alpha_weights_low_entropy_leg_higher() {
    // Leg A concentrates mass on top-1 (low entropy, high confidence).
    // Leg B is uniform (maximum entropy, zero confidence).
    // α should skew toward A (> 0.5). With max entropy on B, confidence_B = 0
    // and the formula returns α = 1 (all weight on A).
    std::vector<float> peaky = {10.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> flat = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    simeon::EntropyAlphaConfig cfg{};
    const float alpha = simeon::entropy_alpha(peaky, flat, /*pool_jaccard=*/0.0f, cfg);
    assert(alpha > 0.5f);
    assert(alpha <= 1.0f);
}

void test_entropy_alpha_deterministic() {
    std::vector<float> a = {3.0f, 1.0f, 0.5f, 0.2f, -0.1f};
    std::vector<float> b = {2.0f, 1.8f, 1.5f, 1.2f, 1.0f};
    simeon::EntropyAlphaConfig cfg{};
    const float first = simeon::entropy_alpha(a, b, 0.3f, cfg);
    for (int i = 0; i < 10; ++i) {
        const float again = simeon::entropy_alpha(a, b, 0.3f, cfg);
        assert(std::memcmp(&first, &again, sizeof(float)) == 0);
    }
}

void test_linear_alpha_entropy_fuse_recovers_equal_weight_at_tie() {
    // Feed identical top-K slices ⇒ α = 0.5. Fuse two orthogonal per-doc
    // score vectors and verify the output is the equal-weight z-score blend.
    constexpr std::size_t n = 12;
    std::vector<float> a_scores(n), b_scores(n);
    for (std::size_t i = 0; i < n; ++i) {
        a_scores[i] = static_cast<float>(i) * 0.5f;
        b_scores[i] = static_cast<float>(n - i) * 0.25f + 1.0f;
    }
    std::vector<float> tk = {3.0f, 2.0f, 1.0f, 0.5f, 0.2f};
    std::vector<float> out(n, 0.0f);
    simeon::EntropyAlphaConfig cfg{};
    simeon::linear_alpha_entropy_fuse(a_scores, b_scores, tk, tk,
                                      /*pool_jaccard=*/0.0f, cfg, out);
    // Recompute equal-weight z-score blend by hand.
    auto zscore = [](std::vector<float>& v) {
        double mean = 0.0;
        for (float x : v)
            mean += x;
        mean /= v.size();
        double var = 0.0;
        for (float x : v)
            var += (x - mean) * (x - mean);
        const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
        for (float& x : v)
            x = static_cast<float>((x - mean) / sd);
    };
    std::vector<float> za = a_scores;
    std::vector<float> zb = b_scores;
    zscore(za);
    zscore(zb);
    for (std::size_t i = 0; i < n; ++i) {
        const float expected = 0.5f * za[i] + 0.5f * zb[i];
        assert(approx(out[i], expected, 1e-4f));
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

// convex_fuse_z: single leg with weight 1 reproduces the z-scored leg.
void test_convex_fuse_z_single_leg_is_zscore() {
    const std::vector<float> leg{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<std::span<const float>, 1> legs{std::span<const float>(leg)};
    const std::array<float, 1> w{1.0f};
    std::vector<float> out(leg.size(), -99.0f);
    simeon::convex_fuse_z(std::span<const std::span<const float>>(legs), w, out);
    // mean 2.5, population sigma = sqrt(1.25)
    const float sd = std::sqrt(1.25f);
    for (std::size_t i = 0; i < leg.size(); ++i)
        assert(std::fabs(out[i] - (leg[i] - 2.5f) / sd) < 1e-4f);
}

// convex_fuse_z: hand-computed two-leg blend.
void test_convex_fuse_z_two_leg_hand_computed() {
    const std::vector<float> a{0.0f, 1.0f}; // z = {-1, +1}
    const std::vector<float> b{2.0f, 0.0f}; // z = {+1, -1}
    const std::array<std::span<const float>, 2> legs{std::span<const float>(a),
                                                     std::span<const float>(b)};
    const std::array<float, 2> w{0.75f, 0.25f};
    std::vector<float> out(2, 0.0f);
    simeon::convex_fuse_z(std::span<const std::span<const float>>(legs), w, out);
    assert(std::fabs(out[0] - (0.75f * -1.0f + 0.25f * 1.0f)) < 1e-4f);
    assert(std::fabs(out[1] - (0.75f * 1.0f + 0.25f * -1.0f)) < 1e-4f);
}

// convex_fuse_z: mismatched sizes / empty legs are a no-op (defensive contract).
void test_convex_fuse_z_degenerate_inputs() {
    std::vector<float> out(3, 7.0f);
    simeon::convex_fuse_z({}, {}, out);
    assert(out[0] == 7.0f); // legs empty -> untouched

    const std::vector<float> short_leg{1.0f, 2.0f};
    const std::array<std::span<const float>, 1> legs{std::span<const float>(short_leg)};
    const std::array<float, 1> w{1.0f};
    simeon::convex_fuse_z(std::span<const std::span<const float>>(legs), w, out);
    // size mismatch -> returns after zero-fill, no partial garbage
    for (float v : out)
        assert(v == 0.0f);
}

} // namespace

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
    test_entropy_alpha_equal_when_top_k_identical();
    test_entropy_alpha_collapses_when_jaccard_above_threshold();
    test_entropy_alpha_weights_low_entropy_leg_higher();
    test_entropy_alpha_deterministic();
    test_linear_alpha_entropy_fuse_recovers_equal_weight_at_tie();
    test_convex_fuse_z_single_leg_is_zscore();
    test_convex_fuse_z_two_leg_hand_computed();
    test_convex_fuse_z_degenerate_inputs();
    return 0;
}
