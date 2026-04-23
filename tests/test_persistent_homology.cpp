#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "simeon/persistent_homology.hpp"

using simeon::phss_select_scale;
using simeon::PhssConfig;

namespace {

// Dense upper-triangular packing: index(i,j) = i*(2*n - i - 1)/2 + (j - i - 1) for i < j
std::vector<float> make_tri(std::uint32_t n, const std::vector<float>& sims) {
    std::vector<float> tri;
    tri.reserve(n * (n - 1) / 2);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = i + 1; j < n; ++j) {
            tri.push_back(sims[i * n + j]);
        }
    }
    return tri;
}

// Three points in a line: 0--1 close, 2 far away.
// sim(0,1) = 0.9, sim(0,2) = 0.5, sim(1,2) = 0.5
// 0D persistence: component {0} born at 1.0, dies at 0.9 (merges with 1).
//                 component {1} born at 1.0, dies at 0.9 (merges with 0).
//                 component {0,1} born at 0.9, dies at 0.5 (merges with 2).
// Deaths sorted: 0.5, 0.9
// LargestGap = 0.9 - 0.5 = 0.4, scale = (0.5 + 0.9) / 2 = 0.7
// MaxPersistence = death of most persistent pair = 0.9
// Elbow on [0.5, 0.9] -> single gap, curvature picks the middle -> 0.9
void test_three_point_line() {
    constexpr std::uint32_t n = 3;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.9f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.5f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.5f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.criterion = PhssConfig::Criterion::LargestGap;
    auto r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.0f);
    // Gap-based scale should be the midpoint between 0.5 and 0.9
    assert(std::fabs(r.selected_scale - 0.7f) < 1e-4f);

    cfg.criterion = PhssConfig::Criterion::LargestGapApprox;
    r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.0f);
    assert(std::fabs(r.selected_scale - 0.7f) < 1e-4f);

    cfg.criterion = PhssConfig::Criterion::MaxPersistence;
    r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.0f);
    // Most persistent pair death is 0.9
    assert(std::fabs(r.selected_scale - 0.9f) < 1e-4f);

    cfg.criterion = PhssConfig::Criterion::Elbow;
    r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.0f);
}

// Three points in a triangle: all pairwise sims identical (0.8).
// Only one unique death value.
// LargestGap with <2 unique deaths should fallback to the single value.
// MaxPersistence should pick 0.8.
void test_equilateral_triangle() {
    constexpr std::uint32_t n = 3;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.8f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.8f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.8f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.criterion = PhssConfig::Criterion::LargestGap;
    auto r = phss_select_scale(tri, n, cfg);
    // With only one unique death, fallback to that value
    assert(std::fabs(r.selected_scale - 0.8f) < 1e-4f);

    cfg.criterion = PhssConfig::Criterion::MaxPersistence;
    r = phss_select_scale(tri, n, cfg);
    assert(std::fabs(r.selected_scale - 0.8f) < 1e-4f);
}

// Two points only: single edge, single component dies at sim.
// All criteria should return that sim.
void test_two_points() {
    constexpr std::uint32_t n = 2;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.6f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    for (auto crit : {PhssConfig::Criterion::LargestGap, PhssConfig::Criterion::MaxPersistence,
                      PhssConfig::Criterion::Elbow}) {
        cfg.criterion = crit;
        auto r = phss_select_scale(tri, n, cfg);
        assert(std::fabs(r.selected_scale - 0.6f) < 1e-4f);
    }
}

// Single point: no edges, no pairs.
// Should return 0.0f (no scale can be selected).
void test_single_point() {
    constexpr std::uint32_t n = 1;
    std::vector<float> tri; // empty

    PhssConfig cfg;
    for (auto crit : {PhssConfig::Criterion::LargestGap, PhssConfig::Criterion::MaxPersistence,
                      PhssConfig::Criterion::Elbow}) {
        cfg.criterion = crit;
        auto r = phss_select_scale(tri, n, cfg);
        assert(r.selected_scale == 0.0f);
        assert(r.n_pairs == 1); // one surviving component
    }
}

// Four points: two close clusters.
// sim(0,1)=0.95, sim(2,3)=0.95, cross-cluster=0.3.
// LargestGap should find the gap between 0.3 and 0.95 -> scale ~0.625.
void test_two_clusters() {
    constexpr std::uint32_t n = 4;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.95f;
    sims[2 * n + 3] = sims[3 * n + 2] = 0.95f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.3f;
    sims[0 * n + 3] = sims[3 * n + 0] = 0.3f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.3f;
    sims[1 * n + 3] = sims[3 * n + 1] = 0.3f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.criterion = PhssConfig::Criterion::LargestGap;
    auto r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.3f);
    assert(r.selected_scale < 0.95f);
    // Should be midpoint of the big gap
    assert(std::fabs(r.selected_scale - 0.625f) < 1e-4f);

    cfg.criterion = PhssConfig::Criterion::LargestGapApprox;
    r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.3f);
    assert(r.selected_scale < 0.95f);
    assert(std::fabs(r.selected_scale - 0.625f) < 1e-4f);
}

// output_diagram=true should populate diagram and n_pairs.
void test_output_diagram() {
    constexpr std::uint32_t n = 3;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.9f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.5f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.5f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.output_diagram = true;
    auto r = phss_select_scale(tri, n, cfg);
    assert(!r.diagram.empty());
    assert(r.n_pairs == r.diagram.size());
}

// output_diagram=false should still compute selected_scale.
void test_no_diagram_still_selects_scale() {
    constexpr std::uint32_t n = 3;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.9f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.5f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.5f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.output_diagram = false;
    auto r = phss_select_scale(tri, n, cfg);
    assert(r.selected_scale > 0.0f);
    assert(r.diagram.empty());
}

// Threshold filtering: if threshold=0.95, edges below 0.95 are ignored.
// For the 3-point line, all edges are filtered out.
// Then all 3 points are isolated -> 3 surviving components -> no finite deaths.
void test_threshold_filters_all_edges() {
    constexpr std::uint32_t n = 3;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.9f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.5f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.5f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.threshold = 0.95f;
    auto r = phss_select_scale(tri, n, cfg);
    // No edges pass threshold -> no merges -> all components survive
    assert(r.n_pairs == 3);
    assert(r.selected_scale == 0.0f); // no finite deaths -> no scale
}

// Threshold filtering: if threshold=0.7, only edge (0,1)=0.9 passes.
// 0 and 1 merge at 0.9; 2 is isolated. One finite death at 0.9.
void test_threshold_filters_partial_edges() {
    constexpr std::uint32_t n = 3;
    std::vector<float> sims(n * n, 0.0f);
    sims[0 * n + 1] = sims[1 * n + 0] = 0.9f;
    sims[0 * n + 2] = sims[2 * n + 0] = 0.5f;
    sims[1 * n + 2] = sims[2 * n + 1] = 0.5f;
    const auto tri = make_tri(n, sims);

    PhssConfig cfg;
    cfg.threshold = 0.7f;
    auto r = phss_select_scale(tri, n, cfg);
    // One merge at 0.9, plus 2 surviving components
    assert(r.n_pairs == 3);
    assert(r.selected_scale > 0.0f); // should pick the 0.9 death
}

} // namespace

int main() {
    test_three_point_line();
    test_equilateral_triangle();
    test_two_points();
    test_single_point();
    test_two_clusters();
    test_output_diagram();
    test_no_diagram_still_selects_scale();
    test_threshold_filters_all_edges();
    test_threshold_filters_partial_edges();
    return 0;
}
