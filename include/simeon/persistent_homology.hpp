#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace simeon {

// Persistent Homology Scale Selection (PHSS)
//
// 0-dimensional persistent homology (connected components) on a similarity
// graph of fragments within a BM25 pool. The persistence diagram is used to
// select a data-driven similarity threshold for building the fragment kNN
// graph, replacing the hand-tuned `min_fragment_sim` and `knn` parameters.
//
// References:
// - Bauer 2021, "Ripser: efficient computation of Vietoris-Rips persistence
//   barcodes", Journal of Applied and Computational Topology.
// - Reininghaus et al. 2015, "A stable multi-scale kernel for topological
//   machine learning", CVPR.
// - Zhu 2013, "Persistent homology: an introduction and a new text
//   representation for NLP".

// A single persistence pair for 0-dimensional homology:
//   birth  = similarity at which the component first appears
//   death  = similarity at which the component merges with another
//            (inf for components that survive to the end)
//   points = fragment indices in this component
struct PersistencePair0D {
    float birth = 0.0f;
    float death = 0.0f;
    std::vector<std::uint32_t> points;
};

// Configuration for PHSS scale selection.
struct PhssConfig {
    // Maximum dimension of homology to compute (0 = connected components only).
    std::uint32_t dim_max = 0;

    // Similarity threshold: edges with similarity < threshold are ignored.
    // 0.0 means use all edges.
    float threshold = 0.0f;

    // Scale selection criterion.
    enum class Criterion : std::uint8_t {
        // Largest gap in the sorted death similarities.
        // This identifies the scale where the most components merge,
        // suggesting a natural cluster boundary.
        LargestGap,

        // Cheap approximation of LargestGap using the raw pairwise similarity
        // distribution rather than the 0D persistence deaths. This avoids the
        // edge sort + union-find pass and is meant as a latency-oriented proxy.
        LargestGapApprox,

        // The death similarity of the most persistent pair.
        // This selects the scale where the strongest cluster survives.
        MaxPersistence,

        // The death similarity at the "elbow" of the persistence curve
        // (sorted deaths vs. rank). Uses the point of maximum curvature.
        Elbow,
    };
    // Default promoted to LargestGapApprox per research findings (see docs/research.md).
    // Equal-or-better nDCG@10 on richcov across 3/3 corpora at ~2× QPS vs
    // the heavy LargestGap path.
    Criterion criterion = Criterion::LargestGapApprox;

    // When true, also return the full persistence diagram for telemetry.
    bool output_diagram = false;
};

// Scale selection result.
struct PhssResult {
    // The selected similarity threshold for building the kNN graph.
    float selected_scale = 0.0f;

    // The largest consecutive gap in the sorted similarity (or death) sequence
    // used by the LargestGap / LargestGapApprox criterion. Reflects clustering
    // quality: large gap = well-separated top cluster = high geometry confidence.
    // Zero when the criterion is MaxPersistence or when fewer than 2 values exist.
    float max_gap = 0.0f;

    // The full 0D persistence diagram (only populated if cfg.output_diagram).
    std::vector<PersistencePair0D> diagram;

    // Diagnostic: number of edges considered.
    std::uint32_t n_edges = 0;

    // Diagnostic: number of persistence pairs.
    std::uint32_t n_pairs = 0;

    // Sub-phase wall-clock timers (µs). Always populated by
    // phss_select_scale(); zero on the LargestGapApprox fast path for
    // phases that don't run (uf_traversal, survivor_scan).
    double edge_gather_us = 0.0;
    double edge_sort_us = 0.0;
    double uf_traversal_us = 0.0;
    double survivor_scan_us = 0.0;
    double death_sort_us = 0.0;
    double criterion_us = 0.0;
};

// Compute 0-dimensional persistent homology on a complete similarity graph
// of `n` points, and select a similarity threshold for graph construction.
//
// `similarities` is a dense upper-triangular matrix in row-major order:
//   similarities[i*(2*n - i - 1)/2 + (j - i - 1)] = sim(i, j) for i < j
// The matrix is symmetric; only the upper triangle is stored.
//
// `n` is the number of points (fragments), typically ≤ 600.
//
// Returns a PhssResult with the selected scale and optional diagram.
PhssResult phss_select_scale(std::span<const float> similarities, std::uint32_t n,
                             const PhssConfig& cfg);

namespace detail {

// Reference O(m log m) LargestGapApprox: threshold-filter + full sort + largest
// adjacent-gap scan. Retained as the differential-test oracle and as the
// degenerate-case fallback for the O(m) bucket path in phss_select_scale.
PhssResult phss_largest_gap_sorted(std::span<const float> similarities, float threshold);

} // namespace detail

} // namespace simeon
