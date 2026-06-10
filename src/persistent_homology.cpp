#include "simeon/persistent_homology.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>

namespace simeon {

namespace {

// Dense upper-triangular matrix accessor.
// Layout: row i has (n - i - 1) entries for columns j > i.
// Index: offset(i, j) = i * (2*n - i - 1) / 2 + (j - i - 1)
struct UpperTriangular {
    std::span<const float> data;
    std::uint32_t n;

    float at(std::uint32_t i, std::uint32_t j) const noexcept {
        if (i == j)
            return 1.0f;
        if (i > j)
            std::swap(i, j);
        const std::uint32_t offset = i * (2 * n - i - 1) / 2 + (j - i - 1);
        return data[offset];
    }
};

struct UnionFind {
    std::vector<std::uint32_t> parent;
    std::vector<std::uint32_t> rank;
    std::vector<std::uint32_t> size;

    explicit UnionFind(std::uint32_t n) : parent(n), rank(n, 0), size(n, 1) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    std::uint32_t find(std::uint32_t x) {
        std::uint32_t root = x;
        while (parent[root] != root)
            root = parent[root];
        while (x != root) {
            const std::uint32_t next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    }

    bool unite(std::uint32_t a, std::uint32_t b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return false;
        if (rank[a] < rank[b])
            std::swap(a, b);
        parent[b] = a;
        size[a] += size[b];
        if (rank[a] == rank[b])
            ++rank[a];
        return true;
    }
};

using Clock = std::chrono::steady_clock;

inline double elapsed_us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

} // anonymous namespace

namespace detail {

PhssResult phss_largest_gap_sorted(std::span<const float> similarities, float threshold) {
    PhssResult result;

    const auto t_gather0 = Clock::now();
    std::vector<float> sims;
    sims.reserve(similarities.size());
    for (float s : similarities) {
        if (threshold <= 0.0f || s >= threshold)
            sims.push_back(s);
    }
    result.n_edges = static_cast<std::uint32_t>(sims.size());
    const auto t_gather1 = Clock::now();
    result.edge_gather_us = elapsed_us(t_gather0, t_gather1);

    if (sims.empty()) {
        result.selected_scale = 0.0f;
        return result;
    }
    const auto t_sort0 = Clock::now();
    std::sort(sims.begin(), sims.end());
    const auto t_sort1 = Clock::now();
    result.edge_sort_us = elapsed_us(t_sort0, t_sort1);

    result.n_pairs = static_cast<std::uint32_t>(sims.size());
    if (sims.size() < 2) {
        result.selected_scale = sims[0];
        return result;
    }
    const auto t_crit0 = Clock::now();
    float max_gap = -1.0f;
    std::size_t best_idx = 0;
    for (std::size_t i = 0; i + 1 < sims.size(); ++i) {
        const float gap = sims[i + 1] - sims[i];
        if (gap > max_gap) {
            max_gap = gap;
            best_idx = i;
        }
    }
    result.max_gap = max_gap > 0.0f ? max_gap : 0.0f;
    result.selected_scale = (sims[best_idx] + sims[best_idx + 1]) * 0.5f;
    const auto t_crit1 = Clock::now();
    result.criterion_us = elapsed_us(t_crit0, t_crit1);
    return result;
}

} // namespace detail

PhssResult phss_select_scale(std::span<const float> similarities, std::uint32_t n,
                             const PhssConfig& cfg) {
    return phss_select_scale(similarities, n, cfg, nullptr);
}

PhssResult phss_select_scale(std::span<const float> similarities, std::uint32_t n,
                             const PhssConfig& cfg, const PhssStats* precomputed) {
    PhssResult result;
    if (n == 0)
        return result;

    if (cfg.criterion == PhssConfig::Criterion::LargestGapApprox) {
        // O(m) pigeonhole maximum-gap. Pass 1 summarizes the (threshold-filtered)
        // similarities; pass 2 buckets them with per-bucket min/max; pass 3 scans
        // consecutive nonempty buckets for the largest cross-bucket gap. With
        // nb = m-1 buckets at least one is empty, so the true largest adjacent
        // gap is a cross-bucket gap whenever it exceeds the mean spacing
        // (vmax-vmin)/(m-1); the gap value and midpoint are then the same float
        // operations the sorted scan performs, so selected_scale/max_gap are
        // bit-identical. In the degenerate regime (gap not strictly above mean
        // spacing, e.g. ~uniform spacing) we defer to the sorted reference to
        // keep first-maximum tie-breaking exact.
        const auto t_gather0 = Clock::now();
        const bool use_threshold = cfg.threshold > 0.0f;
        std::uint32_t m = 0;
        float vmin = std::numeric_limits<float>::infinity();
        float vmax = -std::numeric_limits<float>::infinity();
        if (precomputed != nullptr && !use_threshold) {
            m = precomputed->count;
            if (m > 0) {
                vmin = precomputed->vmin;
                vmax = precomputed->vmax;
            }
        } else {
            for (float s : similarities) {
                if (use_threshold && s < cfg.threshold)
                    continue;
                ++m;
                vmin = std::min(vmin, s);
                vmax = std::max(vmax, s);
            }
        }
        result.n_edges = m;
        const auto t_gather1 = Clock::now();
        result.edge_gather_us = elapsed_us(t_gather0, t_gather1);

        if (m == 0) {
            result.selected_scale = 0.0f;
            return result;
        }
        result.n_pairs = m;
        if (m == 1) {
            result.selected_scale = vmin;
            return result;
        }
        if (vmin == vmax) {
            result.max_gap = 0.0f;
            result.selected_scale = (vmin + vmin) * 0.5f;
            return result;
        }

        // Pass 2: bucket the values (timed under the phase the sort replaced).
        // Pigeonhole uses m-1 bins, but the correctness argument only needs the
        // largest gap to exceed the bin width span/nb; capping nb keeps the bin
        // arrays cache-resident for large edge counts (m ~ n²/2), and any gap
        // narrower than the wider bin width falls through to the sorted oracle.
        const auto t_sort0 = Clock::now();
        const std::uint32_t nb = std::min<std::uint32_t>(m - 1, 8192u);
        std::vector<float> bmin(nb, std::numeric_limits<float>::infinity());
        std::vector<float> bmax(nb, -std::numeric_limits<float>::infinity());
        const double span = static_cast<double>(vmax) - static_cast<double>(vmin);
        // Index math in float: (s - vmin) is exact-signed >= +0 and monotone,
        // scaling by a positive constant and truncating preserve weak
        // monotonicity, so this is a valid bucketing. Float rounding can shift
        // bucket boundaries by a few ulps of nb (< 1% of a bucket width); the
        // fallback margin below absorbs that.
        const float idx_scale = static_cast<float>(static_cast<double>(nb) / span);
        const auto bucket_update = [&](float s) {
            std::uint32_t idx = static_cast<std::uint32_t>((s - vmin) * idx_scale);
            if (idx >= nb)
                idx = nb - 1;
            bmin[idx] = std::min(bmin[idx], s);
            bmax[idx] = std::max(bmax[idx], s);
        };
        if (use_threshold) {
            for (float s : similarities) {
                if (s >= cfg.threshold)
                    bucket_update(s);
            }
        } else {
            for (float s : similarities)
                bucket_update(s);
        }
        const auto t_sort1 = Clock::now();
        result.edge_sort_us = elapsed_us(t_sort0, t_sort1);

        // Pass 3: largest cross-bucket gap, first maximum (strict >).
        const auto t_crit0 = Clock::now();
        float best_gap = -1.0f;
        float best_lo = vmin;
        float best_hi = vmin;
        float prev_max = -std::numeric_limits<float>::infinity();
        bool have_prev = false;
        for (std::uint32_t k = 0; k < nb; ++k) {
            if (bmin[k] > bmax[k])
                continue; // empty bucket
            if (have_prev) {
                const float gap = bmin[k] - prev_max;
                if (gap > best_gap) {
                    best_gap = gap;
                    best_lo = prev_max;
                    best_hi = bmin[k];
                }
            }
            prev_max = bmax[k];
            have_prev = true;
        }

        // Margin 1/64 over the bucket width covers the float-index boundary
        // shift (a few ulps of nb, < 1% of a width); anything closer to the
        // degenerate regime defers to the bit-exact sorted reference.
        const double fallback_width = (span / static_cast<double>(nb)) * (1.0 + 1.0 / 64.0);
        if (best_gap < 0.0f || static_cast<double>(best_gap) <= fallback_width) {
            // Degenerate/tie regime — defer to the bit-exact sorted reference.
            const auto fb = detail::phss_largest_gap_sorted(similarities, cfg.threshold);
            result.selected_scale = fb.selected_scale;
            result.max_gap = fb.max_gap;
            result.criterion_us = elapsed_us(t_crit0, Clock::now());
            return result;
        }
        result.max_gap = best_gap > 0.0f ? best_gap : 0.0f;
        result.selected_scale = (best_lo + best_hi) * 0.5f;
        const auto t_crit1 = Clock::now();
        result.criterion_us = elapsed_us(t_crit0, t_crit1);
        return result;
    }

    const UpperTriangular sim{similarities, n};

    // Phase 1: edge materialization
    const auto t_gather0 = Clock::now();
    struct Edge {
        std::uint32_t i, j;
        float sim;
    };
    std::vector<Edge> edges;
    edges.reserve(static_cast<std::size_t>(n) * (n - 1) / 2);
    for (std::uint32_t i = 0; i < n; ++i) {
        for (std::uint32_t j = i + 1; j < n; ++j) {
            const float s = sim.at(i, j);
            if (cfg.threshold <= 0.0f || s >= cfg.threshold) {
                edges.push_back({i, j, s});
            }
        }
    }
    const auto t_gather1 = Clock::now();
    result.edge_gather_us = elapsed_us(t_gather0, t_gather1);
    result.n_edges = static_cast<std::uint32_t>(edges.size());

    // Phase 2: sort edges by decreasing similarity
    const auto t_sort0 = Clock::now();
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.sim > b.sim; });
    const auto t_sort1 = Clock::now();
    result.edge_sort_us = elapsed_us(t_sort0, t_sort1);

    // Phase 3: UF traversal — 0D persistent homology
    const auto t_uf0 = Clock::now();
    UnionFind uf(n);
    std::vector<float> births(n, 1.0f);
    std::vector<PersistencePair0D> diagram;

    for (const auto& e : edges) {
        const auto ri = uf.find(e.i);
        const auto rj = uf.find(e.j);
        if (ri == rj)
            continue;

        const bool i_younger = births[ri] > births[rj];
        const auto dying = i_younger ? ri : rj;
        const auto surviving = i_younger ? rj : ri;

        // Always record the pair internally for scale selection.
        PersistencePair0D pair;
        pair.birth = births[dying];
        pair.death = e.sim;
        if (cfg.output_diagram) {
            for (std::uint32_t p = 0; p < n; ++p) {
                if (uf.find(p) == dying)
                    pair.points.push_back(p);
            }
        }
        diagram.push_back(std::move(pair));

        uf.unite(dying, surviving);
    }
    const auto t_uf1 = Clock::now();
    result.uf_traversal_us = elapsed_us(t_uf0, t_uf1);

    // Phase 4: survivor scan
    const auto t_surv0 = Clock::now();
    for (std::uint32_t i = 0; i < n; ++i) {
        if (uf.find(i) == i) {
            PersistencePair0D pair;
            pair.birth = births[i];
            pair.death = std::numeric_limits<float>::infinity();
            if (cfg.output_diagram) {
                for (std::uint32_t p = 0; p < n; ++p) {
                    if (uf.find(p) == i)
                        pair.points.push_back(p);
                }
            }
            diagram.push_back(std::move(pair));
        }
    }
    const auto t_surv1 = Clock::now();
    result.survivor_scan_us = elapsed_us(t_surv0, t_surv1);

    result.n_pairs = static_cast<std::uint32_t>(diagram.size());
    if (cfg.output_diagram) {
        result.diagram = std::move(diagram);
    }

    // Scale selection
    if (result.n_pairs == 0) {
        result.selected_scale = 0.0f;
        return result;
    }

    // Phase 5: death collection + sort
    const auto t_death0 = Clock::now();
    const auto& diagram_ref = cfg.output_diagram ? result.diagram : diagram;
    std::vector<float> deaths;
    deaths.reserve(result.n_pairs);
    for (const auto& pair : diagram_ref) {
        if (pair.death != std::numeric_limits<float>::infinity()) {
            deaths.push_back(pair.death);
        }
    }

    if (deaths.empty()) {
        result.selected_scale = 0.0f;
        const auto t_death1 = Clock::now();
        result.death_sort_us = elapsed_us(t_death0, t_death1);
        return result;
    }

    std::sort(deaths.begin(), deaths.end());
    const auto t_death1 = Clock::now();
    result.death_sort_us = elapsed_us(t_death0, t_death1);

    // Phase 6: criterion selection
    const auto t_crit0 = Clock::now();

    switch (cfg.criterion) {
        case PhssConfig::Criterion::LargestGap: {
            if (deaths.size() < 2) {
                result.selected_scale = deaths.empty() ? 0.0f : deaths[0];
                break;
            }
            float max_gap = -1.0f;
            std::size_t best_idx = 0;
            for (std::size_t i = 0; i + 1 < deaths.size(); ++i) {
                const float gap = deaths[i + 1] - deaths[i];
                if (gap > max_gap) {
                    max_gap = gap;
                    best_idx = i;
                }
            }
            result.max_gap = max_gap > 0.0f ? max_gap : 0.0f;
            result.selected_scale = (deaths[best_idx] + deaths[best_idx + 1]) * 0.5f;
            break;
        }
        case PhssConfig::Criterion::LargestGapApprox:
            break;
        case PhssConfig::Criterion::MaxPersistence: {
            float max_persistence = -1.0f;
            float best_death = 0.0f;
            for (const auto& pair : diagram_ref) {
                if (pair.death == std::numeric_limits<float>::infinity())
                    continue;
                const float persistence = pair.death - pair.birth;
                if (persistence > max_persistence) {
                    max_persistence = persistence;
                    best_death = pair.death;
                }
            }
            result.selected_scale = best_death;
            break;
        }
        case PhssConfig::Criterion::Elbow: {
            // Point of maximum curvature on the persistence curve
            // Curve: (rank, death_similarity) for sorted deaths
            float max_curvature = -1.0f;
            std::size_t best_idx = 0;
            for (std::size_t i = 1; i + 1 < deaths.size(); ++i) {
                const float y_prev = deaths[i - 1];
                const float y_curr = deaths[i];
                const float y_next = deaths[i + 1];
                // Approximate curvature via second difference
                const float curvature = std::abs(y_prev - 2.0f * y_curr + y_next);
                if (curvature > max_curvature) {
                    max_curvature = curvature;
                    best_idx = i;
                }
            }
            result.selected_scale = deaths[best_idx];
            break;
        }
    }
    const auto t_crit1 = Clock::now();
    result.criterion_us = elapsed_us(t_crit0, t_crit1);

    return result;
}

} // namespace simeon
