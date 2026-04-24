# PHSS sub-phase profile — `phss_select_scale` hot-spot breakdown

## Experiment

Phase A of the PHSS engineering pass. Six sub-phase timers were added inside
`phss_select_scale()` to decompose the aggregate `phss_select_us` into:

1. **edge_gather**: materialize `std::vector<Edge>` from the upper-triangular
   similarity matrix, thresholded.
2. **edge_sort**: `std::sort` edges by decreasing similarity.
3. **uf_traversal**: Union-Find pass over sorted edges, materializing
   persistence pairs on every merge.
4. **survivor_scan**: O(n) pass for components still alive at the end.
5. **death_sort**: filter finite-lifetime deaths, sort ascending.
6. **criterion**: LargestGap / MaxPersistence / Elbow selection over sorted deaths.

Timers feed through `PhssResult` → `FragmentGeometryProfile` →
`simeon_profile_fragment_geometry` print loop.

Config: `builder=basic`, `mode=phss` (LargestGap criterion),
`pool_size=100`, `top_fragments_per_doc=4`, `iters=3` (606–690 query
evaluations per corpus). Run against the three MiniLM test fixtures.

## Results

Per-query means (µs). `phss_select_mean_us` is the outer wall-clock
(caller-side); sub-phases are measured inside `phss_select_scale`.

| Corpus   | N_frag | phss_select | edge_gather | **edge_sort** | uf | survivor | death_sort | criterion | sum    | Δ vs select |
|----------|-------:|------------:|------------:|--------------:|---:|---------:|-----------:|----------:|-------:|------------:|
| scifact  | 400    | 4578.96     | 61.71       | **4198.06**   | 315.47 | 0.72 | 1.49 | 0.50 | 4577.95 | -0.02% |
| nfcorpus | 331    | 3809.84     | 50.36       | **3487.68**   | 267.44 | 0.62 | 1.22 | 0.37 | 3807.69 | -0.06% |
| fiqa     | 388    | 4119.20     | 57.08       | **3769.45**   | 288.29 | 0.71 | 1.72 | 0.42 | 4117.68 | -0.04% |

Sanity check: all three corpora have sub-phase sums within 0.06% of the
caller-side aggregate, so the timer plumbing is correct.

## Phase dominance

Percentage of `phss_select_mean_us`:

| Corpus   | edge_sort | uf    | edge_gather | others |
|----------|----------:|------:|------------:|-------:|
| scifact  |   **91.7%** |  6.9% |        1.3% |   0.1% |
| nfcorpus |   **91.5%** |  7.0% |        1.3% |   0.1% |
| fiqa     |   **91.5%** |  7.0% |        1.4% |   0.1% |

Every corpus agrees: **edge_sort is the single target**. UF traversal is small
secondary work; the rest is negligible.

## Implications for Phase B

The plan's Phase B optimizations are not equally valuable:

- **B2 (radix sort)** is the real engineering move.
- **B1 (edge packing)** is secondary once B2 exists.
- **B3–B5** only touch the ~7% UF phase.
- **B6 (LargestGapApprox)** is still worth running because it can bypass the
  expensive path entirely.

## Cost model for larger pools (informs Phase C)

Edge count grows as N(N−1)/2. At current `std::sort` efficiency
(~10.5 ns/edge at N=400 × ~80k edges = 4198 µs → 52.5 ns per edge for
the comparator-heavy sort, consistent with cache-hostile float compares):

| pool_size × top_frags | N     | edges   | est. edge_sort µs | est. phss_select µs |
|-----------------------|------:|--------:|------------------:|--------------------:|
| 100 × 4               |   400 |  79.8k  |            ~4200  |              ~4580  |
| 100 × 8               |   800 | 319.6k  |           ~22,000 |             ~24,000 |
| 200 × 4               |   800 | 319.6k  |           ~22,000 |             ~24,000 |
| 500 × 4               |  2000 |   2.0M  |          ~170,000 |            ~184,000 |

At pool=500 the current implementation projects to ~184 ms/query, which is why
the "larger pool" move was blocked at the time.

## Disposition

- Phase A1 complete.
- Phase A2 was rolled into Phase C.
- Phase B is unblocked, with B2 as the obvious first move.

---

*2026-04-23. Code in `src/persistent_homology.cpp`,
`include/simeon/persistent_homology.hpp`,
`include/simeon/fragment_geometry.hpp`, and
`benchmarks/profile_fragment_geometry.cpp`.*
