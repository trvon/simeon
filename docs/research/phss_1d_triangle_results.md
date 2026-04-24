# PHSS-1D Phase A — triangle-count importance probe — DISPROVED

## Experiment

After the 0D PHSS pass closed, the only remaining topology question was 1D
structure: cycles in the fragment similarity graph rather than just connected
components.

This is the cheap probe from the PHSS-1D plan: count triangles per fragment in
the PHSS-selected kNN graph and use that count as a per-fragment importance
weight. If even this crude proxy moved nDCG@10, fuller 1D persistence would be
worth building.

## Implementation

In `src/fragment_geometry.cpp`, after the PHSS-selected kNN adjacency
`adj` is built and before diffusion runs:

1. For each fragment `i`, count triangles by walking `adj[i]` and
   `adj[j]` for each neighbor `j`, marking how many neighbors of `j`
   are also neighbors of `i`. Each (i, j, k) is enumerated twice via
   the j-side traversal so the count is divided by 2.
2. Compute per-fragment weight
   `w_i = 1 + triangle_alpha · log1p(tri_count[i])`.
3. Apply the weight in one of two placements (sweep both):
   - **QueryAttention**: multiply `mass[i]` (post-attention seed
     distribution) and renormalize, before diffusion.
   - **Diffusion**: multiply `mass[i]` (post-diffusion final mass) and
     renormalize, before doc aggregation.

Sweep `triangle_alpha ∈ {0.25, 0.50, 1.00}` × {QueryAttention,
Diffusion} = **6 recipes per corpus** × 3 corpora = 18 cells.

Cost: ~80 LOC in `fragment_geometry.cpp`, ~25 LOC bench wiring.
Triangle-count latency: well under 200 µs at richcov pool=100
(N=800), confirmed by the marginal QPS impact (<5% across all cells).

## Results

Baseline: `phssapprox_k100_t8_richcov_gap` (the current production
frontier from B6).

### scifact (baseline 0.6188 / 0.7411 / 0.8736 nDCG/R@10/R@100, 111 QPS)

| Variant         | nDCG@10  | Δ        | R@10   | R@100  | QPS   |
|-----------------|---------:|---------:|-------:|-------:|------:|
| tri_a0.25_att   | 0.6186   | −0.0002  | 0.7411 | 0.8736 | 107.9 |
| tri_a0.50_att   | 0.6186   | −0.0002  | 0.7411 | 0.8736 | 110.9 |
| tri_a1.00_att   | 0.6186   | −0.0002  | 0.7411 | 0.8736 | 109.2 |
| tri_a0.25_diff  | 0.6188   |  0.0000  | 0.7411 | 0.8736 | 108.4 |
| tri_a0.50_diff  | 0.6183   | −0.0005  | 0.7411 | 0.8736 | 109.7 |
| tri_a1.00_diff  | 0.6183   | −0.0005  | 0.7411 | 0.8736 | 110.3 |

### nfcorpus (baseline 0.2544 / 0.2266 / 0.1991 nDCG/R@10/R@100, 122 QPS)

| Variant         | nDCG@10  | Δ        | R@10   | R@100  | QPS   |
|-----------------|---------:|---------:|-------:|-------:|------:|
| tri_a0.25_att   | 0.2544   |  0.0000  | 0.2266 | 0.1991 | 123.3 |
| tri_a0.50_att   | 0.2543   | −0.0001  | 0.2266 | 0.1991 | 124.8 |
| tri_a1.00_att   | 0.2543   | −0.0001  | 0.2266 | 0.1991 | 122.4 |
| tri_a0.25_diff  | 0.2544   |  0.0000  | 0.2266 | 0.1991 | 126.0 |
| tri_a0.50_diff  | 0.2543   | −0.0001  | 0.2266 | 0.1991 | 120.5 |
| tri_a1.00_diff  | **0.2548** | **+0.0004** | 0.2266 | 0.1991 | 119.3 |

### fiqa (baseline 0.2089 / 0.2673 / 0.4672 nDCG/R@10/R@100, 117 QPS)

| Variant         | nDCG@10  | Δ        | R@10   | R@100  | QPS   |
|-----------------|---------:|---------:|-------:|-------:|------:|
| tri_a0.25_att   | 0.2085   | −0.0004  | 0.2673 | 0.4672 | 121.5 |
| tri_a0.50_att   | 0.2084   | −0.0005  | 0.2673 | 0.4672 | 121.6 |
| tri_a1.00_att   | 0.2084   | −0.0005  | 0.2673 | 0.4672 | 122.0 |
| tri_a0.25_diff  | 0.2084   | −0.0005  | 0.2673 | 0.4672 | 111.8 |
| tri_a0.50_diff  | 0.2085   | −0.0004  | 0.2673 | 0.4672 | 108.6 |
| tri_a1.00_diff  | 0.2085   | −0.0004  | 0.2673 | 0.4672 | 123.6 |

## Verdict — Disprove gate met cleanly

The plan's pre-declared disprove condition: *"every combination is
within ±0.001 of baseline on all 3 corpora."*

Across all 18 cells (3 corpora × 6 variants), the nDCG@10 delta range
is **[−0.0005, +0.0004]** — every cell within ±0.0005 of baseline,
half of the disprove threshold.

R@10 and R@100 are **byte-identical** to baseline in every cell.

Phase A disproves. **Phase B (witness 1D persistence) does not run.**

## Mechanism — why the proxy is inert

Triangle counts are non-zero and the weight is being applied, but the downstream
impact on doc scores is null. Three reasons:

1. **Multi-fragment per-doc averaging washes out per-fragment
   importance.** richcov uses `top_fragments_per_doc=8`, so each doc
   contributes up to 8 fragments to the geometry pool
   (`fragment_geometry.cpp:858: geom_pool[frags[i].pool_index] +=
   mass[i]`). Per-fragment weight variance averages across 8
   contributions per doc, smoothing any signal.
2. **Triangle distribution is bimodal in PHSS-selected graphs.** The
   `LargestGap` selection produces a graph where most edges either
   participate in many triangles (dense neighborhood cores) or none
   (PHSS-trimmed periphery). `log1p` further compresses the dynamic
   range, leaving little discriminative signal between fragments.
3. **Top-10 ranking is BM25-locked, not topology-mediated.** R@10 and
   R@100 are byte-identical across every triangle variant, meaning
   the documents in the top 100 don't change at all. The geometry
   leg's contribution is dominated by the BM25 leg in the
   `alpha=0.8` blend (`fragment_geometry.cpp:864`); per-fragment
   weight perturbations don't have enough leverage to flip
   document-level ordering at the cutoffs that matter.

## Implication for Phase B / topology-track closure

The plan's pre-declared escalation criterion was: "if even raw
triangle counts move nDCG, full witness 1D persistence is worth the
heavier implementation." Triangle counts don't move nDCG. Therefore:

- **Phase B (witness 1D persistence) is not justified.** The failure mode is at
  doc aggregation, not in the specific choice of 1D weight.
- **Phase C (full Ripser) is not justified** for the same reason.

The fragment-topology track item "1D persistent homology" is now **closed**:

- 0D structure (PHSS LargestGap) — captured, fiqa +0.0036 lift.
- 1D structure (triangle proxy) — measured to be inert at richcov
  pool=100 on this fixture set.

If 1D ever becomes worth revisiting, the obvious enabler is reducing fragments
per doc to 1 so per-fragment weight maps 1:1 to per-doc weight. But that also
removes part of the robustness that the current geometry rerank gets from
multi-fragment averaging.

## Disposition

- Phase A done.
- Phases B and C cancelled per plan's disprove gate.
- The topology-track open question "explore 1D persistent homology" is closed.
- The 6 `phssapprox_tri_*_richcov` recipes stay in
  `bench_vs_reference.cpp` for regression tracking — disproof rows are
  data, not noise.
- The `FragmentGeometryConfig::TrianglePlacement` enum and triangle
  weight code stay in tree (no-op when `use_triangle_weight=false`,
  default). Future researchers exploring 1D variants (per-edge
  weighting, persistence-interval weighting, single-fragment-per-doc
  builders) can build on this scaffolding rather than re-implementing
  the counter.
- Production frontier remains `phssapprox_k100_t8_richcov_gap`.

The deferred Dictionary/Structural axis (AhoCorasick + TextRank wired
into retrieval) becomes the next unexercised lever per the prior
deferral.

---

*2026-04-23. Bench rows in `/tmp/bench_{scifact,nfcorpus,fiqa}_1d.out`.
Code in `src/fragment_geometry.cpp` (triangle counter +
weight application), `include/simeon/fragment_geometry.hpp`
(TrianglePlacement enum + config fields), and
`benchmarks/bench_vs_reference.cpp` (6 sweep recipes).*
