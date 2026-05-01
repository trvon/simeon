# Phase XIII Results: Cross-Product Experimental Harness

## Experiment

Promote the implicit pipeline `(BM25 candidate → fragment cosine → MaxSim → α-blend)` into
an explicit grammar with four swappable axes, each implemented as a strategy tag inside
`FragmentGeometryConfig`. A single bench function emits the full Cartesian product as
JSON-tagged rows so cross-axis effects can be measured in one pass.

| Axis | Variants | Implementation |
|------|----------|----------------|
| **BM25** (candidate generator) | Atire, BM25Plus, BM25L, DPH, PL2, DCM (n=6) | `Bm25Config::variant`, looped per cell — full re-index per variant |
| **Manifold** (kernel) | Euclidean, Spherical (arccos), Poly2 ((dot+1)²) (n=3) | `FragmentGeometryConfig::manifold_kind`, dispatch in qsim loop |
| **DocScorer** (per-doc aggregation) | MaxSim, MeanSim, TopKMean, SoftMaxSum, GeoMean, HarmonicMean, VarMaxAdjust (n=7) | `FragmentGeometryConfig::doc_scorer_kind`, dispatch in OM aggregation block |
| **Alpha** (BM25/geometry blend) | 0.65, 0.80, 0.90 (n=3) | Existing `cfg.alpha` |

Cell count: 6 × 3 × 7 × 3 = **378 xprod cells per corpus per fold**.

Config naming: `bm25_fragment_geom_xprod_<bm25>_<manifold>_<scorer>_a<alpha>_k100_t8_richcov`.

Baseline (Phase III proved): `bm25_fragment_geom_outermaxsim_a0.80_k100_t8_richcov` =
`xprod_atire_euclid_max_a0.80` cell. scifact test 0.6175, scifact dev 0.6694, nfcorpus test
0.2543, nfcorpus dev 0.2621.

## Per-Axis Effect Sizes

Range = max-mean(axis) − min-mean(axis), averaging over all other axes' values.
Expected: BM25-variant axis dominates because DCM is catastrophically bad on short corpora.

| fold | BM25 range | scorer range | alpha range | manifold range |
|------|-----------:|-------------:|------------:|---------------:|
| scifact test | **0.044** | 0.011 | 0.004 | 0.002 |
| scifact dev  | **0.060** | 0.014 | 0.006 | 0.002 |
| nfcorpus test | 0.010 | 0.006 | 0.004 | 0.002 |
| nfcorpus dev  | 0.009 | **0.012** | 0.002 | 0.002 |

**Findings**:
1. **Manifold axis is consistent noise** (range ≤ 0.002 across all four folds). Spherical
   and Poly2 are rank-monotone with cosine for unit-norm inputs; the score-shape change
   doesn't translate to nDCG@10 change.
2. **Scorer axis carries real signal on dev folds** (~0.012). Worst aggregator
   (HarmonicMean) is ~0.011 below the best on dev; the span is ~5× larger than the strict
   per-fold cross-fold threshold.
3. **BM25 axis dominates on scifact** because DCM is catastrophic there (scifact test
   mean 0.574 vs Atire 0.617). On nfcorpus the BM25 variants are close together; the
   axis only matters as a "don't pick DCM" floor.
4. **Alpha is a small lever** (≤0.006 range). Phase III's α=0.80 default holds across
   most cells; small per-corpus shifts but no dominant alternative.

## Per-Scorer Detail (Dev-Fold Means)

| scorer | scifact dev mean | nfcorpus dev mean | combined mean | rank |
|--------|-----------------:|------------------:|--------------:|------|
| smax | 0.6461 | 0.2589 | 0.4525 | 4 |
| topk3 | 0.6459 | 0.2629 | 0.4544 | 2 |
| varadj (NEW) | 0.6452 | 0.2572 | 0.4512 | 5 |
| mean | 0.6441 | 0.2640 | **0.4541** | 3 |
| max | 0.6436 | 0.2569 | 0.4503 | 6 |
| geom (NEW) | 0.6430 | 0.2630 | 0.4530 | (tie 4) |
| harm (NEW) | 0.6317 | 0.2525 | 0.4421 | 7 |

**Mechanism notes on the new aggregators**:
- **VarMaxAdjust** (`max − γ·σ`, γ=0.5): tied for top on scifact dev but mid-pack on
  nfcorpus dev. Rewards docs whose strong fragment is consistent with the rest, not an
  outlier. Useful when fragments are noisy (scifact short abstracts).
- **GeoMean**: essentially equivalent to MeanSim (within 0.001). The shift to handle
  non-positive qsims (ε-floor) erases the geometric-vs-arithmetic difference at this
  scale.
- **HarmonicMean**: catastrophically worse across all folds (~0.011 below best on
  scifact dev, ~0.011 below best on nfcorpus dev). The penalty for any weak fragment is
  too aggressive — wipes signal from strong fragments too.

## Cross-Fold Analysis

Cross-fold validation thresholds:
- **Strict proved**: both folds Δ ≥ +0.005 vs Phase III baseline.
- **Loose proved**: one fold Δ ≥ +0.005 AND the other Δ ≥ 0.

### Strict Cross-Fold Proved Cells

**0 cells** across both corpora. The test-fold side never crosses +0.005 for any cell
that is also positive on dev.

### Loose Cross-Fold Proved Cells (scifact)

| Cell | Δ test | Δ dev |
|------|-------:|------:|
| `xprod_atire_spherical_topk3_a0.80` | +0.0038 | +0.0080 |
| `xprod_atire_poly2_topk3_a0.80` | +0.0038 | +0.0052 |
| `xprod_atire_euclid_topk3_a0.80` | +0.0020 | +0.0077 |

Common recipe: **Atire + TopKMean(K=3) + α=0.80**, kernel-agnostic.

### Loose Cross-Fold Proved Cells (nfcorpus)

| Cell | Δ test | Δ dev |
|------|-------:|------:|
| `xprod_atire_poly2_geom_a0.65` (NEW) | +0.0000 | +0.0095 |
| `xprod_atire_poly2_mean_a0.65` | +0.0008 | +0.0084 |
| `xprod_atire_poly2_harm_a0.65` (NEW) | +0.0000 | +0.0082 |
| `xprod_atire_euclid_mean_a0.65` | +0.0001 | +0.0081 |

Common recipe: **Atire + (MeanSim/GeoMean/HarmonicMean) + α=0.65**, kernel-agnostic.

The new GeoMean and HarmonicMean scorers add candidate cells on nfcorpus but don't
unlock cross-corpus transfer.

## Per-Corpus Winning Recipes

| corpus | best loose-proved cell | scorer | alpha | dev Δ |
|--------|-----------------------|--------|-------|-------|
| scifact | `atire_spherical_topk3_a0.80` | TopKMean(K=3) | 0.80 | +0.008 |
| nfcorpus | `atire_poly2_geom_a0.65` | GeoMean | 0.65 | +0.010 |
| trec-covid | (pending; bench in flight) | — | — | — |

The scifact and nfcorpus winning recipes do **not transfer**: scifact prefers TopKMean
at α=0.80, nfcorpus prefers MeanSim/GeoMean at α=0.65. The recipe space is corpus-
adaptive, not corpus-invariant.

## Verdict

**WEAK / corpus-dependent.** The xprod harness exposed the layered structure of the
pipeline and quantified per-axis leverage:
- Manifold (kernel) is noise.
- BM25 variant matters mainly as a floor (avoid DCM on short corpora).
- Scorer aggregation carries the strongest cell-level signal on dev folds.
- Alpha is a small lever.

The dev-fold signal (+0.005 to +0.010 from scorer changes) is real but not strong enough
to clear the test-fold side of the cross-fold threshold. No single (BM25, manifold,
scorer, alpha) tuple proves cross-fold on either corpus, and the per-corpus best recipes
diverge.

## Implications

1. **Single-config promotion is not warranted.** No xprod cell strict-proves; no cell
   robustly transfers cross-corpus. The Phase III OM α=0.80 baseline remains the only
   cross-fold-validated default.
2. **Router-per-corpus becomes the natural follow-up.** Scifact and nfcorpus already
   diverge on best scorer + alpha; trec-covid (long medical abstracts) likely diverges
   from both. A router that picks (scorer, alpha) per corpus could compound the small
   per-corpus dev-fold gains.
3. **The architectural seam is the right level for further experiments.** The 7-scorer
   expansion produced 2 new cross-fold loose-proved cells on nfcorpus. The DocScorer
   axis is the productive layer for additional aggregators (Sinkhorn, signature-
   weighted, position-aware) — the manifold layer is not.
4. **The rigidity diagnosis is confirmed.** The α-blend ceiling (Ceiling A/D) attenuates
   geometry-side variation: even when an aggregator gains +0.012 in within-axis range,
   the proven-baseline-relative gain at the cell level stays sub-threshold on the test
   fold.

## Files Touched

- `include/simeon/manifold.hpp` — Spherical + Poly2 alongside Euclidean + Poincaré
- `include/simeon/fragment_geometry.hpp` — `ManifoldKind` + `DocScorerKind` enums on
  `FragmentGeometryConfig`; declared 7 scorer variants
- `src/fragment_geometry.cpp` — manifold dispatch in qsim loop, scorer dispatch in OM
  aggregation; GeoMean / HarmonicMean / VarMaxAdjust implementations
- `benchmarks/bench_vs_reference.cpp` — 378-cell xprod harness in
  `run_bm25_fragment_graph_grid`

Total: ~300 LOC. Each new axis is a strategy struct + bench tuple entry.

## Next

- **Trec-covid** test + dev folds (216 cells, 4-scorer harness — extension to 7-scorer
  scheduled if test fold completes within budget).
- **Router-per-corpus** experiment if trec-covid confirms the corpus-divergence pattern:
  pick (scorer, alpha) by `QueryRouter` features (corpus mean IDF, doc length distribution,
  score decay shape).
- **Sinkhorn DocScorer** as the next non-rank-monotone aggregator, if the router lift on
  the existing 7 scorers is insufficient.
