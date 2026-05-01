# Phase XII Results: SIF-Radial Poincaré Manifold

## Experiment

Tests whether **fragment information mass** (the magnitude of the IDF-weighted PMI sum,
discarded by L2-normalize) is a useful re-ranking signal when used as the radial
coordinate of vectors in the Poincaré ball model of hyperbolic space.

Construction:
- Encoder: `encode_sif_poincare()` — IDF-weighted PMI accumulation followed by tanh
  radial squash `v ← v · tanh(||v||/τ) / ||v||` with τ=5.0, producing vectors with
  norm ∈ (0, 1) where the norm encodes total IDF-content mass.
- Builder: `build_doc_semantic_fragments_richcov_sif_poincare()` mirrors the SIF
  richcov path; centroid uses `PoincareBallManifold::normalize` (boundary-safe cap)
  instead of L2-normalize.
- Scoring: outer MaxSim with `PoincareBallManifold::similarity` =
  `−acosh(1 + 2·||a−b||² / ((1−||a||²)(1−||b||²)))`. Whitening is skipped on the
  Poincaré path (mean-subtraction + L2-normalize would erase the radial signal).

Reference: Nickel & Kiela (2017), "Poincaré Embeddings for Learning Hierarchical
Representations", NIPS 2017.

## Cross-Fold Results

OM α=0.80 Euclidean baseline: scifact test 0.6175 / dev 0.6694; nfcorpus test 0.2543 /
dev 0.2621. Cross-fold validation threshold: ±0.005 nDCG@10 per fold.

### scifact

| Config | test nDCG | Δ vs OM_eu | dev nDCG | Δ vs OM_eu | Verdict |
|--------|-----------|------------|----------|------------|---------|
| Poincaré α=0.50 | 0.5828 | −0.0347 | 0.5955 | −0.0739 | below threshold |
| Poincaré α=0.65 | 0.6133 | −0.0042 | 0.6497 | −0.0197 | below threshold |
| Poincaré α=0.80 | 0.6199 | +0.0024 | 0.6599 | −0.0095 | sign flip |
| Poincaré α=0.90 | 0.6247 | **+0.0072** | 0.6649 | −0.0045 | **sign flip** |
| Poincaré α=0.95 | 0.6222 | +0.0047 | 0.6611 | −0.0083 | below threshold |

### nfcorpus

| Config | test nDCG | Δ vs OM_eu | dev nDCG | Δ vs OM_eu | Verdict |
|--------|-----------|------------|----------|------------|---------|
| Poincaré α=0.50 | 0.2324 | −0.0219 | 0.2378 | −0.0243 | below threshold |
| Poincaré α=0.65 | 0.2412 | −0.0131 | 0.2532 | −0.0089 | below threshold |
| Poincaré α=0.80 | 0.2481 | −0.0062 | 0.2620 | −0.0001 | below threshold |
| Poincaré α=0.90 | 0.2523 | −0.0020 | 0.2660 | +0.0039 | sign flip |
| Poincaré α=0.95 | 0.2517 | −0.0026 | 0.2651 | +0.0030 | sign flip |

## Verdict

**DISPROVED**. Every α ≥ 0.80 sign-flips across folds; every α < 0.80 is below threshold
on both folds. No configuration cross-fold validates.

The pattern is consistent with prior Ceiling-B-targeting experiments (SIF, BSIF, RFF):
once the fragment representation departs from the proved Euclidean cosine path, the
small-magnitude geometry signal that survives the α blend becomes noise, and the
fold-difference (Ceiling C) dominates.

## Mechanism

SIF magnitude carries some information — the test-fold scifact result at α=0.90 (+0.007)
is real within that fold — but the magnitude signal is fold-correlated rather than
query-relevant. Documents with high IDF-sum magnitude tend to be longer, more
topically-broad documents; whether such documents are over- or under-represented in
relevant sets varies by fold and corpus.

## Latency

Möbius distance computation is approximately 2× slower than `simd::dot`:
- scifact OM Euclidean: 8481 QPS → Poincaré: 5483 QPS (−35%)
- nfcorpus OM Euclidean: 14545 QPS → Poincaré: 8150 QPS (−44%)

Expected: each Möbius distance computes 3 squared norms + a division + an `acosh`.

## Manifold Abstraction Validation

**Validated mechanically** — the Phase XII branch produces measurable, distinct nDCG values
without disturbing the proved Euclidean path (OM α=0.80 baseline preserved bit-identical
across all configs). The `PoincareBallManifold` struct and the `manifold_kind`
config field are reusable for future hyperbolic experiments (e.g., Nickel-Kiela-style
hyperbolic-trained PMI vectors).

## Implications

1. **Magnitude-as-importance hypothesis**: refuted for IDF-weighted SIF on PMI. Magnitude
   carries some signal but it does not generalize across folds.
2. **Hyperbolic geometry per se**: NOT refuted. The simple radial projection is a poor
   proxy for genuine hyperbolic structure (no hierarchy in the underlying PMI vectors).
   A proper test requires hyperbolic-trained word embeddings — out of training-free scope.
3. **Manifold abstraction**: works, ready for the next experiment that doesn't depend on
   the IDF-magnitude hypothesis.

## Files Touched

- `include/simeon/manifold.hpp` — added `PoincareBallManifold` (~50 LOC)
- `include/simeon/fragment_geometry.hpp` — added `ManifoldKind` enum, `manifold_kind`
  field, `build_doc_semantic_fragments_richcov_sif_poincare` declaration
- `src/fragment_geometry.cpp` — added `encode_sif_poincare`,
  `build_doc_semantic_fragments_richcov_sif_poincare`, Poincaré branch in OM fast path
  (~140 LOC)
- `benchmarks/bench_vs_reference.cpp` — built Poincaré fragment set + 6 bench configs
  (α sweep)

Total: ~210 LOC added; no removed/altered functionality on the Euclidean path.
