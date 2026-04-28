# Phase IV Results: C4 RFF Kernel Augmentation

## Experiment

Arc-cosine degree-1 random feature map applied to whitened PMI fragment and query
vectors before similarity computation:

```
Φ(x) = max(0, Wx) / √rff_dim,  W[r,c] ~ N(0,1),  rff_dim = 256
```

Expands dim=128 → 256. W is thread-locally cached (generated once per unique
(seed, in_dim, rff_dim) triplet). Approximates arc-cosine kernel k(x,y) capturing
higher-order PMI co-occurrence beyond linear cosine.

Tested as C1+C4 combo (outer MaxSim + RFF) — the validated scoring path on trec-covid.
Reference: Rahimi & Recht (2007, NIPS).

Implementation: `src/fragment_geometry.cpp` — `rff_matrix()` cache + `apply_rff()`,
inserted after whitening, before qsim. Config flag: `rff_augment = true`, `rff_dim = 256`.

## Alpha Sweep Results (richcov t8, trec-covid)

| Config | Test nDCG@10 | Dev nDCG@10 | Δtest vs BM25 | Δdev vs BM25 | Direction |
|--------|-------------|-------------|---------------|--------------|-----------|
| BM25 baseline | 0.5649 | 0.4943 | — | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5752 | 0.5025 | +0.0103 | +0.0082 | CONSISTENT+ *** |
| RFF+MaxSim α=0.50 | 0.5694 | 0.4865 | +0.0045 | −0.0078 | SIGN FLIP |
| RFF+MaxSim α=0.65 | 0.5780 | 0.4942 | +0.0131 | −0.0001 | SIGN FLIP |
| **RFF+MaxSim α=0.80** | **0.5723** | **0.4993** | **+0.0074** | **+0.0050** | **CONSISTENT+** |
| RFF+MaxSim α=0.90 | 0.5714 | 0.4905 | +0.0065 | −0.0038 | SIGN FLIP |
| RFF+MaxSim α=0.95 | 0.5642 | 0.4946 | −0.0007 | +0.0003 | SIGN FLIP |

## Cross-Fold Assessment vs Outer MaxSim Baseline

| Config | Δtest vs OM | Δdev vs OM | Direction |
|--------|-------------|------------|-----------|
| RFF+MaxSim α=0.50 | −0.0058 | −0.0160 | CONSISTENT− |
| RFF+MaxSim α=0.65 | +0.0028 | −0.0083 | SIGN FLIP |
| **RFF+MaxSim α=0.80** | **−0.0029** | **−0.0032** | **CONSISTENT−** |
| RFF+MaxSim α=0.90 | −0.0038 | −0.0120 | CONSISTENT− |
| RFF+MaxSim α=0.95 | −0.0110 | −0.0079 | CONSISTENT− |

RFF+outer MaxSim at α=0.80 clears the ±0.005 threshold vs raw BM25, but is
**consistently worse than vanilla outer MaxSim on both folds** (−0.003/−0.003).

## Verdict

**DISPROVED. RFF augmentation hurts outer MaxSim on trec-covid.**

The arc-cosine kernel does not add useful signal on top of PMI vectors:

1. **ReLU discards useful signal.** Negative components of the whitened PMI vectors
   carry dissimilarity information. The arc-cosine ReLU sets these to zero, discarding
   ~50% of the feature signal on average.

2. **PMI space is already saturated (Ceiling B).** SPPMI via randomized SVD is the
   optimal linear transformation of the co-occurrence space. A random nonlinear mapping
   on top doesn't recover missing semantic structure — it adds noise.

3. **Arc-cosine kernel mismatch.** The arc-cosine kernel captures angular structure
   between vectors. PMI vectors cluster by vocabulary co-occurrence, not by semantic
   angle. The kernel approximation doesn't align with the natural geometry of the space.

4. **Alpha optimum shifts.** The RFF transform changes the effective geometry, shifting
   the optimal alpha (α=0.65 peaks on test; α=0.80 on dev — different optimal from
   vanilla). This is a sign of degraded signal quality, not improvement.

## Gate Status: C5 (WMD) and C7 (Per-doc 1D Homology)

Both C5 and C7 were gated on C4 showing a positive result. C4 is disproved.

**C5 (WMD) remains gated.** WMD on PMI vectors inherits the PMI ceiling. With the same
Ceiling B constraint, WMD would produce marginal results similar to or worse than outer
MaxSim — the transport metric is computed over PMI vectors whose quality is the
bottleneck, not the aggregation method.

**C7 (Per-doc 1D homology) remains gated.** Requires non-trivial fragment graph
topology, which depends on meaningful pairwise distances. With flat PMI vectors, the
fragment graph topology is uninformative regardless of the persistence dimension used.

## Next

**C6: IDF-Weighted Query-Coverage Reranking** — the only remaining ungated experiment.
Operates at the document level using BM25 term structure (not PMI vectors), so it is
not subject to Ceiling B.
