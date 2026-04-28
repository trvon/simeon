# Phase I Results: C2 (Rank Normalization) and C3 (PHSS Gap-Adaptive Blend)

## Context

Phase I of the post-saturation plan (see plan file). Both experiments target
the blend normalization layer (Ceiling A/D), not the fragment representation
(Ceiling B). Baselines are `phssapprox_a0.75_richcov` and
`phssapprox_a0.85_richcov` (LargestGapApprox, richcov, α=0.75/0.85),
bracketing the production α=0.80 config.

---

## C2: Rank Normalization on BM25 Pool Scores

**Config**: `bm25_fragment_geom_phssapprox_ranknorm_k100_t{4,8}_richcov`

Replaces `zscore_inplace(bm_pool)` with `rank_norm_inplace` — sort descending
by BM25 score, assign rank values [0, -1, ..., -(n-1)], then z-score. Intended
to be distribution-free vs the Gaussian assumption of raw z-score on Pareto BM25
distributions (Bruch et al. 2022, arXiv:2210.11934).

### Results

| Corpus   | Fold | Baseline α=0.80 | ranknorm t4 | ranknorm richcov | Δ richcov |
|----------|------|-----------------|-------------|-----------------|-----------|
| scifact  | test | ≈0.618          | 0.4491      | 0.4832          | **−0.135** |
| scifact  | dev  | ≈0.669          | 0.4884      | 0.4963          | **−0.173** |
| nfcorpus | test | ≈0.254          | 0.2278      | 0.2291          | **−0.025** |
| nfcorpus | dev  | ≈0.263          | 0.2255      | 0.2326          | **−0.031** |
| fiqa     | test | ≈0.205          | 0.1587      | 0.1615          | **−0.044** |
| fiqa     | dev  | ≈0.185          | 0.1512      | 0.1512          | **−0.034** |

### Mechanism

Catastrophic regression on all corpora and folds (−0.025 to −0.173 nDCG).
Cause: rank normalization compresses BM25 scores from a Pareto distribution
(top 1-2 docs at +3..+5 z-score units) to a uniform distribution (all 100 pool
docs at ±1.73 z-score units). With α=0.8, the compressed BM25 advantage
amplifies the geometry side's relative contribution. PMI geometry is at Ceiling
B (static unigram, standalone 0.251 nDCG@10 on scifact), so any increase in
geometry's effective weight causes regression.

Bruch et al. 2022's insight applies to hybrid retrieval where the dense side is
a strong learned model. In this system, the geometry side is weak — making BM25
more distribution-free while amplifying noisy geometry is strictly harmful.

**Verdict: DISPROVED. Consistent catastrophic failure across 3 corpora × 2 folds.**

---

## C3: PHSS Gap-Magnitude Adaptive Blend

**Config**: `bm25_fragment_geom_phssapprox_gapadapt_k100_t8_richcov_d{5,10,15}`

Uses `PhssResult::max_gap` (largest consecutive gap in sorted similarity sequence,
LargestGapApprox criterion) as a geometry-confidence signal. When max_gap is large
(well-clustered fragment pool), reduces alpha by `conf × delta`, down to floor 0.65.
`conf = clamp(max_gap / 0.30, 0, 1)`.

Three delta values tested: 0.05, 0.10, 0.15 (max alpha reduction from 0.80).

### Results

| Corpus   | Fold | Baseline α=0.80 | C3 d5  | C3 d10 | C3 d15 | Δ d10 |
|----------|------|-----------------|--------|--------|--------|-------|
| scifact  | test | 0.618           | 0.6192 | 0.6210 | 0.6213 | +0.003 |
| scifact  | dev  | 0.669           | 0.6716 | 0.6716 | 0.6714 | +0.003 |
| nfcorpus | test | 0.254           | 0.2543 | 0.2546 | 0.2548 | +0.000 |
| nfcorpus | dev  | 0.263           | 0.2619 | 0.2628 | 0.2622 | +0.000 |
| fiqa     | test | 0.205           | 0.2084 | 0.2079 | 0.2063 | +0.003 |
| fiqa     | dev  | 0.185           | 0.1826 | 0.1826 | 0.1822 | −0.003 |

### Cross-Fold Assessment

- **scifact**: +0.003 on both test and dev (consistent direction). Below ±0.005
  per-fold threshold but same-sign on both folds. The PHSS gap signal is meaningful
  on scifact's longer abstracts where fragment pools cluster more distinctly.
- **nfcorpus**: No signal (0.000) on both folds. nfcorpus fragments have enough
  morphological diversity that gap magnitude doesn't track geometric confidence.
- **fiqa**: Sign flip. test +0.003 at d5, dev −0.003. Q&A fragment pools are
  diffuse by nature; the gap signal is noisy for this corpus.

### Verdict

**Weak, corpus-specific, sub-threshold.** Consistent sub-threshold positive on
scifact only. Sign flip on fiqa disqualifies cross-fold validation. Mechanism
confirmed: gap-adaptive blend gives geometry more weight on well-clustered pools,
but only scifact's longer documents generate consistently clustered fragment pools.

The +0.003 scifact signal is consistent (not a fold artifact) but insufficient to
cross the ±0.005 per-fold threshold. C3 remains configured but is not promoted
to production.

**Verdict: WEAK / NOT VALIDATED. Keep flag for future corpus-specific tuning.**

---

## Summary

| Experiment | Verdict | Cross-fold |
|-----------|---------|-----------|
| C2 rank normalization | **DISPROVED** | All corpora, both folds negative |
| C3 gap-adaptive blend | **WEAK** | scifact +0.003 consistent; nfcorpus zero; fiqa sign flip |

Phase I pivot: both Ceiling A/D blend modifications produce sub-threshold or
negative results. Consistent with the saturation analysis: the α=0.8 blend is
the dominant signal and modifying the normalization of either leg cannot overcome
the PMI geometry weakness (Ceiling B). Phase II (SPLATE-style outer MaxSim,
arXiv:2404.13950) restructures the scoring architecture, bypassing the blend
attenuation entirely.
