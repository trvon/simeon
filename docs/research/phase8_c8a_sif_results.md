# Phase VIII Results: C8a SIF-Weighted PMI Fragment Encoding

## Experiment

Addresses Ceiling B (flat PMI bag-of-words space). IDF-weighted token accumulation
replaces the uniform sum in `enc.encode()`:

```
fragment_vec = Σᵢ max(idf(wᵢ), 0.1) · pmi(wᵢ),   L2-normalized
```

High-IDF words (rare content nouns) contribute proportionally more; common function
words (low IDF ≈ 0.1 weight) are down-weighted. Approximates Smooth Inverse Frequency
(SIF, Arora et al. 2017, ICLR) using BM25 IDF as the frequency proxy.

Implementation: `src/fragment_geometry.cpp` — `encode_sif_weighted()` +
`build_doc_semantic_fragments_richcov_sif()`. Builder runs TextRank selection, then
encodes selected sentences + centroid + whole-doc anchor with IDF weighting. Richcov
overlap caps applied post-selection. Config header:
`include/simeon/fragment_geometry.hpp` — `build_doc_semantic_fragments_richcov_sif`.
Falls back to standard richcov on non-PMI encoders.

Tested as C1+C8a (outer MaxSim + SIF fragments) and PHSS+C8a on trec-covid — the
validated corpus from Phase III. Reference: Arora, Liang, Ma (2017).

## Alpha Sweep Results (richcov t8, trec-covid)

### Test Fold (32 queries)

| Config | nDCG@10 | Δ vs BM25 | Δ vs OM α=0.80 |
|--------|---------|-----------|----------------|
| BM25 baseline | 0.5649 | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5752 | +0.0103 | — |
| SIF+MaxSim α=0.50 | 0.5611 | −0.0038 | −0.0141 |
| SIF+MaxSim α=0.65 | 0.5751 | +0.0102 | −0.0001 |
| SIF+MaxSim α=0.80 | 0.5721 | +0.0072 | −0.0031 |
| SIF+MaxSim α=0.90 | 0.5620 | −0.0029 | −0.0132 |
| SIF+MaxSim α=0.95 | 0.5683 | +0.0034 | −0.0069 |
| PHSS+SIF α=0.80 | 0.5805 | +0.0156 | +0.0053 |

### Dev Fold (17 queries)

| Config | nDCG@10 | Δ vs BM25 | Δ vs OM α=0.80 |
|--------|---------|-----------|----------------|
| BM25 baseline | 0.4943 | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5025 | +0.0082 | — |
| SIF+MaxSim α=0.50 | 0.4642 | −0.0301 | −0.0383 |
| SIF+MaxSim α=0.65 | 0.4846 | −0.0097 | −0.0179 |
| SIF+MaxSim α=0.80 | 0.4902 | −0.0041 | −0.0123 |
| SIF+MaxSim α=0.90 | 0.4964 | +0.0021 | −0.0061 |
| SIF+MaxSim α=0.95 | 0.4999 | +0.0056 | −0.0026 |
| PHSS+SIF α=0.80 | 0.4845 | −0.0098 | −0.0180 |

## Cross-Fold Assessment

| Config | Δtest vs OM | Δdev vs OM | Direction | Verdict |
|--------|-------------|------------|-----------|---------|
| SIF+MaxSim α=0.50 | −0.0141 | −0.0383 | consistent negative | DISPROVED |
| SIF+MaxSim α=0.65 | −0.0001 | −0.0179 | consistent negative | DISPROVED |
| SIF+MaxSim α=0.80 | −0.0031 | −0.0123 | consistent negative | DISPROVED |
| SIF+MaxSim α=0.90 | −0.0132 | −0.0061 | consistent negative | DISPROVED |
| SIF+MaxSim α=0.95 | −0.0069 | −0.0026 | consistent negative | DISPROVED |
| **PHSS+SIF α=0.80** | **+0.0053** | **−0.0180** | **FLIP** | **DISPROVED** |

Cross-fold threshold: ±0.005 nDCG@10 per fold.

## Scifact Sanity (test fold)

Scifact results confirm C8a executes correctly and produces consistent small positive
deltas. Not cross-fold validated; scifact is too small for significance.

| Config | scifact test | Δ vs standard |
|--------|-------------|---------------|
| BM25 baseline | 0.6188 | — |
| Outer MaxSim α=0.80 (standard richcov) | 0.6175 | — |
| SIF + MaxSim α=0.80 | 0.6183 | +0.0008 vs OM |
| SIF + PHSS α=0.80 | 0.6196 | +0.0007 vs PHSS |
| SIF + MaxSim α=0.90 | 0.6219 | +0.0044 vs OM α=0.80 |

## Verdict

**DISPROVED.** All SIF variants are negative or sign-flip on cross-fold validation.

PHSS+SIF showed +0.0053 on the test fold (above the ±0.005 threshold), but the dev
fold collapsed to −0.0180 — a catastrophic sign flip of +0.0233. The test result was
a fold artifact, not a genuine improvement.

SIF+MaxSim is consistently negative across all alpha values on both folds. SIF encoding
actively hurts retrieval quality on trec-covid, regardless of the scoring method.

**Ceiling B confirmed for IDF-weighted composition**: The PMI space is already
saturated at the content-word level for COVID abstracts. The uniform sum is a
reasonable approximation because the trec-covid vocabulary is already dominated by
medical nouns (high content-word density). IDF weighting provides no new separability.

## Mechanism

SIF weighting down-weights function words and up-weights content nouns. For trec-covid
COVID abstracts, this means:
- "COVID-19", "mortality", "patients", "treatment" → high IDF → large weight
- "the", "is", "in", "of" → low IDF → near-zero weight (0.1)

**Why SIF fails**: COVID abstracts already have high content-word density. The uniform
PMI sum is already dominated by content nouns because function words have near-zero PMI
signal (they co-occur with everything uniformly). SIF's down-weighting of function words
makes no difference when the PMI space is already content-word saturated.

The test fold's false positive (+0.0053 for PHSS+SIF) reflects query-distribution
differences between folds: test queries may happen to match the SIF-distorted vector
space slightly better, but this is fold-specific and doesn't generalize.

## Next

Both C8a and C8b DISPROVED. Ceiling B confirmed absolute for IDF-based composition.

Redirect options:
- New corpus with lower content-word density (function words carry more signal)
- Encoder approach beyond PMI (dense retrieval, learned sparse)
- DisCoCat grammar-typed composition (C9): fundamentally different representation
  rather than weighted accumulation
