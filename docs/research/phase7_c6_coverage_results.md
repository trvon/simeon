# Phase VII Results: C6 IDF-Weighted Query-Coverage Reranking

## Experiment

IDF-weighted query-coverage reranking adds a third blend term:

```
final_score(d) = α·z(BM25_pool[d]) + (1-α)·z(geometry[d]) + γ·z(coverage[d])
```

where `coverage(d,q)` is the IDF-weighted BM25 score of document `d` against the
**unique** query terms (each concept counted once, weighted by its corpus IDF).
This captures how many distinct query concepts the document addresses — complementary
to MaxSim (best fragment per doc) and BM25 (TF-weighted pool score).

Implemented in `src/fragment_geometry.cpp` via `idx.score_weighted_hashes()`.
Config flag: `idf_coverage = true`, `idf_coverage_gamma = γ`.
Tested as C1+C6 (outer MaxSim + coverage) and PHSS+C6.

References:
- Lin & Bilmes 2010, HLT-NAACL (submodular coverage)
- Carbonell & Goldstein 1998, SIGIR (MMR)

## Gamma Sweep Results (richcov t8, trec-covid)

### Test Fold (32 queries)

| Config | nDCG@10 | Δ vs BM25 | Δ vs OM α=0.80 |
|--------|---------|-----------|----------------|
| BM25 baseline | 0.5649 | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5752 | +0.0103 | — |
| OM+cov α=0.80 γ=0.05 | 0.5764 | +0.0115 | +0.0012 |
| **OM+cov α=0.80 γ=0.10** | **0.5807** | **+0.0158** | **+0.0055** |
| OM+cov α=0.80 γ=0.15 | 0.5790 | +0.0141 | +0.0038 |
| OM+cov α=0.80 γ=0.20 | 0.5758 | +0.0109 | +0.0006 |
| PHSS+cov α=0.80 γ=0.05 | 0.5720 | +0.0071 | −0.0032 |
| PHSS+cov α=0.80 γ=0.10 | 0.5734 | +0.0085 | −0.0018 |
| BM25+cov α=1.00 γ=0.10 | 0.5683 | +0.0034 | −0.0069 |

### Dev Fold (17 queries)

| Config | nDCG@10 | Δ vs BM25 | Δ vs OM α=0.80 |
|--------|---------|-----------|----------------|
| BM25 baseline | 0.4943 | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5025 | +0.0082 | — |
| OM+cov α=0.80 γ=0.05 | 0.4994 | +0.0051 | −0.0031 |
| OM+cov α=0.80 γ=0.10 | 0.4997 | +0.0054 | −0.0028 |
| OM+cov α=0.80 γ=0.15 | 0.4988 | +0.0045 | −0.0037 |
| OM+cov α=0.80 γ=0.20 | 0.5006 | +0.0063 | −0.0019 |
| PHSS+cov α=0.80 γ=0.05 | 0.4995 | +0.0052 | −0.0030 |
| PHSS+cov α=0.80 γ=0.10 | 0.4926 | −0.0017 | −0.0099 |
| BM25+cov α=1.00 γ=0.10 | 0.4902 | −0.0041 | −0.0123 |

## Cross-Fold Assessment

| Config | Δtest vs OM | Δdev vs OM | Direction | Verdict |
|--------|-------------|------------|-----------|---------|
| OM+cov γ=0.05 | +0.0012 | −0.0031 | FLIP | DISPROVED |
| **OM+cov γ=0.10** | **+0.0055** | **−0.0028** | **FLIP** | **DISPROVED** |
| OM+cov γ=0.15 | +0.0038 | −0.0037 | FLIP | DISPROVED |
| OM+cov γ=0.20 | +0.0006 | −0.0019 | FLIP | DISPROVED |

Cross-fold threshold: ±0.005 nDCG@10 per fold.

## Scifact Sanity (test fold)

Scifact results showed sub-threshold behavior (|Δ| < 0.005): coverage adds +0.002–+0.003
on top of outer MaxSim. Not significant on short abstracts.

| Config | scifact test | Δ vs OM |
|--------|-------------|---------|
| BM25 baseline | 0.6188 | — |
| Outer MaxSim α=0.80 | 0.6175 | — |
| OM+cov α=0.80 γ=0.10 | 0.6204 | +0.0029 |

## Verdict

**DISPROVED.** Coverage shows a consistent sign flip: positive on the test fold
(+0.0055 at γ=0.10), negative on the dev fold (−0.0028 at γ=0.10). The effect
does not generalize across folds. PHSS+cov and BM25+cov are negative on both folds.

The test-fold lift at γ=0.10 (+0.0055) crosses the ±0.005 threshold, but the dev
fold being negative rules out generalization. The sign flip is consistent across all
four γ values tested, indicating this is a structural failure, not a tuning artifact.

## Mechanism

Coverage scores a document by the sum of IDF weights of its unique query terms.
This is distribution-free relative to TF weighting in BM25: a document that contains
each of 5 rare query terms once scores the same as one that contains the most common
one 50 times. For multi-concept medical queries on COVID abstracts, this directly
addresses the question: "does this doc cover all the aspects of my query?".

Unlike BM25 (TF-IDF across all terms), coverage uses only the set of **unique query
terms**, making it sensitive to breadth of coverage rather than depth of one term.
This is complementary to MaxSim, which picks the document's strongest fragment.

**Why it flips**: The test fold likely contains more multi-concept queries where
coverage breadth matters. The dev fold may have queries where BM25 TF weighting
already captures the dominant concept, and the coverage term adds noise by giving
equal weight to rare incidental terms. The coverage signal is fold-distribution
dependent, not a stable ranking improvement.

## Next

Phase VI — C8a SIF-weighted PMI fragments (encoding improvement targeting Ceiling B).
