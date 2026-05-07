# Phase L: Weighted RM3 in 4-Way Fusion

## Goal

Phase XLIX's embedding-weighted RM3 is a better scorer than standard RM3
(0.4046 vs 0.4005 macro). Phase XLVI's sigmoid dampening is the best fusion gate.
This phase combines them: swap standard RM3 for weighted RM3 inside the 4-way
sigmoid dampening.

## Method

```text
λ = sigmoid((bm25_entropy10 − 0.48) × 8.0)
score = (1−λ) × z_bm25 + λ × (z_bm25 + z_bm25f + z_rm3_weighted + z_simeon)
```

where `z_rm3_weighted` uses simeon-similarity-weighted pseudo-relevance docs.

The row is `observed_4gen_dampen_weighted_rm3` in `bench_vs_reference.cpp`.

Artifact:

```text
/tmp/simeon_phase50_20260505_013605
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| Weighted RM3 alone | 0.3323 | 0.2034 | 0.2782 | 0.6034 | 0.6059 | **0.4046** |
| Sigmoid 4-way | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | **0.4041** |
| 4-way weighted RM3 | 0.3296 | 0.2071 | 0.2644 | 0.6175 | 0.6017 | 0.4041 |

### 4-way weighted RM3 vs sigmoid

| Corpus | Sigmoid | 4-way wRM3 | Δ |
|---|---:|---:|---:|
| ArguAna | 0.3297 | 0.3296 | −0.0001 |
| FiQA | 0.2058 | 0.2071 | **+0.0013** |
| NFCorpus | 0.2607 | 0.2644 | **+0.0037** |
| SciFact | 0.6204 | 0.6175 | −0.0029 |
| TREC-COVID | 0.6041 | 0.6017 | −0.0024 |
| **Macro** | **0.4041** | **0.4041** | **−0.0001** |

## Interpretation

The 4-way weighted RM3 fusion is **tied** with the standard sigmoid dampening at
0.4041 macro. The improved scorer helps NFCorpus (+0.0037) and FiQA (+0.0013)
but hurts SciFact (−0.0029) and TREC-COVID (−0.0024). The net effect is zero.

### Why the weighted RM3 doesn't improve the 4-way fusion

The weighted RM3 alone (0.4046) **already beats** the best 4-way fusion (0.4041).
Adding BM25F and simeon signals to the weighted RM3 actually **dilutes** its
scoring quality: the additional generators introduce noise that cancels out
the weighted RM3's advantage on NFCorpus and TREC-COVID.

This is a clean signal: **a single better scorer beats a fusion of weaker
scorers.** The weighted RM3 at 0.4046 macro is the best individual row in the
entire benchmark.

### The remaining problem

The weighted RM3 still regresses SciFact by −0.0154 vs BM25. The entropy gate
is not a good safety mechanism for weighted RM3 because:

- SciFact entropy (0.21) is too close to FiQA (0.45), where weighted RM3 helps
- ArguAna entropy (0.0006) is very low, but weighted RM3 actually IMPROVES there

Entropy alone cannot separate "safe to use RM3" from "unsafe."

## Full hierarchy of best rows

| Row | Type | Macro | Worst Δ | SciFact Δ |
|---|---|---|---|---|
| Weighted RM3 alone | 1 generator, 1 scorer | **0.4046** | −0.0154 | −0.0154 |
| Sigmoid 4-way | 4 gen, entropy gate | 0.4041 | **+0.0004** | **+0.0016** |
| 4-way weighted RM3 | 4 gen, entropy gate | 0.4041 | −0.0013 | −0.0029 |
| Raw 4-way z-equal | 4 gen, no gate | 0.4045 | −0.0054 | −0.0054 |
| RM3 standard | 1 generator | 0.4005 | −0.0190 | −0.0190 |

## Impact on the theorem

Two clear system configurations emerge:

1. **Maximum raw quality:** Weighted RM3 alone (0.4046 macro). Best if SciFact
   regression is acceptable.

2. **Maximum safety:** Sigmoid 4-way dampening (0.4041 macro). Best if zero
   regressions are required.

The gap to the 4-way oracle remains 0.4043 (oracle 0.8089 − weighted RM3 0.4046).

The next task is to find a qrel-free signal that gates weighted RM3 safely on
SciFact-like queries without losing TREC-COVID/NFCorpus gains. The current
entropy signal cannot do this because SciFact and FiQA have similar entropies.

## Next direction

**SciFact-gating feature search:** use `SIMEON_4GEN_WINNER_DUMP` data to find
a qrel-free feature (or feature pair) that separates queries where weighted RM3
helps vs hurts, with special attention to the low-entropy-high-idf regime of
SciFact fact-claim queries.
