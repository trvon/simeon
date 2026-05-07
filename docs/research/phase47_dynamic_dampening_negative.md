# Phase XLVII: Dynamic Multi-Signal Dampening (Negative)

## Goal

Phase XLVI's sigmoid dampening uses only `bm25_entropy10` to control the 4-way
fusion weight. This phase tests whether adding **complementarity** (BM25/simeon
Jaccard) and **simeon confidence** (margin) as additional dampening signals
improves ordering beyond the simple entropy gate.

## Method

Three per-query dampening variants, each computing λ ∈ [0,1] as a product of
sigmoid gates:

| Row | Dampening signals | Formula |
|---|---|---|
| `observed_4gen_dampen_sigmoid` | entropy only | σ(ent − 0.48) |
| `observed_4gen_dampen_complement` | entropy × diversity | σ(ent − 0.48) × σ((1−jacc50) − 0.50) |
| `observed_4gen_dampen_dynamic` | entropy × diversity × confidence | σ(ent − 0.48) × σ((1−jacc50) − 0.50) × σ(margin − 0.05) |

where `jacc50` = BM25/simeon top-50 Jaccard and `margin` = simeon top-2 margin.

The fused score is `(1 − λ) × z_BM25 + λ × z_4way`.

## Artifact

```text
/tmp/simeon_dynamic_20260504_163948
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro | Worst Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 | — |
| 4-way raw | 0.3286 | 0.2055 | 0.2623 | 0.6134 | 0.6128 | 0.4045 | −0.0054 |
| **sigmoid** | **0.3297** | **0.2058** | **0.2607** | **0.6204** | **0.6041** | **0.4041** | **+0.0004** |
| complement | 0.3297 | 0.2058 | 0.2611 | 0.6195 | 0.6011 | 0.4034 | +0.0004 |
| dynamic | 0.3297 | 0.2060 | 0.2615 | 0.6208 | 0.5851 | 0.4006 | +0.0004 |

### Deltas vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid | +0.0004 | +0.0005 | +0.0086 | +0.0016 | +0.0392 | +0.0101 |
| complement | +0.0004 | +0.0005 | +0.0090 | +0.0007 | +0.0362 | +0.0094 |
| dynamic | +0.0004 | +0.0007 | +0.0094 | +0.0020 | +0.0202 | +0.0065 |

## Interpretation

**All multi-signal gates are worse than the simple entropy sigmoid.**

The Jaccard gate (`complement`) slightly helps NFCorpus (+0.0090 vs +0.0086)
but hurts TREC-COVID and SciFact. The simeon confidence gate (`dynamic`) is
disastrous: it cuts TREC-COVID's gain by half (+0.0202 vs +0.0392) because
simeon margin is low on TREC-COVID — the embedding is genuinely uncertain
about medical abstracts, yet its signal is still complementary.

The key insight: **BM25 entropy captures the safety dimension that matters.**
When BM25 is uncertain, simeon helps regardless of whether simeon itself is
confident or whether its rankings overlap with BM25. Adding more conditions
makes the gate too conservative without improving discrimination.

This is a clean negative result: **simple entropy dampening is the Pareto-optimal
safety gate** among tested corpus-agnostic formulations.

## Impact on the theorem

The dynamic gate failure confirms that the remaining gap (0.4048 macro) is not
about _which_ generator to trust per-query. It is about **how to score the
union pool**: the current linear z-score combination of 4 generators is the
scoring bottleneck, not the generator-selection logic.

## Next direction

The next attack on OrderingGap should:

1. **Change the scorer `S`**, not the fusion `F`. The current scorer is
   `zscore(BM25) + zscore(BM25F) + zscore(RM3) + zscore(simeon)`. A different
   scoring function over the same 4 signals might capture non-linear
   interactions.

2. **Embedding-weighted RM3**: use simeon embedding similarity to weight the
   pseudo-relevance feedback terms, so expansion is guided by both BM25
   relevance AND topical coherence.

3. **Score-list interaction features**: instead of adding z-scores, use the
   per-document (BM25_rank, simeon_rank, RM3_rank) tuples as evidence for
   a non-linear rank combiner.
