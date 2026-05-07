# Phase XLVI: Continuous 4-Generator Dampening

## Goal

Phase XLV's entropy hard gate (`bm25_entropy10 >= 0.48 ? 4-way : BM25`) achieved
safety but left 0.0031 macro points on the table vs raw 4-way z-equal. This phase
replaces the hard on/off switch with a smooth sigmoid dampening that transitions
gradually from BM25-only to 4-way z-fusion as BM25 uncertainty increases.

## Method

```text
lambda = sigmoid((bm25_entropy10 − 0.48) × 8.0)
```

This maps entropy to 4-way blend weight λ:

| BM25 entropy | λ | Behavior |
|---|---:|---|
| 0.00 (ArguAna mean) | 0.021 | nearly BM25-only |
| 0.20 (SciFact mean) | 0.096 | mostly BM25 |
| 0.45 (FiQA mean) | 0.440 | roughly half-half |
| 0.48 (center) | 0.500 | exact balance |
| 0.57 (NFCorpus mean) | 0.673 | mostly 4-way |
| 0.68 (TREC-COVID mean) | 0.832 | nearly 4-way |

The fused score per document is:

```text
score = (1 − λ) × zscore_bm25 + λ × zscore_4way
```

where `zscore_4way = zscore_bm25 + zscore_bm25f + zscore_rm3 + zscore_simeon`.

## Added row

`observed_4gen_dampen_sigmoid` in `benchmarks/bench_vs_reference.cpp`.

Artifact:

```text
/tmp/simeon_dampen_20260504_160104
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| 4-way z-equal raw | 0.3286 | 0.2055 | 0.2623 | 0.6134 | 0.6128 | 0.4045 |
| entropy hard gate | 0.3293 | 0.2046 | 0.2567 | 0.6204 | 0.5962 | 0.4014 |
| **sigmoid dampen** | **0.3297** | **0.2058** | **0.2607** | **0.6204** | **0.6041** | **0.4041** |
| shape-risk hard | 0.3327 | 0.2089 | 0.2597 | 0.6186 | 0.5835 | 0.4007 |

### Deltas vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| **sigmoid dampen** | **+0.0004** | **+0.0005** | **+0.0086** | **+0.0016** | **+0.0392** | **+0.0101** |
| 4-way z-equal raw | −0.0007 | +0.0002 | +0.0102 | −0.0054 | +0.0479 | +0.0104 |
| entropy hard gate | +0.0000 | −0.0007 | +0.0046 | +0.0016 | +0.0313 | +0.0074 |

### Headline comparison

| Row | Macro | Worst Δ vs BM25 | SciFact Δ |
|---|---:|---:|---:|
| 4-way z-equal raw | 0.4045 | **−0.0054** | −0.0054 |
| entropy hard gate | 0.4014 | −0.0007 | +0.0016 |
| **sigmoid dampen** | **0.4041** | **+0.0004** | **+0.0016** |
| shape-risk hard | 0.4007 | −0.0002 | −0.0002 |

## Interpretation

The sigmoid dampening achieves the best of both worlds:

1. **Nearly matches raw 4-way macro** (0.4041 vs 0.4045, −0.0004)
2. **No regressions** — all corpus deltas are positive (worst +0.0004 on ArguAna)
3. **Eliminates the raw SciFact regression** — SciFact goes from −0.0054 (raw) to +0.0016
4. **Recovers TREC-COVID** — +0.0392 vs +0.0479 for raw, markedly better than hard gate's +0.0313
5. **Beats hard gate by +0.0027 macro** with equivalent safety

The continuous dampening is strictly better than the hard gate. It allows partial
4-way influence at moderate entropy (FiQA, SciFact) rather than a binary decision,
which naturally hedges risk.

## Why sigmoid works better than hard gate

The hard gate at 0.48 is binary: it either uses full 4-way or plain BM25. This
means:
- Queries near the threshold (entropy ≈ 0.48) flip discontinuously
- Some FiQA and SciFact queries just above 0.48 get full 4-way and may regress
- Some TREC-COVID and NFCorpus queries just below 0.48 get no simeon benefit

The sigmoid avoids both problems: near-threshold queries get a balanced blend,
and the transition is smooth. This preserves most of the raw 4-way upside while
recovering the hard gate's safety.

## Limitations

- The sigmoid center (0.48) and steepness (8.0) are dev-fit heuristics
- The gap to the 4-way oracle (0.8089 − 0.4041 = 0.4048) is still enormous
- The dampening controls _when_ to use 4 generators, not _how_ to score them

## Next direction

The dampening row confirms that generator choice is no longer the bottleneck.
The remaining 0.4048 macro gap to the oracle is dominated by **OrderingGap**:
better within-pool scoring to sort the exposed candidates, not more generators
or better generator routing.
