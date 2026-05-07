# Phase XL: Observed Shape-Risk Fusion

## Goal

Phase XXXIX found a corpus-agnostic rank-shape gate in per-query diagnostics. This
phase turns that diagnostic into actual observed benchmark rows, so the result is
measured from score-level rankings rather than from a winner dump.

The rows are still research rows because the hard threshold came from dev-qrel
search, but they test whether the mechanism survives real fusion:

```text
prefer BM25 by default; use z-fusion only when BM25 has non-trivial top margin
and BM25/RM3 rank neighborhoods differ enough to expose complementary evidence.
```

## Added rows

`benchmarks/bench_vs_reference.cpp` now emits two shape-risk fusion rows when
`--generator-slices` is enabled:

- `observed_shape_risk_z_gate_hard_devfit`
- `observed_shape_risk_z_blend_continuous`

The hard gate is:

```text
z_equal iff bm25_margin2 >= 0.01939 AND bm25_rm3_jaccard100 <= 0.7544;
otherwise BM25
```

The continuous variant replaces the hard switch with a damped interpolation from
BM25 z-score to z-equal using the same margin/drift axes.

Artifact:

```text
/tmp/simeon_shape_risk_rows_20260504_095120
```

## Test results

| Corpus | BM25 | z-equal | hard shape-risk | continuous shape-risk | RM3 | union oracle@100 |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 0.3293 | 0.3334 | 0.3327 | 0.3333 | 0.3243 | 0.9646 |
| FiQA | 0.2053 | 0.2074 | 0.2089 | 0.2067 | 0.1996 | 0.5523 |
| NFCorpus | 0.2521 | 0.2603 | 0.2597 | 0.2614 | 0.2726 | 0.5286 |
| SciFact | 0.6188 | 0.6094 | 0.6186 | 0.6192 | 0.5998 | 0.8939 |
| TREC-COVID | 0.5649 | 0.5796 | 0.5835 | 0.5825 | 0.6064 | 0.9847 |
| **Macro** | **0.3941** | **0.3980** | **0.4007** | **0.4006** | **0.4005** | **0.7848** |

## Interpretation

The hard and continuous shape-risk rows improve macro nDCG@10 by about +0.0065
without the large SciFact regression seen in static z-equal fusion:

```text
static z-equal:  SciFact -0.0094 vs BM25
shape-risk hard: SciFact -0.0002 vs BM25
shape-risk cont: SciFact +0.0004 vs BM25
```

This is the first score-level row that converts the rank-shape diagnostic into a
nearly non-regressing general fusion mechanism over the five English fixtures.

## What is still unsolved

The row remains far from the diagnostic union oracle:

```text
macro union oracle@100 - macro hard shape-risk ≈ 0.3841 nDCG@10
```

So Phase XL does not close the theorem gap. It reduces `RoutingApproxGap` for the
observed fusion layer while leaving most of `OrderingGap` and `ExposureGap` open.

## Next direction

The next iteration should avoid fixed dev-tuned constants and learn a
label-free/risk-calibrated formula from score-list shape itself:

- normalize margin thresholds by score-list entropy;
- damp z-fusion continuously as generator drift becomes extreme;
- add `S` improvements for within-pool ordering, because fusion alone cannot
  approach the oracle projection.
