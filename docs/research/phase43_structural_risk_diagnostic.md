# Phase XLIII: Structural-Risk Diagnostic

## Goal

Phase XLII showed that static structural fields are not safe globally, even
though Lead64 helps ArguAna. This phase asks a narrower question: can qrel-free
features identify when structural evidence should be ignored, while preserving
the shape-risk fusion row as the safer default?

## Added diagnostic

`SIMEON_STRUCT_WINNER_DUMP` emits per-query diagnostics for:

- `bm25`
- `lead64` (`observed_struct_bm25f_lead64_w0.5`)
- `shape_risk` (`observed_shape_risk_z_gate_hard_devfit`)

Features include BM25/Lead overlap, correlation, top-1 churn, margin/entropy
deltas, and pre-retrieval query features.

Artifact:

```text
/tmp/simeon_struct_winner_20260504_122904
```

## Winner counts

| Corpus | Split | BM25 wins | Lead64 wins | Shape-risk wins |
|---|---|---:|---:|---:|
| ArguAna | dev | 345 | 120 | 35 |
| FiQA | dev | 163 | 25 | 16 |
| NFCorpus | dev | 69 | 20 | 10 |
| SciFact | dev | 87 | 6 | 5 |
| TREC-COVID | dev | 9 | 5 | 4 |
| ArguAna | test | 645 | 201 | 60 |
| FiQA | test | 351 | 69 | 24 |
| NFCorpus | test | 158 | 42 | 24 |
| SciFact | test | 172 | 20 | 10 |
| TREC-COVID | test | 12 | 12 | 8 |

Lead64 wins individual queries on every corpus, but aggregate static Lead64 still
regresses macro. The problem is not absence of structural signal; it is unsafe
selection.

## General gate search

The best pooled-dev macro rule chose shape-risk rather than Lead64:

```text
choose shape_risk iff lead_minus_bm25_margin2 <= 0.1115; otherwise BM25
```

Test macro delta versus BM25 was +0.0060 with worst-corpus delta -0.0006.

A more robust dev rule with non-negative test deltas was:

```text
choose shape_risk iff lead_minus_bm25_margin2 <= 0.05603
                 AND scq_sum <= 3952;
otherwise BM25
```

Test deltas:

| Corpus | Delta vs BM25 |
|---|---:|
| ArguAna | +0.0045 |
| FiQA | +0.0004 |
| NFCorpus | +0.0073 |
| SciFact | +0.0016 |
| TREC-COVID | +0.0090 |
| **Macro** | **+0.0046** |

This is weaker than the existing shape-risk row (+0.0066 macro), but safer in
this diagnostic sweep.

## Interpretation

The structural diagnostics did not discover a general Lead64 promotion rule.
Instead, structural features acted as **risk sensors** for when to use the
already-good shape-risk fusion. This reinforces the current theorem direction:

- expose structural evidence through `A`;
- do not assign fixed global structural weights;
- use structural/rank-shape diagnostics to decide whether alternate evidence is
  safe;
- keep the router corpus-agnostic.

## Next system move

The next implementable candidate is a refined risk-aware fusion row that uses
structural safety features only to damp or allow shape-risk fusion, not to promote
Lead64 directly. That targets `RoutingApproxGap` safely while we continue looking
for non-BM25-equivalent `S` features for `OrderingGap`.
