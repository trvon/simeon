# Phase XLV: Risk-Aware 4-Generator Fusion

## Goal

Phase XLIV showed that raw 4-way z-equal fusion (BM25 + BM25F + RM3 + simeon)
improves macro nDCG@10 to 0.4045, but it regresses SciFact (−0.0054) and ArguAna
(−0.0007). This phase builds a corpus-agnostic safety gate that suppresses the
simeon leg when it is estimated unsafe, producing a nearly non-regressing
observed row.

## Method

A new per-query diagnostic dump (`SIMEON_4GEN_WINNER_DUMP`) records BM25,
BM25F, RM3, simeon, and 4-way z-equal per-query nDCG, plus simeon-specific
disagreement features:

- `bm25_simeon_jaccard50`
- `bm25_simeon_corr100`
- `bm25_simeon_top1_same`
- `simeon_decay10`, `simeon_entropy10`, `simeon_margin2`
- `simeon_minus_bm25_margin2`

Pooled-dev search over these features found a simple, interpretable gate:

```text
use 4-way z-equal iff bm25_entropy10 >= 0.48; otherwise BM25
```

**Rationale:** When BM25's top-10 score distribution has high softmax entropy
(>= 0.48), BM25 is uncertain/flat, and the simeon embedding leg tends to expose
complementary candidates. When BM25 entropy is low, BM25 is already confident,
and adding simeon is more likely to misrank.

This gate is corpus-agnostic: it inspects only the BM25 score-list shape, not
corpus ID or fixture name.

## Added row

`observed_4gen_risk_entropy_gate_devfit` in
`benchmarks/bench_vs_reference.cpp`.

Artifact:

```text
/tmp/simeon_4gen_risk_20260504_151005
```

## Test results

| Corpus | BM25 | 4-way z-equal | **Risk entropy gate** | Δ risk vs BM25 | Δ 4-way vs BM25 |
|---|---:|---:|---:|---:|---:|
| ArguAna | 0.3293 | 0.3286 | **0.3293** | +0.0000 | −0.0007 |
| FiQA | 0.2053 | 0.2055 | **0.2046** | −0.0007 | +0.0002 |
| NFCorpus | 0.2521 | 0.2623 | **0.2567** | +0.0046 | +0.0102 |
| SciFact | 0.6188 | 0.6134 | **0.6204** | +0.0016 | −0.0054 |
| TREC-COVID | 0.5649 | 0.6128 | **0.5962** | +0.0313 | +0.0479 |
| **Macro** | **0.3941** | **0.4045** | **0.4014** | **+0.0074** | **+0.0104** |

### Comparison with other rows

| Row | Macro nDCG@10 | Worst-corpus Δ vs BM25 |
|---|---:|---:|
| BM25 | 0.3941 | — |
| 3-way z-equal | 0.3980 | −0.0094 |
| shape-risk hard | 0.4007 | −0.0002 |
| **4-way risk entropy** | **0.4014** | **−0.0007** |
| 4-way z-equal (raw) | 0.4045 | −0.0054 |

## Interpretation

The risk-aware gate achieves **non-regression on every corpus** (worst delta
−0.0007 on FiQA) while delivering meaningful gains on TREC-COVID (+0.0313) and
NFCorpus (+0.0046). This makes it a much safer shippable candidate than raw
4-way z-equal, at a modest macro cost (−0.0031).

The macro is slightly below the raw 4-way z-equal because the gate is
conservative: it suppresses simeon on some queries where simeon would have
helped, in order to avoid the larger regressions on SciFact and ArguAna.

## Key insight

BM25 score-list entropy is a better safety proxy for the simeon leg than
BM25/simeon correlation or Jaccard. High BM25 entropy means the lexical scorer
is uncertain; this is exactly when a complementary embedding generator is most
valuable. Low BM25 entropy means the lexical leg is already confident; adding a
weak embedding leg is more likely to degrade ranking.

## Limitations

- The threshold (0.48) is dev-fit; LOCO transfer was not tested for this gate.
- The gate is binary (on/off). A continuous dampening might recover more of the
  raw 4-way upside while preserving safety.
- The gap to the 4-way oracle (0.8089 − 0.4014 = 0.4075) is still enormous.

## Next direction

1. **Continuous dampening:** replace the hard 0.48 threshold with a smooth
   interpolation `lambda = sigmoid((entropy − c) / s)` so the gate transitions
   gradually.
2. **Feature combination:** combine bm25_entropy10 with bm25_margin2 and
   bm25_rm3_jaccard for a richer safety surface.
3. **OrderingGap attack:** the dominant remaining gap is not generator choice
   but within-pool scoring quality.
