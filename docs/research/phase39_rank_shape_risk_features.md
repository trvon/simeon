# Phase XXXIX: Rank-Shape Risk Features

## Goal

Phase XXXVIII moved the oracle out of the retriever and into the objective. The
remaining bridge from theorem to system is to reduce the routing approximation
gap without corpus-specific rules:

```text
RoutingApproxGap = M(best answer-key generator choice) - M(BestCorpusAgnosticGate)
```

This phase adds corpus-agnostic, qrel-free rank-list shape features. The intent
is not to predict a corpus label, but to estimate risk: when is it safe to leave
BM25 for RM3 or z-fusion?

## Added diagnostics

`SIMEON_GENERATOR_WINNER_DUMP` now emits additional generator-output features:

- BM25/RM3 overlap at multiple depths:
  - `bm25_rm3_jaccard10`
  - `bm25_rm3_jaccard50`
  - `bm25_rm3_jaccard100`
- BM25/BM25F overlap:
  - `bm25_bm25f_jaccard10`
  - `bm25_bm25f_jaccard50`
- top-1 churn:
  - `bm25_rm3_top1_same`
- score-list confidence / flatness:
  - `bm25_entropy10`
  - `rm3_entropy10`
  - `z_equal_entropy10`
  - `rm3_minus_bm25_entropy10`
- score-list margin:
  - `bm25_margin2`
  - `rm3_margin2`
  - `z_equal_margin2`
  - `rm3_minus_bm25_margin2`

Artifact:

```text
/tmp/simeon_winner_diag_shape_20260504_092810
```

## Pooled-dev macro search

The best macro rule fit on pooled dev remained RM3-oriented:

```text
choose RM3 iff bm25_rm3_jaccard100 <= 0.6949
             AND rm3_minus_bm25_decay10 >= -0.1398;
otherwise BM25
```

Test result:

| Corpus | Delta vs BM25 |
|---|---:|
| ArguAna | +0.0009 |
| FiQA | -0.0042 |
| NFCorpus | +0.0159 |
| SciFact | -0.0124 |
| TREC-COVID | +0.0339 |
| **Macro** | **+0.0068** |

This improves macro but is not robust; it still regresses on FiQA and SciFact.

## Risk-aware robust search

Optimizing for minimum dev-corpus delta surfaces a safer z-fusion rule:

```text
choose z_equal iff bm25_margin2 >= 0.01939
               AND bm25_rm3_jaccard100 <= 0.7544;
otherwise BM25
```

Test result:

| Corpus | Delta vs BM25 |
|---|---:|
| ArguAna | +0.0035 |
| FiQA | +0.0037 |
| NFCorpus | +0.0075 |
| SciFact | -0.0002 |
| TREC-COVID | +0.0186 |
| **Macro** | **+0.0066** |

This is the best evidence so far for a general mechanism: the rule is still
qrel-tuned on dev, but its form is corpus-agnostic and its test regressions are
near-zero in this fixture set.

## Leave-one-corpus-out transfer

LOCO transfer remains weak:

| Held-out corpus | Delta vs BM25 | Learned rule family |
|---|---:|---|
| ArguAna | -0.0052 | RM3 overlap gate |
| FiQA | -0.0042 | RM3 overlap/decay gate |
| NFCorpus | +0.0047 | RM3 BM25F-overlap/margin gate |
| SciFact | -0.0142 | RM3 overlap/decay gate |
| TREC-COVID | +0.0088 | z-fusion correlation/entropy gate |

So the theorem is still not solved. Shape features improve pooled robust
behavior, but the learned gate does not yet transfer when an entire corpus is
unseen.

## Interpretation

The added features reduce the empirical routing approximation gap under pooled
macro validation, but not under LOCO. The next system move should not ship these
thresholds. It should convert the insight into an implementation principle:

1. prefer BM25 by default;
2. use alternate fusion only when rank-neighborhood drift is moderate and the
   BM25 score list is sufficiently confident;
3. validate with pooled-dev and LOCO before promoting a gate to the library.

In theorem terms, we have a stronger diagnostic estimator for
`RoutingApproxGap`, but not yet a corpus-invariant proof that the gap is closed.
