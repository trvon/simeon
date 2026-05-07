# Phase XXXVII: Corpus-Agnostic Router Invariance

## Correction

The previous phase found useful corpus diagnoses, but a production router cannot
be corpus-specific. The admissible router must be a single qrel-free function:

```text
F(q, G(q)) -> generator/fusion choice
```

It may inspect the query and the generator outputs, but it must not branch on
fixture name, corpus ID, or hand-authored corpus rules. Otherwise the result is
an adapter experiment, not a general router theorem.

## Evaluation protocol

Using the Phase XXXVI winner dumps:

- dev artifact: `/tmp/simeon_winner_diag_post_dev_20260504_082727`
- test artifact: `/tmp/simeon_winner_diag_post_20260504_081400`

Two corpus-agnostic probes were run:

1. **Pooled-dev macro fit -> pooled-test macro evaluation**
   - one global rule shared by all corpora
   - objective is macro average across corpora, not query-count-weighted average
2. **Leave-one-corpus-out (LOCO)**
   - fit the same rule family on four dev corpora
   - evaluate on the held-out corpus test split

These probes are still diagnostic because thresholds are selected using qrels on
dev, but they test whether a *general form* transfers.

## Best pooled-dev macro rule

The highest dev-macro rule in the searched family was:

```text
choose RM3 iff idf_stddev <= 2.528 AND scq_sum <= 367.6; otherwise BM25
```

| Corpus | Test score | Delta vs BM25 |
|---|---:|---:|
| ArguAna | 0.3282 | +0.0000 |
| FiQA | 0.2022 | -0.0030 |
| NFCorpus | 0.2682 | +0.0161 |
| SciFact | 0.6097 | -0.0091 |
| TREC-COVID | 0.6026 | +0.0377 |
| **Macro** | **0.4022** | **+0.0083** |

A nearby rule using generator disagreement did slightly better on test macro:

```text
choose RM3 iff scq_sum <= 367.6 AND bm25_rm3_jaccard50 <= 0.7857; otherwise BM25
```

It achieved macro +0.0102 over BM25, with the same pattern: gains on NFCorpus and
TREC-COVID, regressions on FiQA and SciFact.

## Robust dev rule

When the objective maximizes the minimum dev-corpus delta, the best rule was:

```text
choose z_equal iff avg_term_chars <= 5.632; otherwise BM25
```

| Corpus | Test delta vs BM25 |
|---|---:|
| ArguAna | +0.0046 |
| FiQA | +0.0035 |
| NFCorpus | -0.0010 |
| SciFact | -0.0036 |
| TREC-COVID | +0.0114 |
| **Macro** | **+0.0029** |

This is safer but too weak to count as meaningful progress toward the ceiling.

## LOCO transfer

Single-threshold leave-one-corpus-out transfer remained mixed:

| Held-out corpus | Learned rule family result | Test delta vs BM25 |
|---|---:|---:|
| ArguAna | RM3 gate | -0.0043 |
| FiQA | RM3 gate | -0.0061 |
| NFCorpus | RM3 gate | +0.0060 |
| SciFact | z-equal gate | -0.0062 |
| TREC-COVID | z-equal gate | +0.0042 |

## Result

The general-router theorem is **not solved** by one-threshold gates. The current
features contain some invariant signal, but not enough to choose alternate
generators without regressions.

The corrected proof obligation is therefore:

```text
GeneralRoutingGap = Oracle_generator_choice - BestCorpusAgnosticGate
```

where `BestCorpusAgnosticGate` excludes corpus IDs and corpus-specific rules. The
next useful experiment should improve the feature representation, not add more
per-corpus branches.

## Next general mechanisms to test

1. **Risk-aware fallback rather than winner prediction**
   - choose BM25 unless the alternate generator has low estimated downside
   - optimize for non-regression, not maximum win count
2. **Rank-neighborhood stability features**
   - overlap/correlation at multiple K values
   - top-document churn and relevant-neighborhood drift proxies
3. **Score-list shape calibration**
   - entropy, Gini, margin ratios, score mass concentration
   - compare generator confidence rather than raw query text features
4. **Pairwise fusion rather than hard switching**
   - dynamically damp RM3 when drift is high
   - dynamically damp z-fusion when generators disagree too much

This preserves the goal: a single general router that can ship in the library,
with corpus adapters reserved only for explicit corpus structure.
