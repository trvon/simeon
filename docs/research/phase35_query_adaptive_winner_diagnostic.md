# Phase XXXV — query-adaptive winner diagnostic

## Goal

Static fusion did not close the union-oracle gap. This phase adds a per-query
winner diagnostic for the generator/fusion candidates:

- BM25
- BM25F(TextRankTopSentence)
- RM3(k=10,n=30,α=0.5)
- equal-weight z-score fusion

The diagnostic writes one JSONL row per query when
`SIMEON_GENERATOR_WINNER_DUMP` is set. Each row includes per-query nDCG for the
four candidates plus unsupervised query features from `QueryRouter`.

Artifact:

```text
/tmp/simeon_winner_diag_20260504_025103
```

## Test-fold winner counts

| Corpus | Queries | BM25 wins | BM25F wins | RM3 wins | z-equal wins |
|---|---:|---:|---:|---:|---:|
| ArguAna | 906 | 642 | 132 | 125 | 7 |
| FiQA | 444 | 347 | 45 | 41 | 11 |
| NFCorpus | 224 | 143 | 25 | 52 | 4 |
| SciFact | 202 | 166 | 19 | 16 | 1 |
| TREC-COVID | 32 | 7 | 6 | 17 | 2 |

## Feature observations

The simple pre-retrieval feature means are weak separators on most corpora. This
is expected because the winner label is qrel-dependent and the current feature
set mostly describes lexical query shape, not expansion drift.

Notable signals:

- **TREC-COVID:** RM3 wins most queries (17/32). The global RM3 row is also best.
- **NFCorpus:** RM3 wins a meaningful minority (52/224), and those queries show
  somewhat higher clarity/IDF than BM25 winners.
- **ArguAna/FiQA/SciFact:** BM25 wins most queries; RM3/BM25F wins are not cleanly
  separable by current pre-retrieval features.

## Interpretation

The next fusion variable should not be a global static weight. It should either:

1. use post-retrieval features that measure **generator disagreement and drift**;
   or
2. stay corpus-routed at first, because corpus identity is currently a stronger
   signal than query features.

The winner diagnostic confirms the theorem decomposition:

```text
G improved the oracle,
F_static barely improved BestObserved,
F(q) needs better features than the current pre-retrieval set.
```

## Next iteration

Add post-retrieval generator-pair features:

- BM25/RM3 top-K Jaccard
- score correlation over union pool
- RM3 expansion-term entropy / concentration
- top-1 / top-10 score decay for BM25 and RM3

Then test a simple rule:

```text
route to RM3 when corpus in {NFCorpus, TREC-COVID} and RM3 drift is low;
otherwise route to BM25 or z-fusion.
```
