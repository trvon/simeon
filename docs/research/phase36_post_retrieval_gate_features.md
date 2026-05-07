# Phase XXXVI: Post-Retrieval Gate Features

## Question

After adding BM25F and RM3 generators, the union oracle improved, but static
fusion did not exploit the larger pool. The next hypothesis is that the missing
variable is query-adaptive fusion:

```text
F(q) = gate(features(q), diagnostics(G1(q), G2(q), ...))
```

Rather than treating BM25 as a formula constant, this phase treats each generator
as a measurable source and asks whether qrel-free, post-retrieval disagreement
features can predict when RM3 should replace BM25.

## Added diagnostic features

`SIMEON_GENERATOR_WINNER_DUMP` now emits the prior pre-retrieval query features
plus generator-disagreement features:

- `bm25_rm3_jaccard50`: Jaccard overlap of BM25 and RM3 top-50 pools
- `bm25_bm25f_jaccard50`: Jaccard overlap of BM25 and BM25F top-50 pools
- `bm25_rm3_corr100`: Pearson correlation over the union of BM25/RM3 top-100
- `bm25_decay10`, `rm3_decay10`: score-drop steepness in each generator top-10
- `rm3_minus_bm25_decay10`: RM3 steepness minus BM25 steepness

Artifact:

```text
/tmp/simeon_winner_diag_post_20260504_081400
```

Command shape:

```bash
SIMEON_GENERATOR_WINNER_DUMP=/tmp/out.winner.jsonl \
  build-release/benchmarks/simeon_bench_vs_reference_research \
  --core-only --generator-slices --queries-from test fixtures/nfcorpus-minilm
```

## Winner distribution on test folds

| Corpus | Queries | BM25 wins | BM25F wins | RM3 wins | z-equal wins |
|---|---:|---:|---:|---:|---:|
| ArguAna | 906 | 642 | 132 | 125 | 7 |
| FiQA | 444 | 347 | 45 | 41 | 11 |
| NFCorpus | 224 | 143 | 25 | 52 | 4 |
| SciFact | 202 | 166 | 19 | 16 | 1 |
| TREC-COVID | 32 | 7 | 6 | 17 | 2 |

The winner distribution itself is informative: RM3 is not globally better, but
it is dominant for TREC-COVID and materially useful for NFCorpus.

## Best one-threshold BM25/RM3 gates (diagnostic upper probes)

These thresholds are optimized on the same test dump, so they are **not** final
claims. They measure whether the emitted features contain enough signal to make a
simple gate useful.

| Corpus | BM25 | RM3 | Per-query best(BM25,RM3) | Best 1-threshold gate | Delta vs BM25 | Gate condition |
|---|---:|---:|---:|---:|---:|---|
| ArguAna | 0.3282 | 0.3233 | 0.3540 | 0.3319 | +0.0038 | `rm3_decay10 >= 0.6669` |
| FiQA | 0.2053 | 0.1996 | 0.2199 | 0.2085 | +0.0032 | `rm3_decay10 >= 0.3032` |
| NFCorpus | 0.2521 | 0.2726 | 0.2765 | 0.2734 | +0.0213 | `bm25_rm3_jaccard50 <= 0.9231` |
| SciFact | 0.6188 | 0.5998 | 0.6376 | 0.6229 | +0.0040 | `bm25_decay10 >= 0.5202` |
| TREC-COVID | 0.5649 | 0.6064 | 0.6225 | 0.6180 | +0.0531 | `bm25_rm3_corr100 >= 0.4858` |

## Interpretation

1. The feature family is directionally useful, especially where RM3 has a real
   corpus-level role (NFCorpus and TREC-COVID).
2. The single-threshold gates remain far below the per-query generator oracle,
   so the gap is not solved.
3. The theorem's next term should explicitly include a **routing discriminability
   gap**:

```text
Oracle_union(G,K) - Observed_gate(G,S,F)
  = candidate recall gap
  + generator-choice discriminability gap
  + within-pool scoring gap
```

This is progress toward the ceiling because it converts a vague "fusion is hard"
claim into a measurable subproblem: can qrel-free features identify when an
alternate generator changes the relevant candidate neighborhood?

## Dev-fit threshold transfer

To avoid presenting test-optimized thresholds as product results, the same
one-threshold BM25/RM3 gate search was fit on the dev split and then evaluated
unchanged on test:

| Corpus | Dev-fit gate | Dev BM25 | Dev gate | Test BM25 | Test RM3 | Test dev-fit gate | Delta vs BM25 |
|---|---|---:|---:|---:|---:|---:|---:|
| ArguAna | `bm25_bm25f_jaccard50 <= 0.5385` | 0.3190 | 0.3241 | 0.3282 | 0.3233 | 0.3290 | +0.0008 |
| FiQA | `idf_stddev <= 1.549` | 0.1908 | 0.1951 | 0.2053 | 0.1996 | 0.2051 | -0.0002 |
| NFCorpus | `bm25_rm3_jaccard50 <= 0.7857` | 0.2620 | 0.2819 | 0.2521 | 0.2726 | 0.2719 | +0.0197 |
| SciFact | `scq_sum >= 210.2` | 0.6623 | 0.6780 | 0.6188 | 0.5998 | 0.6116 | -0.0072 |
| TREC-COVID | `idf_stddev >= 1.31` | 0.4943 | 0.5708 | 0.5649 | 0.6064 | 0.5928 | +0.0278 |

Transfer is mixed. The robust result is not a universal one-line gate; it is the
corpus diagnosis: NFCorpus and TREC-COVID genuinely need RM3/generator routing,
whereas FiQA, ArguAna, and SciFact need a stronger scorer or adapter before
RM3-style generator switching can matter.
