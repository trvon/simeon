# Phase XXXII — RM3 generator-slice oracle

## Goal

Continue testing the corrected formula where BM25 is only one generator variable:

```text
G ∈ GeneratorSpace
Observed(G,A,S,F,B) ≤ Oracle(G,K,C,S)
```

Phase XXXII adds a second non-BM25 slice:

```text
G = BM25 with RM3 pseudo-relevance feedback
```

and union oracles:

```text
BM25 ∪ BM25F(TextRankTopSentence)
BM25 ∪ RM3
BM25 ∪ BM25F(TextRankTopSentence) ∪ RM3
```

Run command:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --core-only --generator-slices --queries-from test fixtures/nfcorpus-minilm
```

Artifact:

```text
/tmp/simeon_generator_slices_rm3_20260503_205152
```

## Results

| Corpus | Fold | BM25@100 | RM3@100 | Union all@100 | Lift@100 | BM25@500 | Union all@500 | Lift@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | dev | 0.9217 | 0.9478 | 0.9598 | +0.0381 | 0.9699 | 0.9880 | +0.0181 |
| ArguAna | test | 0.9269 | 0.9557 | 0.9646 | +0.0377 | 0.9701 | 0.9845 | +0.0144 |
| FiQA | dev | 0.4804 | 0.4873 | 0.5391 | +0.0587 | 0.6207 | 0.6866 | +0.0659 |
| FiQA | test | 0.5122 | 0.5167 | 0.5523 | +0.0401 | 0.6608 | 0.6972 | +0.0364 |
| NFCorpus | dev | 0.4726 | 0.5595 | 0.5714 | +0.0988 | 0.5689 | 0.7403 | +0.1714 |
| NFCorpus | test | 0.4355 | 0.5114 | 0.5286 | +0.0931 | 0.5446 | 0.7052 | +0.1606 |
| SciFact | dev | 0.8492 | 0.8634 | 0.8838 | +0.0346 | 0.9388 | 0.9592 | +0.0204 |
| SciFact | test | 0.8755 | 0.8876 | 0.8939 | +0.0184 | 0.9224 | 0.9464 | +0.0240 |
| TREC-COVID | dev | 0.9484 | 0.9669 | 0.9781 | +0.0297 | 1.0000 | 1.0000 | +0.0000 |
| TREC-COVID | test | 0.9613 | 0.9795 | 0.9847 | +0.0234 | 0.9928 | 0.9990 | +0.0062 |

## Interpretation

This is strong evidence that `G` is a major limiting variable. RM3 raises the
measured oracle on every corpus/fold, and the union of BM25 + BM25F + RM3 raises
Oracle@100 by +0.018 to +0.099.

NFCorpus is the clearest recall-bound case: the union-all K=500 oracle reaches
0.7052 on test versus BM25 K=500 at 0.5446. This means the previous BM25-pool
oracle understated training-free headroom by ~0.16 nDCG on NFCorpus test.

FiQA also moves meaningfully, but less dramatically. ArguAna moves despite being
structural, which means query expansion changes which argument-neighborhood docs
enter the pool, but discrimination remains the larger problem.

## Formula update

The old slice:

```text
Oracle_BM25@K
```

is now just:

```text
Oracle_G@K where G = BM25
```

The active measured bound is better represented as:

```text
Oracle_{G_union}@K, G_union = BM25 ∪ BM25F ∪ RM3
```

and will keep increasing as new training-free generator variables are added.
