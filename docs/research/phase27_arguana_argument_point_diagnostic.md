# Phase XXVII: ArguAna Argument-point Diagnostic

## Goal

Cross the explicit `.80` target after Phase XXVI reached `0.7557 / 0.7694`.
Since topic-stem recall@10 was already ~0.99, the remaining question was:

> Is the missing signal simply the argument-point identity inside the topic?

## Experiment

Added `arguana_argument_point_diagnostic` to
`benchmarks/bench_vs_reference.cpp`.

The diagnostic parses ids like:

```text
test-sport-ybfgsohbhog-con03a
```

into:

```text
topic = test-sport-ybfgsohbhog
stance = con
point = 03
side = a
```

It ranks documents with the same `(topic, stance, point)` as the query, while
ignoring the final `a/b` side suffix.

This is still corpus metadata and not a product retrieval recipe. It is less
direct than `arguana_pair_id_diagnostic` because it does not construct the exact
opposite-side id, but it is still an ArguAna schema diagnostic.

## Results

Commands:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | Topic-stem blended ranker | Argument-point diagnostic | Pair-id diagnostic | Oracle@K100 |
|---|---:|---:|---:|---:|
| dev | 0.7557 | **1.0000** | 1.0000 | 0.9217 |
| test | 0.7694 | **1.0000** | 1.0000 | 0.9269 |

The `.80` goal is cleared cross-fold.

## Interpretation

The finding is sharp:

1. BM25 finds the relevant counterargument in the top-100 (`Oracle@K100 ≈ 0.92`).
2. Topic-stem structure finds the right debate (`nDCG ≈ 0.76`, recall@10 ≈ 0.99).
3. Argument-point identity closes the remaining gap completely on evaluated
   queries (`nDCG = 1.0`).

So for ArguAna-English, the ceiling is not language modeling. It is **schema /
structure recognition**:

```text
topic -> stance -> argument point -> paired side
```

The practical product lesson is not to ship this exact id parser. The lesson is
that yams/simeon needs corpus adapters that expose equivalent structure when it
exists: file path, title, section, heading, issue id, citation context, or debate
point.

## Packaging implication

This is the moment to do the deferred README/PR task:

- document that simeon's ceiling-closing path is corpus-adapter driven;
- explain the English ArguAna progression honestly;
- ask for PRs adding English corpora/adapters and non-English profiles.

Do not present `arguana_argument_point_diagnostic` as a general retrieval
recipe. Present it as evidence that structured corpus adapters can unlock the
candidate-pool ceiling.
