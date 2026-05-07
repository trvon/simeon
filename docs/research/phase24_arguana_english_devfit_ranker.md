# Phase XXIV: English Dev-fit ArguAna Pair Ranker

## Goal

The user asked to stay with English and see how far we can push before doing
README/PR-facing documentation. Phase XXIII made language profiles explicit;
Phase XXIV asks whether an English relation ranker can pass the next gate:

```text
>= 0.70 nDCG@10 on ArguAna dev and test
```

## Experiment

Added `arguana_text_pair_ranker_devfit_p5` to
`benchmarks/bench_vs_reference.cpp`.

It keeps the same English debate-neighborhood adapter:

1. locate source/self document by query-text containment;
2. use first-five-token page header to define the candidate cluster;
3. rank non-self candidates in that cluster.

Unlike the Phase XXII hand-set discriminator, this row uses fixed weights from a
small pairwise-perceptron fit on the ArguAna dev fold. Features are still
interpretable English/text/order features:

- proximity and signed offset;
- exact offset buckets for ±1..±8 inside the debate cluster;
- full-query, title, body, and first-35-token content Jaccard;
- rebuttal cue count;
- `counterpoint` marker;
- title/body shape marker;
- relative concision.

This row is therefore **not training-free**. It is a supervised ceiling-closure
diagnostic showing how much relation ranking can recover once the English
debate-neighborhood structure is known.

## Results

Commands:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | Phase XXII hand-set discriminator | Phase XXIV dev-fit ranker | Oracle@K100 |
|---|---:|---:|---:|
| dev | 0.6267 | **0.7022** | 0.9217 |
| test | 0.6373 | **0.7025** | 0.9269 |

The ranker clears the `>= 0.70` gate cross-fold.

Relative to BM25:

- dev: `+0.3820`
- test: `+0.3732`

Relative to the training-free hand-set discriminator:

- dev: `+0.0755`
- test: `+0.0652`

## Interpretation

The English ArguAna runway now decomposes into:

1. **Candidate recall**: solved by BM25 (`Oracle@K100 ≈ 0.92`).
2. **Topic/debate neighborhood**: solved by page-header structure
   (`nDCG ≈ 0.44`).
3. **Relation ranking**: hand-set English features reach `≈0.63`; dev-fit
   weights reach `≈0.70`.
4. **Remaining `.80` gap**: likely requires better rank-1 relation inference,
   not more K or more topic similarity.

The fact that dev-fit weights transfer to test almost exactly (`0.7022` /
`0.7025`) means the English relation feature shape is stable inside ArguAna.
The next risk is overfitting to this corpus, so before presenting as product
work we still need a second English debate/counterargument corpus.

## Next target

Phase XXV target:

```text
>= 0.75 nDCG@10 on ArguAna dev and test
```

Promising directions:

- add a low-confidence fallback to the Phase XXII hand-set ranker;
- add title-negation templates (`not/no/does not/cannot/<title term>`);
- add source/target shape classifier (`a`-style title+body vs `b`-style
  rebuttal body) without using ids;
- inspect remaining rank>1 misses from `SIMEON_PER_QUERY_DUMP`.
