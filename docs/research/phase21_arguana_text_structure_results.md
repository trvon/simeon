# Phase XXI: ArguAna Text-Neighborhood Structure Probe

## Goal

Phase XX cleared the `.80` target with `arguana_pair_id_diagnostic`, but that
row uses fixture id structure. Phase XXI asks the stricter question:

> Can we move toward the ArguAna ceiling without using qrels or the exact
> `a ↔ b` id-pair key?

## Experiment

Added `arguana_text_neighborhood_p5` to `benchmarks/bench_vs_reference.cpp`.

The scorer uses only observable text plus corpus order:

1. Normalize query text and locate the source/self document whose normalized
   text contains the query.
2. Take the self document's first five word tokens as a debate-page/topic
   header.
3. Rank other documents with the same header by proximity to the self document
   in corpus order.

This is still structure-aware and ArguAna-specific, but it does not read qrels
and does not use the exact opposite-suffix pair id.

## Results

Commands:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | BM25 | Best prior similarity row | Text-neighborhood | Pair-id diagnostic | Oracle@K100 |
|---|---:|---:|---:|---:|---:|
| dev | 0.3202 | 0.3196 | **0.4480** | 1.0000 | 0.9217 |
| test | 0.3293 | 0.3214 | **0.4345** | 1.0000 | 0.9269 |

The non-id structure probe improves over BM25 by:

- dev: `+0.1278` nDCG@10
- test: `+0.1052` nDCG@10

It does not reach `.80`, but it is the first legitimate move that spends a
large, cross-fold chunk of the ArguAna runway without qrels or exact id-pair
resolution.

## Interpretation

This confirms the Phase XX mechanism:

- BM25/topical similarity already lands in the correct debate neighborhood.
- Ranking within that neighborhood is the real problem.
- Local page structure alone places the relevant counterargument in top-10
  frequently enough to lift recall@10 to ~0.88, but nDCG remains ~0.44 because
  the correct counterargument is often not rank 1.

The remaining gap is therefore a **within-topic pairing/reranking** problem, not
a candidate generation problem.

## Next move

The next scorer should keep the text-neighborhood candidate set and replace the
distance-only order with a deterministic pair discriminator. Candidate features:

1. **Opening-title asymmetry**: source `a` documents usually contain a short
   title followed by body text; paired `b` counterarguments often skip that
   title form or begin with rebuttal language.
2. **Contradiction/rebuttal markers**: `counterpoint`, `however`, `but`, `no
   guarantee`, `not`, `does not`, `instead`, `rather`.
3. **Topic-retention with novelty**: high overlap with the self document's
   debate header, moderate overlap with the query body, and high introduction
   of opposing cue terms.

The promote gate for Phase XXII should be:

```text
arguana_text_pair_discriminator >= 0.60 nDCG@10 on dev and test
```

Reaching `.80` without ids probably requires this discriminator to recover the
rank-1 pair inside the current top-10 neighborhood candidates.
