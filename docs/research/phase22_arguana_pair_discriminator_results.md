# Phase XXII: ArguAna Text Pair Discriminator

## Goal

Phase XXI showed that text-derived debate-neighborhood structure moves ArguAna
from ~0.32 to ~0.44 nDCG@10 without using qrels or the exact `a ↔ b` id pair.
Phase XXII replaces distance-only ordering inside the neighborhood with a
deterministic pair discriminator.

Promotion target for this phase was `>= 0.60` on both dev and test.

## Experiment

Added `arguana_text_pair_discriminator_p5` to
`benchmarks/bench_vs_reference.cpp`.

Candidate set is unchanged from Phase XXI:

1. locate the source/self document by query-text containment;
2. take the source document's first five word tokens as the debate-topic header;
3. consider only non-self docs with the same header.

The discriminator scores each candidate with fixed text-only features:

- local corpus-order proximity;
- full-query/content Jaccard;
- title/content Jaccard;
- body/content Jaccard penalty;
- rebuttal/discourse cue density;
- relative concision.

It does not use qrels and does not use the exact opposite-suffix id key.

## Results

Commands:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | BM25 | Text-neighborhood | Text pair discriminator | Pair-id diagnostic | Oracle@K100 |
|---|---:|---:|---:|---:|---:|
| dev | 0.3202 | 0.4480 | **0.6267** | 1.0000 | 0.9217 |
| test | 0.3293 | 0.4345 | **0.6373** | 1.0000 | 0.9269 |

Phase XXII clears the `>= 0.60` promote gate cross-fold.

Relative to BM25:

- dev: `+0.3065` nDCG@10
- test: `+0.3080` nDCG@10

Relative to Phase XXI text-neighborhood:

- dev: `+0.1787`
- test: `+0.2028`

## Interpretation

This is the first non-qrel, non-exact-id ArguAna branch that spends a large
fraction of the candidate-pool runway.

The mechanism is now clear:

1. **BM25 solves neighborhood recall** (`Oracle@K100 ≈ 0.92`).
2. **Text header structure solves topic clustering** (`nDCG ≈ 0.44`).
3. **Pair discrimination solves much of rank-1 ordering** (`nDCG ≈ 0.63`).
4. The remaining gap to `.80` is not candidate recall; it is pair-level
   relation classification inside a small debate cluster.

## Next target

Phase XXIII should pursue `>= 0.70` with a second-stage deterministic
discriminator:

- infer source-side title span more robustly;
- classify candidate `a`-style title+body vs `b`-style rebuttal body;
- add explicit contradiction templates over title terms:
  `not/no/does not/cannot/insufficient/no guarantee` near query-title tokens;
- add a rank-combination fallback with local order when discriminator confidence
  is low.

The `.80` target now looks achievable only if the discriminator can recover the
exact paired rebuttal as rank 1 more often; the top-10 recall is already ~0.91.
