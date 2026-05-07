# Phase XXV: English Blended Pair Ranker

## Goal

Continue the English-only ArguAna push toward `.80` without touching README/PR
packaging yet. The local gate was improvement over Phase XXIV's dev-fit ranker.

## Experiment

Added `arguana_text_pair_ranker_blend_p5` to
`benchmarks/bench_vs_reference.cpp`.

The row blends:

- Phase XXIV dev-fit relation ranker;
- Phase XXII hand-set relation discriminator;
- a small proximity fallback.

Because it includes the dev-fit leg, this remains a supervised diagnostic row,
not the training-free product candidate.

## Results

Commands:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | Dev-fit ranker | Blended ranker | Oracle@K100 |
|---|---:|---:|---:|
| dev | 0.7022 | **0.7087** | 0.9217 |
| test | 0.7025 | **0.7097** | 0.9269 |

The blend improves cross-fold but only modestly:

- dev: `+0.0065`
- test: `+0.0072`

## Interpretation

We are now in a harder regime. Simple score blending no longer buys large
chunks of the runway. The stable English progression is:

| Stage | nDCG@10 |
|---|---:|
| BM25 | ~0.32 |
| text-neighborhood structure | ~0.44 |
| hand-set pair discriminator | ~0.63 |
| dev-fit pair ranker | ~0.70 |
| blended pair ranker | ~0.71 |
| BM25-pool oracle | ~0.92 |

The remaining `.09` to the `.80` target is not likely to come from convex score
mixing. It needs new information about pair identity or relation direction.

## Next target

Phase XXVI should use `SIMEON_PER_QUERY_DUMP` on `arguana_text_pair_ranker_blend_p5`
and inspect the rank-2/rank-3 misses. The goal is to find a new deterministic
feature that separates the paired rebuttal from nearby same-topic rebuttals.

Candidate feature families:

1. **same-point title alignment**: the correct paired doc often answers the
   source title, not just the broader debate topic;
2. **first-sentence contradiction**: contradiction cues near source title terms;
3. **local cluster pairing pattern**: learn structure at the argument-point
   level without using exact `a/b` suffix ids.
