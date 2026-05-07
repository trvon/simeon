# Phase XXVI: ArguAna Topic-stem Ranker

## Goal

Continue the English-only push after Phase XXV saturated simple score blending.
The next hypothesis was that some remaining misses come from the text-header
neighborhood merging duplicate debate topics across ArguAna categories. A
corpus-structure adapter can use the **topic stem** of the ArguAna id without
using the exact `a ↔ b` pair suffix.

## Experiment

Added `arguana_topic_stem_ranker_blend_p5` to
`benchmarks/bench_vs_reference.cpp`.

Candidate set:

- extract the topic stem from ids like:

```text
test-sport-ybfgsohbhog-con03a -> test-sport-ybfgsohbhog
```

- rank all non-self docs with the same topic stem;
- do **not** use the final `a/b` suffix and do **not** directly construct the
  paired id.

Scoring reuses the Phase XXV blended English relation ranker.

This is a corpus-structure diagnostic. It is less exploitative than
`arguana_pair_id_diagnostic`, but still not a universal product recipe because
it depends on ArguAna's id schema.

## Results

Commands:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | Text-header blend | Topic-stem blend | Oracle@K100 |
|---|---:|---:|---:|
| dev | 0.7087 | **0.7557** | 0.9217 |
| test | 0.7097 | **0.7694** | 0.9269 |

This clears the next `>= 0.75` gate cross-fold.

Relative to Phase XXV:

- dev: `+0.0470`
- test: `+0.0597`

Recall@10 is now essentially saturated:

- dev: `0.9960`
- test: `0.9934`

## Interpretation

The `.80` gap is now almost purely rank-1 order inside the correct topic stem.
Candidate recall is solved.

Updated progression:

| Stage | nDCG@10 |
|---|---:|
| BM25 | ~0.32 |
| text-neighborhood structure | ~0.44 |
| hand-set pair discriminator | ~0.63 |
| dev-fit pair ranker | ~0.70 |
| text-header blended ranker | ~0.71 |
| topic-stem blended ranker | **~0.76** |
| BM25-pool oracle | ~0.92 |

The result supports the corpus/language-support conclusion: English relation
ranking needs a corpus structure adapter. For ArguAna, the useful adapter is the
debate topic stem; for yams or another English corpus, the equivalent may be
path, title, section, or citation context.

## Next target

Phase XXVII target:

```text
>= 0.80 nDCG@10 on dev and test
```

Since recall@10 is already ~0.99, the next feature must reorder within the
topic-stem cluster. Candidate directions:

1. argument-point alignment without exact suffix ids;
2. better first-sentence/title contradiction matching;
3. supervised ranker with non-linear feature crosses over offset bucket ×
   title/body shape.
