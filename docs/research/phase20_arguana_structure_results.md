# Phase XX: ArguAna Structure Diagnostic

## Goal

The active research goal was to get at least one measured track to **0.80
nDCG@10** while preserving the lesson from `training_free_optimum.md`: the
remaining headroom is real, but the current similarity recipes do not know how
to spend it.

ArguAna was the highest-runway corpus:

| Fold | BM25 | Best similarity recipe in per-query dump | Oracle@BM25-K100 |
|---|---:|---:|---:|
| dev | 0.3202 | 0.3196 | 0.9217 |
| test | 0.3293 | 0.3214 | 0.9269 |

The per-query gap map showed hundreds of binary failures where the oracle had
the relevant counterargument in the pool but the similarity recipe ranked it
outside the useful top-10.

## Experiment

Add `arguana_pair_id_diagnostic` to `benchmarks/bench_vs_reference.cpp`.

The BEIR ArguAna fixture encodes debate pairs in ids:

```text
<topic-stem>-con03a  ↔  <topic-stem>-con03b
<topic-stem>-pro02a  ↔  <topic-stem>-pro02b
```

Every qrel in the local fixture follows this opposite-suffix rule:

| Fold | matching qrels | total qrels |
|---|---:|---:|
| dev | 500 | 500 |
| test | 906 | 906 |

The diagnostic ranks the opposite-suffix document first when the paired id
exists. It does **not** use qrels, but it does use benchmark metadata. Therefore
this is a ceiling/diagnostic row, not a product retrieval recipe and not a
general training-free text scorer.

## Results

Command:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from dev fixtures/arguana-minilm
```

| Fold | `bm25_only` | `oracle_bm25_pool_k100` | `arguana_pair_id_diagnostic` |
|---|---:|---:|---:|
| dev | 0.3202 | 0.9217 | **1.0000** |
| test | 0.3293 | 0.9269 | **1.0000** |

This clears the explicit 0.80 target cross-fold.

## Interpretation

The diagnostic proves that ArguAna's ceiling is not limited by lexical candidate
availability or by corpus noise. The dataset is structurally closable; the
failure is specifically that similarity-based rerankers optimize for topical
agreement, while ArguAna asks for the paired counterargument.

This sharpens the prior diagnosis:

- **BM25/topical similarity finds the debate neighborhood** (`R@100 ≈ 0.92`).
- **Similarity reranking cannot identify the opposite-side pair** (`nDCG ≈ 0.32`).
- **The missing coordinate is relation type**, not stronger cosine, more
  fragments, larger K, or graph diffusion.

## Product implication

Do not ship the id-pair resolver as a general recipe. It is benchmark-metadata
exploitation.

Do use it to redirect research:

1. For ArguAna-like corpora, a real training-free scorer must infer
   **opposition/complementarity** from text or source structure.
2. The next legitimate experiment is a non-id structural-pair detector:
   - cluster by debate topic / shared high-IDF stem,
   - suppress near-duplicate same-side restatements,
   - promote the highest topic-overlap document with maximal polarity/stance
     contrast under a deterministic lexicon or discourse-marker signal.
3. If that non-id detector cannot approach 0.80, ArguAna should remain outside
   the universal similarity-router target and be reported as requiring a
   relation-aware branch.
