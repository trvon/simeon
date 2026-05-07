# Phase XLIV: Simeon Embedding Generator Slice

## Goal

Add simeon's own training-free embedding as a first-class generator `G` inside the
`--generator-slices` research harness. Prior phases used only lexical generators
(BM25, BM25F, RM3). This phase tests whether the library's embedding signal
exposes different relevant documents and whether union fusion can exploit them.

## Added rows

`benchmarks/bench_vs_reference.cpp` now encodes docs and queries with an
`AchlioptasSparse 4096→384` simeon encoder inside
`run_first_generator_slice_oracles` and emits:

**Observed**
- `observed_gen_simeon_achlioptas_4096_384`
- `observed_union_bm25_simeon_z_equal`
- `observed_union_bm25_bm25f_rm3_simeon_z_equal`
- `observed_union_bm25_simeon_rrf_k100`
- `observed_union_bm25_bm25f_rm3_simeon_rrf_k100`

**Oracle**
- `oracle_gen_simeon_achlioptas_4096_384_k{50,100,200,500}`
- `oracle_union_bm25_simeon_k{50,100,200,500}`
- `oracle_union_bm25_bm25f_rm3_simeon_k{50,100,200,500}`

Artifact:

```text
/tmp/simeon_embedding_slice_20260504_140812
```

## Test results

| Corpus | BM25 | simeon | BM25+simeon z | 3-way z | **4-way z** | shape-risk | 3-way oracle | 4-way oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | 0.3293 | 0.1401 | 0.3120 | 0.3334 | **0.3286** | 0.3327 | 0.9646 | 0.9668 |
| FiQA | 0.2053 | 0.0838 | 0.1996 | 0.2074 | **0.2055** | 0.2089 | 0.5523 | 0.5804 |
| NFCorpus | 0.2521 | 0.1472 | 0.2652 | 0.2603 | **0.2623** | 0.2597 | 0.5286 | 0.5973 |
| SciFact | 0.6188 | 0.3475 | 0.6113 | 0.6094 | **0.6134** | 0.6186 | 0.8939 | 0.9098 |
| TREC-COVID | 0.5649 | 0.3427 | 0.5893 | 0.5796 | **0.6128** | 0.5835 | 0.9847 | 0.9901 |
| **Macro** | **0.3941** | **0.2123** | **0.3955** | **0.3980** | **0.4045** | **0.4007** | **0.7848** | **0.8089** |

### Deltas vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| simeon standalone | −0.1892 | −0.1215 | −0.1049 | −0.2713 | −0.2222 | −0.1818 |
| BM25+simeon z | −0.0173 | −0.0057 | +0.0131 | −0.0075 | +0.0244 | +0.0014 |
| 3-way z | +0.0041 | +0.0021 | +0.0082 | −0.0094 | +0.0147 | +0.0039 |
| **4-way z** | **−0.0007** | **+0.0002** | **+0.0102** | **−0.0054** | **+0.0479** | **+0.0104** |
| shape-risk hard | +0.0034 | +0.0036 | +0.0076 | −0.0002 | +0.0186 | +0.0066 |

### Deltas vs 3-way z-equal

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| **4-way z** | −0.0048 | −0.0019 | +0.0020 | +0.0040 | +0.0332 | +0.0065 |
| shape-risk hard | −0.0007 | +0.0015 | −0.0006 | +0.0092 | +0.0039 | +0.0027 |

## Interpretation

### 1. Standalone simeon is weak

The training-free embedding alone (0.2123 macro) is far below BM25 alone
(0.3941). This is expected: simeon captures topical structure, not paraphrase or
semantic equivalence. On BEIR-style tasks with graded relevance, lexical signal
is stronger.

### 2. The embedding raises the union oracle

The 4-way union oracle@100 is **0.8089**, up from the 3-way oracle **0.7848**:

| Corpus | 3-way oracle | 4-way oracle | Lift |
|---|---:|---:|---:|
| ArguAna | 0.9646 | 0.9668 | +0.0022 |
| FiQA | 0.5523 | 0.5804 | +0.0281 |
| NFCorpus | 0.5286 | 0.5973 | +0.0687 |
| SciFact | 0.8939 | 0.9098 | +0.0159 |
| TREC-COVID | 0.9847 | 0.9901 | +0.0054 |

The embedding exposes **different relevant documents** that BM25/BM25F/RM3 miss,
especially on FiQA and NFCorpus. This is evidence that `G = BM25 ∪ simeon` is a
valid generator expansion.

### 3. Z-fusion outperforms RRF for this combination

| Row | Macro |
|---|---:|
| 4-way z-equal | 0.4045 |
| 4-way RRF@100 | 0.3718 |

Score-level z-fusion preserves magnitude information; RRF discards it and
performs worse when one generator is much weaker than the others.

### 4. 4-way z-equal is the new best observed row

At **0.4045 macro nDCG@10**, the 4-way z-equal beats:
- BM25 alone: +0.0104
- 3-way z-equal: +0.0065
- shape-risk hard: +0.0038

The gains are concentrated on **TREC-COVID** (+0.0479) and **NFCorpus**
(+0.0102), with small regressions on ArguAna (−0.0007) and SciFact (−0.0054).

### 5. The gap to the oracle is still large

```text
4-way oracle@100 - 4-way observed z-equal = 0.8089 - 0.4045 = 0.4044
```

So Phase XLIV does not close the theorem gap. It confirms that:
- `ExposureGap` can be reduced by adding the embedding generator (oracle rises)
- `RoutingApproxGap` is still the dominant bottleneck (observed is far below oracle)
- `OrderingGap` remains large because the fusion layer cannot rank the union pool
  well enough.

## Risk note

The 4-way z-equal regresses SciFact by −0.0054 vs BM25 and ArguAna by −0.0007.
A production system would need a **risk-aware gate** that dampens or suppresses
the embedding leg on corpora where it hurts, rather than fusing it uniformly.
Phase XLIII's shape-risk framework is the right axis for that gate, but it must
be extended with embedding-specific risk features (BM25/simeon Jaccard, simeon
score-list shape).

## Next direction

1. **Risk-aware 4-generator fusion**: extend the shape-risk gate to condition on
   BM25/simeon disagreement and simeon confidence, so the embedding leg is active
   only when it is estimated safe.
2. **OrderingGap attack**: the 4-way oracle is 0.8089 while observed is 0.4045.
   Better within-pool scoring (not just more generators) is needed to close this.
3. **Budget check**: the embedding adds encode cost. A PQ-compressed or
   matryoshka-short-prefix variant could reduce `BudgetGap`.
