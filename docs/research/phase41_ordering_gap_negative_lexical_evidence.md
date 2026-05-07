# Phase XLI: Ordering Gap Negative Result — Lexical Evidence Rerank

## Goal

After Phase XL reduced `RoutingApproxGap` with observed shape-risk fusion, the
next hypothesis was that a simple corpus-agnostic within-pool lexical evidence
scorer could reduce `OrderingGap`:

```text
OrderingGap = M(O_M restricted to P_G@K) - M(BestObserved_G)
```

The experiment reranks the BM25 top-500 pool with additional qrel-free evidence:

- IDF-weighted query-term containment between query and document signatures;
- ordered/unordered query bigram proximity within a small window;
- BM25 z-score as the base score.

## Added rows

`benchmarks/bench_vs_reference.cpp` now emits these research rows under
`--generator-slices`:

- `observed_ordering_bm25_pool500_overlap_w0.1_phrase_w0.05`
- `observed_ordering_bm25_pool500_overlap_w0.5_phrase_w0.5`
- `observed_ordering_bm25_pool500_overlap_w1.0_phrase_w0.25`

Artifacts:

```text
/tmp/simeon_ordering_rows_20260504_103602
/tmp/simeon_ordering_small_rows_20260504_105757
```

## Test results

| Corpus | BM25 | shape-risk fusion | weak lexical rerank | medium lexical rerank | strong lexical rerank | BM25 oracle@500 |
|---|---:|---:|---:|---:|---:|---:|
| ArguAna | 0.3293 | 0.3327 | 0.3265 | 0.3246 | 0.3026 | 0.9701 |
| FiQA | 0.2053 | 0.2089 | 0.2038 | 0.1892 | 0.1691 | 0.6608 |
| NFCorpus | 0.2521 | 0.2597 | 0.2526 | 0.2514 | 0.2512 | 0.5446 |
| SciFact | 0.6188 | 0.6186 | 0.6186 | 0.6018 | 0.5840 | 0.9224 |
| TREC-COVID | 0.5649 | 0.5835 | 0.5642 | 0.5041 | 0.4665 | 0.9928 |
| **Macro** | **0.3941** | **0.4007** | **0.3931** | **0.3742** | **0.3547** | **0.8181** |

## Interpretation

The simple lexical evidence reranker does **not** reduce `OrderingGap`. It is
mostly redundant with BM25 at low weights and actively harmful at higher weights.
The result is strongest on TREC-COVID: even though the BM25 top-500 oracle is
near perfect, extra query-term containment and phrase proximity push the ranking
away from the answer-key ordering.

This says the remaining ordering signal is not merely more exact lexical overlap.
Likely missing axes are:

- relation/stance/discourse cues;
- section/field-specific evidence;
- entity or citation structure;
- query intent and answer-type matching;
- learned-like pair discrimination approximated without labels.

## Consequence for the theorem program

Phase XLI is a useful negative result: the large BM25-pool oracle does not imply
that any monotone lexical-overlap reranker can approach it. The gap is a genuine
ordering/discrimination gap, not a failure to count enough query terms.

Next work should either:

1. add richer qrel-free `S` features that are not BM25-equivalent, or
2. expose more structure through `A` while keeping the general router
   corpus-agnostic.
