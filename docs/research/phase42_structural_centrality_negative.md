# Phase XLII: Structural Centrality Rows

## Goal

Phase XLI showed that more query-term overlap and phrase proximity do not close
`OrderingGap`. This phase tests a less BM25-equivalent structural hypothesis:
maybe relevant documents are better identified by matching the query against
central or lead document fields.

The rows are corpus-agnostic and qrel-free:

- TextRank top sentence as an auxiliary BM25F field.
- First 64 document tokens as a lead-field auxiliary BM25F field.

## Added rows

`benchmarks/bench_vs_reference.cpp` now emits:

- `observed_struct_bm25f_textrank_w1.0`
- `observed_struct_bm25f_lead64_w0.5`
- `observed_struct_bm25f_lead64_w1.0`

These sit alongside the existing `observed_gen_bm25f_textrank_w0.5` generator
slice row.

Artifact:

```text
/tmp/simeon_structural_rows_20260504_112323
```

## Test results

| Corpus | BM25 | TextRank w0.5 | TextRank w1.0 | Lead64 w0.5 | Lead64 w1.0 | Shape-risk fusion | Union oracle@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | 0.3293 | 0.3262 | 0.3070 | 0.3498 | 0.3493 | 0.3327 | 0.9646 |
| FiQA | 0.2053 | 0.1783 | 0.1506 | 0.2057 | 0.1914 | 0.2089 | 0.5523 |
| NFCorpus | 0.2521 | 0.2358 | 0.2229 | 0.2500 | 0.2478 | 0.2597 | 0.5286 |
| SciFact | 0.6188 | 0.5898 | 0.5546 | 0.6034 | 0.5802 | 0.6186 | 0.8939 |
| TREC-COVID | 0.5649 | 0.5300 | 0.5070 | 0.5471 | 0.5307 | 0.5835 | 0.9847 |
| **Macro** | **0.3941** | **0.3720** | **0.3484** | **0.3912** | **0.3799** | **0.4007** | **0.7848** |

## Interpretation

Structural centrality is not a general ordering solution. The lead64 field is
useful for ArguAna (+0.0205) and neutral on FiQA, but it regresses on SciFact and
TREC-COVID. TextRank top sentence is worse across the macro average.

This is still informative:

- Some structure exists and can help a corpus without qrels.
- A single static structural field is not invariant across corpora.
- Shape-risk fusion remains the best general observed row so far.

## Consequence for the theorem program

The remaining `OrderingGap` is not closed by:

1. more exact lexical overlap;
2. generic phrase proximity;
3. static central/lead document fields.

The next viable path is not another global field weight. It is a **risk-aware
structural adapter**: expose structural evidence as candidate/scorer inputs, but
let a corpus-agnostic router decide when that evidence is safe. That keeps the
router general while allowing `A` to surface observable structure.
