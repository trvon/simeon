# BM25F auxiliary-field plumbing

`Bm25Index` now supports an optional auxiliary text field at ingest and a
BM25F-style scorer:

- `add_doc(body, aux)` indexes a parallel aux posting list with its own length statistics.
- `score_bm25f(body_query, aux_query, ...)` linearly combines body and aux scores, and the one-query overload reuses the same query text for both fields.
- `bench_vs_reference --aux-from {textrank,ac}` runs the structural-only slice: `reference`, `bm25_only`, and the corresponding BM25F rows.

## E1 sanity

Acceptance criterion: `weight_aux=0` must recover the body-only BM25 baseline byte-identically.

Result: **passed on all six runs** (`scifact`, `nfcorpus`, `fiqa` × `textrank`, `ac`).

| Corpus | Baseline `bm25_only` | Aux row | nDCG@10 |
|--------|----------------------:|---------|--------:|
| scifact | 0.6188 | `bm25_textrank_title_w0.0` | 0.6188 |
| scifact | 0.6188 | `bm25_ac_entity_w0.0` | 0.6188 |
| nfcorpus | 0.2521 | `bm25_textrank_title_w0.0` | 0.2521 |
| nfcorpus | 0.2521 | `bm25_ac_entity_w0.0` | 0.2521 |
| fiqa | 0.2053 | `bm25_textrank_title_w0.0` | 0.2053 |
| fiqa | 0.2053 | `bm25_ac_entity_w0.0` | 0.2053 |

Notes:

- The entity-field path requires a distinct auxiliary query string because the
  aux field stores matched pattern IDs (`entNNN`).
- The plumbing itself is sound even though the downstream structural experiments
  do not meet their promote thresholds.
