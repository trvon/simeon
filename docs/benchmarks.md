# Benchmark guide

This documents the stable benchmarks shipped with simeon and how to interpret their output.

## Microbenchmark (`simeon_microbench`)

Measures encoding throughput. Runs N iterations of the encode pipeline at sketch dims
from 256 to 32768, emitting a JSONL row per (projection, dim, iteration).

```sh
./build/benchmarks/simeon_microbench 1000 256 > microbench.jsonl
```

Arguments: `(n_docs, doc_len_chars)`. Run as a CI smoke test with `(200, 256)`.

## Accuracy benchmark (`simeon_accuracy_bench`)

Generates a clustered synthetic corpus, encodes with multiple projection modes, and
measures intra/inter-cluster cosine separation. Emits per-projection per-cluster
statistics as JSONL. Used to validate that projection modes maintain topical structure.

```sh
./build/benchmarks/simeon_accuracy_bench 10 40 0.05 > accuracy.jsonl
```

Arguments: `(docs_per_cluster, words_per_doc, leakage)`. CI smoke: `(10, 40, 0.05)`.

## Reference embedding comparison (`simeon_bench_vs_reference`)

**Requires `-Denable_research=true`.** Compares simeon against a frozen learned-embedding
model on a BEIR-style benchmark. The fixture format is documented in
[reference_fixture.md](reference_fixture.md). Fixture provisioning is out of tree.

```sh
./build/benchmarks/simeon_bench_vs_reference fixtures/scifact-minilm > vs_reference.jsonl
```

### Held-out dev/test workflow (router tuning)

```sh
# Sweep router thresholds on dev fold
./build/benchmarks/simeon_bench_vs_reference fixtures/scifact-minilm \
    --queries-from dev --router-per-query scifact_dev_per_query.jsonl \
    > scifact_router_dev.jsonl

# Confirm on test fold
./build/benchmarks/simeon_bench_vs_reference fixtures/scifact-minilm \
    --queries-from test --router-per-query scifact_test_per_query.jsonl \
    > scifact_router_test.jsonl
```

## Research sweeps (`simeon_bench_vs_reference_research`)

**Requires `-Denable_research=true`.** Archived probe-style ablations for experimental
features that did not ship. Sweeps documented in [research.md](research.md):

| Flag | Description |
|---|---|
| `--aux-from {textrank,ac}` | BM25F with synthetic title/entity aux fields |
| `--softmatch-only` | PMI-neighbor query expansion |
| `--transport-only` | Phrase/concept transport via SDM + z-scored fusion |
| `--graph-only` | PPR graph reranking |
| `--cluster-only` | TextRank-fragment cluster rescoring |
| `--fragment-only` | PMI fragment graphs + geometric neighborhoods + DocScorer grid |

```sh
./build/benchmarks/simeon_bench_vs_reference_research \
    --aux-from textrank fixtures/scifact-minilm > scifact_textrank_bm25f.jsonl
```

## Aho-Corasick benchmark (`simeon_aho_corasick_bench`)

Throughput benchmark for the dense Aho-Corasick dictionary matcher. Builds the automaton
from synthesized patterns and scans synthetic input.

```sh
./build/benchmarks/simeon_aho_corasick_bench [input_size_bytes]
```

## TextRank benchmark (`simeon_text_rank_bench`)

Throughput benchmark for the TextRank sentence ranker on synthetic documents.

```sh
./build/benchmarks/simeon_text_rank_bench [n_docs]
```

## Profiling harnesses

| Binary | Purpose |
|---|---|
| `simeon_profile_sab_smooth` | Isolated SAB-smooth BM25 query loop for xctrace/sample profiling |
| `simeon_profile_fragment_geometry` | Isolated fragment-geometry pipeline for profiling |

### Fragment-geometry rerank, phaseD-perf engineering pass

`simeon_profile_fragment_geometry`, richcov builder, mode=approx, 50 queries ×
5 iters, Apple Silicon (arm64/NEON), mean µs per query. Differential signals
(`phss_scale_mean`, `graph_edges_mean`) identical before/after on every
fixture × mode; see research.md for mechanism.

| Phase | SciFact before | SciFact after | NFCorpus before | NFCorpus after |
|---|---|---|---|---|
| query_total_mean_us | 8729.3 | 3321.1 | 8533.5 | 3273.0 |
| phss_select_mean_us | 5567.8 | 925.8 | 5474.3 | 918.6 |
| phss_select_edge_sort_mean_us | 5170.8 | 764.0 | 5090.7 | 758.0 |
| phss_pairwise_mean_us | 2016.4 | 1538.2 | 1989.0 | 1521.8 |
| adjacency_mean_us | 737.3 | 502.2 | 718.4 | 544.3 |

## Research outcomes

See [research.md](research.md) for the full research summary including:
- Which sweeps produced gains and were promoted to production
- Which paths were negative and why
- Component gating via `enable_research`
