# Benchmark guide

This documents the stable benchmarks shipped with simeon and how to interpret their output.

## Manifest-driven embedding retrieval (`simeon_embedding_experiment`)

Runs exact retrieval for every encoder variant in a versioned `.exp` manifest.
Unlike the synthetic accuracy smoke, it consumes qrels, fingerprints the fixture
and manifest, enforces declared dev/holdout policy, and emits one self-describing
JSONL record per variant.

```sh
./build/benchmarks/simeon_embedding_experiment \
  experiments/embedding-foundation.exp \
  --fixture /path/to/fixture --split dev
```

See [experimentation.md](experimentation.md) for the fixture schema, metric
semantics, manifest parameters, result contract, and frozen-holdout workflow.
The same executable also owns the `wsdm_idf_fusion` provider, which constructs
the established six-leg union once and replays IDF candidate/score variants.
It reports candidate recall separately from ranked Recall@100 so a new leg's
retrieval and ordering effects cannot be conflated.

`coordinate_calibrated_idf` builds one maximum-width hashed-IDF/FWHT workspace,
measures coordinate utilization, and compares nested prefix charts. It supports
corpus centering/standardization plus fixed, blended, score-admitted, and
query-energy-admitted routing. Exact prefix/reference scores are cached so a
manifest threshold sweep replays policy decisions rather than recomputing the
corpus matrix.

`feature_family_atlas_idf` gives character and word evidence separate
hashed-IDF/FWHT charts, enforces an exact combined vector-storage budget, and
replays fixed block weights. Independent, joint, and corpus-RMS-calibrated
normalization reveal whether collision-free feature partitions transfer. It
reports standalone family quality, chart overlap/distortion, family energy,
and evaluation-only union recall without feeding qrels into the policy.

The provider also supports safe-base residual experiments. `base_only` retains
the combined hashed-IDF chart; `residual_blend` and `selective` add a separately
priced family chart using raw cosine, query-wise z-score, or rank RRF. Results
report the exact deployable dimension total, three IDF artifacts, complete
research score-cache bytes, base/family metrics, route counts, and policy
replay time. A rejected selective route returns the base ranking exactly.

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

| Phase | SciFact before | pass 1 | pass 2 | NFCorpus before | pass 1 | pass 2 |
|---|---|---|---|---|---|---|
| query_total_mean_us | 8729.3 | 3321.1 | 1896.2 | 8533.5 | 3273.0 | 1822.8 |
| phss_select_mean_us | 5567.8 | 925.8 | 221.6 | 5474.3 | 918.6 | 218.8 |
| phss_select_edge_sort_mean_us | 5170.8 | 764.0 | 207.2 | 5090.7 | 758.0 | 204.4 |
| phss_pairwise_mean_us | 2016.4 | 1538.2 | 991.6 | 1989.0 | 1521.8 | 983.1 |
| adjacency_mean_us | 737.3 | 502.2 | 323.9 | 718.4 | 544.3 | 355.9 |

## Fusion pass (dev-tuned, frozen test validation)

nDCG@10, MiniLM fixtures, `--fusion-only` research sweep. Winner frozen on dev,
read once on test.

| Config | SciFact dev | SciFact test | NFCorpus dev | NFCorpus test | FiQA dev | FiQA test |
|---|---|---|---|---|---|---|
| best fixed (prior; FiQA = bm25_only) | — | 0.6714 | — | 0.3182 | 0.2378 | 0.2375 |
| cc_wsdmsab0.60_wsdmat0.40 (promoted) | 0.6950 | 0.6885 | 0.2977 | 0.3220 | 0.2480 | 0.2512 |
| cc_atire0.20_sab0.80 | 0.6777 | 0.6767 | 0.3000 | 0.3210 | 0.2391 | 0.2442 |
| fusion_combsum_atire_wsdmat_sab | 0.6885 | 0.6891 | 0.2939 | 0.3212 | 0.2448 | 0.2451 |
| fusion_rrf_atire_wsdmat_sab | 0.6828 | 0.6752 | 0.2722 | 0.3097 | — | — |
| ccprf pf0.30 (promoted ⊕ fused-feedback RM3; opt-in) | — | 0.6990 | — | 0.3261 | 0.2481 | 0.2469 |
| 6-leg union pool oracle | 0.9368 | 0.9365 | 0.5760 | 0.5918 | 0.5880 | 0.6073 |

## Geometry-leg hubness correction (CSLS)

nDCG@10, `simeon_recipe_accuracy_bench`, standalone pure-geometry leg
(richcov + PHSS approx, α=0) over the 6-leg union pool. Knobs:
`FragmentGeometryConfig::csls_k` / `csls_beta`. Dev-tuned, single test read.

| Leg | SciFact dev | SciFact test | NFCorpus dev | NFCorpus test | FiQA dev |
|---|---|---|---|---|---|
| geom_pure | 0.2382 | 0.2830 | 0.1697 | 0.1997 | 0.0678 |
| geom_csls_k8_b1.0 (promoted, opt-in) | 0.2732 | 0.3168 | 0.1761 | 0.2115 | 0.0695 |
| geom_csls_k16_b1.0 | 0.2678 | 0.3149 | 0.1794 | 0.2105 | 0.0715 |
| promoted fusion ⊕ g0.10·z(geom_csls_k8) | 0.6951 | 0.6980 | 0.3012 | 0.3208 | — |

Fusion contribution stays within the ±0.005 dev gate — promoted fusion config
unchanged; csls is an opt-in knob for the standalone geometry rerank path.

## Research outcomes

See [research.md](research.md) for the full research summary including:
- Which sweeps produced gains and were promoted to production
- Which paths were negative and why
