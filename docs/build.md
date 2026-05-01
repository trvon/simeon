# Building simeon

## Requirements

- C++20 toolchain — Clang 16+ or GCC 13+ (MSVC 19.36+ untested but expected to work).
- Meson ≥ 0.63.
- Ninja (or any other Meson backend).

simeon has zero third-party runtime dependencies.

## Standard build

```sh
meson setup build
meson compile -C build
meson test -C build
```

Output:

- `build/libsimeon.a` — the static library.
- `build/tests/test_*` — unit tests, runnable individually or via `meson test`.
- `build/benchmarks/simeon_microbench` — throughput microbench.
- `build/benchmarks/simeon_accuracy_bench` — clustered-corpus retrieval bench (emits JSONL).

## Build options

| Option                | Default | Effect                                                   |
|-----------------------|--------:|----------------------------------------------------------|
| `enable_simd_neon`    |  `true` | Enable NEON kernels on aarch64 (auto-probed).            |
| `enable_simd_avx2`    |  `true` | Enable AVX2+FMA kernels on x86_64 (auto-probed). Set `false` for distribution builds targeting older x86. |
| `enable_tests`        |  `true` | Build the unit-test suite.                               |
| `enable_benchmarks`   |  `true` | Build the microbench + accuracy bench.                   |
| `enable_fuzz`         | `false` | Build libFuzzer harnesses (clang only).                  |
| `enable_coverage`     | `false` | Inject `-fprofile-instr-generate -fcoverage-mapping` (clang) or `--coverage` (gcc). |

Examples:

```sh
# x86 native, AVX2 on
meson setup build -Denable_simd_avx2=true

# library only, no tests / benches
meson setup build -Denable_tests=false -Denable_benchmarks=false

# clang coverage build
CC=clang CXX=clang++ meson setup build-cov -Denable_coverage=true --buildtype=debug
meson compile -C build-cov
LLVM_PROFILE_FILE=build-cov/meson-logs/profraw/test-%p-%m.profraw meson test -C build-cov
```

To render reports, merge the `.profraw` files with `llvm-profdata` and feed the test binaries to `llvm-cov`. Latest run on Apple M-series, NEON, 12 tests:

| Scope                           | Region | Function | Line   | Branch |
|---------------------------------|-------:|---------:|-------:|-------:|
| `src/` + `include/` overall     |  91.8% |    83.7% |  90.9% |  92.5% |

100% coverage: `tokenizer.cpp`, `hash_xxh64.cpp`, `arch/scalar.cpp`, `arch/neon.cpp`. Weakest is `hash_crc32c.cpp` (60% function / 59.5% line) — the slice-by-1 software fallback never fires on aarch64 because the hardware `__crc32cd` path takes everything. A CI run on x86_64 with hardware CRC disabled would close that gap.

## Sanitizers

Standard Meson `b_sanitize` works because simeon has no external libs:

```sh
# AddressSanitizer + UndefinedBehaviorSanitizer
meson setup build-asan -Db_sanitize=address,undefined -Db_lundef=false
meson test -C build-asan

# ThreadSanitizer (validates the thread-local sketch buffer in encode_one)
meson setup build-tsan -Db_sanitize=thread -Db_lundef=false
meson test -C build-tsan
```

Both should run clean. simeon is small enough that the full test suite finishes in <30 s under any sanitizer.

## Consuming simeon from another Meson project

```meson
simeon_proj = subproject('simeon')
simeon_dep  = simeon_proj.get_variable('simeon_dep')

executable('myapp', 'main.cpp', dependencies: simeon_dep)
```

Or, after `meson install`:

```meson
simeon_dep = dependency('simeon')
```

## Vendoring as plain sources

If your build system isn't Meson, copy `include/simeon/` and `src/` into your tree and compile every `.cpp` under `src/` with `-Iinclude`. Architecture-specific files (`src/arch/neon.cpp`, `src/arch/avx2.cpp`) should only be compiled when the corresponding ISA is available; gate them with `__aarch64__` / `__AVX2__` or your build system's equivalent. AVX2 sources require `-mavx2 -mfma`.

## Reproducing benchmarks

```sh
./build/benchmarks/simeon_microbench 1000 256 > microbench.jsonl
./build/benchmarks/simeon_accuracy_bench 50 60 0.05 > accuracy.jsonl
```

The accuracy bench takes `(docs_per_cluster, words_per_doc, leakage)` as positional args. See [research/benchmarks.md](research/benchmarks.md) for interpretation and published summary tables.

## Reference embedding comparison

The default `bench_vs_reference` binary is the stable comparison surface: it compares simeon configurations against a frozen learned-embedding model on a real IR benchmark and keeps the public router / production rows only. It requires an external fixture (corpus, queries, qrels, pre-computed reference embeddings); the fixture format is documented in [reference_fixture.md](reference_fixture.md). Fixture provisioning is out of tree.

```sh
./build/benchmarks/simeon_bench_vs_reference fixtures/scifact-minilm > vs_reference.jsonl
```

### Held-out dev/test workflow (router tuning)

The bench supports honest router-threshold tuning when the fixture provides a held-out dev fold (`queries_dev.tsv`, `qrels_dev.tsv`, `reference_dev.bin` siblings of the test files). Pick a fold with `--queries-from {test,dev}` and optionally dump per-query router telemetry with `--router-per-query <path.jsonl>`:

```sh
# Sweep on dev (router_grid_4096_768_passA_*, _passB_* rows, plus oracle):
./build/benchmarks/simeon_bench_vs_reference fixtures/scifact-minilm \
    --queries-from dev \
    --router-per-query scifact_router_dev_per_query.jsonl \
    > scifact_router_dev_grid.jsonl

# Confirm the top-N grid configs on test:
./build/benchmarks/simeon_bench_vs_reference fixtures/scifact-minilm \
    --queries-from test \
    --router-per-query scifact_router_test_per_query.jsonl \
    > scifact_router_test_grid.jsonl
```

The oracle row (`router_oracle_4096_768`) reports the per-query argmax over the three recipes — an upper bound on what any pre-retrieval router can achieve at `(pool_size, alpha) = (500, 0.75)`. The dev→test gap on the top-3 grid configs is the honest tuning generalization estimate.

## Research benchmark binary

Archived probe-style ablations now live in a separate research binary:

```sh
./build/benchmarks/simeon_bench_vs_reference_research fixtures/scifact-minilm > research.jsonl
```

Use this binary when reproducing the experimental grids documented under [research/](research/).

### Structural BM25F sweeps

`--aux-from {textrank,ac}` emits `reference`, `bm25_only`, and the corresponding BM25F rows (`w_aux = 0.0, 0.2, 0.5, 1.0`) for the requested auxiliary field.

```sh
# TextRank synthetic-title field
./build/benchmarks/simeon_bench_vs_reference_research \
    --aux-from textrank \
    fixtures/scifact-minilm > scifact_textrank_bm25f.jsonl

# Aho-Corasick self-bootstrapped entity field
./build/benchmarks/simeon_bench_vs_reference_research \
    --aux-from ac \
    fixtures/fiqa-minilm > fiqa_ac_bm25f.jsonl
```

### PMI soft-match sweeps

`--softmatch-only` runs a corpus-derived, training-free semantic expansion slice: `reference`, `bm25_only`, and PMI-neighbor soft-match BM25 rows. The current harness learns PMI embeddings from the fixture corpus, expands each query term with its nearest lexical neighbors, and rescales them through `score_weighted_hashes()`.

```sh
./build/benchmarks/simeon_bench_vs_reference_research \
    --softmatch-only \
    fixtures/scifact-minilm > scifact_softmatch.jsonl
```

### Phrase/document transport sweeps

`--transport-only` runs the transport-structure ablation slice: `reference`, `bm25_only`, then calibrated phrase/concept transport rows built from existing sparse retrieval primitives. The current implementation uses BM25 top-K pooling, SDM bigram legs as phrase transport, optional mined concepts, and z-scored fusion inside the pool.

```sh
./build/benchmarks/simeon_bench_vs_reference_research \
    --transport-only \
    fixtures/fiqa-minilm > fiqa_transport.jsonl
```

### Graph transport sweeps

`--graph-only` runs the pool-restricted graph reranking slice: `reference`, `bm25_only`, then personalized-PageRank-style graph rows built over query, phrase, and BM25 top-K document nodes. The current implementation tests phrase-only vs phrase+doc-doc graphs, `K ∈ {100,300}`, and damping in `{0.70,0.85}`.

```sh
./build/benchmarks/simeon_bench_vs_reference_research \
    --graph-only \
    fixtures/fiqa-minilm > fiqa_graph.jsonl
```

### Cluster topology sweeps

`--cluster-only` runs the document-anchored fragment/cluster slice: `reference`, `bm25_only`, then rows that extract top TextRank fragments per document, build sparse fragment signatures, cluster them by weighted overlap inside the BM25 pool, and rescore docs through query-cover plus fragment-containment mass.

```sh
./build/benchmarks/simeon_bench_vs_reference_research \
    --cluster-only \
    fixtures/fiqa-minilm > fiqa_cluster.jsonl
```

### Fragment graph sweeps

`--fragment-only` runs the document-anchored semantic fragment graph slice: `reference`, `bm25_only`, then rows that learn in-corpus PMI embeddings, encode top TextRank fragments per document, build fragment-to-fragment semantic edges inside the BM25 pool, and diffuse query-seeded mass through `doc -> fragment -> fragment -> doc`.

```sh
./build/benchmarks/simeon_bench_vs_reference_research \
    --fragment-only \
    fixtures/fiqa-minilm > fiqa_fragment.jsonl
```

The current `--fragment-only` grid includes:

- PMI-only fragment graph rows,
- hybrid fragment rows with lexical bridge edges,
- geometric fragment rows with query-centered soft neighborhoods over locally whitened fragment vectors,
- DocScorer cross-product rows named
  `bm25_fragment_geom_xprod_<bm25>_euclid_<scorer>_a<alpha>_k100_t8_richcov`
  over BM25 ∈ {atire, bm25plus, bm25l, dph, pl2, dcm, layered, layeredw},
  scorer ∈ {max, mean, topk3, smax, geom}, alpha ∈ {0.65, 0.80, 0.90},
- per-corpus recipe-router rows (`xprod_v17{short,medium,long}_recipe_*`)
  that select BM25 variant + DocScorer + dual-stage gating from observable
  corpus features (`avg_dl`, `n_docs`).
