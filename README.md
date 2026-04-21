<p align="center">
<h1 align="center">simeon — Training-free SIMD text embeddings</h1>
<h6 align="center">Deterministic, model-free text→vector for dense retrieval. NEON/AVX2 kernels, single-digit microseconds per doc.</h6>
</p>
<p align="center">
<img alt="license" src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue?style=flat-square">
<img alt="language" src="https://img.shields.io/badge/language-C%2B%2B20-informational?style=flat-square">
<img alt="isa" src="https://img.shields.io/badge/SIMD-NEON%20%7C%20AVX2%20%7C%20scalar-success?style=flat-square">
</p>

> [!WARNING]
> **Experimental.** simeon is a research vehicle; API and defaults are not yet stable. It is not a semantic replacement for learned embeddings — see [Positioning](#positioning).

## What it does

simeon maps arbitrary UTF-8 text to a fixed-width float vector without any learned model:

```
text → char/word n-grams → hashed count-sketch → (optional) random projection
     → (optional) matryoshka row weights → L2-normalize
     → (optional) Product Quantization → m-byte code
```

The whole pipeline is deterministic, seeded, and byte-identical across architectures for the same input.

## Features

- **Three projection heads** — Achlioptas sparse, very-sparse (Li 2006), dense Gaussian
- **Matryoshka-style nested output** — one projection, queryable at any prefix width with `1/sqrt(1 + r/decay)` row weights (training-free analog of Kusupati et al. 2022)
- **Product Quantization + ADC** — 8–192× index compression with near-baseline R@10 (Jégou, Douze, Schmid 2010)
- **BM25 index + RRF fusion + cascade rerank helpers** — pure-lexical first stage and reciprocal-rank fusion primitives, so simeon can plug into hybrid retrieval pipelines without an external IR library
- **5 BM25 variants** including the novel **SubwordAwareBackoff** (training-free morphological backoff via a parallel char-n-gram inverted index)
- **Per-query router** picks among `Bm25Atire` / `Bm25SabSmooth` / `CascadeLinearAlpha` from cheap pre-retrieval predictors (Carmel & Yom-Tov 2010 family) — matches MiniLM-L6 nDCG@10 on BEIR scifact, training-free, no GPU
- **Densified MinHash head** (Shrivastava 2017) for Jaccard-space three-way fusion on duplicate-heavy corpora
- **BPE-lite subword tokenizer** with caller-supplied merges (no built-in vocab; deterministic across runs)
- **Three hash families** — SplitMix64, XXH64 (canonical), CRC32C (hardware on x86 SSE4.2 / aarch64 +crc, slice-by-1 fallback), plus Mixed Tabulation (Houen & Thorup 2023, sparse-JL with practical hashing)
- **SIMD kernels** — aarch64 NEON, x86 AVX2+FMA, portable scalar
- Standalone Meson build, zero runtime deps, GPLv3
- Cross-arch byte-identity tests, SIMD-vs-scalar parity tests, determinism KATs

## Positioning

simeon captures **lexical and topical** structure, not paraphrase or semantic equivalence. It is **not** a drop-in replacement for a learned bi-encoder like MiniLM. The intended deployments are:

- First-stage ANN recall, with a learned reranker on top.
- BM25 ⊕ dense late-fusion (RRF), where lexical signal is what wins on hard / short queries.
- Self-contained retrieval where shipping a 200MB model is impractical (CLI tools, embedded, on-device).

In YAMS, simeon is used as the default retrieval embedding backend and lexical companion. ONNX remains reserved for opt-in plugin tasks such as GLiNER / ColBERT rather than the default embedding path.

simeon is best read as an engineering and evaluation layer over several lines of prior work: NUMEN-style training-free retrieval, ColBERT-era late-interaction framing, sparse random projection, matryoshka-style nested representations, Product Quantization, and classical query-difficulty routing. It is not a new retrieval algorithm. The shipped benchmark and router notes include negative findings, so this repository should not be read as blanket validation of any single upstream paper's headline claims.

## Documentation

| Topic              | File                                                  |
|--------------------|-------------------------------------------------------|
| Build              | [docs/build.md](docs/build.md)                        |
| Benchmarks         | [docs/benchmarks.md](docs/benchmarks.md)              |
| Research notes     | [docs/research.md](docs/research.md)                  |
| Research saturation | [docs/training_free_saturation.md](docs/training_free_saturation.md) |
| Works cited        | [docs/works_cited.md](docs/works_cited.md)            |
| Reference fixture  | [docs/reference_fixture.md](docs/reference_fixture.md)|
| Headers            | [include/simeon/](include/simeon/)                    |
| Source             | [src/](src/) and [src/arch/](src/arch/)               |
| Tests              | [tests/](tests/)                                      |
| Microbench         | [benchmarks/](benchmarks/)                            |

## Build

```sh
meson setup build
meson compile -C build
meson test -C build
```

Requires a C++20 toolchain (Clang 16+ or GCC 13+), Meson ≥ 0.63, and Ninja. No third-party dependencies. Full options in [docs/build.md](docs/build.md).

## Quick start

```cpp
#include <simeon/simeon.hpp>

simeon::EncoderConfig cfg;
cfg.sketch_dim = 4096;
cfg.output_dim = 384;
cfg.projection = simeon::ProjectionMode::AchlioptasSparse;

simeon::Encoder enc(cfg);
std::vector<float> out(enc.output_dim());
enc.encode("training-free text embeddings", out.data());
// out is a unit-norm 384-d float vector, deterministic for this (cfg, seed, text).
```

Matryoshka — one vector, queryable at any prefix:

```cpp
cfg.matryoshka = true;
simeon::Encoder enc(cfg);
std::vector<float> out(384);
enc.encode("nested representation", out.data());

std::vector<float> coarse(out.begin(), out.begin() + 64);
simeon::matryoshka_prefix_normalize(coarse.data(), 64);  // unit-norm 64-d coarse query
```

Product Quantization — 96–192× smaller index:

```cpp
#include <simeon/pq.hpp>

simeon::PQConfig pcfg{.dim = 384, .m = 8, .k = 256};
simeon::ProductQuantizer pq(pcfg);
pq.train(corpus_embs, n_train);
std::vector<std::uint8_t> codes(n_docs * pcfg.m);
pq.encode_batch(corpus_embs, n_docs, codes.data());

simeon::PQQuery q(pq, query_vec);
float score = q.inner_product(codes.data() + i * pcfg.m);  // O(m) per db code
```

## Status

Stable: tokenizer, hasher, three projection heads, L2 normalization, NEON + AVX2 + scalar dispatch, matryoshka, Product Quantization + ADC. Defaults and the public surface may still change.

## Citation

If you use simeon in research or benchmarks, cite the software via
[CITATION.cff](CITATION.cff), use [docs/research.md](docs/research.md) for
claim-to-document mapping, and use [docs/works_cited.md](docs/works_cited.md)
when you need the underlying prior-work list.

## License

GPL-3.0-or-later.
