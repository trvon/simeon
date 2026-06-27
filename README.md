# simeon

Training-free SIMD text embeddings. Deterministic, model-free `text -> vector`
for dense retrieval. NEON/AVX2 kernels. Single-digit microseconds per doc.

![license](https://img.shields.io/badge/license-GPL--3.0--or--later-blue?style=flat-square)
![language](https://img.shields.io/badge/language-C%2B%2B20-informational?style=flat-square)
![isa](https://img.shields.io/badge/SIMD-NEON%20%7C%20AVX2%20%7C%20scalar-success?style=flat-square)

> [!WARNING]
> Experimental. API and defaults may change. `simeon` is not learned semantic encoder.

## What It Is

`simeon` maps UTF-8 text to fixed-width float vectors with no learned model.

```text
text -> char/word n-grams -> hashed count sketch -> optional random projection
     -> optional matryoshka weights -> L2 normalize -> optional PQ code
```

Same input + same config + same seed = same bytes across supported architectures.

The retrieval side composes these primitives into a first-pass + rerank pipeline.
A first-pass BM25 (SAB-smooth by default) produces the candidate scores; a routed
recipe then either returns them, expands them with RM3 pseudo-relevance feedback,
or reranks with fragment-geometry topology:

```mermaid
flowchart LR
    D[docs] --> IDX[Bm25Index<br/>SAB-smooth and variants]
    D --> FR[semantic fragments<br/>PMI embeddings]
    Q[query] --> FP[first-pass BM25]
    IDX --> FP
    FP --> R{routed recipe}
    R -->|lexical| OUT[ranked docs]
    R -->|RM3 PRF| PRF[score_with_prf<br/>forward-index RM1] --> OUT
    R -->|topology rerank| GEO[fragment-geometry<br/>PHSS LargestGapApprox plus outer-MaxSim] --> OUT
    FR --> GEO
```

## What Ships

- Embedding core: Achlioptas sparse, very-sparse JL, dense Gaussian.
- Nested output: matryoshka-style prefix use from one projection.
- Compression: Product Quantization + ADC.
- Retrieval core: BM25 variants, fusion helpers, query router.
- Structured retrieval hooks: corpus adapters.
- Runtime: aarch64 NEON, x86 AVX2+FMA, scalar fallback.
- Build: C++20, Meson, zero runtime deps.

## Positioning

- Good fit: first-stage recall, lexical+dense fusion, model-free or on-device retrieval.
- Not good fit: paraphrase or semantic equivalence. Not drop-in replacement for MiniLM-class bi-encoder.
- Benchmarks, router notes, ablations, and negative results live in [docs/research.md](docs/research.md).

## Build

```sh
meson setup build
meson compile -C build
meson test -C build
```

Requires C++20, Meson, Ninja. Full build options: [docs/build.md](docs/build.md).

## Quick Start

```cpp
#include <simeon/simeon.hpp>

simeon::EncoderConfig cfg;
cfg.sketch_dim = 4096;
cfg.output_dim = 384;
cfg.projection = simeon::ProjectionMode::AchlioptasSparse;

simeon::Encoder enc(cfg);
std::vector<float> out(enc.output_dim());
enc.encode("training-free text embeddings", out.data());
```

Public headers:

- `<simeon/simeon.hpp>`: encoder
- `<simeon/pq.hpp>`: PQ + ADC
- `<simeon/retrieval.hpp>`: BM25, fusion, router
- `<simeon/corpus_adapter.hpp>`: structured corpus hooks

## Docs

- Build: [docs/build.md](docs/build.md)
- Research notes: [docs/research.md](docs/research.md)
- Works cited: [docs/works_cited.md](docs/works_cited.md)

## Status

- Stable surface: tokenizer, hashing, projection heads, normalization, SIMD dispatch, matryoshka, PQ, retrieval core.
- Opt-in surface: BM25F, SDM/WSDM, RM3 pseudo-relevance feedback, concept mining, fragment-geometry rerank.

## Citation

Cite software via [CITATION.cff](CITATION.cff). Use [docs/research.md](docs/research.md) for claim-to-document mapping.

## License

GPL-3.0-or-later.
