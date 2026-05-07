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

**Embeddings**
- Three projection heads — Achlioptas sparse, very-sparse (Li 2006), dense Gaussian
- Matryoshka-style nested output — one projection, queryable at any prefix width with `1/sqrt(1 + r/decay)` row weights
- Product Quantization + ADC — 8–192× index compression with near-baseline R@10

**Retrieval**
- Stable retrieval core via [`<simeon/retrieval.hpp>`](include/simeon/retrieval.hpp) — BM25 variants, fusion helpers, query router
- 5 BM25 variants including the novel **SubwordAwareBackoff** (training-free morphological backoff via a parallel char-n-gram inverted index)
- Per-query router among `Bm25Atire` / `Bm25SabSmooth` / `CascadeLinearAlpha` — matches MiniLM-L6 nDCG@10 on BEIR scifact, training-free, no GPU
- Densified MinHash head (Shrivastava 2017) for Jaccard-space three-way fusion on duplicate-heavy corpora
- Opt-in extensions: BM25F, SDM/WSDM, RM3, concept mining (corpus-sensitive; not part of the default routed path)

**Quantization & tokenization**
- BPE-lite subword tokenizer with caller-supplied merges; no built-in vocab, deterministic across runs
- Three hash families — SplitMix64, XXH64, CRC32C, Mixed Tabulation (Houen & Thorup 2023)

**Build & correctness**
- SIMD kernels — aarch64 NEON, x86 AVX2+FMA, portable scalar
- Standalone Meson build, zero runtime deps, GPLv3
- Cross-arch byte-identity tests, SIMD-vs-scalar parity tests, determinism KATs

## Positioning

simeon captures **lexical and topical** structure, not paraphrase or semantic equivalence. It is **not** a drop-in replacement for a learned bi-encoder like MiniLM. The intended deployments are:

- First-stage ANN recall, with a learned reranker on top.
- BM25 ⊕ dense late-fusion (RRF), where lexical signal is what wins on hard / short queries.
- Self-contained retrieval where shipping a 200MB model is impractical (CLI tools, embedded, on-device).

In YAMS, simeon is used as the default retrieval embedding backend and lexical companion. ONNX remains reserved for opt-in plugin tasks such as GLiNER / ColBERT rather than the default embedding path.

simeon is best read as an engineering and evaluation layer over several lines of prior work: NUMEN-style training-free retrieval, ColBERT-era late-interaction framing, sparse random projection, matryoshka-style nested representations, Product Quantization, and classical query-difficulty routing. It is not a new retrieval algorithm. The shipped benchmark and router notes include negative findings, so this repository should not be read as blanket validation of any single upstream paper's headline claims.

## Corpus adapters and ceiling research

Recent ArguAna experiments in `docs/research/` show that the gap between a
training-free recipe and the BM25 candidate-pool oracle is often **structure
recognition**, not more vector similarity. On the full English ArguAna fixture,
BM25 reaches ~0.32 nDCG@10 while the BM25-top-100 oracle is ~0.92. Adding
corpus-specific structure progressively closes the gap:

| Stage | nDCG@10 | Scope |
|---|---:|---|
| BM25 baseline | ~0.32 | general lexical retrieval |
| text-neighborhood structure | ~0.44 | text-derived debate cluster |
| English pair discriminator | ~0.63 | English relation cues |
| topic-stem blended ranker | ~0.76 | ArguAna topic adapter |
| argument-point diagnostic | 1.00 | ArguAna schema diagnostic, not a product recipe |

The product lesson is: expose **corpus adapters** when structure exists — paths,
titles, sections, headings, issue IDs, citations, debate points — and keep them
separate from universal embedding claims. YAMS now has a first-class
`CorpusAdapter` component; its built-in native adapter performs English-first
query seeding for path/content fragments plus structured metadata filters, then
fuses the result as `corpus_adapter` evidence.

The ArguAna diagnostic rows are not shippable general retrieval algorithms;
they are evidence that structured adapters can unlock headroom already present
in the candidate set. They are no longer emitted by default from the reference
bench; set `SIMEON_ARGUANA_DIAGNOSTICS=1` only when reproducing the phase20–28
research notes.

PRs are welcome for additional English corpora, corpus adapters, language
profiles, and qrel-backed fixtures that let these findings transfer beyond
ArguAna.

The current theorem work now treats BM25 as one generator variable inside
`T=(G,A,S,F,B)`, not as the ceiling. BM25F and RM3 generator slices raise the
measured union oracle across the English fixtures, while Phase XXXVI shows that
qrel-free post-retrieval gate features recover part of the BM25/RM3 per-query
oracle on NFCorpus and TREC-COVID. Phase XXXVII adds the stricter constraint that
router rules must be corpus-agnostic: a shippable router may inspect query and
rank-neighborhood features, but not corpus IDs. Phase XXXVIII clarifies that
oracle rows are answer-key diagnostics, not callable system components. Phase
XXXIX adds rank-shape risk features; pooled robust gates reduce regressions, but
LOCO transfer remains weak. Phase XL turns those diagnostics into observed
score-level rows: shape-risk fusion improves macro nDCG@10 over BM25/static
z-fusion while avoiding the large SciFact regression. Phase XLI shows that simple
lexical-overlap reranking does not reduce the remaining ordering gap. Phase XLII
shows static central/lead fields are also not a general fix. Phase XLIII shows
structural features are currently better as risk sensors for shape-risk fusion
than as directly promoted fields. Phase XLIV adds simeon's own training-free
embedding as a generator slice. Standalone simeon is weak (0.2123 macro), but the
4-way union oracle (BM25 + BM25F + RM3 + simeon) rises to 0.8089 from 0.7848,
proving the embedding exposes different relevant documents. The observed 4-way
z-equal fusion reaches 0.4045 macro nDCG@10, a new best observed row, with the
largest gains on TREC-COVID (+0.0479) and NFCorpus (+0.0102). Phase XLV adds a
corpus-agnostic safety gate for the 4-way fusion: `bm25_entropy10 >= 0.48`
decides when BM25 is uncertain enough for the simeon leg to help safely. The
resulting `observed_4gen_risk_entropy_gate_devfit` reaches 0.4014 macro nDCG@10
with **non-negative deltas on every corpus** (worst −0.0007 on FiQA), making it
a safer shippable candidate than raw 4-way z-equal. Phase XLVI replaces the hard
gate with a continuous sigmoid dampening that transitions gradually as BM25
uncertainty rises, reaching 0.4041 macro (+0.0101 over BM25) with **positive
deltas on every corpus** (worst +0.0004 on ArguAna). The dampened row nearly
matches raw 4-way z-equal while eliminating its regressions entirely. Phase XLVII
finds that adding complementarity and simeon-confidence gates makes things worse:
simple entropy dampening is Pareto-optimal. Phase XLVIII shows consensus filtering
also fails. Phase XLIX makes the first successful scorer-level improvement:
embedding-weighted RM3 filters pseudo-relevance docs by simeon similarity,
achieving 0.4046 macro (+0.0106 over BM25) as a single generator — matching the
4-way dampening and fixing RM3's ArguAna regression (+0.0080 Δ vs standard RM3).
The remaining proof obligation is rich within-pool ordering and a safety gate
for the SciFact regression (−0.0154). Phase L swaps weighted RM3 into the 4-way
sigmoid dampening; the result is tied at 0.4041 macro — the improved scorer helps
NFCorpus but the additional generator signals dilute the gain elsewhere. The
clean conclusion: a single better scorer beats a fusion of weaker ones.

## Documentation

| Topic              | File                                                  |
|--------------------|-------------------------------------------------------|
| Build              | [docs/build.md](docs/build.md)                        |
| Benchmarks         | [docs/research/benchmarks.md](docs/research/benchmarks.md)              |
| Research notes     | [docs/research/index.md](docs/research/index.md)                  |
| Research saturation | [docs/research/training_free_saturation.md](docs/research/training_free_saturation.md) |
| Training-free ceiling | [docs/research/training_free_optimum.md](docs/research/training_free_optimum.md) |
| Training-free space | [docs/research/training_free_space_redefinition.md](docs/research/training_free_space_redefinition.md) |
| Generator-slice oracle | [docs/research/phase31_theorem_redefinition_plan.md](docs/research/phase31_theorem_redefinition_plan.md) |
| RM3 generator slice | [docs/research/phase32_rm3_generator_slice.md](docs/research/phase32_rm3_generator_slice.md) |
| Constructive union ranker | [docs/research/phase33_constructive_union_ranker.md](docs/research/phase33_constructive_union_ranker.md) |
| Weighted fusion rows | [docs/research/phase34_weighted_fusion_rows.md](docs/research/phase34_weighted_fusion_rows.md) |
| Query-adaptive diagnostic | [docs/research/phase35_query_adaptive_winner_diagnostic.md](docs/research/phase35_query_adaptive_winner_diagnostic.md) |
| Post-retrieval gate features | [docs/research/phase36_post_retrieval_gate_features.md](docs/research/phase36_post_retrieval_gate_features.md) |
| General router invariance | [docs/research/phase37_general_router_invariance.md](docs/research/phase37_general_router_invariance.md) |
| Oracle-external theorem | [docs/research/phase38_oracle_external_theorem.md](docs/research/phase38_oracle_external_theorem.md) |
| Rank-shape risk features | [docs/research/phase39_rank_shape_risk_features.md](docs/research/phase39_rank_shape_risk_features.md) |
| Observed shape-risk fusion | [docs/research/phase40_observed_shape_risk_fusion.md](docs/research/phase40_observed_shape_risk_fusion.md) |
| Ordering-gap lexical negative | [docs/research/phase41_ordering_gap_negative_lexical_evidence.md](docs/research/phase41_ordering_gap_negative_lexical_evidence.md) |
| Structural centrality rows | [docs/research/phase42_structural_centrality_negative.md](docs/research/phase42_structural_centrality_negative.md) |
| Structural-risk diagnostic | [docs/research/phase43_structural_risk_diagnostic.md](docs/research/phase43_structural_risk_diagnostic.md) |
| Simeon embedding generator slice | [docs/research/phase44_simeon_embedding_generator_slice.md](docs/research/phase44_simeon_embedding_generator_slice.md) |
| Risk-aware 4-generator fusion | [docs/research/phase45_risk_aware_4gen_fusion.md](docs/research/phase45_risk_aware_4gen_fusion.md) |
| Continuous 4-generator dampening | [docs/research/phase46_continuous_dampening.md](docs/research/phase46_continuous_dampening.md) |
| Dynamic multi-signal dampening (negative) | [docs/research/phase47_dynamic_dampening_negative.md](docs/research/phase47_dynamic_dampening_negative.md) |
| Cross-generator consensus booster (negative) | [docs/research/phase48_consensus_booster_negative.md](docs/research/phase48_consensus_booster_negative.md) |
| Embedding-weighted RM3 query expansion | [docs/research/phase49_embedding_weighted_rm3.md](docs/research/phase49_embedding_weighted_rm3.md) |
| Weighted RM3 in 4-way fusion | [docs/research/phase50_weighted_rm3_4way_fusion.md](docs/research/phase50_weighted_rm3_4way_fusion.md) |
| Adaptive α weighted RM3 | [docs/research/phase51_adaptive_alpha_rm3.md](docs/research/phase51_adaptive_alpha_rm3.md) |
| Diversity-aware MMR RM3 | [docs/research/phase52_diversity_aware_rm3.md](docs/research/phase52_diversity_aware_rm3.md) |
| MMR β sensitivity sweep | [docs/research/phase53_beta_sensitivity_sweep.md](docs/research/phase53_beta_sensitivity_sweep.md) |
| Query-type gate + diverse RM3 fusion | [docs/research/phase54_query_type_gate.md](docs/research/phase54_query_type_gate.md) |
| Embedding-based scoring (negative) | [docs/research/phase55_embedding_scoring_negative.md](docs/research/phase55_embedding_scoring_negative.md) |
| Gated 3-way ensemble | [docs/research/phase56_gated_ensemble.md](docs/research/phase56_gated_ensemble.md) |
| PPR graph re-ranking | [docs/research/phase57_ppr_graph_rerank.md](docs/research/phase57_ppr_graph_rerank.md) |
| Theorem correction (LM framework) | [docs/research/theorem_correction_lm_framework.md](docs/research/theorem_correction_lm_framework.md) |
| Corpus/language adapters | [docs/research/language_corpus_support.md](docs/research/language_corpus_support.md) |
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

Stable: tokenizer, hasher, three projection heads, L2 normalization, NEON + AVX2 + scalar dispatch, matryoshka, Product Quantization + ADC, and the retrieval core in [`<simeon/retrieval.hpp>`](include/simeon/retrieval.hpp) (BM25 variants + fusion + query router).

Opt-in / corpus-sensitive: BM25F auxiliary fields, SDM/WSDM, RM3, concept mining, and the other research surfaces documented under [docs/research/](docs/research/). Defaults and the public surface may still change.

## Citation

If you use simeon in research or benchmarks, cite the software via
[CITATION.cff](CITATION.cff), use [docs/research/index.md](docs/research/index.md) for
claim-to-document mapping, and use [docs/works_cited.md](docs/works_cited.md)
when you need the underlying prior-work list.

## License

GPL-3.0-or-later.
