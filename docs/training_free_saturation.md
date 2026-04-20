# Training-free literature saturation

This doc is the artifact answer to "did we squeeze all juice from the
related literature?" It groups every thread simeon has touched into three
tables: **shipped**, **closed with negative result**, and **declined**.
Open residuals at the bottom.

Verdict as of 2026-04-20: ~85–90% of the training-free literature juice
against scifact-style corpora has been extracted. Remaining threads are
either non-lever (documentation-only variance bounds), out-of-scope for
simeon's exhaustive-cosine design (ANN / LSH), or obsoleted by a prior
shipped result.

## Shipped

| Literature result                                     | Realization in simeon                                             |
|-------------------------------------------------------|-------------------------------------------------------------------|
| Robertson BM25 (Atire IDF)                            | `Bm25Variant::Atire` — [include/simeon/bm25.hpp](../include/simeon/bm25.hpp) |
| Lv & Zhai 2011a — BM25+ floor                         | `Bm25Variant::BM25Plus`                                           |
| Lv & Zhai 2011b — BM25L long-doc form                 | `Bm25Variant::BM25L`                                              |
| Amati DFR DLH13                                       | `Bm25Variant::DLH13`                                              |
| Madsen-Kauchak-Elkan 2005 — word burstiness / DCM     | `Bm25Variant::Dcm` (Zhai-Lafferty Dirichlet-LM form, log(1 + tf·T / (α·ttf))) |
| Novel training-free — SubwordAwareBackoff             | `Bm25Variant::SubwordAwareBackoff` (parallel char-n-gram inverted index) |
| Cormack RRF fusion                                    | `rrf_fuse` — [src/fusion.cpp](../src/fusion.cpp)                  |
| z-scored linear-α fusion                              | `linear_alpha_fuse`                                               |
| Carmel & Yom-Tov 2010 — pre-retrieval predictor family | `QueryFeatures` / `QueryRouter` — [include/simeon/query_router.hpp](../include/simeon/query_router.hpp) |
| Cronen-Townsend 2002, Hauff 2008 — post-retrieval-lite | `features_with_pool()` (score decay, entropy, pool-Jaccard)      |
| Achlioptas sparse random projection                   | `ProjectionMode::AchlioptasSparse` — [src/projection.cpp](../src/projection.cpp) |
| Li 2006 very-sparse Gaussian                          | `ProjectionMode::VerySparse`                                      |
| Kane-Nelson sparse-JL                                 | `ProjectionMode::SparseJL`                                        |
| Ailon-Chazelle / Nelson-Nguyên FWHT                   | `ProjectionMode::Fwht`, `OsnapFwht` — [src/projection.cpp](../src/projection.cpp) |
| Levy-Goldberg 2014 PMI factorization                  | `PmiEmbeddings` — [src/pmi.cpp](../src/pmi.cpp) (**negative result**, see below) |
| Halko-Martinsson-Tropp randomized SVD                 | Used inside `PmiEmbeddings::learn()`                              |
| Kusupati 2022 Matryoshka (training-free analog)       | `matryoshka_decay`, data-aware weights — [src/projection.cpp](../src/projection.cpp) |
| Shrivastava 2017 densified MinHash                    | `MinHashEncoder` — [src/minhash.cpp](../src/minhash.cpp)          |
| Jégou-Douze-Schmid 2010 Product Quantization          | `ProductQuantizer` — [src/pq.cpp](../src/pq.cpp)                  |
| Sennrich 2016 / Gage 1994 BPE (training-free)         | `BpeMerges` — [src/tokenizer_bpe.cpp](../src/tokenizer_bpe.cpp)   |
| Houen-Thorup 2023 Mixed Tabulation                    | `HashFamily::MixedTabulation` — [src/hasher.cpp](../src/hasher.cpp) |

## Closed with negative result

| Thread                                     | Finding                                                  | Pointer                                      |
|--------------------------------------------|----------------------------------------------------------|----------------------------------------------|
| PMI projection on scifact                  | Standalone rank-256 PMI = 0.251 nDCG@10 (target 0.50–0.55). Mechanism failure: scifact wins are contextual + multi-word-term. | [docs/pmi_projection.md](pmi_projection.md)  |
| Step 1f df-only router enrichment          | `min_idf` / `idf_stddev` saturated on scifact; realized +0.005 nDCG@10 on test fold. | [docs/router_design.md](router_design.md)    |
| SAB-pool cascade on semantic corpora       | Scifact headline cascade fell to 0.303 on FiQA vs BM25-alone 0.226 — cascade helps but tuning transfer is weak. | [docs/benchmarks.md](benchmarks.md) FiQA section |

## Declined

| Thread                                    | Why declined                                                             |
|-------------------------------------------|--------------------------------------------------------------------------|
| PL2 / DPH / IFB2 (DFR family)             | DLH13 already ablated the DFR family; no clear quality lever from more variants |
| Andoni-Razenshteyn 2015 data-dependent LSH | simeon uses exhaustive cosine by design; no ANN index to host the structure |
| Count-Min / HyperLogLog / Cuckoo          | Postings compression — scifact 5K, FiQA 57K, NFCorpus 3.6K all too small to benefit |
| Cohen-Nelson-Woodruff 2016 sparse-JL      | Kane-Nelson ε=0.10 already shipped at 0.385 nDCG; incremental tightening only |
| Weinberger 2009 hash kernels              | Variance bounds are documentation-only; no shipped metric lever          |
| Carmel & Yom-Tov catalog completion       | IDF-family predictors saturated per Step 1f; remaining candidates need a 3rd corpus |
| Quora-QQP PMI                             | PMI mechanism failure is unigram-vs-contextual, not corpus-specific — re-run adds no information |
| Fang-Zhai 2005 axiomatic-IR for SAB       | Proof exercise; no code lever. Candidate research note, not code.       |
| Labeled-data / learned routing            | Violates simeon's training-free contract                                 |
| Dense bi-encoders (DPR, BERT, MiniLM)     | Not training-free. simeon's comparison point, not an import target.     |
| Cross-encoders / late-interaction rerank  | Require a model; out of scope                                            |

## Open residuals

1. **Per-corpus router tuner.** Three-corpus evidence (Step 1j Closure A) confirmed the need: scifact `router_default_4096_768` holds above BM25-alone on all three corpora but falls short of the per-corpus oracle by ~5 points on NFCorpus and ~4 points on FiQA. Promote to a follow-on product step.
2. **Axiomatic-IR formalization of SubwordAwareBackoff** (Fang-Zhai 2005). Research paper material. Optional companion `docs/sab_axioms.md` if/when pursued; low priority vs further benchmark work.

## Three-corpus transfer summary (2026-04)

| Finding                                         | Scifact  | FiQA     | NFCorpus | Verdict           |
|-------------------------------------------------|----------|----------|----------|-------------------|
| SAB-smooth standalone matches/beats BM25-Atire  | +0.00 (ties) | −0.01 | +0.05    | Corpus-agnostic on morphology |
| SAB-smooth matches MiniLM-L6 nDCG@10            | +0.00 (0.612 vs 0.641) | −0.15 | **+0.001 (match)** | Scientific-text win |
| Scifact-tuned `router_default` beats BM25-alone | +0.02    | +0.00    | +0.02    | Transfers directionally |
| Scifact-tuned `router_default` reaches oracle   | −0.05    | −0.04    | −0.06    | Per-corpus tuning needed |
| SAB→simeon-cos cascade beats BM25-alone         | −0.21    | −0.09    | −0.06    | **Scifact-specific; do not ship elsewhere** |
