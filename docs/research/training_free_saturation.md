# Training-free literature saturation

This doc is the artifact answer to "did we squeeze all juice from the
related literature?" It groups every thread simeon has touched into three
tables: **shipped**, **closed with negative result**, and **declined**.
Open residuals at the bottom.

Verdict as of 2026-04-20: ~85–90% of the training-free literature juice
against scifact-style corpora has been extracted. Remaining threads are
either non-lever (documentation-only variance bounds), out-of-scope for
simeon's exhaustive-cosine design (ANN / LSH), or obsoleted by a prior
shipped result. Step 1m closed the genuine-innovation entropy-α frontier as
a documented null-result-with-safety-property. Step 1n closed the
latent-concept-model frontier as a pure null (regression across every
corpus × every base variant × every weight). Remaining innovation gaps
(Gaps 2–5 in the Step 1n plan) are deferred until a corpus arrives where
the surviving paraphrase-gap mechanism admits a corpus-statistic lever.

## Shipped

| Literature result                                     | Realization in simeon                                             |
|-------------------------------------------------------|-------------------------------------------------------------------|
| Robertson BM25 (Atire IDF)                            | `Bm25Variant::Atire` — [include/simeon/bm25.hpp](../include/simeon/bm25.hpp) |
| Lv & Zhai 2011a — BM25+ floor                         | `Bm25Variant::BM25Plus`                                           |
| Lv & Zhai 2011b — BM25L long-doc form                 | `Bm25Variant::BM25L`                                              |
| Amati DFR DLH13                                       | `Bm25Variant::DLH13`                                              |
| Amati & van Rijsbergen 2002 — PL2 (DFR Poisson)       | `Bm25Variant::PL2` (Step 1k B1)                                   |
| Amati 2007 — DPH (DFR hypergeometric)                 | `Bm25Variant::DPH` (Step 1k B1)                                   |
| Madsen-Kauchak-Elkan 2005 — word burstiness / DCM     | `Bm25Variant::Dcm` (Zhai-Lafferty Dirichlet-LM form, log(1 + tf·T / (α·ttf))) |
| Novel training-free — SubwordAwareBackoff             | `Bm25Variant::SubwordAwareBackoff` (parallel char-n-gram inverted index) |
| Lavrenko & Croft 2001 — RM3 pseudo-relevance feedback | `score_with_prf()` — [src/prf.cpp](../src/prf.cpp) (Step 1k A1)   |
| Cormack RRF fusion                                    | `rrf_fuse` — [src/fusion.cpp](../src/fusion.cpp)                  |
| z-scored linear-α fusion                              | `linear_alpha_fuse`                                               |
| Carmel & Yom-Tov 2010 — pre-retrieval predictor family | `QueryFeatures` / `QueryRouter` — [include/simeon/query_router.hpp](../include/simeon/query_router.hpp) |
| Zhao, Scholer, Tsegay 2008 — Sum-SCQ                  | `QueryFeatures::scq_sum`, `atire_min_scq` gate (Step 1k B2)       |
| Cronen-Townsend & Croft 2002 — simplified clarity     | `QueryFeatures::simplified_clarity`, `atire_max_clarity` gate (Step 1k B2) |
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
| Metzler & Croft 2005 — Sequential Dependence Model    | `score_sdm()` with parallel word-bigram postings (Step 1l, opt-in via `build_word_bigrams=true`) — [src/bm25.cpp](../src/bm25.cpp) |

## Closed with negative result

| Thread                                     | Finding                                                  | Pointer                                      |
|--------------------------------------------|----------------------------------------------------------|----------------------------------------------|
| PMI projection on scifact                  | Standalone rank-256 PMI = 0.251 nDCG@10 (target 0.50–0.55). Mechanism failure: scifact wins are contextual + multi-word-term. | [docs/pmi_projection.md](pmi_projection.md)  |
| Step 1f df-only router enrichment          | `min_idf` / `idf_stddev` saturated on scifact; realized +0.005 nDCG@10 on test fold. | [docs/router_design.md](router_design.md)    |
| SAB-pool cascade on semantic corpora       | Scifact headline cascade fell to 0.303 on FiQA vs BM25-alone 0.226 — cascade helps but tuning transfer is weak. | [docs/benchmarks.md](benchmarks.md) FiQA section |
| Step 1l SDM on FiQA (Metzler 2005)         | `bm25_atire_sdm_l0.85_0.10_0.05` = +0.006 nDCG@10 vs Atire (target +0.015 to +0.040). Bigram signal is real but below promote threshold; Metzler defaults beat unigram-heavier weights. Infrastructure shipped behind `build_word_bigrams=false` default (zero runtime cost). | [docs/sdm_results.md](sdm_results.md) |
| Step 1m entropy-α fusion (simeon novel)    | `bm25_pool500_entropy_alpha_4096_768` = −0.006 to −0.010 nDCG@10 vs matched static α=0.75 across all three corpora. Self-correcting safety property kept (rescues +0.07 to +0.21 vs pure cosine), so infrastructure ships in `simeon::` namespace; not wired into router recipes. | [docs/fusion_entropy_alpha.md](fusion_entropy_alpha.md) |
| Step 1n latent-concept mining (Bendersky 2008) | `bm25_atire_concepts_l0.50` regresses −0.108 / −0.011 / −0.068 nDCG@10 on scifact / NFCorpus / FiQA vs Atire; lower weight (l0.25) halves but doesn't close the gap. SAB-smooth base is ~neutral on scifact (+0.002) but still negative on NFCorpus (−0.003) and FiQA (−0.016). PMI × BM25-bigram score scale dominates base BM25 on matched queries, reordering around exact-phrase hits — actively harmful on paraphrase-bound FiQA. Infrastructure ships opt-in in `simeon::` namespace; not wired into router. | [docs/concept_mining.md](concept_mining.md) |

## Declined

| Thread                                    | Why declined                                                             |
|-------------------------------------------|--------------------------------------------------------------------------|
| IFB2 (DFR family completion)              | PL2 / DPH / DLH13 bracket the DFR family's scifact+NFCorpus behavior (0.590–0.600 / 0.249); IFB2 adds no measurement lever |
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
| Step 1k `atire_max_clarity=3.0` gate lift vs `router_default` | +0.00 (no-op) | **+0.006 (0.202→0.208)** | **+0.028 (0.270→0.298)** | Morphology-heavy corpora win; scifact-neutral |
| Step 1k RM3 lift vs base BM25 variant           | −0.015 (Atire) / +0.018 (SAB-smooth) | −0.009 (Atire) / −0.002 (SAB-smooth) | +0.019 (Atire) / −0.012 (SAB-smooth) | Per-corpus, per-variant; not a universal lever |
| PL2 / DPH (DFR family completion)               | 0.598 / 0.600 (below Atire 0.619) | 0.189 / 0.200 (below Atire 0.205) | 0.249 (tied DLH13) | DFR internally consistent, caps below tuned Robertson |
| Step 1l SDM lift vs base BM25 variant (Metzler 0.85/0.10/0.05) | −0.007 (Atire) / −0.00 (SAB-smooth) | **+0.006 (Atire)** / +0.008 (SAB-smooth, 0.198→0.206) | +0.001 (Atire) / +0.00 (SAB-smooth) | FiQA-only signal; below +0.010 promote threshold. Infra shipped opt-in, not routed |
| Step 1m entropy-α vs matched static α=0.75 (Atire pool 500, 4096→768) | −0.006 (0.625→0.619) | −0.006 (0.211→0.205) | −0.010 (0.261→0.252) | Negative on quality; safety rail vs pure cosine: +0.21 / +0.09 / +0.07 |
| Step 1n concepts l=0.50 vs base (Atire / SAB-smooth) | **−0.108 (Atire)** / +0.002 (SAB) | **−0.068 (Atire)** / −0.016 (SAB) | **−0.011 (Atire)** / −0.003 (SAB) | Null across the board; exact-phrase reward overweights base BM25. Infra opt-in in `simeon::`, not routed |
