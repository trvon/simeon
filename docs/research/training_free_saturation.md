# Training-free literature saturation

This doc answers the narrow question: which literature-grounded, training-free
threads were shipped, disproved, or deliberately declined?

Verdict as of 2026-04-24: **BEIR-3 training-free track is saturated**. The
remaining literature-grounded directions identified in
`literature_synthesis.md` have now all been either shipped or closed:

- Bottleneck 1 (multi-fragment averaging) — MaxSim probe disproved by
  cross-fold (`phss_maxsim_results.md`).
- Bottleneck 2 (BM25-pool R@100 ceiling / per-corpus α) — disproved by
  cross-fold (`phss_alpha_sweep_results.md`).
- Bottleneck 3 (per-corpus routing) — subsumed by Bottleneck 2.
- Bottleneck 4 (Dictionary axis: AhoCorasick + TextRank fields) —
  disproved on BEIR-3 (`ac_entity_results.md`,
  `textrank_title_results.md`).

Production frontier: `phssapprox_k100_t8_richcov_gap` — scifact 0.6188
(= BM25), nfcorpus 0.2544 (+0.0023), fiqa 0.2089 (+0.0036) at ~120 QPS. The
next increment now requires either a new corpus or leaving the training-free
constraint.

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
| Mihalcea & Tarau 2004 — TextRank                      | `simeon::TextRank` — [include/simeon/text_rank.hpp](../include/simeon/text_rank.hpp) (sentence ranking, used in `build_doc_semantic_fragments`) |
| Aho & Corasick 1975 — multi-pattern dictionary match  | `simeon::AhoCorasick` + double-array variant — [include/simeon/aho_corasick.hpp](../include/simeon/aho_corasick.hpp) |
| Edelsbrunner-Letscher-Zomorodian 0D persistent homology | `simeon::phss_select_scale` — [src/persistent_homology.cpp](../src/persistent_homology.cpp). LargestGap criterion + LargestGapApprox fast path (validated B6) |
| Santhanam et al. 2022 ColBERTv2 MaxSim aggregation (training-free analog) | `FragmentGeometryConfig::DocAggregator` enum (Sum / Max) — [src/fragment_geometry.cpp](../src/fragment_geometry.cpp) (no-op-safe at default Sum; **negative result** when promoted, see below) |
| Bruch & Gai 2023 convex-fusion analysis               | Already-shipping convex blend `α·z(BM25) + (1−α)·z(geom)` at `fragment_geometry.cpp:864`, validated as the right form by Bruch-Gai (per-corpus α tuning attempted, **negative result**, see below) |
| PHSS `LargestGapApprox` as production default         | Promoted 2026-04-24 per B6 cross-fold validation. `PhssConfig::criterion` default now `LargestGapApprox` — ~2× QPS at equal-or-better quality on richcov across 3/3 corpora vs heavy LargestGap. See `docs/research/phss_largest_gap_approx_results.md`. |
| Corpus-as-self-KB candidate-set expansion (nfcorpus)  | `FragmentGeometryConfig::doc_doc_neighbors` span + `selfkb_*` knobs (cap/filter/gate). Validated on nfcorpus cross-fold (+0.004 nDCG / +0.027 R@100 both folds, `n10_gate25` recipe). Corpus-specific; scifact and fiqa stay on the default production frontier. See `docs/research/self_kb_results.md`. |
| RRF of 5 BM25 variants (nfcorpus)                     | `simeon::score_bm25_variants_rrf` — [include/simeon/fusion.hpp](../../include/simeon/fusion.hpp). Fuses Atire + BM25+ + BM25L + DLH13 + SAB-smooth rankings via Cormack-style RRF (k=60). Validated on nfcorpus cross-fold (+0.005/+0.009 nDCG both folds vs production frontier `phssapprox_k100_t8_richcov_gap`) at 1300 QPS — ~500× cheaper than other validated nfcorpus levers. Scifact and fiqa fold-flip. See `docs/research/rrf_variants_results.md`. |

## Closed with negative result

| Thread                                     | Finding                                                  | Pointer                                      |
|--------------------------------------------|----------------------------------------------------------|----------------------------------------------|
| PMI projection on scifact                  | Standalone rank-256 PMI = 0.251 nDCG@10 (target 0.50–0.55). Mechanism failure: scifact wins are contextual + multi-word-term. | [docs/pmi_projection.md](pmi_projection.md)  |
| Step 1f df-only router enrichment          | `min_idf` / `idf_stddev` saturated on scifact; realized +0.005 nDCG@10 on test fold. | [docs/router_design.md](router_design.md)    |
| SAB-pool cascade on semantic corpora       | Scifact headline cascade fell to 0.303 on FiQA vs BM25-alone 0.226 — cascade helps but tuning transfer is weak. | [docs/benchmarks.md](benchmarks.md) FiQA section |
| Step 1l SDM on FiQA (Metzler 2005)         | `bm25_atire_sdm_l0.85_0.10_0.05` = +0.006 nDCG@10 vs Atire (target +0.015 to +0.040). Bigram signal is real but below promote threshold; Metzler defaults beat unigram-heavier weights. Infrastructure shipped behind `build_word_bigrams=false` default (zero runtime cost). | [docs/sdm_results.md](sdm_results.md) |
| Step 1m entropy-α fusion (simeon novel)    | `bm25_pool500_entropy_alpha_4096_768` = −0.006 to −0.010 nDCG@10 vs matched static α=0.75 across all three corpora. Self-correcting safety property kept (rescues +0.07 to +0.21 vs pure cosine), so infrastructure ships in `simeon::` namespace; not wired into router recipes. | [docs/fusion_entropy_alpha.md](fusion_entropy_alpha.md) |
| Step 1n latent-concept mining (Bendersky 2008) | `bm25_atire_concepts_l0.50` regresses −0.108 / −0.011 / −0.068 nDCG@10 on scifact / NFCorpus / FiQA vs Atire; lower weight (l0.25) halves but doesn't close the gap. SAB-smooth base is ~neutral on scifact (+0.002) but still negative on NFCorpus (−0.003) and FiQA (−0.016). PMI × BM25-bigram score scale dominates base BM25 on matched queries, reordering around exact-phrase hits — actively harmful on paraphrase-bound FiQA. Infrastructure ships opt-in in `simeon::` namespace; not wired into router. | [docs/concept_mining.md](concept_mining.md) |
| T1 NQC predictor correlation (Shtok-Kurland-Carmel 2012) | Spearman ρ vs oracle = 0.16 / 0.30 / 0.16 (scifact / nfcorpus / fiqa). Below the +0.4 routing-utility gate on all 3. Routing-discriminator ρ ≈ 0 — pre-disproved T6 learned-router. | [docs/qpp_post_retrieval_results.md](qpp_post_retrieval_results.md) |
| T2 Full WIG vs WIG-lite (Zhou-Croft 2007) | Full Zhou-Croft WIG underperforms shipped WIG-lite (top-drop decay) on 2/3 corpora. Stop investing in this signal family. | [docs/qpp_post_retrieval_results.md](qpp_post_retrieval_results.md) |
| T3 Weighted SDM (Bendersky-Croft 2010 IDF reweighting) | Best Atire WSDM Δ across β sweep: scifact +0.0015, nfcorpus +0.0009, fiqa **−0.0020** at canonical β=1.0. Three different β-monotonicity shapes across corpora rule out useful default. | [docs/wsdm_results.md](wsdm_results.md) |
| T4 Adaptive RM3 K(clarity) (Bendersky-Metzler-Croft 2011) | Adaptive K vs n=20 baseline ΔR@100 = {−0.24, −0.09, −0.45}pp across 3 corpora. Clarity saturates n_max anchor on 2/3 corpora; optimal K direction corpus-dependent. | [docs/rm3_adaptive_results.md](rm3_adaptive_results.md) |
| T5 Axiomatic LTD (Fang-Zhai 2005) | Plan's "FiQA is long-doc" premise factually wrong (avg_dl 132.9 — shortest of 3). LTD α<1 helps R@100 on long-doc scifact / nfcorpus but hurts fiqa nDCG by −0.0171 at α=0.5. Plan validate gate rejected; corpus-bound recall-track lever subfinding (LTD α=0.7 on long-doc avg_dl > 150). | [docs/ltd_results.md](ltd_results.md) |
| PHSS pool_size scaling (Phase C) | pool=100 is the optimum. scifact and fiqa regress nDCG@10 at pool=200/500 (richcov: −0.005 to −0.012). Subfinding: nfcorpus pool=500 R@100 +0.0122 is real but at 5 QPS (latency-prohibitive). | [docs/research/phss_pool_scaling.md](phss_pool_scaling.md) |
| PHSS-1D triangle weighting (cycle proxy probe) | nDCG@10 delta range [−0.0005, +0.0004] across 18 cells (3 corpora × 3 alpha × 2 placements). R@10/R@100 byte-identical. Multi-fragment per-doc averaging washes per-fragment weighting. Phases B (witness 1D) + C (full Ripser) cancelled per disprove gate. | [docs/research/phss_1d_triangle_results.md](phss_1d_triangle_results.md) |
| MaxSim aggregation (training-free ColBERTv2 analog) | Test-fold scifact richcov +0.0028 nDCG@10 inverted on dev fold (−0.0045). 3 cells flip sign between folds; 3 consistently regress; **0 consistently improve**. Disproved at the population level. `DocAggregator` enum stays in tree (no-op at default Sum). | [docs/research/phss_maxsim_results.md](phss_maxsim_results.md) |
| Per-corpus α sweep (Bruch & Gai 2023) | Dev winners α=0.85 for nfcorpus/fiqa don't transfer to test (lose −0.0002 / −0.0019 vs default α=0.80). Cross-fold validation gate not met on 2/3 corpora. α=0.80 universal default confirmed. Bruch-Gai's "sample efficient" assumption fails because BEIR-3 dev folds have 5+ point absolute baseline gaps to test. | [docs/research/phss_alpha_sweep_results.md](phss_alpha_sweep_results.md) |
| RRF-5 as universal default | Lifts bm25_atire baseline cross-fold on 2/3 corpora (scifact +0.004, nfcorpus +0.007-0.009), but only nfcorpus cross-fold-validates vs production frontier `phssapprox_k100_t8_richcov_gap`. scifact/fiqa flip between dev/test folds. Shipped as corpus-specific library API for nfcorpus-style corpora; not a universal default. | [docs/research/rrf_variants_results.md](rrf_variants_results.md) |
| Plan 4 single-fragment-per-doc (argmax before softmax) | 0 of 6 (corpus × builder) cells cross-fold lift nDCG ≥0.005. Three cells flip sign between folds. Mechanism finding: production-frontier attention_scale=8 softmax already does soft-argmax (~80% mass on best fragment); hard argmax only redistributes 20% which gets further smoothed by diffusion + α=0.8 blend. Combined with MaxSim and PHSS-1D triangle disproofs, Bottleneck 1 (multi-fragment averaging) is three-way closed. | [docs/research/plan4_singlefrag_results.md](plan4_singlefrag_results.md) |
| AhoCorasick self-bootstrapped entity field (BM25F) | Best fiqa +0.0026 at w=0.2 (below +0.005 promote bar); scifact / nfcorpus flat-to-negative. Self-bootstrapped dictionary produces tiny lexical signal; needs external domain dict (UMLS / Wikidata) or structured-doc corpus. | [docs/research/ac_entity_results.md](ac_entity_results.md) |
| TextRank synthetic-title field (BM25F) | Every (corpus × weight) cell regresses: scifact −0.0135, nfcorpus −0.0050, fiqa −0.0029 at best. Top-1 sentence isn't acting like a useful title field on body-only short docs. Needs structured-document corpus. | [docs/research/textrank_title_results.md](textrank_title_results.md) |
| T6 Learned router feasibility (gated by Phase A) | Skipped — Phase A routing-discriminator ρ ≈ 0 pre-disproved cheap-features LR closing the predictor→oracle gap. | [docs/qpp_post_retrieval_results.md](qpp_post_retrieval_results.md) |

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

All residuals below require either a new corpus or an off-training-free move.
There are no remaining within-scope universal levers on the BEIR-3 fixture set.

1. **New corpus: trec-covid (avg_dl ~290) for true long-doc test.**
   T5 LTD's "FiQA is long-doc" premise was factually wrong; LTD's
   recall lift on long-doc corpora (scifact/nfcorpus, +0.005 R@100
   at α=0.7) is real but corpus-bound to the longer end of BEIR-3's
   range. trec-covid would test the prediction at the actual long
   end. Also tests PHSS pool=500 R@100 hypothesis at a corpus where
   long fragments may actually structure into 1D loops. Out of scope
   for current fixture set; explicit user decision required (~1-2
   day fixture-engineering effort: corpus.tsv + qrels +
   reference.bin).
2. **New corpus: structured-document fixture for dictionary axis.**
   Both `ac_entity_results.md` and `textrank_title_results.md` reach
   identical conclusions: AhoCorasick + TextRank fields fail because
   BEIR-3 docs are body-only short text with no genuine title /
   entity-tag structure. A structured-doc corpus (e.g., yams's own
   file-store with paths-as-titles) would let the
   GLiNER-replacement primitives be exercised in their designed
   regime. Out of scope per "simeon primitives first, yams
   integration second" memory rule.
3. **1D persistent homology on a richer fragment substrate.** The
   triangle-count proxy disproved (`phss_1d_triangle_results.md`),
   but the mechanism finding was: multi-fragment per-doc averaging
   washes per-fragment weighting. A fragment substrate where each
   doc has only 1 anchor fragment (instead of richcov's 8) would
   change the averaging math. Open research question; no current
   plan.
4. **Per-corpus router tuner** (legacy residual). Three-corpus
   evidence (Step 1j Closure A) confirmed the need: scifact
   `router_default_4096_768` holds above BM25-alone on all three
   but falls short of per-corpus oracle by ~5 / ~4 nDCG points.
   Promotion to product step deferred indefinitely — Phase A QPP
   work showed the cheap-features routing-discriminator ρ ≈ 0,
   meaning even with this tuner the gap-closure ceiling is
   information-bound not feature-bound.
5. **Axiomatic-IR formalization of SubwordAwareBackoff**
   (Fang-Zhai 2005). Research paper material; no code lever.
   Optional companion `docs/sab_axioms.md` if pursued; low priority.

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
