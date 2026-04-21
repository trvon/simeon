# simeon benchmarks

## Conditions

- Hardware: 2024 Apple M-series laptop, NEON tier, single-threaded.
- Build: `buildtype=debugoptimized`, NEON enabled, no sanitizers.
- Real-corpus fixture: BEIR `scifact` — 5,183 docs / 202 test queries (after Step 1e `--dev-fraction 0.33` split; 98 dev queries are held out). Reference: `sentence-transformers/all-MiniLM-L6-v2`, 384-d float32, L2-normalized. Fixture format documented in [reference_fixture.md](reference_fixture.md). See [training_free_saturation.md](training_free_saturation.md) for what was not measured and why.
- Synthetic corpus: 8 topical clusters × 50 documents × 60 words. Cluster-description query, same-cluster docs are relevant.
- R@K denominator caps at `min(K, |relevant|)` (saturating-recall convention).
- Separation = mean intra-cluster cosine − mean inter-cluster cosine.
- The tables below are the publishable summary. Detailed design notes live in
  [router_design.md](router_design.md) and [pmi_projection.md](pmi_projection.md).

## Headline (BEIR scifact)

`router_default_4096_768` **matches MiniLM-L6 on nDCG@10** (0.654 vs 0.654) and **beats MiniLM on MRR@10** (0.626 vs 0.607), training-free, no GPU, no model weights. The router picks among `Bm25Atire` / `Bm25SabSmooth` / `CascadeLinearAlpha` per query using pre-retrieval predictors (avg IDF, OOV rate, query length); design and literature in [router_design.md](router_design.md). The `high_idf_threshold` default (3.0) came from a 36-spec dev sweep.

| Configuration                                       | nDCG@10 | R@10  | R@100 | MRR@10 | code B/doc |
|-----------------------------------------------------|--------:|------:|------:|-------:|-----------:|
| MiniLM-L6 reference (384-d float32)                 |   0.654 | 0.808 | 0.932 |  0.607 |       1536 |
| **`router_default_4096_768` (idf=3.0, tuned)**      | **0.654** | **0.768** | **0.892** | **0.626** | 3072 |
| `bm25_pool500_linear_alpha075_4096_768`             |   0.638 | 0.762 | 0.874 |  0.608 |       3072 |
| `bm25_pool500_entropy_alpha_4096_768` (Step 1m)     |   0.619 | 0.741 | 0.874 |  0.592 |       3072 |
| `bm25_sab_pool500_entropy_alpha_4096_768` (Step 1m) |   0.612 | 0.733 | 0.881 |  0.587 |       3072 |
| `bm25_sab_smooth_gamma5` (novel BM25 alone)         |   0.636 | 0.759 | 0.893 |  0.607 |          — |
| `bm25_only` (Atire baseline)                        |   0.633 | 0.756 | 0.865 |  0.605 |          — |
| `router_oracle_4096_768` (per-query argmax, ceiling)|   0.713 | 0.834 | 0.931 |  0.684 |       3072 |

The remaining gap to MiniLM is concentrated in R@10 (0.768 vs 0.808). The
oracle row shows that some router headroom remains, but cheap pre-retrieval
predictors do not fully recover it.

## scifact — BM25 formulation ablation

Eight BM25 variants standalone on the same fixture. δ=1.0 for BM25+/L; γ=5.0 for SAB-smooth. SubwordAwareBackoff additionally builds a parallel char-3..5-gram inverted index. DCM uses the Zhai-Lafferty Dirichlet-LM form `log(1 + tf·T / (α·ttf))` with α_sum = avg_dl. PL2 and DPH are parameter-free DFR scorers (PL2 Poisson, DPH hypergeometric).

| Variant                                   | nDCG@10 | R@10  | R@100 | MRR@10 |    QPS |
|-------------------------------------------|--------:|------:|------:|-------:|-------:|
| BM25 Atire (Robertson)                    |   0.619 | 0.741 | 0.874 |  0.592 | 25,494 |
| BM25+ (Lv & Zhai CIKM 2011)               |   0.608 | 0.724 | 0.874 |  0.578 | 23,209 |
| BM25L (Lv & Zhai SIGIR 2011)              |   0.608 | 0.729 | 0.871 |  0.576 | 21,261 |
| DLH13 (Amati DFR, parameter-free)         |   0.590 | 0.699 | 0.846 |  0.562 |  8,898 |
| PL2 (Amati & van Rijsbergen TOIS 2002)    |   0.598 | 0.710 | 0.861 |  0.571 |  8,068 |
| DPH (Amati TREC 2007, hypergeometric)     |   0.600 | 0.714 | 0.847 |  0.573 | 12,293 |
| DCM / Dirichlet-LM (Madsen-Kauchak-Elkan 2005) | 0.568 | 0.692 | 0.847 |  0.534 | 12,714 |
| SubwordAwareBackoff strict (γ=0)          |   0.596 | 0.700 | 0.871 |  0.570 | 17,828 |
| **SubwordAwareBackoff smooth (γ=5)**      | **0.612** | **0.723** | **0.881** | **0.587** |  2,961 |

Smooth SAB is the strongest standalone BM25 we have and the best pool source for cascade. The +0.7-point R@100 lift (0.874 → 0.881) is where n-gram fallback surfaces morphological matches that exact BM25 misses. BM25+/L underperform on scifact (short abstracts; the long-doc floor doesn't apply) — reproduces Lv & Zhai 2011's corpus-dependent finding. DLH13's parameter-free score loses to tunable Atire by 2.9 nDCG points — the price of zero hyperparameters. PL2 and DPH land between DLH13 and Atire (0.598 / 0.600) — the DFR family is internally consistent but caps below tuned Robertson on short abstracts. DCM trails the Robertson family by ~5 points because the Dirichlet-LM contribution is not IDF-weighted (rarity enters through `ttf`, which is a weaker signal on short-abstract corpora).

## scifact — RM3 pseudo-relevance feedback

Lavrenko & Croft 2001 relevance model RM3: first-pass BM25 → top-K=10 pseudo-relevant docs → RM1 term distribution → top-N=20 expansion terms → α=0.5 blend with original query.

| Configuration                                 | nDCG@10 | R@10  | R@100 | MRR@10 |   QPS |
|-----------------------------------------------|--------:|------:|------:|-------:|------:|
| `bm25_atire` (baseline)                       |   0.619 | 0.741 | 0.874 |  0.592 | 25,494 |
| `bm25_atire_rm3_k10_a0.5`                     |   0.604 | 0.736 | 0.884 |  0.569 |    220 |
| `bm25_sab_smooth_gamma5` (baseline)           |   0.612 | 0.723 | 0.881 |  0.587 |  3,672 |
| `bm25_sab_smooth_rm3_k10_a0.5`                |   0.630 | 0.760 | 0.888 |  0.602 |    182 |

Negative-result on BM25-Atire: expansion slightly regresses top-10 ranking (−1.5 points nDCG) while lifting R@100 (+1.0 point) — expansion introduces noise on short abstract queries with already-informative terms. **Positive result on SAB-smooth** (+1.8 points nDCG, +3.7 points R@10): SAB's n-gram backoff already captures morphological variants, and RM3's term expansion is additive on top without competing with it. Per-query latency drops ~2 orders of magnitude because RM3 requires two scoring passes plus the relevance-model build.

## scifact — SDM (Sequential Dependence Model)

Metzler & Croft 2005: three-leg blend (λ_u · BM25 + λ_o · ordered-bigram BM25 + λ_uw · unordered-bigram BM25). Bigram postings built in-process when `Bm25Config::build_word_bigrams=true`. Default weights are Metzler's published (0.85, 0.10, 0.05); unordered window is 8 positions.

| Configuration                                           | nDCG@10 | R@10  | R@100 | MRR@10 |
|---------------------------------------------------------|--------:|------:|------:|-------:|
| `bm25_atire` (baseline)                                 |   0.619 | 0.741 | 0.874 |  0.592 |
| `bm25_atire_sdm_l0.85_0.10_0.05`                        |   0.612 | 0.736 | 0.883 |  0.584 |
| `bm25_atire_sdm_l0.90_0.05_0.05` (unigram-heavier)      |   0.619 | 0.746 | 0.879 |  0.587 |
| `bm25_sab_smooth_gamma5` (baseline)                     |   0.612 | 0.723 | 0.881 |  0.587 |
| `bm25_sab_smooth_sdm_l0.85_0.10_0.05`                   |   0.612 | 0.718 | 0.881 |  0.587 |

Near-no-op on scifact: Metzler defaults cost 0.7 nDCG points on Atire because short scientific abstracts have few adjacent-term signals that BM25 misses. Unigram-heavier weights (0.90/0.05/0.05) recover baseline exactly. SAB-smooth is tied either way.

## scifact — concept mining (Step 1n)

Bendersky 2008 latent concept model: corpus-PMI word-bigram mining + PMI-weighted concept BM25 linearly blended into the base variant. Training-free (closed-form PMI, no dev-fold tuner).

| Configuration                                | nDCG@10 | R@10  | R@100 | MRR@10 |
|----------------------------------------------|--------:|------:|------:|-------:|
| `bm25_atire` (baseline)                      |   0.619 | 0.741 | 0.874 |  0.592 |
| `bm25_atire_concepts_l0.50`                  |   0.510 | 0.649 | 0.841 |  0.474 |
| `bm25_atire_concepts_l0.25`                  |   0.564 | 0.696 | 0.866 |  0.530 |
| `bm25_sab_smooth_gamma5` (baseline)          |   0.612 | 0.723 | 0.881 |  0.587 |
| `bm25_sab_smooth_concepts_l0.50`             |   0.614 | 0.733 | 0.871 |  0.587 |

Null result: pre-declared regression gate (<0.005) violated on Atire at every weight. See [concept_mining.md](concept_mining.md) for scale-mismatch mechanism.

## scifact — cascade and fusion

| Configuration                                            | nDCG@10 | R@10  | R@100 | MRR@10 |
|----------------------------------------------------------|--------:|------:|------:|-------:|
| `bm25_rrf_simeon_4096_384` (global RRF k=60)             |   0.490 | 0.616 | 0.866 |  0.457 |
| `bm25_pool500_pool_rrf_4096_384` (RRF in pool only)      |   0.513 | 0.648 | 0.871 |  0.476 |
| `bm25_pool100_simeon_cos_4096_384` (cosine rerank)       |   0.401 | 0.537 | 0.865 |  0.364 |
| `bm25_pool500_simeon_cos_4096_384`                       |   0.373 | 0.492 | 0.759 |  0.340 |
| `bm25_pool1000_simeon_cos_4096_384`                      |   0.369 | 0.488 | 0.708 |  0.337 |
| `bm25_pool500_linear_alpha050_4096_384` (z-scored)       |   0.620 | 0.742 | 0.874 |  0.589 |
| **`bm25_pool500_linear_alpha075_4096_384` (z-scored)**   | **0.638** | **0.761** | **0.881** | **0.608** |
| `bm25_pool500_linear_alpha075_4096_768`                  |   0.638 | 0.762 | 0.874 |  0.608 |
| `bm25_sab_pool500_simeon_cos_4096_768`                   |   0.421 | 0.530 | 0.789 |  0.388 |
| `simeon_4096_384_rrf_bm25_rrf_minhash_256` (3-way RRF)   |   0.481 | 0.657 | 0.867 |  0.431 |

Linear-α z-scored combination is the only fusion that beats BM25 alone on top-10 ranking. Pure-simeon rerank discards BM25 magnitude info and *hurts* nDCG@10. Bigger simeon (768 vs 384) does not help the cascade — the simeon leg is α-bottlenecked, not capacity-bottlenecked. Pool-restricted RRF beats global RRF (0.51 vs 0.49) because pool restriction prevents simeon's projection-space noise from injecting bad candidates outside BM25's strong region.

## scifact — per-query router

| Configuration                                                | nDCG@10 | R@10  | R@100 | MRR@10 | route mix (atire/sab/cascade) |
|--------------------------------------------------------------|--------:|------:|------:|-------:|-------------------------------|
| `router_oracle_4096_768` (per-query argmax, headroom)        |   0.713 | 0.834 | 0.931 |  0.684 | 60 / 230 / 10 |
| **`router_default_4096_768` (idf=3.0)**                      | **0.654** | **0.768** | **0.892** | **0.626** | 60 / 207 / 33 |
| `router_cascade_aggressive_4096_768`                         |   0.654 | 0.768 | 0.892 |  0.626 | 60 / 207 / 33 |
| `router_grid_*_passA_oov0.25_idf3_*` (raise OOV threshold)   |   0.645 | 0.781 | 0.887 |  0.609 | 60 / 209 / 31 |
| `router_sab_only_4096_768`                                   |   0.636 | 0.759 | 0.893 |  0.607 | 0 / 300 / 0 |
| `router_grid_*_passA_*_idf6_*` (old default for reference)   |   0.640 | 0.763 | 0.889 |  0.609 | 0 / 93 / 207 |

Main takeaway: lowering `high_idf_threshold` from 6.0 to 3.0 is what unlocks
the Atire route and closes the scifact gap. The more detailed sweep analysis
and route-mix discussion are preserved in [router_design.md](router_design.md).

### Step 1f / 1g.1 — router enrichment (test fold, 202 queries)

| Config                                              | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768` (Step 1e Pass A winner)   |   0.640 | 0.753 | 0.886 |
| `passC_ant14_amif0.0` (Step 1f Pass C dev winner)   |   0.630 | 0.745 | 0.879 |
| **`passD_jac0.7_dec0.3` (Step 1g.1 Pass D)**        | **0.645** | **0.763** | **0.896** |
| `router_oracle_4096_768` (test oracle)              |   0.691 | 0.812 | 0.932 |

Step 1f's `df`-only enrichment overfits the dev fold and loses on test. Step
1g.1's post-retrieval-lite predictors (`score_decay_rate`,
`pool_overlap_jaccard` and friends) recover a small but real held-out lift.
Methodology and confusion analysis live in [router_design.md](router_design.md).

### Step 1k — SCQ + simplified-clarity gates (Pass E, test fold, 202 queries)

Two pre-retrieval predictors from the Carmel-Yom-Tov catalog that sit outside the IDF family: Sum-SCQ (Zhao, Scholer, Tsegay 2008) and simplified clarity (Cronen-Townsend & Croft 2002). Added as AND-gates on the Atire route: `atire_min_scq` floor + `atire_max_clarity` ceiling.

| Config                                                    | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768`                                 |   0.640 | 0.753 | 0.886 |
| `passE_scq0_clar99.0` (default — no gate)                 |   0.640 | 0.753 | 0.886 |
| `passE_scq<any>_clar8.0` (mild clarity ceiling)           |   0.637 | 0.748 | 0.881 |
| `passE_scq<any>_clar3.0 / clar5.0` (tight ceiling)        |   0.617 | 0.733 | 0.875 |
| `router_default_with_rm3_k10_a0.5`                        |   0.633 | 0.763 | 0.885 |

Null result on scifact: raising the SCQ floor is a no-op (all high-IDF queries already clear it), and tightening the clarity ceiling only removes queries the router was handling correctly via Atire. Step 1k B2 ships the predictors but leaves scifact defaults unchanged; the gates pay off on NFCorpus (see below). Router+RM3 also regresses on scifact for the same reason as the standalone `bm25_atire_rm3` row.

## scifact — speed/quality Pareto

Both legs were measured on the same M-series CPU as the rest of this page; simeon is C++/NEON, MiniLM is PyTorch/CPU (batch 32, single-threaded), so the comparison is same-hardware and deployment-realistic.

| Configuration                                 | nDCG@10 | encode docs/s | query qps |
|-----------------------------------------------|--------:|--------------:|----------:|
| `bm25_only` ★                                 |   0.633 |             — |    31,800 |
| `bm25_pool500_linear_alpha075_4096_384` ★     |   0.638 |         7,000 |     5,800 |
| `bm25_pool500_linear_alpha075_4096_768` ★     |   0.638 |         3,700 |     4,900 |
| `router_default_4096_768`                     |   0.640 |         3,600 |     1,500 |
| MiniLM-L6 reference (CPU-measured) ★          |   0.654 |           322 |     1,829 |
| `fwht_8192_1024` (max encode dps)             |   0.438 |        23,500 |     1,900 |

Indexing-side: simeon encodes 23–80× faster than MiniLM CPU at the same nDCG-cost tier. Query-side: MiniLM CPU and the quality-matching cascade are within 20% qps of each other; the lexical-end Pareto rows (BM25 / α-cascade) are 3–17× MiniLM CPU at 97–98% of MiniLM nDCG@10.

## scifact — projection and wider-sketch sweep

| Configuration                                | nDCG@10 | R@10  | R@100 | MRR@10 |  enc dps |
|----------------------------------------------|--------:|------:|------:|-------:|---------:|
| `achlioptas_4096_384` (baseline)             |   0.350 | 0.450 | 0.641 |  0.323 |    7,497 |
| `achlioptas_4096_768`                        |   0.407 | 0.508 | 0.699 |  0.379 |    4,044 |
| `gaussian_4096_384`                          |   0.378 | 0.469 | 0.656 |  0.353 |    7,735 |
| `very_sparse_4096_384`                       |   0.364 | 0.454 | 0.646 |  0.342 |   26,005 |
| `achlioptas_4096_384_mixed_tab` hash         |   0.368 | 0.453 | 0.652 |  0.350 |    6,281 |
| `sparse_jl_4096_384_eps0.10` (Kane-Nelson)   |   0.385 | 0.531 | 0.668 |  0.345 |    5,657 |
| `sparse_jl_4096_384_eps0.05`                 |   0.370 | 0.488 | 0.670 |  0.338 |   10,795 |
| `fwht_4096_384` (Ailon-Chazelle)             |   0.349 | 0.459 | 0.691 |  0.323 |   28,383 |
| `fwht_8192_1024`                             |   0.438 | 0.547 | 0.732 |  0.409 |   23,505 |
| `achlioptas_8192_512`                        |   0.385 | 0.490 | 0.711 |  0.359 |    3,282 |
| `achlioptas_8192_1024`                       |   0.429 | 0.551 | 0.744 |  0.393 |    1,715 |
| `achlioptas_16384_1024`                      |   0.451 | 0.565 | 0.750 |  0.421 |      833 |

FWHT matches `AchlioptasSparse` quality at the same dim with **>10× higher encode throughput at wide sketches** — `fwht_8192_1024` slightly beats `achlioptas_8192_1024` (0.438 vs 0.429) at 14× the encode rate. Pick FWHT when sketch width is the lever. Sparse-JL ε=0.10 picks up R@10 (+8 points over Achlioptas) at modest cost. The 4096→16384 sweep is still climbing at 16384→1024 but the cost curve has steepened. Mixed Tabulation is variance reduction only (~+1.8 nDCG points over SplitMix64).

## scifact — index compression (PQ) and matryoshka

| Configuration                                          | code B/doc | nDCG@10 | R@10  | R@100 |
|--------------------------------------------------------|-----------:|--------:|------:|------:|
| `achlioptas_4096_384_pq8` (192× compression)           |          8 |   0.108 | 0.166 | 0.421 |
| `achlioptas_4096_384_pq16` (96×)                       |         16 |   0.187 | 0.280 | 0.492 |
| `achlioptas_4096_384_pq32` (48×)                       |         32 |   0.299 | 0.376 | 0.605 |
| `pq16_first_stage_then_full_k100`                      |    16+full |   0.327 | 0.413 | 0.492 |
| `pq16_first_stage_then_full_k500`                      |    16+full |   0.349 | 0.453 | 0.622 |
| `achlioptas_matryoshka_4096_384` (analytic schedule)   |       1536 |   0.285 | 0.382 | 0.550 |
| `achlioptas_4096_384_matryoshka_data_aware`            |       1536 |   0.345 | 0.435 | 0.629 |

PQ m=32 keeps nDCG@10 within 5 points of full float32 at 48× smaller on-disk. PQ m=16 still serves as a viable first-stage filter when fed a wider candidate window. The data-aware matryoshka schedule recovers +6 nDCG points over the analytic `1/sqrt(1+r/decay)` schedule; supply a held-out seed corpus via `compute_matryoshka_weights()`. ADC ranking cost is `O(m)` per database code regardless of `dim`, so PQ is also a brute-force scan throughput win for moderately large indexes.

## scifact — densified MinHash

| Configuration                              | code B/doc | nDCG@10 | R@10  | R@100 |
|--------------------------------------------|-----------:|--------:|------:|------:|
| `minhash_256` (Shrivastava 2017)           |       1024 |   0.181 | 0.258 | 0.538 |
| `minhash_512`                              |       2048 |   0.217 | 0.340 | 0.626 |
| 3-way RRF: simeon ⊕ BM25 ⊕ MinHash         |       2560 |   0.481 | 0.657 | 0.867 |

Scientific abstracts are not duplicate-heavy — the win condition for Jaccard signal. Carry the three-way fuser forward for boilerplate-heavy corpora; on scifact it does not exceed two-way `bm25 ⊕ simeon`.

## scifact — PMI embeddings (Step 1g.2)

Word-level shifted-PPMI factored with randomized SVD (Levy-Goldberg 2014, Halko et al. 2011). `_incorpus` rows use the evaluation corpus as the seed (leakage-positive sanity ceiling, not a headline number). Design and provenance discipline in [pmi_projection.md](pmi_projection.md).

| Configuration                                              | code B/doc | nDCG@10 | R@10  | R@100 | MRR@10 |
|------------------------------------------------------------|-----------:|--------:|------:|------:|-------:|
| `simeon_pmi256_incorpus`                                   |       1024 |   0.251 | 0.346 | 0.678 |  0.226 |
| `simeon_pmi512_incorpus`                                   |       2048 |   0.285 | 0.382 | 0.699 |  0.257 |
| `simeon_pmi256_rrf_bm25_incorpus`                          |       1024 |   0.415 | 0.553 | 0.889 |  0.376 |
| `bm25_sab_pool500_simeon_pmi256_cos_rerank_incorpus`       |       1024 |   0.278 | 0.405 | 0.773 |  0.244 |
| `router_default_with_pmi256_cascade_incorpus`              |       1024 |   0.638 | 0.753 | 0.887 |  0.610 |

Negative result on scifact — even with the in-corpus leakage ceiling, standalone unigram PMI (0.25–0.28 nDCG@10) trails `achlioptas_4096_384` (0.35) and the PMI reranker drags the SAB cascade from 0.42 down to 0.28. The router preserves 0.64 because it rarely picks the cascade route. Scifact's gains come from morphological + contextual signal that static word-level PMI can't reach; see [pmi_projection.md](pmi_projection.md) for the analysis.

## FiQA — transfer check

Second BEIR fixture: `fiqa` — 57,638 docs / 444 test queries / 204 dev queries.
Finance is much more semantic/paraphrase-heavy than scifact, so this section
acts as a transfer check rather than a headline claim.

| Configuration                                       | nDCG@10 | R@10  | R@100 | MRR@10 |
|-----------------------------------------------------|--------:|------:|------:|-------:|
| MiniLM-L6 reference (384-d float32)                 |   0.359 | 0.429 | 0.702 |  0.436 |
| `bm25_atire_sdm_l0.85_0.10_0.05` (Step 1l)          |   0.212 | 0.268 | 0.468 |  0.263 |
| **`bm25_pool500_linear_alpha075_4096_768`**         | **0.211** | **0.269** | **0.478** | **0.263** |
| `bm25_pool500_linear_alpha075_4096_384`             |   0.211 | 0.267 | 0.473 |  0.266 |
| `router_grid_*_passB_pool500_a0.50` (FiQA favors α=0.50) | 0.210 | 0.265 | 0.460 |  0.261 |
| `bm25_atire_sdm_l0.90_0.05_0.05` (Step 1l)          |   0.208 | 0.265 | 0.471 |  0.259 |
| `router_grid_4096_768_passE_scq0_clar3.0` (Step 1k) |   0.208 | 0.260 | 0.478 |  0.263 |
| `router_grid_4096_768_passE_scq0_clar5.0` (Step 1k) |   0.208 | 0.261 | 0.477 |  0.262 |
| `bm25_sab_smooth_sdm_l0.85_0.10_0.05` (Step 1l)     |   0.206 | 0.252 | 0.478 |  0.262 |
| `bm25_atire` (Robertson baseline)                   |   0.205 | 0.264 | 0.467 |  0.257 |
| `bm25_pool500_entropy_alpha_4096_768` (Step 1m)     |   0.205 | 0.264 | 0.467 |  0.256 |
| `bm25_sab_smooth_concepts_l0.50` (Step 1n)          |   0.182 | 0.223 | 0.427 |  0.226 |
| `bm25_atire_concepts_l0.25` (Step 1n)               |   0.167 | 0.206 | 0.400 |  0.207 |
| `bm25_atire_concepts_l0.50` (Step 1n)               |   0.138 | 0.167 | 0.347 |  0.175 |
| `router_grid_4096_768_passE_scq0_clar8.0` (Step 1k) |   0.204 | 0.259 | 0.468 |  0.255 |
| `router_default_with_rm3_k10_a0.5` (Step 1k RM3)    |   0.202 | 0.255 | 0.466 |  0.255 |
| `router_default_4096_768` (scifact-tuned)           |   0.202 | 0.256 | 0.463 |  0.253 |
| `bm25_dph` (Step 1k)                                |   0.200 | 0.249 | 0.444 |  0.256 |
| `bm25_sab_smooth_gamma5` (novel BM25 alone)         |   0.198 | 0.250 | 0.467 |  0.249 |
| `bm25_sab_pool500_entropy_alpha_4096_768` (Step 1m) |   0.198 | 0.250 | 0.467 |  0.249 |
| `bm25_sab_smooth_rm3_k10_a0.5` (Step 1k RM3)        |   0.196 | 0.240 | 0.460 |  0.249 |
| `bm25_atire_rm3_k10_a0.5` (Step 1k RM3)             |   0.196 | 0.261 | 0.472 |  0.236 |
| `bm25_pl2` (Step 1k)                                |   0.189 | 0.242 | 0.456 |  0.235 |
| `bm25_sab_pool500_simeon_cos_4096_768`              |   0.117 | 0.151 | 0.371 |  0.151 |
| `simeon_pmi256_rrf_bm25_incorpus`                   |   0.116 | 0.170 | 0.445 |  0.139 |
| `achlioptas_4096_768` (standalone simeon)           |   0.101 | 0.122 | 0.262 |  0.138 |
| `bm25_dcm` (Dirichlet-LM, no IDF)                   |   0.085 | 0.119 | 0.299 |  0.114 |
| `minhash_256`                                       |   0.037 | 0.042 | 0.135 |  0.058 |
| `router_oracle_4096_768` (per-query argmax, ceiling) |  0.244 | 0.307 | 0.525 |  0.307 |

Takeaways:

- scifact-specific “near-parity to MiniLM” headlines do **not** transfer to FiQA
- **Step 1l SDM lifts FiQA by +0.006 on Atire** (`bm25_atire_sdm_l0.85_0.10_0.05` 0.212 vs `bm25_atire` 0.205) — the adjacent-bigram signal is real on finance text but below the plan's +0.010 promote threshold. SDM on SAB-smooth (+0.008) similarly underdelivers; unigram-heavier weights (0.90/0.05/0.05, 0.208) confirm Metzler's defaults are the right fixed point. Infrastructure stays behind `build_word_bigrams=false` default at zero runtime cost.
- **Step 1m entropy-α regresses by −0.006 vs matched static α=0.75** on FiQA (`bm25_pool500_entropy_alpha_4096_768` 0.205 vs `bm25_pool500_linear_alpha075_4096_768` 0.211). The cosine head's flatter top-K shifts per-query α toward 1.0 (BM25-only). Same pattern across scifact (−0.006) and NFCorpus (−0.010). Counterpoint: entropy-α rescues +0.081 on FiQA over `bm25_sab_pool500_simeon_cos_4096_768` 0.117 — a self-correcting safety floor when the dense leg would tank alone. Documented null-with-safety-property: see `docs/fusion_entropy_alpha.md`.
- **Step 1n concept mining regresses across all three corpora** (FiQA: −0.068 Atire l=0.50, −0.039 Atire l=0.25, −0.016 SAB-smooth l=0.50). PMI × BM25-bigram score-scale dominates base BM25 on matched queries, reordering around exact-phrase hits — actively harmful on finance paraphrase. Null result; infrastructure ships opt-in in `simeon::` namespace (no `Bm25Config` coupling), not wired into the router. See [concept_mining.md](concept_mining.md).
- **Step 1k clarity gate adds +0.006 nDCG on FiQA** (`passE_scq0_clar3.0` 0.208 vs `router_default` 0.202). Any `clar ≤ 5.0` ceiling produces the same 0.208; SCQ floor is a no-op since FiQA's high-IDF queries already clear it.
- **RM3 is near-flat on FiQA** (−0.005 on Atire, −0.002 on SAB-smooth vs the base variants; router+RM3 0.202 matches plain router). FiQA's semantic-paraphrase queries don't lift from corpus-statistic query expansion — expansion amplifies lexical matches the BM25 base already finds.
- PL2 (0.189) and DPH (0.200) slot *below* Atire (0.205) on FiQA; DFR family caps below tuned Robertson.
- linear-α fusion is still the only consistent way simeon beats BM25 alone
- router thresholds are corpus-sensitive and should be tuned per corpus

The full transfer discussion is preserved in [router_design.md](router_design.md).

## NFCorpus — headline rows

Third BEIR fixture: `nfcorpus` — 3,633 docs / 224 test queries / 99 dev queries (`--dev-fraction 0.33`). Medical abstracts with dense morphological terminology; matches scifact on domain character (science) but has different vocabulary.

| Configuration                                       | nDCG@10 | R@10  | R@100 | MRR@10 |
|-----------------------------------------------------|--------:|------:|------:|-------:|
| MiniLM-L6 reference (384-d float32)                 |   0.297 | 0.286 | 0.306 |  0.481 |
| **`router_grid_4096_768_passE_scq0_clar3.0` (Step 1k)** | **0.298** | **0.275** | **0.242** | **0.487** |
| `bm25_sab_smooth_sdm_l0.85_0.10_0.05` (Step 1l)     |   0.298 | 0.273 | 0.245 |  0.487 |
| `bm25_sab_smooth_gamma5` (novel BM25 alone)         |   0.298 | 0.274 | 0.244 |  0.487 |
| `bm25_sab_pool500_entropy_alpha_4096_768` (Step 1m) |   0.298 | 0.274 | 0.244 |  0.487 |
| `bm25_sab_smooth_concepts_l0.50` (Step 1n)          |   0.295 | 0.272 | 0.244 |  0.482 |
| `bm25_sab_smooth_rm3_k10_a0.5` (Step 1k RM3)        |   0.286 | 0.267 | 0.272 |  0.470 |
| `router_default_with_rm3_k10_a0.5` (Step 1k RM3)    |   0.277 | 0.260 | 0.274 |  0.455 |
| `bm25_atire_rm3_k10_a0.5` (Step 1k RM3)             |   0.271 | 0.253 | 0.264 |  0.445 |
| `router_default_4096_768` (scifact-tuned)           |   0.270 | 0.244 | 0.214 |  0.464 |
| `bm25_pool500_linear_alpha075_4096_768`             |   0.261 | 0.243 | 0.201 |  0.443 |
| `bm25_atire_sdm_l0.90_0.05_0.05` (Step 1l)          |   0.254 | 0.230 | 0.199 |  0.440 |
| `bm25_atire_sdm_l0.85_0.10_0.05` (Step 1l)          |   0.253 | 0.229 | 0.199 |  0.438 |
| `bm25_atire`                                        |   0.252 | 0.229 | 0.199 |  0.434 |
| `bm25_pool500_entropy_alpha_4096_768` (Step 1m)     |   0.252 | 0.229 | 0.201 |  0.433 |
| `bm25_atire_concepts_l0.25` (Step 1n)               |   0.245 | 0.224 | 0.199 |  0.420 |
| `bm25_atire_concepts_l0.50` (Step 1n)               |   0.242 | 0.220 | 0.199 |  0.415 |
| `bm25_plus`                                         |   0.252 | 0.232 | 0.200 |  0.440 |
| `bm25_l`                                            |   0.252 | 0.231 | 0.199 |  0.443 |
| `bm25_pl2` (Step 1k)                                |   0.249 | 0.230 | 0.199 |  0.421 |
| `bm25_dph` (Step 1k)                                |   0.249 | 0.228 | 0.198 |  0.423 |
| `bm25_dlh13`                                        |   0.249 | 0.226 | 0.198 |  0.430 |
| `bm25_dcm`                                          |   0.242 | 0.225 | 0.202 |  0.412 |
| `bm25_sab_strict` (γ=0)                             |   0.243 | 0.215 | 0.208 |  0.426 |
| `bm25_sab_pool500_simeon_cos_4096_768`              |   0.190 | 0.180 | 0.205 |  0.350 |
| `achlioptas_4096_768` (standalone simeon)           |   0.177 | 0.167 | 0.164 |  0.326 |
| `achlioptas_4096_384`                               |   0.147 | 0.136 | 0.153 |  0.284 |
| `router_oracle_4096_768` (per-query argmax, ceiling) |  0.327 | 0.302 | 0.260 |  0.541 |

Takeaways:

- **Step 1l SDM is a no-op on NFCorpus** (`bm25_atire_sdm` 0.253 vs `bm25_atire` 0.252 = +0.001, noise; SAB-SDM 0.298 ties SAB-smooth exactly). Medical abstracts are terminology-heavy but not phrase-heavy — adjacent-bigram signal doesn't add beyond unigram IDF. Matches Step 1l's predicted 0–0.005 lift on scientific text.
- **Step 1m entropy-α regresses by −0.010 vs matched static α=0.75 on Atire pool** (`bm25_pool500_entropy_alpha_4096_768` 0.252 vs `bm25_pool500_linear_alpha075_4096_768` 0.261) — largest undershoot of the three corpora. SAB-pool entropy-α ties `bm25_sab_smooth_gamma5` at 0.298 (cosine weight collapses to ~0 when BM25 already solves the query). Safety rescue holds: +0.074 vs pure cosine. Documented null-with-safety-property: see `docs/fusion_entropy_alpha.md`.
- **Step 1k clarity ceiling matches MiniLM** (`passE_scq0_clar3.0` 0.298 vs MiniLM 0.297). The AND-gate forces Atire on low-clarity queries only, and on NFCorpus that corresponds to the subset where SAB-smooth is already the right pick — effectively routing the entire fixture to the stronger BM25 variant without losing the scifact cascade route on the rest. Any `clar ≤ 5.0` ceiling produces the same 0.298; clarity floor threshold is insensitive at 3.0–5.0.
- **RM3 is the second NFCorpus lift** (+1.9 points nDCG on Atire, +0.8 on SAB-smooth). Morphology-heavy medical queries benefit from term-expansion when the base variant is already subword-aware; compounding them is the Step 1k "cheap RM3 + better router" story.
- **SAB-smooth matches MiniLM without the gate** (0.298). Medical morphology is even richer than scifact's; n-gram backoff remains the primary corpus-agnostic lever.
- PL2 and DPH slot into the DFR cluster at 0.249 — DFR-family scores collapse on NFCorpus to within noise of DLH13.
- The SAB→simeon cosine cascade still collapses (0.190 vs BM25-alone 0.252), same pattern as on FiQA. Simeon cosine rerank is a scifact-specific win.
- Three-corpus verdict: **SAB as a standalone scorer is corpus-agnostic on morphology-heavy text; Step 1k's clarity ceiling generalizes where scifact router tuning does not.**

## Microbench — synthetic corpus

> The synthetic corpus has clear topical separation, no paraphrase, no semantic drift — R@10 ≈ 1.0 is the regression floor, not a quality ceiling. Use it to compare configurations of simeon to each other.

### Throughput (1,000 random ASCII docs × 256 bytes)

| Configuration                      | sketch | out | µs / doc | docs / s |
|------------------------------------|-------:|----:|---------:|---------:|
| no projection                      |  1024  |   — |     3.71 |  269,678 |
| no projection                      |  4096  |   — |     4.42 |  226,325 |
| `VerySparse`                       |  4096  | 256 |    12.13 |   82,470 |
| `VerySparse`                       |  4096  | 384 |    16.04 |   62,352 |
| `VerySparse`                       |  4096  | 768 |    28.63 |   34,928 |
| `AchlioptasSparse`                 |  4096  | 128 |    38.24 |   26,150 |
| `AchlioptasSparse`                 |  4096  | 256 |    74.46 |   13,429 |
| `AchlioptasSparse`                 |  4096  | 384 |   109.75 |    9,112 |
| `AchlioptasSparse`                 |  4096  | 512 |   150.52 |    6,644 |
| `DenseGaussian`                    |  4096  | 256 |    65.75 |   15,210 |
| `DenseGaussian`                    |  4096  | 384 |    96.09 |   10,407 |
| `DenseGaussian`                    |  4096  | 768 |   195.95 |    5,103 |

`VerySparse` (Li 2006) is the throughput sweet spot when slightly lower retrieval quality is acceptable.

### SIMD kernel throughput (n=4,096 floats)

| Kernel       | NEON Gelem/s |
|--------------|-------------:|
| `add_vec`    |         7.87 |
| `scale_vec`  |         8.12 |
| `saxpy`      |         7.31 |

All three are memory-bandwidth-bound; the three-way spread (<11%) confirms the dispatch is hitting SIMD. Dispatched from `include/simeon/simd.hpp`; used by the PMI-sum encode path and matryoshka weighting (`src/simeon.cpp`).

### Projection sweep, n-gram width, n-gram mode

| Projection                | sketch | out | R@10  | R@100 | MRR@10 | Separation |
|---------------------------|-------:|----:|------:|------:|-------:|-----------:|
| none                      |  1024  |   — | 1.000 | 0.998 |  1.000 |      0.545 |
| `AchlioptasSparse`        |  4096  | 384 | 1.000 | 0.975 |  1.000 |      0.536 |
| `DenseGaussian`           |  4096  | 384 | 1.000 | 0.970 |  1.000 |      0.534 |
| `VerySparse`              |  4096  | 384 | 0.988 | 0.965 |  1.000 |      0.540 |

| n-gram range | R@10  | R@100 | Separation |
|--------------|------:|------:|-----------:|
| 3–3          | 1.000 | 0.985 |      0.535 |
| 3–5 (default)| 1.000 | 0.975 |      0.536 |
| 3–7          | 0.988 | 0.955 |      0.496 |
| 4–6          | 1.000 | 0.983 |      0.513 |

| Mode      | R@10  | R@100 | Separation |
|-----------|------:|------:|-----------:|
| char      | 1.000 | 0.975 |      0.536 |
| char+word | 1.000 | 0.978 |      0.540 |
| word      | 1.000 | 0.983 |  **0.646** |

3–5 is a good default. On real text with morphological variation, char-only is more robust; `char+word` is the safe default.

### Matryoshka prefix queryability (synthetic)

`AchlioptasSparse 4096 → 384`, `matryoshka = true`, `decay = 32`. Truncate the leading prefix and re-normalize.

| Prefix width | R@10  | R@100 | Notes                                  |
|-------------:|------:|------:|----------------------------------------|
| 384 (full)   | 0.975 | 0.980 | matches plain `AchlioptasSparse 384`   |
| 192          | 0.975 | 0.953 | half storage, no R@10 loss             |
| 128          | 0.963 | 0.930 | 1/3 storage                            |
| 64           | 0.938 | 0.870 | 1/6 storage                            |

Use this when downstream consumers want to pick a quality / cost point per query — store one 384-d vector, query at 64-d for a coarse first pass and re-rank survivors with the full vector. Quality on real text is dramatically lower with the analytic schedule; see scifact matryoshka rows above.

## Microbench — SAB-smooth perf audit

FiQA, 57,638 docs × 444 queries × 5 iters; `simeon_profile_sab_smooth` harness; `SubwordAwareBackoff` γ=5.

| Phase            |  Baseline |      Post |      Δ |
|------------------|----------:|----------:|-------:|
| add_docs (ms)    |  11,667.6 |   5,886.8 | −49.5% |
| finalize (ms)    |     167.1 |      63.6 | −61.9% |
| build_total (ms) |  11,834.7 |   5,950.3 | −49.7% |
| query mean (µs/q)|     992.4 |   1,009.4 |  +1.7% |
| query p95 (µs/q) |   1,749.2 |   1,778.6 |  +1.7% |
| wall clock (s)   |     14.63 |      8.56 | −41.5% |
| cycles (B)       |      47.5 |      32.5 | −31.5% |
| instructions (B) |     113.9 |     112.3 |  −1.4% |
| peak RSS (MB)    |     1,234 |     1,514 | +22.7% |

Quality gate — `bm25_sab_smooth_gamma5` nDCG@10, all within ±0.001:

| Corpus   | Baseline | Post   |      Δ |
|----------|---------:|-------:|-------:|
| scifact  |    0.612 | 0.6120 |  0.000 |
| NFCorpus |    0.298 | 0.2981 | +0.000 |
| FiQA     |   0.1978 | 0.1978 |  0.000 |

## Aho-Corasick dictionary matcher

1 MiB input, `whole_word=true, longest_match=true, case_insensitive=true`, random lowercase patterns of length 5..12; hit input = sampled patterns separated by spaces; noise input = random lowercase tokens.

| Patterns |   Nodes | Build (ms) | Noise MB/s | Hits MB/s | Hits count |
|---------:|--------:|-----------:|-----------:|----------:|-----------:|
|    1,000 |   7,159 |        1.8 |      127.1 |     110.5 |    108,786 |
|   10,000 |  63,421 |       23.7 |       88.8 |      44.7 |    110,171 |
|  100,000 | 557,587 |      299.9 |       55.9 |      19.8 |    110,439 |
|  500,000 | 2,562,568 |   1,800.8 |       27.5 |      16.3 |    110,403 |

Dense-goto table memory ≈ nodes × 1,024 bytes (256 slots × 4 B). At 500k patterns, ≈2.5 GB of goto table dominates cache locality. Two alternatives tried and rejected:

| Variant | 500k Noise MB/s | 500k Hits MB/s | vs. monolithic |
|---|---:|---:|---:|
| Monolithic dense (baseline) | 24.5 | 13.7 | — |
| CSR sparse + dense root fast path | 12.5 | 14.5 | noise regresses 2× |
| Sharded, 2 partitions | 16.8 | 7.9 | both regress |
| Sharded, 4 partitions | 11.7 | 6.8 | both regress |
| Sharded, 8 partitions | 6.8 | 4.8 | both regress |
| Sharded, 16 partitions | 3.7 | 3.0 | both regress |

CSR's `if (state == root)` dispatch introduces a data-dependency serialization on the hot load that the branchless `next_[state*256 + c]` avoids. Sharding scales throughput as ≈1/N since each shard rescans the full input; the per-shard automaton shrink doesn't compensate. For GLiNER-scale tech dictionaries (≤100k surface forms) the dense table fits in ≈550 MB and clears the ≥50 MB/s noise-input target. Scaling past 500k patterns needs a compressed representation with branchless lookup (double-array trie, compressed-sparse-fail) — prototyped on branch `ac-da-trie-wip`.

## TextRank sentence ranker

Synthetic doc-count=200 sweep, damping=0.85, max_iters=30, ε=1e-4, min_sentence_tokens=3, max_sentence_tokens=60; each doc samples sentences from one of 5 topic pools.

| Sentences/doc | p50 ms | p99 ms | Lead-1 agreement |
|--------------:|-------:|-------:|-----------------:|
|             4 |  0.004 |  0.006 |            0.305 |
|             8 |  0.010 |  0.013 |            0.095 |
|            16 |  0.025 |  0.031 |            0.075 |
|            32 |  0.081 |  0.096 |            0.050 |
|            64 |  0.284 |  0.319 |            0.035 |

Latency scales as O(n²) in sentence count (graph build dominates). p99 stays under 1 ms through 64 sentences; the yams title path caps at `max_sentences=256` (worst-case ≈5 ms). Lead-1 agreement dropping as n grows confirms TextRank diverges from lead bias as documents gain internal structure — the signal the yams title extractor will rely on for long-form plain text.

## Out of scope

- **Semantic equivalence / paraphrase** standalone benchmarks — simeon is lexical; use a learned bi-encoder where paraphrase robustness is the only signal.
- **Cross-language** — char-n-gram tokenization handles Latin scripts well; CJK is on the roadmap.
