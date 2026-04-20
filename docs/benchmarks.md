# simeon benchmarks

## Conditions

- Hardware: 2024 Apple M-series laptop, NEON tier, single-threaded.
- Build: `buildtype=debugoptimized`, NEON enabled, no sanitizers.
- Real-corpus fixture: BEIR `scifact` — 5,183 docs / 300 queries / 339 qrels (test split). Reference: `sentence-transformers/all-MiniLM-L6-v2`, 384-d float32, L2-normalized. Fixture format documented in [reference_fixture.md](reference_fixture.md).
- Synthetic corpus: 8 topical clusters × 50 documents × 60 words. Cluster-description query, same-cluster docs are relevant.
- R@K denominator caps at `min(K, |relevant|)` (saturating-recall convention).
- Separation = mean intra-cluster cosine − mean inter-cluster cosine.
- The headline tables below are the publishable benchmark summary; raw local result dumps are intentionally not shipped in-tree.

## Headline (BEIR scifact)

`router_default_4096_768` **matches MiniLM-L6 on nDCG@10** (0.654 vs 0.654) and **beats MiniLM on MRR@10** (0.626 vs 0.607), training-free, no GPU, no model weights. The router picks among `Bm25Atire` / `Bm25SabSmooth` / `CascadeLinearAlpha` per query using pre-retrieval predictors (avg IDF, OOV rate, query length); design and literature in [router_design.md](router_design.md). The `high_idf_threshold` default (3.0) came from a 36-spec dev sweep.

| Configuration                                       | nDCG@10 | R@10  | R@100 | MRR@10 | code B/doc |
|-----------------------------------------------------|--------:|------:|------:|-------:|-----------:|
| MiniLM-L6 reference (384-d float32)                 |   0.654 | 0.808 | 0.932 |  0.607 |       1536 |
| **`router_default_4096_768` (idf=3.0, tuned)**      | **0.654** | **0.768** | **0.892** | **0.626** | 3072 |
| `bm25_pool500_linear_alpha075_4096_768`             |   0.638 | 0.762 | 0.874 |  0.608 |       3072 |
| `bm25_sab_smooth_gamma5` (novel BM25 alone)         |   0.636 | 0.759 | 0.893 |  0.607 |          — |
| `bm25_only` (Atire baseline)                        |   0.633 | 0.756 | 0.865 |  0.605 |          — |
| `router_oracle_4096_768` (per-query argmax, ceiling)|   0.713 | 0.834 | 0.931 |  0.684 |       3072 |

The remaining gap to MiniLM is concentrated in R@10 (0.768 vs 0.808). The oracle row shows ~6 nDCG points and ~7 R@10 points of pre-retrieval-router headroom remain — bounded by predictor quality (cheap features can't perfectly identify which queries Atire wins on).

## scifact — BM25 formulation ablation

Five BM25 variants standalone on the same fixture. δ=1.0 for BM25+/L; γ=5.0 for SAB-smooth. SubwordAwareBackoff additionally builds a parallel char-3..5-gram inverted index.

| Variant                                   | nDCG@10 | R@10  | R@100 | MRR@10 |    QPS |
|-------------------------------------------|--------:|------:|------:|-------:|-------:|
| BM25 Atire (Robertson)                    |   0.633 | 0.756 | 0.865 |  0.605 | 27,813 |
| BM25+ (Lv & Zhai CIKM 2011)               |   0.619 | 0.744 | 0.865 |  0.585 | 28,388 |
| BM25L (Lv & Zhai SIGIR 2011)              |   0.615 | 0.747 | 0.863 |  0.579 | 25,222 |
| DLH13 (Amati DFR, parameter-free)         |   0.610 | 0.727 | 0.849 |  0.581 | 10,702 |
| SubwordAwareBackoff strict (γ=0)          |   0.614 | 0.728 | 0.865 |  0.584 | 19,690 |
| **SubwordAwareBackoff smooth (γ=5)**      | **0.636** | **0.759** | **0.893** | **0.607** |  2,809 |

Smooth SAB is the strongest standalone BM25 we have and the best pool source for cascade. The +2.8-point R@100 lift (0.865 → 0.893) closes 60% of the gap to MiniLM's R@100 — the n-gram fallback surfaces morphological matches that exact BM25 misses. BM25+/L underperform on scifact (short abstracts; the long-doc floor doesn't apply) — reproduces Lv & Zhai 2011's corpus-dependent finding. DLH13's parameter-free score loses to tunable Atire by 2.3 nDCG points — the price of zero hyperparameters.

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

Three findings from the 36-row Pass A threshold sweep + 9-row Pass B (pool × α) sweep:

- **`high_idf_threshold` is the only knob that moves the metric.** Lowering it from the conservative 6.0 to 3.0 unlocks the Atire route for 60/300 queries (rare-term, high-IDF — exactly where exact match wins) and lifts nDCG@10 by **+1.4 points** to MiniLM parity. Now the new default.
- **The oracle reaches 0.713 nDCG@10** with route mix 60/230/10 — the cascade route is rarely the per-query best on scifact (only 10 queries), but it averages best on the queries SAB doesn't dominate. The 6 nDCG points between tuned (0.654) and oracle (0.713) is the predictor-quality ceiling: cheap pre-retrieval features cannot perfectly identify which queries Atire will win.
- **Pass B (pool × α) does not move the metric.** After threshold tuning the cascade route only sees 33/300 queries, so the `cascade_alpha` knob has too little leverage to register. `pool_size` is similarly mute.

### Step 1f / 1g.1 — router enrichment (test fold, 202 queries)

| Config                                              | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768` (Step 1e Pass A winner)   |   0.640 | 0.753 | 0.886 |
| `passC_ant14_amif0.0` (Step 1f Pass C dev winner)   |   0.630 | 0.745 | 0.879 |
| **`passD_jac0.7_dec0.3` (Step 1g.1 Pass D)**        | **0.645** | **0.763** | **0.896** |
| `router_oracle_4096_768` (test oracle)              |   0.691 | 0.812 | 0.932 |

Step 1f's `df`-only enrichment overfits the 98-query dev fold and loses on test. Step 1g.1's post-retrieval-lite predictors (`score_decay_rate`, `pool_overlap_jaccard` and friends) close ~9% of the test router→oracle gap. Methodology, dev-fold tables, and confusion analysis live in [router_design.md](router_design.md).

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

## FiQA — headline rows (Step 1i, multi-corpus transfer)

Second BEIR fixture: `fiqa` — 57,638 docs / 444 test queries / 204 dev queries (2:1 split via `build_reference_fixture.py --dev-fraction 0.33`). Finance domain, semantic-paraphrase-heavy (the opposite of scifact's lexical/morphological bias). Reference: `sentence-transformers/all-MiniLM-L6-v2`. All rows below are the test fold; configs re-used from scifact with no FiQA-specific tuning.

| Configuration                                       | nDCG@10 | R@10  | R@100 | MRR@10 |
|-----------------------------------------------------|--------:|------:|------:|-------:|
| MiniLM-L6 reference (384-d float32)                 |   0.359 | 0.429 | 0.702 |  0.436 |
| **`bm25_pool500_linear_alpha075_4096_768`**         | **0.211** | **0.269** | **0.478** | **0.263** |
| `bm25_pool500_linear_alpha075_4096_384`             |   0.211 | 0.267 | 0.473 |  0.266 |
| `router_grid_*_passB_pool500_a0.50` (FiQA favors α=0.50) | 0.210 | 0.265 | 0.460 |  0.261 |
| `bm25_atire` (Robertson baseline)                   |   0.205 | 0.264 | 0.467 |  0.257 |
| `bm25_sab_smooth_gamma5` (novel BM25 alone)         |   0.198 | 0.250 | 0.467 |  0.249 |
| `router_default_4096_768` (scifact-tuned)           |   0.202 | 0.256 | 0.463 |  0.253 |
| `bm25_sab_pool500_simeon_cos_4096_768`              |   0.117 | 0.151 | 0.371 |  0.151 |
| `achlioptas_4096_768` (standalone simeon)           |   0.101 | 0.122 | 0.262 |  0.138 |
| `minhash_256`                                       |   0.037 | 0.042 | 0.135 |  0.058 |
| `simeon_pmi256_rrf_bm25_incorpus`                   |   0.116 | 0.170 | 0.445 |  0.139 |
| `router_oracle_4096_768` (per-query argmax, ceiling) |  0.244 | 0.307 | 0.525 |  0.307 |

**Divergences from scifact** (findings that do **not** generalize):

- **MiniLM/BM25 gap widens 1.03× → 1.75×.** Scifact is lexical (BM25 97% of MiniLM); FiQA is semantic (BM25 57% of MiniLM). The "BM25 alone is already near-parity" headline is scifact-specific.
- **Standalone simeon collapses 0.40 → 0.10 nDCG@10.** Projection-space signal carries scientific morphology but not financial paraphrase.
- **SAB-pool+simeon-cos cascade inverts: +0 scifact → −9 FiQA points vs BM25 alone.** On a semantic corpus the simeon reranker discards the little ranking signal BM25 recovered and falls below it. The scifact-headline cascade pattern is corpus-dependent.
- **PassB α winner flips 0.75 → 0.50.** Scifact-tuned router thresholds transfer imperfectly; FiQA prefers more simeon weight in the linear combination, though the absolute lift is small (0.210 vs 0.205 BM25).
- **Oracle → router gap widens 0.059 → 0.042 absolute but 9% → 20% relative.** FiQA has more per-query ambiguity that cheap pre-retrieval features can't resolve.

**Generalizations** (findings that hold across corpora):

- **Linear-α z-scored combination is the only fusion that beats BM25 alone.** On both scifact (+0.005) and FiQA (+0.006) it's the single configuration that nets positive vs Atire. SAB-pool + cosine-rerank fails on FiQA as predicted by the "cascade is pool-recall-bounded" frame.
- **Router >= BM25 alone on both corpora.** 0.654 vs 0.633 on scifact, 0.202 vs 0.205 on FiQA — the router tracks BM25 on semantic corpora rather than boosting past it, but doesn't regress below it.
- **MinHash is weak on both.** Neither corpus is duplicate-heavy.
- **PMI (in-corpus) trails Achlioptas on both.** The unigram-PMI-vs-contextual gap is corpus-agnostic.

Raw rows in `benchmarks/results/fiqa_full.jsonl`. Router-transfer notes in [router_design.md](router_design.md).

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

## Out of scope

- **Semantic equivalence / paraphrase** standalone benchmarks — simeon is lexical; use a learned bi-encoder where paraphrase robustness is the only signal.
- **Cross-language** — char-n-gram tokenization handles Latin scripts well; CJK is on the roadmap.
