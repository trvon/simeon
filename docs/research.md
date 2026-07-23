# Research notes

This file records what the historical retrieval experiments found and which
components were promoted to the shipped surface. The old driver
(`bench_vs_reference`) is retired, but experimentation is not concluded. New
work uses the versioned, manifest-driven contract in
[experimentation.md](experimentation.md); findings below remain a historical
ledger and are not automatically comparable to new `trec-v1` results.

## Reusable experiment foundation

`simeon_embedding_experiment` now provides the first shared experiment provider:
strict manifests, model-independent fixtures, semantic data/config fingerprints,
exact retrieval, standard metric semantics, source/SIMD provenance, JSONL
results, and enforced dev/holdout state. The initial study lives in
`experiments/embedding-foundation.exp` and separates token-unit changes from
projection/compression changes.

The intended expansion is provider-based: PMI and other corpus-adaptive
representations, lexical candidate legs, fragment geometry, fusion, and routing
should reuse the same fixture, metric, provenance, and holdout contract. See
[experimentation.md](experimentation.md) for the training-regime taxonomy and
model-space outline.

The first instrument-driven optimization is opt-in, deterministic ASCII case
folding before hashing. It improves development nDCG@10 on SciFact, NFCorpus,
and FiQA in both the 32,768-dimensional and projected 384-dimensional spaces,
without fitted state. The compact relative gains are 4.75%, 16.96%, and 14.46%
respectively. The selected compact variant is frozen but has not consumed any
holdout during that selection; it was later read as the frozen baseline for the
compact FWHT validation. See the [controlled report](../experiments/reports/2026-07-20-ascii-lower.md)
for fingerprints, absolute scores, complexity changes, and residual risks.

The next controlled iteration added deterministic signed-square-root weighting
over accumulated sketch counts. In the 32,768-dimensional lowercase char+word
space it improves development nDCG@10 by 14.93% on SciFact, 17.06% on NFCorpus,
and 24.08% on FiQA. It consistently regresses the 4,096-to-384 compact path,
where the nonlinear transform magnifies bucket collisions, so it remains a
wide-only opt-in rather than a default. See the
[weighting report](../experiments/reports/2026-07-20-signed-sqrt.md).

A collision-safe follow-up applied `sqrt(tf)` to distinct feature identities
before sketch aggregation. That corrected the ordering problem but regressed
compact nDCG@10 by 12.30–29.29% and was roughly equivalent to the cheaper
post-sketch transform in the wide space. It remains a replayable experimental
axis, not a promoted recipe. A later interaction check on the selected 8k FWHT
geometry also regressed all three dev fixtures, so the old projection was not
the sole cause. See the
[negative-result report](../experiments/reports/2026-07-20-feature-sqrt.md).

The first promoted compact geometry recipe is
`compact_retrieval_config()`: lowercase char 3–5 grams plus word tokens, an
8,192-bucket signed sketch, and fixed FWHT projection to 384 dimensions. Against
the previous 4,096-to-384 Achlioptas compact baseline, it improved frozen
holdout nDCG@10 by 9.40% on SciFact, 4.16% on NFCorpus, and 1.92% on FiQA while
encoding 3.8–5.0x faster. Legacy defaults stay unchanged so stored embedding
identities do not move implicitly. See the
[compact projection report](../experiments/reports/2026-07-20-compact-fwht.md).

The next promoted iteration makes those character grams word-bounded: split on
ASCII punctuation and whitespace, preserve non-ASCII UTF-8 bytes, wrap each
word with fixed `<` and `>` markers, and never emit a character gram across a
word boundary. Relative dev nDCG@10 improved by 11.52% on SciFact, 9.77% on
NFCorpus, and 22.12% on FiQA. The single frozen holdout read improved by 10.02%,
8.41%, and 0.04%, respectively; FiQA is effectively flat on the primary metric,
while recall@100 improved on all three holdouts. Median dev encoding time also
fell by 0.9–2.7%. The mode is now part of `compact_retrieval_config()`, while
the generic configuration default remains whole-text to preserve existing
embedding identities. See the
[word-boundary report](../experiments/reports/2026-07-20-word-boundaries.md).

The first promoted corpus-adaptive embedding recipe adds a bounded hashed-IDF
artifact learned from deployment documents only. A 65,536-bucket table occupies
128 KiB and uses no qrels, pretrained vectors, or gradient fitting. Against the
word-bounded artifact-free preset, frozen holdout nDCG@10 improves by 25.77% on
SciFact, 33.06% on NFCorpus, and 57.57% on FiQA; recall@100 also improves on all
three. Exact scoring and 384-float vector storage are unchanged, while encoding
cost rises by 12–13% after removing an unnecessary second sketch pass. This is
exposed separately as `compact_hashed_idf_retrieval_config()` so the artifact-
free identity remains intact. See the
[hashed-IDF report](../experiments/reports/2026-07-20-hashed-idf.md).

Scaling only the output geometry shows that the 384-coordinate projection was
still discarding useful weighted-sketch information. Development nDCG@10 rises
consistently through 768 dimensions; 1,024 adds small SciFact/NFCorpus gains but
slightly regresses FiQA and carries materially higher scoring/storage cost. The
frozen 768-dimensional choice improves test nDCG@10 over the 384-dimensional
hashed-IDF encoder by 5.13% on SciFact, 4.59% on NFCorpus, and 8.97% on FiQA,
with recall@100 gains on all three. It is exposed as
`quality_hashed_idf_retrieval_config()` while the compact identity remains 384
dimensions. See the
[dimension-scaling report](../experiments/reports/2026-07-20-hashed-idf-dimensions.md).

The follow-up fusion experiment tested that 768-dimensional encoder as both a
candidate leg and a z-normalized score leg inside the exact frozen six-leg WSDM
workbench. Development selected a fixed 0.30 contribution: it was the only
choice that cleared +0.005 nDCG@10 on all three corpora while increasing
Recall@100 everywhere. Frozen holdout gains were only +0.0032 SciFact, +0.0016
NFCorpus, and +0.0029 FiQA, however—all below the repository's promotion gate
and below the corpus-specific PRF path on SciFact/NFCorpus. The recipe is not
promoted. The reusable `wsdm_idf_fusion` provider remains because it cleanly
separates candidate recall from score-fusion quality and amortizes expensive
corpus evidence across a manifest sweep. See the
[fusion report](../experiments/reports/2026-07-20-hashed-idf-fusion.md).

The next iteration imported the admission structure from YAMS'
`RetrievalCoordinates.lean` and asked how the 1,024-dimensional FWHT space is
actually used. Across the three development corpora, diagonal variance has
1,021.7–1,022.9 effective coordinates out of 1,024 (99.77–99.89%); individual
axes are not collapsing. Corpus mean-centering and diagonal standardization
regress every corpus. Treating 768 as a local chart and 1,024 as its reference
produces useful uncertainty observations—67.3–83.1% top-100 overlap and
0.0150–0.0181 mean similarity distortion—but the best routing thresholds are
corpus-specific and below the promotion gate. A pre-scoring query-energy gate
also fails to improve either fixed endpoint. Holdout was not opened. The
reusable `coordinate_calibrated_idf` provider and its cached evidence replay are
retained; no transform or route is shipped. See the
[coordinate-space report](../experiments/reports/2026-07-20-coordinate-space.md).

The follow-up treated character and word evidence as the semantically distinct
charts suggested by that result. The new `feature_family_atlas_idf` provider
builds separate IDF/FWHT spaces, enforces one 768-float concatenated storage
budget, and replays fixed weights with independent, joint, or corpus-RMS joint
normalization. The charts are genuinely complementary—their top-100 overlap is
only 6–26%, and their union adds up to 0.021 Recall@100 on FiQA. Independent
normalization improves the per-corpus dev maximum by +0.00509 SciFact and
+0.00386 FiQA, but every split regresses NFCorpus; the winners also select
different allocations. RMS calibration reaches +0.00493 on FiQA but still
regresses NFCorpus by at least 0.00489. No fixed recipe transfers, so holdout
was not opened and the combined 768-dimensional preset remains unchanged. The
atlas is retained because it makes family complementarity, raw energy (about
98% character at 512/256), exact storage, and evaluation-only union headroom
observable. See the [feature-family report](../experiments/reports/2026-07-20-feature-family-atlas.md).

The next iteration kept that combined 768-dimensional chart as a safe base and
priced a 256-dimensional family chart as a residual, for exactly 1,024 floats
per document. Raw cosine, query-wise score normalization, rank fusion, and
label-free admission gates are all replayable from cached evidence. The most
balanced dev result, a 192-character/64-word residual with 0.15 query-z-score
weight, improves both nDCG@10 and Recall@100 on all three corpora, but its
nDCG@10 gains are only +0.00484 SciFact, +0.00094 NFCorpus, and +0.00079 FiQA.
No policy clears the +0.005 cross-corpus promotion gate; routing thresholds do
not predict benefit portably, and RRF regresses NFCorpus. The 768-dimensional
preset therefore remains unchanged and holdout stays unopened. The instrument
is retained because raw cosine supplies an exact 1,024-dimensional deployment
geometry, while score-normalized modes explicitly expose their full-corpus
scoring/cache cost. See the
[feature-residual report](../experiments/reports/2026-07-20-feature-residual.md).

## Production surface (shipped retrieval recipes)

- **BM25 variants** (Atire, AtireLTD, SAB-smooth, BM25+, BM25L, DPH, PL2, DCM, Layered, LayeredW)
- **BM25 ↔ simeon RRF fusion** at pool sizes 100/300/500 and α ∈ {0.65, 0.75, 0.85, 0.95}
- **Fragment-geometry rerank** (richcov, PHSS approx, MaxSim SPLATE)
- **Query router** (pre-retrieval predictors → Atire vs SAB vs CascadeLinearAlpha)
- **Quality router** (short vs medium vs long semantic query → Bm25Only vs FragmentRichCovPhssApprox vs FragmentRichCovPhssApproxMax)
- **Step 1g.1 post-retrieval gates** (pool Jaccard, score decay) and **Step 1k B2 pre-retrieval gates** (SCQ, clarity)
- **BM25-variant RRF** (`score_bm25_variants_rrf`)
- **RM3 pseudo-relevance feedback** (`score_with_prf`)
- **Concept mining** (`mine_concepts` + BM25-concept blend)
- **Corpus-class recipe routing** (AtireLTD for long-doc corpora, Atire for short-doc)

### Explored and not shipped

| Approach | Description | Outcome |
|---|---|---|
| Synthetic aux fields | BM25F with TextRank titles / Aho-Corasick entities | Marginal; not shipped |
| PMI-neighbor expansion | corpus-derived, training-free query expansion | Neutral-to-negative; not shipped |
| Phrase/concept transport | SDM bigrams + mined concepts + z-scored BM25 pool fusion | Mixed; concept leg occasionally helped, phrase leg did not |
| Graph reranking | Personalized-PageRank over query-phrase-doc graphs | Negative; not shipped |
| Cluster rescoring | TextRank-fragment clusters by query-cover and containment mass | Neutral; not shipped |
| Fragment graphs | semantic + lexical bridge edges, geometric neighborhoods, DocScorer grid | Fragment geometry (richcov) promoted; geometric neighborhoods and bridge edges were not |

### Key research findings

1. **Fragment geometry with rich coverage (richcov) is the strongest training-free dense signal.**
   PHSS approximate scale selection (LargestGapApprox) is equal-or-better than the exact LargestGap
   path at ~2× throughput. Shipped as `build_doc_semantic_fragments_rich_covered` +
   `FragmentGeometryConfig::use_phss = true` with `PhssConfig::Criterion::LargestGapApprox`.

2. **BM25 variants matter, but corpus-specifically.**
   AtireLTD lifts long-doc corpora (avg_dl > 250). Atire is safest for short-doc.
   Use `recommend_recipe_by_avg_dl()` as a starting point.
   *Fixture-version note:* the earlier claim that BM25-variant RRF fusion
   "regresses on SciFact/FiQA" does not reproduce on the regenerated MiniLM
   fixtures — `bm25_rrf_variants5` tops the SciFact test fold (0.6714). All
   fusion conclusions were re-validated under the dev/test workflow in the
   fusion pass below.

3. **The pre-retrieval router (QueryRouter) works across the three shipped fixtures.**
   Step 1g.1 post-retrieval gates (pool Jaccard, score decay) and Step 1k B2 pre-retrieval
   gates (SCQ, clarity) are the recommended config for production.

4. **Graph-based re-ranking (PPR, cluster) did not transfer.**
   The PPR graph experiments (phrase-doc, doc-doc edges, damping 0.70-0.85) and cluster-topology
   experiments were negative across all three fixtures. These paths were not promoted.

5. **RM3 PRF is corpus-sensitive.**
   It helps NFCorpus but stays flat or regresses on SciFact/FiQA. Keep opt-in and
   tune per corpus.

6. **Concept mining is fragile but occasionally useful.**
   Works on FiQA (structured product descriptions) but the PMI-floor threshold
   needs per-corpus tuning.

7. **Double-array Aho-Corasick was rejected.**
   The dense 256-col goto table outperforms the double-array variant in throughput
   despite larger memory footprint. The DA prototype was removed (commit 7faf23e+).

## PHSS engineering pass (phaseD-perf)

The production `LargestGapApprox` criterion previously copied all m ≈ n(n−1)/2
pairwise similarities and ran a full `std::sort` — only to locate one largest
adjacent gap. Profiling on extracted SciFact/NFCorpus corpora put that sort at
~59% of richcov rerank time. Three bit-identical changes landed:

1. **O(m) pigeonhole maximum-gap** replaces gather+sort in
   `phss_select_scale` (`src/persistent_homology.cpp`). With nb bins over
   [vmin, vmax], any adjacent gap wider than the bin width span/nb must cross a
   bin boundary, so per-bin min/max suffice to find it; the gap value and
   midpoint are then computed from the same two adjacent sorted values the
   sorted scan would subtract, giving bit-identical `selected_scale`/`max_gap`
   with first-maximum (strict `>`) tie-breaking. The bin count is capped at
   8192 to stay cache-resident; whenever the best cross-bin gap is not strictly
   above the mean spacing (≈ uniform spacing, where the sort's tie-break could
   pick an intra-bin pair), the code falls back to the retained sorted
   reference `detail::phss_largest_gap_sorted`, so the degenerate regime is
   bit-exact by construction rather than by argument. NaN inputs remain
   unsupported (they were already UB under `std::sort`). Differential tests
   (`tests/test_persistent_homology.cpp`) assert bit equality against the
   sorted oracle across edge cases and 1,500 seeded randomized trials.
2. **Blocked `dot4` kernel** for the pairwise fragment similarity loop
   (`simd.hpp`, `src/arch/*`). One query row against four candidate rows; each
   output keeps the exact accumulator structure of `dot` (NEON: 2 accumulators,
   8 floats/iter, `vaddvq` reduction; AVX2: 16 floats/iter, 8-lane scalar
   reduction; scalar: double accumulation), so results are bit-identical to
   four independent `dot` calls — the win is amortized `a` loads and ILP, not
   reassociation. Parity asserted bit-exactly in `tests/test_simd_parity.cpp`,
   validated on x86 (Zen 2, Debian 13) under both gcc and clang in addition to
   the arm64 development host.
3. **Adjacency loop**: per-row temp vector hoisted out of the loop and the
   upper-triangle row for j > i read contiguously instead of through the
   per-element triangular index; iteration order unchanged, outputs
   bit-identical.

A second pass extended the same bit-identity discipline:

4. **Sparse adjacency via `simd::scan_ge`** (`simd.hpp`, `src/arch/*`).
   Survivor density above the PHSS scale is typically <1%, so the PHSS-graph
   build extracts surviving pairs with a SIMD `>=` index scan over the
   contiguous triangle rows and does the softmax bookkeeping on the sparse
   edge list. A row-major edge stream delivers each row's partners in the
   same ascending order the per-row scan used, so `row_sum` accumulation and
   the normalized weights are bit-identical; `row_max` is an order-independent
   float max. The kNN (non-PHSS) graph path is untouched —
   `std::partial_sort`'s tie behavior is implementation-defined and can't be
   replicated provably.
5. **PHSS range stats fused into the pairwise loop** (`PhssStats`,
   `persistent_homology.hpp`). The pairwise loop computes count/min/max via
   `simd::range` while each row is cache-hot (min/max are associative, so
   lane-parallel reduction returns identical values) and hands them to
   `phss_select_scale`, eliminating the select gather pass when no edge
   threshold is set.
6. **2x4 blocked pairwise kernel `dot2x4`** (NEON; AVX2 dispatches to two
   `dot4` calls — 16 ymm registers can't hold 16 accumulators without
   spills). Sharing each b-row load across two a-rows cuts load traffic ~40%
   versus `dot4`; every output keeps `dot`'s accumulator structure, so results
   stay bit-identical. 1.55× measured on the pairwise phase.
7. **Float-index bucketing** in the max-gap pass. (s − vmin) is exact and
   monotone in float; scaling by a positive constant and truncating preserve
   weak monotonicity, so float index math is a valid bucketing — it only
   shifts bin boundaries by a few ulps of nb (<1% of a bin width). The
   degenerate fallback margin is widened to bin width × (1 + 1/64) to absorb
   that shift. Removing the float→double convert chain was worth 3.7× on the
   bucket pass.

Bit-identity is a per-architecture guarantee: `dot_neon` and `dot_avx2` use
different accumulator widths by design, so cross-architecture similarity values
(and hence `phss_scale_mean`, e.g. 0.538512 arm64 vs 0.538508 Zen 2 on the
SciFact profile corpus) differ at the 1e-5 level. This predates the engineering
pass; determinism holds within an architecture.

Effect (`simeon_profile_fragment_geometry`, richcov, 50 queries × 5 iters,
Apple Silicon; per-phase tables in benchmarks.md): approx-mode rerank mean
8729 → 1896 µs/query on SciFact and 8533 → 1823 µs on NFCorpus (~4.6×
cumulative); `phss_select` itself 5568 → 222 µs (25×), pairwise 2016 → 992 µs,
adjacency 737 → 324 µs. `phss_scale_mean` and `graph_edges_mean` are unchanged
to all printed digits on every fixture × mode, and the exact `LargestGap`
select path is untouched (timings within noise), so nDCG is provably
unaffected. The remaining approx-mode profile is pairwise dot products (~52%,
near the NEON FMA roofline for this layout), adjacency (~17%, dominated by the
kNN-path queries), BM25 pool scoring (~14%), and scale selection (~12%, one
streaming bucket pass).

### graph_prefix_dim: prefix-dim graph similarities (neutral, not promoted)

`FragmentGeometryConfig::graph_prefix_dim` computes the fragment-graph pairwise
cosines (PHSS selection + adjacency) on a renormalized prefix of the whitened
fragment dims, hoping to trade graph fidelity for the dominant pairwise cost.
The PMI-SVD energy-concentration argument does not survive whitening: whitening
equalizes per-dim variance, so a 64-of-128 prefix carries ~half the norm and
prefix cosines have a markedly different distribution. PHSS adapts by selecting
a much lower scale (SciFact profile corpus: 0.539 → 0.104 at gpfx=64), the
graph densifies ~2× and adjacency + diffusion costs grow, eating most of the
2× pairwise saving (total 1897 → 1750 µs at gpfx=64; gpfx=96 is *slower* than
baseline at 2062 µs).

Quality on regenerated MiniLM fixtures (test split, richcov + LargestGapApprox,
nDCG@10): SciFact baseline 0.6657 vs gpfx96/64/48 = 0.6644/0.6647/0.6652;
NFCorpus baseline 0.3028 vs 0.3022/0.3031/0.3037 — deltas within the ±0.005
noise band on both corpora, so neutral rather than harmful. Verdict:
no quality cost but no meaningful speed win either; the knob stays default-off
(0 = full dim) as a research config. Any future prefix-similarity attempt
should operate on pre-whitening PMI coordinates, where the SVD ordering
actually concentrates energy.

## Fusion pass: soft per-query fusion over shipped legs

Motivated by the router-oracle gap (best-of-3-recipes per query = 0.7219/0.3439
vs best fixed 0.6714/0.3182 on SciFact/NFCorpus test) and by the routing-paradox
literature: hard per-query routing underperforms the best fixed method because
QPP signals are too weak for discrete selection, while *soft fusion* (per-query
convex weights over methods, Bruch-Gai 2022) captures part of the oracle gap by
hedging. `--fusion-only` in the research bench sweeps convex combinations of
pool-restricted z-normalized legs — Atire, WSDM(Atire), SAB-smooth, WSDM(SAB),
fragment geometry (pure, α=0), rrf_variants5 — against RRF / CombSUM / CombMNZ
baselines over identical legs, with dev-fold tuning and frozen test validation.

Findings (MiniLM fixtures, dev→test workflow):

- **Promoted: `z(WSDM_sab)·0.6 + z(WSDM_atire)·0.4`** — dev winner on SciFact
  (0.6950 dev), frozen test 0.6885 vs 0.6714 best fixed (+0.0171, > 3× the
  ±0.005 noise gate), entire α-plateau above the old best. On NFCorpus the same
  config is +0.0038 (within noise, no regression). Exported as
  `convex_fuse_z()` in fusion.hpp. Proximity evidence fused across two
  tokenization regimes (exact word vs subword-backoff) is the mechanism: the
  two WSDM legs agree on topicality but disagree usefully on term-match
  confidence.
- Convex combination > CombSUM/CombMNZ > RRF over the same legs on both
  corpora (dev), reproducing Bruch-Gai's ordering in a purely lexical setting.
- **Signal-conditioned α (clarity/NQC quantile interpolation) never beat fixed
  α** on dev — consistent with the QPP-correlation literature; not promoted.
- The fragment-geometry leg did not enter winning combinations on these two
  corpora; its value remains the quality-router path (semantic-tier queries).
- **Union-pool recall**: the 6-leg union pool's oracle is 0.9365/0.5918 vs
  0.8845/0.5275 for the BM25-only k100 pool — leg-diversified candidate
  generation materially raises the rerank ceiling and is the natural input for
  future rerank-precision work.
- NFCorpus's own dev winner (`z(atire)·0.2+z(sab)·0.8`, test +0.0028) stays
  below the promotion gate; treated as neutral.
- **FiQA cross-check**: the promoted config is the top non-oracle row on the
  FiQA test fold (0.2512 vs 0.2375 bm25_only, +0.0137), with the same
  WSDM-pair plateau shape. Final tally: promoted on SciFact + FiQA, neutral on
  NFCorpus, regression on none. FiQA's 6-leg union-pool oracle is 0.6073.

### Fused-feedback RM3 (`prf_fused`): promoted as a blend leg

The rerank workbench's gap decomposition showed only +0.03–0.05 of the
union-pool-oracle gap is recoverable by combining the six existing legs; the
rest needs new document-level evidence. First candidate family tested via
workbench feature legs:

- **Fused-feedback RM3 beats classic RM3 as a standalone leg on every
  corpus** (single-leg: 0.6648 vs 0.5747 SciFact dev, 0.2884 vs 0.2633
  NFCorpus dev, 0.2249 vs 0.1971 FiQA test — three for three, both folds).
  Mechanism: RM3's known weak point is feedback quality; anchoring
  the relevance model on the *promoted fusion's* top-10 (softmax(z) doc
  weights) instead of the BM25 first pass gives a cleaner pseudo-relevant
  set. Library support: the `score_with_prf` overload taking an explicit
  feedback set (prf.hpp).
- **Blend (corpus-sensitive, opt-in)**:
  `0.3·z(prf_fused) + 0.7·(0.6·z(wsdm_sab)+0.4·z(wsdm_at))` — test nDCG@10
  0.6990 SciFact (+0.0105 over the WSDM fusion), 0.3261 NFCorpus (+0.0042),
  0.2469 FiQA (−0.0043). The pf-weight plateau (0.1–0.5) clears the baseline
  on both SciFact and NFCorpus test folds, but FiQA is flat-to-negative —
  the same corpus profile RM3 has always shown (short scientific corpora
  benefit; financial QA does not). Keep opt-in per corpus, like RM3 itself.
  Cross-fold note: SciFact's train-proxy dev fold had the sign *flipped*
  (−0.002 dev vs +0.0105 test) — the ±0.005 fold-disagreement rule in action.
- **MaxSim-family doc scorers are flat as features** (MaxSim / TopKMean /
  SoftMaxSum / GeoMean over fragment qsims: ≤ +0.0015 in any blend) — the
  fragment signal's aggregation is not the bottleneck on these corpora.
- The best-leg-or-feat oracle (0.7678 vs 0.7382 legs-only, SciFact dev) shows
  features add per-query headroom that fixed blends barely capture — the
  routing-paradox pattern recurs at the feature level.
- **Second screening round, three more negatives** (dev, both corpora):
  iterated PRF (`prf_iter2`, re-anchoring on the prf_fused blend's top-10) is
  dominated by the single fused-feedback round; Callan-style passage windows
  (`passage_w50`, max window-50 saturating-tf score) and Tao-Zhai pair
  proximity (`prox_pair`, exp(−min distance)) are flat-to-negative as blend
  features and nearly signal-free standalone. With the MaxSim family this
  exhausts the obvious within-pool lexical-evidence axis: the remaining
  ~0.2-0.3 pool-oracle gap is not reachable by reweighting or re-reading the
  pool with bag/window/proximity heuristics. The feature columns remain in
  the workbench dump as instrumentation.

### Hubness correction on the geometry leg (CSLS): promoted opt-in

First lever from the score-space-geometry axis (distinct from the exhausted
lexical pool-re-reading axis). High-dimensional similarity spaces grow hubs —
fragments near everything, query included — whose raw qsim overstates
relevance (Radovanović et al. 2010). CSLS (Conneau et al. 2018; QB-Norm
family) subtracts each fragment's mean top-k pairwise similarity — its pool
centrality, already computed in `sims_tri` for PHSS — from its query
similarity before attention. Softmax attention is shift-invariant, so exactly
the per-fragment hub term moves the mass distribution; cost is one O(m·k)
scan of the existing triangle.

Cross-fold result (`csls_k=8, csls_beta=1.0`): standalone geometry leg
+0.035/+0.034 dev/test SciFact, +0.006/+0.012 NFCorpus, flat FiQA, regression
nowhere — promoted as an opt-in `FragmentGeometryConfig` knob for the
standalone rerank path (the FFI rerank surface). Fused into the promoted WSDM
pair the contribution stays within the dev noise gate (SciFact test showed
+0.012 at g=0.20 but dev was flat — the fold-disagreement rule keeps the
fusion config unchanged). Numbers: benchmarks.md "Geometry-leg hubness
correction".

### Screened negatives from the same pass (dev, scifact + nfcorpus)

- **Spectral tempering cannot replace pool whitening.** Corpus-level
  `PmiEmbeddings::temper_spectrum(alpha)` (per-dim sd^-α scaling of the PMI
  coordinates) with per-query whitening off collapses the geometry leg:
  scifact dev 0.093–0.130 across α∈[0,1] vs 0.238 whiten-on; nfcorpus
  0.136–0.145 vs 0.170. Monotone toward α=1 but never close — the
  load-bearing mechanism in whitening is the *pool-local mean centering*,
  which no corpus-level scaling reproduces; tempering + whitening-on is a
  no-op by construction (whitening divides out per-dim constant scales).
  `temper_spectrum` stays as a research primitive (`recipe_accuracy_bench
  ... spectral` mode).
- **Min-max vs z-score fusion normalization is a wash** on the promoted WSDM
  pair (the untested half of Bruch-Gai 2022): scifact dev −0.0017, nfcorpus
  dev +0.0019 — z stays.
- **Query-independent doc-topology priors are inert as fusion legs.** Per-doc
  0-D persistence gap and mean pairwise coherence over each doc's own
  fragments, z-fused into the promoted pair at w=0.1/0.2: scifact dev −0.003
  to −0.011, nfcorpus within noise. Per-query survivor-component topology
  (features of the PHSS graph at the selected scale) remains unexplored.

### The learning-free ceiling: beating MiniLM except on FiQA

The pool-oracle gap (≈0.2–0.3) is a recall ceiling, not an achievable ranking
target — the honest yardstick is the frozen learned dense reference
(`all-MiniLM-L6-v2`). Against it, the promoted fusion stack already wins on the
two lexical-signal-rich corpora and trails only on the paraphrase-heavy one:

| Test nDCG@10 | learning-free best | MiniLM reference | Δ |
|---|---|---|---|
| SciFact | 0.6990 | 0.6451 | **+0.054** |
| NFCorpus | 0.3261 | 0.3167 | **+0.009** |
| FiQA | 0.2512 | 0.3687 | −0.117 |

FiQA (financial QA, short colloquial questions, paraphrase-heavy) is the one
corpus where a learned model demonstrably achieves what the promoted lexical
fusion stack does not. The semantic lever screened in that pass was in-corpus
PMI; a whole-doc **PMI-256 dense cosine** leg was screened
as the strongest variant of the fragment-pooled geometry leg. Result: it is the
weakest signal in the entire workbench on FiQA (single-feature nDCG@10 **0.0805**
vs MiniLM 0.3687) and adds nothing in fusion on any corpus (best blend gain
≤ +0.0024 dev, at near-zero weight). This is the documented confirmation of
Peat & Willett 1991 ("the limitations of term co-occurrence data for query
expansion"): the paraphrase bridges FiQA needs are not present in the corpus
co-occurrence statistics — they require the broad-world pretraining a learned
encoder carries and a learning-free system structurally cannot. Higher PMI rank
cannot recover information the corpus does not contain, so no rank-512 follow-up
was run. **Conclusion of the precision arc**: learning-free fusion is at or above
the learned-dense frontier wherever lexical signal carries relevance; PMI does
not close the vocabulary-mismatch gap.

The later hashed-IDF experiment does not create paraphrase information, but it
shows that the earlier standalone hashed encoder left substantial lexical
discrimination unused: FiQA test nDCG@10 rises from 0.0899 to 0.1416 and
recall@100 from 0.2706 to 0.4043. This closes 18.6% of that encoder's gap to the
frozen MiniLM reference while remaining below the existing full learning-free
fusion score of 0.2512. Retaining 768 FWHT coordinates subsequently raises the
standalone FiQA score to 0.1543 and recall@100 to 0.4330, closing another 5.6%
of the remaining MiniLM gap. Adding that signal to the full fusion stack later
produced 0.2541 FiQA nDCG@10, only +0.0029 over the frozen WSDM pair and below
the promotion gate. The extra candidate recall is real, but the fixed blend
does not convert enough of it into top-ten precision; a learned-style semantic
bridge still does not emerge from corpus statistics alone.

### Engineering: O(corpus) → O(feedback) RM1 build

The fused-feedback RM3 leg (`prf_fused`, promoted via the `CcPrf` rows) calls
`Bm25Index::build_relevance_model`, which originally scanned **every** posting
list in the corpus to harvest expansion terms from a ~10-doc feedback set —
O(total postings) per query regardless of feedback size. A lazily-built forward
index (`doc_id → [(term_hash, tf)]`, constructed once on first use, paid only on
the PRF path) turns this into a forward scan over the feedback docs' own term
lists. Feedback docs are visited in ascending doc-id order so each term's
contributions accumulate in the same order the posting walk used — the RM term
set is **bit-identical** (verified at every scale by `simeon_bench_prf`), so all
BEIR rankings and nDCG are unchanged.

Per-call `build_relevance_model` cost (NEON, release, `bench_prf` synthetic
corpus, k=10 feedback):

| n_docs | inverted µs/call (before) | forward µs/call (after) | speedup |
|---|---|---|---|
| 2,000 | 1,991 | 19.5 | 102× |
| 10,000 | 6,960 | 20.4 | 342× |
| 50,000 | 35,298 | 23.1 | 1,530× |
| 250,000 | 167,198 | 17.8 | 9,373× |

The after-curve is flat in corpus size (the win grows with the corpus); the
one-time forward-index build is ≈ one old call, amortized across all queries.
The reference inverted scan is retained as `build_relevance_model_inverted` for
the bit-identity regression gate. The lazy build is thread-safe (double-checked
locking on an atomic ready flag), so concurrent first-time PRF queries against
one shared index serialize the build and publish it safely.

## Negative results and dead ends

These were investigated and did not produce consistent gains:

- **PMI soft-match query expansion**: did not generalize across corpora.
- **SDM/WSDM bigram legs**: helped some corpora modestly but not enough for default routing.
- **PPR graph reranking**: negative; the back-propagation overestimated authority.
- **Cluster-topology rescoring**: neutral; the fragment-containment signal was too noisy.
- **TextRank/Aho-Corasick aux fields for BM25F**: marginal; the synthetic titles/entities
  didn't carry enough discriminative signal.
- **Double-array Aho-Corasick trie**: slower than dense 256-col goto table despite
  theoretical memory advantages; the dense table is bandwidth-bound, not capacity-bound,
  on the tested dictionary sizes.

## Works cited

See [works_cited.md](works_cited.md) for the paper-level citations backing each
component and experiment in this library.
