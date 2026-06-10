# Research notes

This file summarizes what the research benchmark infrastructure (`bench_vs_reference`
and `bench_vs_reference_research`) probes, what each sweep found, and which
components were promoted to the production surface.

Build with `-Denable_research=true` to enable research-only components (GloVe loader,
ArguAna adapters, SelfAssessRouter, Rm3DiverseStrategy) and the reference-comparison
benchmark binaries.

## Research benchmark sweeps

All sweeps run against an external fixture (corpus + queries + qrels + reference
embeddings). The fixture format is documented in [reference_fixture.md](reference_fixture.md).
Both the production bench (`bench_vs_reference`) and the research bench
(`bench_vs_reference_research`) consume the same fixture layout.

### Production surface (shipped router recipes)

The default `bench_vs_reference` binary keeps the stable comparison surface:
- **BM25 variants** (Atire, AtireLTD, SAB-smooth, BM25+, BM25L, DPH, PL2, DCM, Layered, LayeredW)
- **BM25 ↔ simeon RRF fusion** at pool sizes 100/300/500 and α ∈ {0.65, 0.75, 0.85, 0.95}
- **Fragment-geometry rerank** (richcov, PHSS approx, MaxSim SPLATE)
- **Query router** (pre-retrieval predictors → Atire vs SAB vs CascadeLinearAlpha)
- **Quality router** (short vs medium vs long semantic query → Bm25Only vs FragmentRichCovPhssApprox vs FragmentRichCovPhssApproxMax)
- **Step 1g.1 post-retrieval gates** (pool Jaccard, score decay) and **Step 1k B2 pre-retrieval gates** (SCQ, clarity)
- **BM25-variant RRF** (`score_bm25_variants_rrf`) and entropy-weighted linear-α fusion (`linear_alpha_entropy_fuse`)
- **RM3 pseudo-relevance feedback** (`score_with_prf`)
- **Concept mining** (`mine_concepts` + BM25-concept blend)
- **Corpus-class recipe routing** (AtireLTD for long-doc corpora, Atire for short-doc)

### Research sweeps (opt-in, `bench_vs_reference_research`)

| Sweep flag | Description | Outcome |
|---|---|---|
| `--aux-from {textrank,ac}` | BM25F with synthetic aux fields (TextRank titles, Aho-Corasick entities) | Marginal; not shipped |
| `--softmatch-only` | PMI-neighbor query expansion (corpus-derived, training-free) | Neutral-to-negative; not shipped |
| `--transport-only` | Phrase/concept transport via SDM bigrams + mined concepts + z-scored BM25 pool fusion | Mixed; concept leg occasionally helped, phrase leg did not |
| `--graph-only` | Personalized-PageRank-style graph reranking over query-phrase-doc graphs | Negative; not shipped |
| `--cluster-only` | TextRank-fragment cluster rescoring by query-cover and containment mass | Neutral; not shipped |
| `--fragment-only` | PMI-encoded fragment graphs with semantic edges + lexical bridge edges + geometric query-centered neighborhoods + DocScorer cross-product grid | Fragment geometry (richcov) promoted to production; geometric neighborhoods and hybrid bridge edges were not |

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

## Components gated behind `enable_research`

When `-Denable_research=false` (production default), the following are excluded:

| Component | Files | Reason |
|---|---|---|
| GloVe/fastText loader | `glove_embeddings.hpp/cpp` | Only used by research benchmarks |
| SelfAssessRouter | `retrieval_strategy.hpp/cpp` | Only used by research benchmarks |
| Rm3DiverseStrategy | `retrieval_strategy.hpp/cpp` | Only used by research benchmarks |
| ArguanaAdapter | `corpus_adapter.hpp/cpp` | Corpus-specific; never used in production |
| ArguanaTextPairAdapter | `corpus_adapter.hpp/cpp` | Corpus-specific; never used in production |
| `bench_vs_reference` | `benchmarks/bench_vs_reference.cpp` | Requires GloVe + SelfAssessRouter |
| `bench_vs_reference_research` | `benchmarks/bench_vs_reference_research.cpp` | Research-only sweeps |

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

For detailed per-sweep results including nDCG tables, run:
```sh
./build/benchmarks/simeon_bench_vs_reference fixtures/<dataset> > results.jsonl
./build/benchmarks/simeon_bench_vs_reference_research fixtures/<dataset> --<flag> > research.jsonl
```

## Works cited

See [works_cited.md](works_cited.md) for the paper-level citations backing each
component and experiment in this library.
