# Phase XV Results: Dual-Stage Candidate Generation

## Experiment

The first 14 phases all attacked the **rescoring layer** — given the BM25 top-100,
how do we rerank it? Phase XV attacks the **candidate-set layer** itself.

Architecture:
```
                          ┌── BM25 top-100 ──┐
query ─► encode_query ─►  ├── union dedupe ──┼─► fragment-rerank ─► top-10
                          └── Dense top-K ───┘

where Dense top-K = top-K docs by cosine(query_vec, doc_dense_vec),
      doc_dense_vec = the centroid fragment from rich_cov_doc_frags[doc]
```

Re-uses the **already-computed centroid fragment** that lives at index `n-2`
of every doc's `rich_cov_doc_frags` (built by `build_doc_semantic_fragments_rich`
as `L2_normalize(mean(sentence_fragment_vectors))`). No new fragment work needed.

The motivation: **Ceiling A** says `nDCG@10 ≤ nDCG@10(oracle_rerank(BM25_top_100))`.
On scifact the gap is 0.094 (oracle 0.713 vs production 0.619). Every prior
phase has competed for slices of that 0.094 by re-scoring; XV expands the pool
so the geometry leg sees documents BM25 missed.

## Cross-Fold Results

OM Atire baseline (Phase III): scifact test 0.6175 / dev 0.6694; nfcorpus test 0.2543 / dev 0.2621. Cross-fold validation thresholds: **strict** = both folds Δ ≥ +0.005; **loose** = one fold Δ ≥ +0.005 AND the other Δ ≥ 0.

### nfcorpus — **2 STRICT proved + 10 LOOSE proved**

This is the **first STRICT cross-fold proved cell of the entire research arc.**

| Cell | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|------|----------:|--:|---------:|--:|---------|
| `dual_layered_spherical_geom_dp200_a0.80` | 0.2640 | **+0.0097** | 0.2673 | **+0.0052** | **STRICT** |
| `dual_layered_euclid_geom_dp200_a0.80` | 0.2636 | **+0.0093** | 0.2673 | **+0.0052** | **STRICT** |
| `dual_layered_spherical_geom_dp100_a0.80` | 0.2639 | +0.0096 | 0.2670 | +0.0049 | LOOSE |
| `dual_layered_euclid_geom_dp100_a0.80` | 0.2636 | +0.0093 | 0.2670 | +0.0049 | LOOSE |
| `dual_layered_euclid_geom_dp50_a0.80` | 0.2629 | +0.0086 | 0.2664 | +0.0043 | LOOSE |
| ... (7 more LOOSE cells) |

Pattern: **bigger dense pool wins.** dp200 > dp100 > dp50. α=0.80 (default) outperforms α=0.65 across all dual cells. Both Spherical and Euclidean kernel give equivalent results (manifold remains a noise axis, consistent with prior phases).

For comparison, the best **Layered** cell from Phase XIV on nfcorpus: test −0.0027 / dev +0.0049. The dual-stage architecture lifts the test fold from −0.003 to +0.010 — that's a **+0.0124 swing** from pool expansion alone.

### scifact — Dense noise

| Cell (best dual) | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|------------------|----------:|--:|---------:|--:|---------|
| `dual_layered_spherical_geom_dp50_a0.80` | 0.6193 | +0.0018 | 0.6605 | −0.0089 | mixed |
| `dual_layered_spherical_geom_dp200_a0.80` | 0.6136 | −0.0039 | 0.6628 | −0.0066 | mixed |
| ... (10 more, all mixed/negative) |

Pattern: **expansion hurts on scifact.** Larger dpool → worse nDCG. α=0.65 is dramatically worse than α=0.80. None of the 12 dual cells beats the best Phase-XIV Layered cell (test +0.0048 / dev +0.0036).

## Mechanism

**Why it works on nfcorpus**: nfcorpus is medical literature (~3K docs) with diverse vocabulary. The BM25 top-100 has limited recall — many relevant docs use synonyms or related-but-not-overlapping terms with the query. Dense retrieval via the per-doc centroid (which is a content-summary in PMI space) recovers some of these missing relevant docs. The unified pool (BM25 ∪ dense) gives the geometry leg access to recall it didn't have before.

**Why it fails on scifact**: scifact is short abstracts (~5K docs). BM25's TF-IDF on short text is already a near-optimal retrieval signal — the relevant docs almost always share surface vocabulary with the query. Dense retrieval can't add to recall (already saturated by BM25); it just contaminates the pool with cosine-similar but irrelevant docs. The geometry rerank then competes "noise vs. real" docs, and noise wins enough times to drag nDCG down.

The mechanism is consistent with **Ceiling A**'s structure: pool expansion only helps when there's actual recall opportunity. For corpora where BM25 top-100 already captures the relevant set, expansion just dilutes precision.

## Verdict

**PROVED on nfcorpus** — first STRICT cross-fold proved cells of the entire research arc.

**DENSE NOISE on scifact** — BM25 pool already saturates recall there.

The corpus-dependent pattern means this is **NOT a universal win** but **IS the first config that achieves cross-fold validation on a real corpus**. A router-per-corpus that picks dual-stage on nfcorpus and Layered-only on scifact would compound: both corpora cross-fold-validated, just by different recipes.

## Files Touched

- `include/simeon/fragment_geometry.hpp` — added `dense_pool_size`, `doc_dense_vecs`
  fields to `FragmentGeometryConfig`; exposed `read_frag_vec` in public API
- `src/fragment_geometry.cpp` — pool augmentation block in
  `score_fragment_geometry_profiled`; renamed internal `read_frag_vec` to
  `read_frag_vec_impl`, added public wrapper at TU end
- `benchmarks/bench_vs_reference.cpp` — extracted `doc_dense_vecs` from
  `rich_cov_doc_frags` centroid fragments at xprod harness setup; added 12
  dual cells (3 dpool × 2 alpha × 2 manifold)

Total: ~85 LOC. No API breaks for non-dual callers; existing xprod cells
identical to prior phases.

## Implications

1. **Ceiling A is real and partially traversable.** On a corpus where BM25
   under-recalls (nfcorpus), pool expansion via dense retrieval lifts nDCG
   above the historical sub-threshold ceiling.

2. **PMI-centroid dense retrieval is good enough on diverse-vocabulary corpora.**
   Phases XI, XII, XIV-C disproved PMI-based semantic operations at the
   *fragment* level. But at the **doc** level (centroid of fragments), PMI
   captures enough corpus-wide structure to recover missing recall.

3. **The "right" answer is corpus-aware retrieval.** Different corpora have
   different recall ceilings under BM25 alone. The next experiment should be
   a router that detects "BM25 likely under-recalls" (e.g. via score-decay
   shape, query-vocabulary entropy, doc-length distribution) and conditionally
   enables dual-stage pool expansion.

4. **trec-covid is now genuinely interesting.** Phase XV results suggest
   dual-stage will help on long-document corpora. trec-covid (171K docs,
   medical abstracts/full-texts) is the largest unsampled corpus and its
   long docs have the diverse vocabulary that nfcorpus's win predicts.

## Trec-Covid Bench Optimization (2026-04-30)

Initial trec-covid attempts under `--dual-only` stalled in setup for 100-130+
minutes with no cells beyond the 2 baseline rows. Profiling with `sample`
identified the actual bottleneck: **the unordered bigram window** in
`Bm25Index::add_doc`. For a doc of length N tokens with window w=8, the inner
loop emits `N × w` pair counts. trec-covid's full-text papers (~3K-50K tokens
each) blow this up to ~24K-400K ops per doc × 171K docs = **billions of
operations per BM25 reindex**, with the global unordered bigram posting map
ballooning to tens of millions of entries (50GB+ RSS).

Three optimizations applied:

1. **`FlatHashMapU64::clear()` added** (include/simeon/flat_hash_map_u64.hpp):
   resets entries while preserving allocated capacity, enabling thread-local
   reuse across docs.
2. **Per-doc TF maps swapped** (src/bm25.cpp): 4 `std::unordered_map<u64, u32>`
   thread-local accumulators in `add_doc` replaced with `FlatHashMapU64<u32>`.
   Cache locality + faster insertion path. 0 nDCG drift on validation.
3. **Initial-`idx` bigram build skipped under `--dual-only`** (benchmarks/bench_vs_reference.cpp):
   the dual block has its own dedicated `vidx` for Layered scoring; the initial
   `idx` only feeds the bm25_only baseline (no bigrams needed). Setting
   `bcfg.build_word_bigrams = !dual_only` skips the expensive O(N×w) work
   entirely on the initial pass.
4. **Dual-block Layered's L3 layer disabled** (`bigram_unordered_window=0`,
   `layered_lambda_unordered=0`): L3 contributes only λ=0.05; skipping the
   O(N×w) inner loop on the dual-block reindex is the bigger win. Introduces
   ±0.003 nDCG drift on scifact dual cells (acceptable for trec-covid run).

**Wall-time impact**: scifact dual-only went from 98 sec → 12.6 sec (8×
speedup). trec-covid both folds completed in **~10 min total** vs the prior
130+ min stuck pattern.

## Trec-Covid Cross-Fold Result (2026-04-30) — DENSE NOISE

OM α=0.80 baseline (Phase III): trec-covid test 0.5752 / dev 0.5025.

| Cell | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|------|----------:|--:|---------:|--:|---------|
| `dual_layered_spherical_geom_dp200_a0.80` | 0.5486 | −0.0266 | 0.4913 | −0.0112 | mixed |
| `dual_layered_euclid_geom_dp200_a0.80` | 0.5480 | −0.0272 | 0.4913 | −0.0112 | mixed |
| `dual_layered_*_geom_dp50_a0.80` | ~0.5410 | ~−0.034 | ~0.4720 | ~−0.030 | mixed |
| `dual_layered_*_geom_*_a0.65` | 0.4940-0.4990 | −0.077 to −0.081 | 0.4480-0.4550 | −0.048 to −0.054 | mixed |
| `bm25_only` (baseline) | 0.5649 | −0.0103 | 0.4943 | −0.0082 | (BM25 floor) |

**Strict proved: 0. Loose proved: 0. Best dual cell underperforms OM α=0.80 by 0.027 on test.**

## Cross-Corpus Phase XV Picture (Final)

| Corpus | Verdict | Best dual cell vs OM α=0.80 (test / dev) |
|--------|---------|------------------------------------------|
| **nfcorpus** (3K docs, medical) | **STRICT proved** | **+0.0097 / +0.0052** |
| scifact (5K docs, short abstracts) | dense noise | +0.0018 / −0.0089 |
| trec-covid (171K docs, full-text papers) | dense noise (worst) | −0.0266 / −0.0112 |

The Phase XV mechanism is corpus-dependent and predicted by the doc-vector
quality:
- **nfcorpus**: BM25 under-recalls medical synonym phrases; PMI rank-128
  centroid retrieval recovers some missed docs. Expansion is positive-sum.
- **scifact**: short abstracts; BM25 saturates recall. Expansion adds noise.
- **trec-covid**: long full-text papers; PMI centroid (mean of fragment vecs)
  becomes diffuse over many sentences, losing discriminative signal. Dense
  retrieval pulls in topically-near-but-irrelevant docs. Expansion is
  catastrophic.

Phase XV's STRICT cross-fold proved cells **stand on nfcorpus**. The right
production strategy is **per-corpus router** (Phase XVI) — enable dual-stage
on corpora with under-recall-prone characteristics, fall back to Layered-only
otherwise.

## Next Steps

- **Phase XVI: per-corpus router.** Detect under-recall heuristic, conditionally
  enable dual-stage. Target: cross-fold validation on ≥2 of 3 corpora.
- **trec-covid validation.** With dual-stage as a proven config on long-doc
  corpora, the trec-covid bench (still infrastructure-blocked) becomes worth
  the wall-time investment.
- **Yams promotion.** `Bm25Variant::Layered + dense_pool_size=200` on
  nfcorpus-like corpora is the first promotable configuration since Phase III.
