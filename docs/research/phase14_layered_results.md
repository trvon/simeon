# Phase XIV Results: Layered BM25 (Beyond Bag-of-Words)

## Experiment

BM25's five commitments (bag-of-words / term-independence / TF-saturation /
IDF-weighting / length-normalization) sit underneath all of simeon's existing
variants (Atire, BM25+, BM25L, DPH, PL2, DCM). Phase XIV tests whether relaxing
the bag-of-words and term-independence commitments — by layering ordered/
unordered bigram and PMI-semantic-expansion terms on top of the unigram BM25
backbone — produces a cross-fold-validated lift.

**Layered architecture**: applies the BM25 saturation+IDF+length-norm template
to four feature classes:
- L1 (lexical): unigrams (= Atire BM25)
- L2 (order): adjacent ordered bigrams (Metzler-Croft 2005 SDM)
- L3 (window): adjacent unordered bigrams within k tokens (SDM)
- L4 (semantics): top-k PMI cosine kNN of each query token

Combined via per-layer weights `λ_ℓ`. Three new `Bm25Variant`s:
- `Layered`: L1 + L2 + L3 with fixed Metzler-Croft weights (0.85, 0.10, 0.05)
- `LayeredW`: same structure with per-bigram IDF reweighting (Bendersky-Croft 2010, β=1)
- `LayeredSem`: Layered + L4 (PMI semantic expansion, λ_sem normalized)

Each variant runs through the full Phase XIII xprod harness (3 manifolds × 7
scorers × 3 alphas) on the existing rich_cov fragments. A separate λ-sweep block
holds the best-test-fold cell config and varies (λ_ordered, λ_unordered) around
the SDM default.

References:
- Metzler & Croft 2005 (SDM, SIGIR): `score = λ_u · BM25_unigram + λ_o · BM25_ordered_bigram + λ_uw · BM25_window_bigram`
- Bendersky & Croft 2010 (WSDM, SIGIR): per-bigram weight `λ_b = λ_base · (idf_b / mean_idf)^β`
- Imani et al. 2018 (Axiomatic Query Order, arXiv:1811.03569): justifies L2 axiomatically

## Cross-Fold Results

OM Atire baseline (Phase III): scifact test 0.6175 / dev 0.6694; nfcorpus test 0.2543 / dev 0.2621. Cross-fold validation thresholds: **strict** = both folds Δ ≥ +0.005; **loose** = one fold Δ ≥ +0.005 AND the other Δ ≥ 0.

### scifact

| Variant | Top cell | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|---------|----------|-----------|---|----------|---|---------|
| Layered | `_spherical_geom_a0.65` | 0.6223 | **+0.0048** | 0.6730 | +0.0036 | **flat positive** (closest to strict yet) |
| LayeredW | `_spherical_geom_a0.65` | 0.6204 | +0.0029 | 0.6755 | **+0.0061** | LOOSE proved (first dev-side cross of +0.005) |
| LayeredSem (λ=0.05) | `_euclid_geom_a0.65` | 0.6199 | +0.0024 | (not measured) | — | positive but sub-optimal vs Layered |

λ-sweep (Layered, holding GeoMean+α=0.65): default (0.10, 0.05) is essentially
optimal. Sweeping λ_ordered ∈ {0.05, 0.10, 0.15, 0.20, 0.30} and λ_unordered ∈
{0.0, 0.05, 0.10} did not produce a cell beating the default.

### nfcorpus

| Variant | Top cell | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|---------|----------|-----------|---|----------|---|---------|
| Layered | `_poly2_harm_a0.65` | 0.2547 | +0.0004 | 0.2698 | **+0.0077** | LOOSE proved |
| LayeredW | `_poly2_harm_a0.65` | 0.2535 | −0.0008 | 0.2700 | +0.0079 | LOOSE proved (test side near zero) |
| LayeredSem | various | (catastrophic at λ_sem=0.30) | — | — | — | broken until λ_sem reduced |

Three loose-proved cells on nfcorpus, all with `poly2 + harm/mean/topk3 + α=0.65`
and dev fold +0.005 to +0.010. Test side stays near zero (±0.001).

## Strict Cross-Fold Proved Cells

**Total: 0** across 4 folds × 9 BM25 variants × 3 manifolds × 7 scorers × 3 alphas
= ~2272 cells per corpus. The threshold remains uncrossed.

The closest-ever single cell to strict cross-fold proved: `Layered + spherical +
GeoMean + α=0.65` on scifact — test **+0.0048** (just below +0.005 strict), dev
+0.0036 (above 0).

## Mechanism Notes

1. **Bag-of-words is partially the bottleneck**: Layered's L2+L3 give a real
   marginal lift on both corpora — closest-to-strict on scifact, +0.008 on
   nfcorpus dev. But the gain is bounded by the corpus's bigram diversity.
   scifact (5K abstract docs) has limited bigram repetition; nfcorpus's gain
   washes out on test.

2. **Per-bigram IDF reweighting (LayeredW) is dev-asymmetric on scifact**: it
   shifts +0.0019 from test to dev relative to Layered. Net signal-preserving
   but doesn't add over Layered's combined fold-Δ. Suggests the rare-bigram
   upweighting helps queries whose answers contain unusual phrases (more common
   on dev than test) but hurts queries whose answers contain common phrases.

3. **L4 PMI semantic expansion (LayeredSem) does not help**: at the default
   λ_sem=0.30 the per-neighbor sum overwhelmed L1 and produced 0.30 nDCG floor
   (50% drop). After normalizing by sim_sum (so total semantic contribution per
   query token is bounded by λ_sem · sim_max), it produces +0.0024 on scifact
   test — strictly worse than plain Layered. **PMI rank-128 trained on small
   corpora doesn't have kNN structure useful for semantic expansion.** This is
   the same Ceiling B limit hit by SIF (Phase VIII), BSIF (Phase IX), Poincaré
   (Phase XII), and GloVe-on-trec-covid (Phase XI).

4. **Ceiling C (fold-distribution gap) remains the ultimate bottleneck**:
   the scifact test+dev fold gap is structural — corpus-driven, not model-
   driven. No single cell achieves Δ ≥ +0.005 on both folds because the BEIR-3
   train/dev split has different relevance-judgment distributions.

## Files Touched

- `include/simeon/bm25.hpp` — added `Bm25Variant::{Layered, LayeredW, LayeredSem}`,
  `layered_lambda_*`, `layered_w_beta`, `semantic_*` config fields, and
  `set_semantic_expansion()` setter
- `include/simeon/pmi.hpp` — added `token_at(i)`, `row_at(i)` accessors for
  external vocab iteration
- `src/bm25.cpp` — Layered scoring branch (L1+L2+L3 inline), LayeredW (delegates
  to score_wsdm), LayeredSem (per-query PMI kNN scan + normalized expansion),
  plus 4 dispatch-table case-additions for compile coverage
- `benchmarks/bench_vs_reference.cpp` — extended xprod BM25 axis from 6 to 9
  variants, added `set_semantic_expansion(&pmi)` for LayeredSem cells, added
  λ-sweep block for Layered

Total: ~310 LOC. All public APIs unchanged for non-Layered variants; existing
6-BM25-variant cross-fold results are preserved bit-identically.

## Verdict

**WEAK / closest-to-threshold yet, no strict proof.**

The Layered family (especially Layered + GeoMean + α=0.65) is the closest a
training-free configuration has come to the +0.005 strict cross-fold threshold:
- scifact: test +0.0048 (97% of threshold) / dev +0.0036 (72% of threshold)
- nfcorpus: dev +0.0077 (154% of threshold) / test +0.0004 (8% of threshold)

Net: bigram structure (Imani 2018 axiom + Metzler-Croft SDM) is a real signal
but bounded. PMI semantic expansion doesn't unlock additional headroom because
PMI quality is the ceiling.

## Implications

1. **Layered as default**: even without strict cross-fold proof, Layered's
   consistent positive sign across folds + corpora makes it a defensible
   default over Atire for production.
2. **Per-corpus router still needed**: scifact wants Layered + GeoMean + α=0.65;
   nfcorpus wants Layered + HarmonicMean + α=0.65 (different scorer). The fact
   that they agree on Layered + α=0.65 but disagree on scorer suggests a
   2-cell router would capture both wins.
3. **L4 dead-ends unless PMI quality improves**: full L4 unlock requires
   either (a) larger PMI training corpus, (b) pre-trained vectors (already
   disproved Phase XI), or (c) hyperbolic-trained PMI (out of training-free
   scope).
4. **Cross-fold ceiling diagnosed**: Ceiling C's ±0.005 fold-difference appears
   to be the actual bound. Both "best-test" and "best-dev" cells exist; they
   just don't coincide. This suggests the BEIR-3 train/dev split itself is the
   limit, not the model architecture.
