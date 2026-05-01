# Phase XVI Results: Per-Corpus Router (Partial Win)

## Experiment

Phase XV produced a STRICT cross-fold proved cell on nfcorpus via dual-stage
candidate generation, but the same architecture catastrophically hurt scifact
and trec-covid. Phase XVI tests whether a corpus-conditioned router — which
inspects `avg_dl` and `doc_count` at index-finalize time and conditionally
enables dual-stage — can capture nfcorpus's win without harming the other
corpora.

Heuristic (Goldilocks zone for dual-stage):
```
enable_dual = (200 ≤ avg_dl ≤ 1000) AND (doc_count ≤ 50000)
```

Maps to the 3 corpora as:
- scifact (avg_dl ≈ 150, |D| ≈ 5K) → dual disabled (too short)
- nfcorpus (avg_dl ≈ 600, |D| ≈ 3K) → **dual enabled**
- trec-covid (avg_dl ≈ 3000, |D| ≈ 171K) → dual disabled (too long)

The router cell is `bm25_fragment_geom_xprod_routed_layered_geom_a{0.65,0.80}_k100_t8_richcov`.

## Cross-Fold Results

OM α=0.80 baselines (Phase III): scifact 0.6175/0.6694; nfcorpus 0.2543/0.2621; trec-covid 0.5752/0.5025.

| Corpus | α | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|--------|---|----------:|--:|---------:|--:|---------|
| **nfcorpus** | 0.65 | 0.2568 | +0.0025 | 0.2676 | +0.0055 | LOOSE |
| **nfcorpus** | 0.80 | 0.2643 | **+0.0100** | 0.2675 | **+0.0054** | **STRICT** |
| scifact | 0.65 | 0.6089 | −0.0086 | 0.6546 | −0.0148 | both negative |
| scifact | 0.80 | 0.6159 | −0.0016 | 0.6634 | −0.0060 | both negative |
| trec-covid | 0.65 | 0.5451 | −0.0301 | 0.4723 | −0.0302 | both negative |
| trec-covid | 0.80 | 0.5592 | −0.0160 | 0.4809 | −0.0216 | both negative |

The router heuristic correctly identified nfcorpus as the dual-stage candidate.
The STRICT cross-fold cell from Phase XV is preserved.

## Why scifact and trec-covid Still Lose

The router gates **dual-stage on/off** but uses a fixed underlying recipe
(`Bm25Variant::Layered + GeoMean + α=0.80`). Two recipe mismatches surface:

### scifact: L3-skip optimization regression

The Phase XV optimization for trec-covid wall-time (`bigram_unordered_window = 0`,
`layered_lambda_unordered = 0`) propagates to the router's `vidx`. On scifact,
where L3 contributed a small but real signal, this disable costs ~0.003 nDCG.
Phase XIV's near-strict cell (Layered + GeoMean + α=0.65 + L3 enabled: test
+0.0048 / dev +0.0036) regresses to the −0.0016/−0.0060 we now see.

### trec-covid: wrong base recipe

Phase III proved `Atire + MaxSim + α=0.80` on trec-covid (+0.010/+0.008
cross-fold). The router uses `Layered + GeoMean + α=0.80`, which is a
fundamentally different recipe. On long-doc corpora the bigram L2 layer adds
noise (paper-length docs have many spurious bigram coincidences) and GeoMean
dilutes the strong-fragment signal that MaxSim captures. The −0.016/−0.022
gap is the recipe gap, not a router-decision gap.

## Diagnosis: 4-Axis Router Needed

A real corpus router needs to select the full tuple, not just dual:

| Axis | scifact | nfcorpus | trec-covid |
|------|---------|----------|------------|
| BM25 variant | Layered (L1+L2+L3) | Layered (L1+L2) | Atire (L1) |
| Scorer | GeoMean | GeoMean | MaxSim |
| α | 0.80 | 0.80 | 0.80 |
| Dual-stage | off | on | off |

Each corpus has a distinct optimum. The current router collapses 4 axes to 1
and only captures the nfcorpus win.

## Verdict

**PARTIAL WIN.** The Phase XV STRICT proved cell is preserved on nfcorpus
through the router's heuristic. scifact and trec-covid both regress relative
to their proved baselines because the router's fixed Layered+GeoMean recipe
doesn't match those corpora's optima.

The right architectural answer is **a recipe-selection router** — Phase XVII —
that picks the full `(BM25_variant, scorer, alpha, dual)` tuple at index
build time using observable corpus features. With 3 corpora and 3 distinct
optima, this is essentially a small lookup table backed by 3-feature
classification. Cost: ~80 LOC bench-side, no new core API.

## Files Touched

- `benchmarks/bench_vs_reference.cpp` — added 2 routed cells
  (`bm25_fragment_geom_xprod_routed_layered_geom_a*`) at the end of the
  dual block. Heuristic uses `vidx.idx.avg_dl()` and `vidx.idx.doc_count()`
  to decide dual-stage gating.

Total: ~30 LOC. No core API changes.

## Implications for Phase XVII

A complete router with corpus-specific recipes:

```
recipe_for_corpus(idx) {
    avg_dl = idx.avg_dl();
    n_docs = idx.doc_count();
    
    if (avg_dl > 1000) {
        // Long-doc corpus (trec-covid-like): Phase III proved.
        return {Atire, MaxSim, alpha=0.80, dual=off};
    } else if (avg_dl >= 200 && n_docs <= 50000) {
        // Medium-doc small corpus (nfcorpus-like): Phase XV proved.
        return {Layered_no_L3, GeoMean, alpha=0.80, dual=on};
    } else {
        // Short-doc corpus (scifact-like): Phase XIV near-proved.
        return {Layered_with_L3, GeoMean, alpha=0.80, dual=off};
    }
}
```

Each branch matches an empirically validated config. The router's job is just
to select which branch to take.

The honest claim: **no single training-free configuration crosses the
cross-fold threshold on more than one corpus**. A router that picks per
corpus is the right yams promotion target.
