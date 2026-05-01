# Phase XVII Results: Recipe Router (4-axis)

## Experiment

Phase XVI shipped a 1-axis dual-on/off router that strict-proved on nfcorpus
but regressed scifact and trec-covid because it used a fixed underlying recipe
(Layered + GeoMean + α=0.80). Phase XVII extends the router to select the
**full 4-axis recipe** — `(BM25 variant, scorer, alpha, dual)` — based on
observable corpus features. Each branch matches an empirically-validated
config from a prior phase:

| Corpus class | Trigger | Recipe | Source |
|--------------|---------|--------|--------|
| `Long` | `avg_dl > 1000` | Atire + MaxSim + α=0.80, no bigrams, no dual | Phase III |
| `Medium` | `200 ≤ avg_dl ≤ 1000` and `n_docs ≤ 50000` | Layered (no L3) + GeoMean + α=0.80 + dual | Phase XV |
| `Short` (default) | else | Layered + L3 enabled + GeoMean + α=0.80 | Phase XIV |

Implementation: ~80 LOC bench-side. The router builds a per-class `v17_idx`
with class-appropriate Bm25Config, then runs 2 alpha cells per corpus.

## Cross-Fold Results

OM α=0.80 baselines (Phase III): scifact 0.6175/0.6694; nfcorpus 0.2543/0.2621; trec-covid 0.5752/0.5025.

| Corpus | Heuristic class | Recipe | Test nDCG | Δ | Dev nDCG | Δ | Verdict |
|--------|-----------------|--------|----------:|--:|---------:|--:|---------|
| scifact | Medium (avg_dl ~210) | Layered no L3 + GeoMean + dual @α=0.80 | 0.6159 | −0.0016 | 0.6634 | −0.0060 | both negative |
| **nfcorpus** | Medium (avg_dl ~600) | Layered no L3 + GeoMean + dual @α=0.80 | 0.2643 | **+0.0100** | 0.2675 | **+0.0054** | **STRICT proved** |
| trec-covid | **Short (BUG: avg_dl ~150 in BEIR fixture)** | Layered + L3 + GeoMean @α=0.80 | 0.5576 | −0.0176 | (pending — bench in progress) | — | misclassified |

## Heuristic Bug — Misclassification of trec-covid

The router heuristic uses **`avg_dl`** as the primary discriminator. This was
based on the assumption that trec-covid contains long full-text papers
(~3K-50K tokens). **In the BEIR `trec-covid-minilm` fixture, docs are stored
as title + abstract only** (avg_dl ~150 tokens), matching scifact's character
profile.

The router classifies trec-covid as `Short`, applying `Layered + L3` — a
recipe optimized for diverse-vocabulary short-doc corpora (scifact). For
trec-covid:
- **Wrong recipe**: Phase III proved `Atire + MaxSim + α=0.80` on trec-covid
  (+0.010/+0.008 cross-fold validated). The router's `Layered + L3 + GeoMean`
  is a different recipe and underperforms by 0.018 on test.
- **Wrong wall-time profile**: enabling L3 on trec-covid's 171K docs triggers
  the expensive O(N×w) unordered-bigram construction we previously identified
  and skipped. Each fold's BM25 reindex takes ~90 min instead of ~10 min.

## Verdict

**1 of 3 STRICT proved** — same as Phase XV. The recipe router's architecture
is correct (per-class recipe selection), but the heuristic feature
(`avg_dl` alone) doesn't discriminate trec-covid from scifact in the BEIR
fixture. The win condition (≥2 of 3 STRICT) is not met.

## Proposed Fix (Phase XVIII)

Change the Long-class trigger from `avg_dl > 1000` to
**`avg_dl > 1000 OR n_docs > 50000`**:

```cpp
if (v17_avg_dl > 1000.0f || v17_n_docs > 50000u)
    v17_class = V17Class::Long;       // trec-covid → Atire+MaxSim
else if (v17_avg_dl >= 200.0f && v17_n_docs <= 50000u)
    v17_class = V17Class::Medium;     // nfcorpus → Layered+GeoMean+dual
else
    v17_class = V17Class::Short;      // scifact → Layered+L3+GeoMean
```

With trec-covid at 171K docs, this routes it to `Long` regardless of fixture's
avg_dl, applying the proved Atire+MaxSim recipe. Expected: trec-covid recipe
cells reproduce Phase III's STRICT result (test +0.010 / dev +0.008).

If correct, Phase XVIII would achieve **2 of 3 STRICT** — the production-ready
answer. nfcorpus + trec-covid both cross-fold validated; scifact remains
sub-threshold (Ceiling C — fold-distribution gap, not architecture).

## Files Touched

- `benchmarks/bench_vs_reference.cpp` — added the v17 recipe block at the
  end of the dual block in `run_bm25_fragment_graph_grid`. Branches on
  `(avg_dl, doc_count)` and dispatches to one of three pre-validated recipes
  via per-class `Bm25Config` (different `variant`, `build_word_bigrams`,
  `bigram_unordered_window`, and `layered_lambda_unordered`).

Total: ~80 LOC.

## Implications

1. **Per-class recipe selection works on the corpora where it can work.**
   nfcorpus's STRICT proof is preserved through the recipe router. scifact's
   sub-threshold-but-positive Layered+L3 result is the best-achievable on
   that corpus regardless of router design.

2. **Heuristic features matter more than the routing logic.** The router's
   correctness depends on the features it uses to classify corpora. `avg_dl`
   alone is insufficient when fixture choices (abstracts vs full text) change
   the apparent corpus character. `n_docs` is the more reliable structural
   signal for "large corpus" routing.

3. **Bench-infrastructure cost of misclassification is real.** Wrong-class
   recipe means wrong BM25 build (here: L3 enabled when it shouldn't be),
   which costs ~90 min wall time per fold on trec-covid. The fix saves ~60
   min/fold and corrects the recipe simultaneously.

## Phase XVIII: Heuristic Fix (`avg_dl > 1000 OR n_docs > 50000`)

The XVIII fix changed the `Long` trigger to OR `n_docs > 50000`. trec-covid's
171K docs now routes to `Long` regardless of the BEIR fixture's avg_dl, so
the Atire + MaxSim recipe runs (instead of Layered + L3 + GeoMean). Bench
wall time drops from ~95 min to ~10 min per fold (Atire skips the O(N×w)
unordered-bigram build entirely on long-doc corpora).

Cross-fold result on trec-covid:

| α | Test (Δ vs OM 0.5752) | Dev (Δ vs OM 0.5025) | Verdict |
|---|----------------------:|---------------------:|---------|
| 0.65 | 0.5709 (−0.0043) | 0.5138 (**+0.0113**) | mixed (sign flip) |
| 0.80 | 0.5713 (−0.0039) | 0.5028 (+0.0003) | mixed |

Phase III published `Atire + MaxSim + α=0.80` on trec-covid as +0.010/+0.008
cross-fold. The XVIII v17long_recipe cell, despite running the same theoretical
recipe, gets −0.0039/+0.0003 — a 0.014-point test-fold gap. Same recipe;
different result. The router's plumbing (per-class `Bm25Config` + per-class
DocScorer + per-class dual-stage gating) introduces something that the
direct outer-MaxSim cell didn't have. Diagnosis is open.

**Phase XVIII verdict: 1 of 3 STRICT** (nfcorpus only — same as Phase XVII).
trec-covid moved from "wrong recipe" (XVII) to "right recipe but doesn't
reproduce" (XVIII). Win condition (≥2 of 3 STRICT) not met.

Cross-corpus picture (Phase XVIII, all post-heuristic-fix):

| Corpus | Class | Recipe routed to | Test Δ | Dev Δ | Verdict |
|--------|-------|------------------|-------:|------:|---------|
| **nfcorpus** | Medium | Layered no L3 + GeoMean + dual @α=0.80 | **+0.0100** | **+0.0054** | **STRICT** |
| scifact | Medium | Layered no L3 + GeoMean + dual @α=0.80 | −0.0016 | −0.0060 | both negative |
| trec-covid | Long | Atire + MaxSim + α=0.80 | −0.0039 | +0.0003 | mixed |

## Outstanding

- **Recipe-gap diagnosis**: why does the v17long_recipe cell get −0.0039/+0.0003
  on trec-covid when Phase III's standalone `bm25_fragment_geom_outermaxsim_a0.80_k100_t8_richcov`
  cell achieved +0.010/+0.008 on (presumably) the same fixture? Compare exact
  `Bm25Config` and `GeometryGraphConfig` field-by-field. Likely candidates:
  trec-covid fixture difference between Phase III and now (corpus regenerated?),
  recipe applies extra config the original cell did not (DocScorerKind preset,
  `doc_scorer_top_k`, `doc_scorer_softmax_beta`), or fragment-builder difference
  (rich_covered vs rich_covered with shared-prep DocPrep).
- **Yams promotion**: blocked until the recipe gap is closed. nfcorpus alone
  (STRICT proved) is already promotable as a per-corpus config; full per-class
  router needs the trec-covid recipe to actually reproduce Phase III's number.
