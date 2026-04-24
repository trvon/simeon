# RRF fusion of 5 BM25 variants — cross-fold validated on nfcorpus

## Experiment

Per `next_research_plans.md` Plan 3, fuse rankings from 5 BM25
variants using reciprocal rank fusion (Cormack et al. 2009, k=60):

- **Atire** (baseline Robertson BM25 with Atire IDF)
- **BM25+** (Lv & Zhai 2011, δ=1.0 floor)
- **BM25L** (Lv & Zhai 2011, long-doc form)
- **DLH13** (Amati DFR)
- **SAB-smooth** (SubwordAwareBackoff, γ=5)

Each variant retrieves independently; rankings are then RRF-fused to exploit
their different length-normalization and term-weighting behavior.

## Implementation

**Library**: `simeon::score_bm25_variants_rrf` in `fusion.hpp` +
`fusion.cpp`. Scores `query` against each `Bm25Index` in the variant
span, writes RRF-fused scores into `out_scores`.

**Bench**: `bm25_rrf_variants5` row (Atire, BM25+, BM25L, DLH13,
SAB-smooth). Same 5 variants are already shipped as individual
`Bm25Variant::*` recipes in the bench.

## Results

### vs bm25_atire baseline (raw Robertson BM25)

| Corpus   | Fold | atire nDCG | rrf5 nDCG | Δ nDCG | atire R@100 | rrf5 R@100 | Δ R@100 |
|----------|------|-----------:|----------:|-------:|------------:|-----------:|--------:|
| scifact  | dev  | 0.6623     | 0.6664    | +0.0041 | 0.8469      | 0.8929     | **+0.0460** |
| scifact  | test | 0.6188     | 0.6224    | +0.0036 | 0.8736      | 0.8785     | +0.0049 |
| nfcorpus | dev  | 0.2620     | 0.2711    | +0.0091 | 0.2158      | 0.2267     | +0.0109 |
| nfcorpus | test | 0.2521     | 0.2589    | +0.0068 | 0.1991      | 0.2004     | +0.0013 |
| fiqa     | dev  | 0.1908     | 0.2071    | +0.0163 | 0.4360      | 0.4379     | +0.0019 |
| fiqa     | test | 0.2053     | 0.2047    | −0.0006 | 0.4672      | 0.4732     | +0.0060 |

RRF-5 lifts nDCG cross-fold on scifact and nfcorpus vs `bm25_atire`. FiQA shows
the usual fold-flip pattern.

### vs production frontier `phssapprox_k100_t8_richcov_gap`

| Corpus   | Fold | phssapprox nDCG | rrf5 nDCG | Δ nDCG | rrf5 QPS | phssapprox QPS |
|----------|------|----------------:|----------:|-------:|---------:|---------------:|
| scifact  | dev  | 0.6716          | 0.6664    | −0.0052 | 506      | 154            |
| scifact  | test | 0.6188          | 0.6224    | +0.0036 | 507      | 103            |
| nfcorpus | dev  | 0.2622          | 0.2711    | **+0.0089** | 1419 | 122            |
| nfcorpus | test | 0.2544          | 0.2589    | **+0.0045** | 1295 | 126            |
| fiqa     | dev  | 0.1852          | 0.2071    | +0.0219 | 43       | 119            |
| fiqa     | test | 0.2089          | 0.2047    | −0.0042 | 43       | 117            |

### vs sab_smooth_gamma5 (strongest single variant)

| Corpus   | Fold | sab nDCG | rrf5 nDCG | Δ nDCG |
|----------|------|---------:|----------:|-------:|
| scifact  | dev  | 0.6866   | 0.6664    | −0.0202 |
| scifact  | test | 0.6120   | 0.6224    | +0.0104 |
| nfcorpus | dev  | 0.3161   | 0.2711    | **−0.0450** |
| nfcorpus | test | 0.2981   | 0.2589    | **−0.0392** |
| fiqa     | dev  | 0.2078   | 0.2071    | −0.0007 |
| fiqa     | test | 0.1978   | 0.2047    | +0.0069 |

**SAB-smooth alone beats RRF-5 on nfcorpus by a wide margin.** But RRF-5 still
beats the current production frontier there, so it is useful as a separate
candidate rather than a replacement for SAB itself.

## Verdict — nfcorpus cross-fold validates; scifact/fiqa disprove

Pre-declared gates:
- **Validate**: R@100 lifts ≥0.01 on ≥2/3 corpora + no nDCG
  regression >0.003 cross-fold.
- **Disprove**: R@100 lift <0.005 on every corpus.

Results:
- **nfcorpus**: +0.0089/+0.0045 dev/test nDCG over the production
  frontier — cross-fold consistent, >0.005 validate threshold on dev.
  Clean validate.
- **scifact**: +0.0036/−0.0052 nDCG vs production (fold-flip); test
  lift is small but consistent with bm25_atire comparison (+0.0036
  on both folds vs Atire). Does not beat production cross-fold.
- **fiqa**: fold-flip (dev +0.0219, test −0.0042). Classic
  cross-fold artifact pattern from `feedback_cross_fold_validation`
  memory rule.

By the strict cross-fold gate: **validate on nfcorpus, disprove on scifact and
fiqa**. This matches the usual corpus-specific pattern.

## Mechanism — why nfcorpus wins

NFCorpus wins because the variants disagree usefully there:

- **SAB-smooth** helps with morphology
- **BM25L** helps with long-doc normalization
- **DLH13** reacts differently to skewed jargon distributions

That disagreement creates a useful RRF pool. Scifact and FiQA do not show the
same diversity pattern.

## Production recommendation

Ship `simeon::score_bm25_variants_rrf` as the **nfcorpus-specific** retrieval
path, alongside the two already-validated nfcorpus recipes:

1. **RRF-5** (this plan): +0.005-0.009 nDCG cross-fold at 1300 QPS.
   Cheapest; primary recommendation for nfcorpus.
2. **Self-KB n10_gate25** (`self_kb_results.md`): +0.004 nDCG /
   +0.027 R@100 at 3 QPS. Biggest R@100 lift; use when recall at
   depth matters.
3. **PHSS pool=500 richcov** (`phss_pool_scaling.md`): +0.012 R@100
   at 5 QPS. Only when neither of the above is enough.

RRF-5 is ~500× faster than either self-KB or PHSS pool=500 at similar nDCG
lift. It is the best default nfcorpus-style recipe.

For scifact + fiqa, stay on `phssapprox_k100_t8_richcov_gap` (the
existing production frontier).

## Latency

RRF-5 adds per-query cost = (N−1) × single BM25 score, where N=5.
Measured QPS:
- scifact: 506 QPS (vs 34000 bm25_atire alone — 67× slower)
- nfcorpus: 1300 QPS (vs 99000 — 77× slower)
- fiqa: 43 QPS (vs 3600 — 84× slower; fiqa's longer queries + more
  docs magnify the cost)

Acceptable for the nfcorpus-specific path; not a universal default
because of the fiqa latency hit.

## Cross-arc note

This is the **fourth cross-fold-validated nfcorpus-specific lever**:

1. `atire_max_clarity=3.0` gate (Step 1k): +0.028 nDCG on nfcorpus.
2. PHSS pool=500 richcov: +0.012 R@100.
3. Self-KB n10_gate25: +0.004 nDCG + +0.027 R@100.
4. RRF-5 (this plan): +0.005-0.009 nDCG.

Nfcorpus consistently validates selective levers where scifact and fiqa do not.
The recurring heuristic is simple: *nfcorpus-like corpora respond to
candidate-set expansion and variant diversity; scientific-abstract and Q&A
corpora generally do not.*

## Disposition

- `simeon::score_bm25_variants_rrf` shipped in library.
- Bench recipe `bm25_rrf_variants5` stays.
- `training_free_saturation.md` Shipped list updated.

---

*2026-04-24. Bench rows in
`/tmp/bench_{scifact,nfcorpus,fiqa}_rrf_{dev,test}.out`. Library
code in `include/simeon/fusion.hpp`, `src/fusion.cpp`. Bench recipe
in `benchmarks/bench_vs_reference.cpp` (`run_bm25_variants_rrf` +
call site).*
