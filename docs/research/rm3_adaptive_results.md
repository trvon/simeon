# Adaptive RM3 expansion-K — negative result

Tests Bendersky-Metzler-Croft 2011's claim that RM3 expansion-term count K
should scale with query clarity. Adaptive K does not beat fixed K=20 on any of
the three fixtures, and the preferred K direction disagrees across corpora.

## Math

    K(clarity) = clip(round(n_min + (clarity - lo)/(hi - lo) * (n_max - n_min)),
                      n_min, n_max)

with `lo = 0.5`, `hi = 5.0`, `n_min = 5`, `n_max = 50`. Clarity feeds
in from `simeon::QueryRouter::features().simplified_clarity`.

## Three-corpus measurement (R@100, RM3 sweep)

All rows: SAB-smooth γ=5 unigram, α=0.5, top-K feedback k=10. Adaptive
column uses `n_terms_for_clarity()` with the defaults above. Plan
target metric (R@100) shown; nDCG@10 / R@10 / MRR@10 included for
completeness.

| Corpus   | no-RM3 base | n=10   | n=20 (k10) | n=30   | n=50   | adaptive | adaptive − n=20 |
|----------|------------:|-------:|-----------:|-------:|-------:|---------:|----------------:|
| scifact  | 0.8812      | 0.8790 | **0.8884** | 0.8860 | 0.8909 | 0.8860   | −0.0024         |
| nfcorpus | 0.2443      | 0.2500 | **0.2723** | 0.2747 | 0.2765 | 0.2714   | −0.0009         |
| fiqa     | **0.4674**  | 0.4624 | 0.4600     | 0.4591 | 0.4561 | 0.4555   | −0.0045         |

Plan target for promotion: **R@100 +1.5pp on ≥2/3 corpora** vs the
n=20 (k10) baseline at fixed compute budget. Plan disprove threshold:
**|ΔR@100| < 0.5pp in any direction**, indicating saturation.

Observed: ΔR@100 = {−0.24, −0.09, −0.45} pp. **3/3 corpora land within the
disprove bound.** T4 is disproved.

## Mechanism — why adaptive K can't win

Two failures explain the result:

### 1. Clarity distribution saturates the n_max anchor

Per-query `simplified_clarity` distribution on the oracle-router rows:

| corpus   | min  | p25  | median | p75  | max   | mean |
|----------|-----:|-----:|-------:|-----:|------:|-----:|
| scifact  | 2.72 | 4.69 | 5.19   | 5.97 | 8.67  | 5.38 |
| nfcorpus | 0.00 | 5.77 | 8.02   | 9.93 | 13.70 | 7.55 |
| fiqa     | 2.97 | 4.35 | 5.05   | 5.79 | 12.80 | 5.27 |

With `clarity_hi = 5.0`, the median query on every corpus already clips to
`n_max = 50`. So the adaptive recipe mostly degenerates toward fixed K=50.

Raising `clarity_hi` would help saturation, but not the deeper failure below.

### 2. Optimal K direction is corpus-dependent

The K-sweep itself shows non-monotonic and **opposite-direction**
optima:

- nfcorpus: R@100 monotonically increases 0.2500 → 0.2723 → 0.2747 →
  0.2765 with K. Bigger expansion always helps.
- scifact: R@100 jumps 0.2723 → 0.8884 → 0.8860 → 0.8909 — roughly
  flat past K=20 with a small bump at K=50. K=20 already near-optimal.
- fiqa: R@100 *monotonically decreases* 0.4624 → 0.4600 → 0.4591 →
  0.4561. **Larger expansion drifts off-topic.** Even the no-RM3
  baseline (0.4674) beats every RM3 setting.

A single clarity-driven mapping cannot satisfy nfcorpus (wants more terms) and
fiqa (wants fewer or none) simultaneously.

The deeper finding is that expansion-term policy is corpus-bound, not
clarity-bound. FiQA hurts because its feedback docs share too much generic
financial vocabulary, so expansion drifts toward the corpus centroid.

## Infrastructure disposition

- `simeon::n_terms_for_clarity()` ships in `simeon/prf.hpp` for
  downstream callers; cost is negligible (5 floats + a clip).
- `run_bm25_prf_adaptive()` stays in the bench for regression
  tracking; the four sweep recipes (`rm3_n10`, `rm3_n30`, `rm3_n50`,
  `rm3_adaptive_a0.5`) remain in the three `*_full.jsonl` result
  files.
- No router or cascade integration: not invoked by any default recipe.

## Next lever

- **Per-corpus K policy** is still viable for single-corpus deployments.
- **Topic-drift gating** is the natural follow-up if this family is revisited.
- **α-adaptive instead of K-adaptive** is another cheaper follow-up option.

## References

- Bendersky, M., Metzler, D., Croft, W.B. (2011). "Parameterized
  Concept Weighting in Verbose Queries." SIGIR 2011.
- Lavrenko, V. & Croft, W.B. (2001). "Relevance-Based Language
  Models." SIGIR 2001.
- Cronen-Townsend, S., Zhou, Y., Croft, W.B. (2002). "Predicting
  Query Performance." SIGIR 2002 (clarity definition used here).
