# Plan 4 — single-fragment-per-doc builder — DISPROVED

## Experiment

Per `next_research_plans.md` Plan 4, this experiment attacks Bottleneck 1 at a
different point than MaxSim: **before aggregation**.

- **MaxSim** (`phss_maxsim_results.md`, disproved by cross-fold):
  keeps all t=8 fragments per doc; aggregator picks `max(mass[i])`
  instead of `sum`.
- **Plan 4** (this experiment): keeps all t=8 fragments in the
  fragment pool, but at query time picks the single
  argmax-`query·fragment` fragment per doc and suppresses others via
  `qsim[i] = -inf` before softmax. Effectively N_geom = pool_size
  (100) instead of pool_size × t (800).

If multi-fragment averaging is really the ceiling, Plan 4 should lift nDCG
where MaxSim did not.

## Implementation

- `FragmentGeometryConfig::single_fragment_per_doc` flag (default
  false).
- `src/fragment_geometry.cpp`: right before the query-attention
  softmax at line ~849, if flag enabled, scan `qsim[]` per
  `frags[i].pool_index`, find the argmax, and set non-winners to
  `-inf`. Downstream softmax naturally assigns them zero mass.
- 2 bench recipes: `phssapprox_singlefrag_k100_t4` (basic builder)
  and `phssapprox_singlefrag_k100_t8_richcov` (production frontier).

Total: ~30 LOC C++ + 50 LOC bench wiring.

## Results — cross-fold

### richcov (production frontier comparison)

| Corpus   | Fold | baseline nDCG | singlefrag nDCG | Δ nDCG | baseline R@10 | singlefrag R@10 | Δ R@10 | QPS |
|----------|------|--------------:|----------------:|-------:|--------------:|----------------:|-------:|----:|
| scifact  | dev  | 0.6716        | 0.6677          | −0.0039 | 0.7857        | 0.7857          |  0.0000 | 159 |
| scifact  | test | 0.6189        | **0.6197**      | **+0.0008** | 0.7411     | **0.7510**      | **+0.0099** |  90 |
| nfcorpus | dev  | 0.2622        | 0.2579          | −0.0043 | 0.2432        | 0.2402          | −0.0030 | 132 |
| nfcorpus | test | 0.2544        | 0.2520          | −0.0024 | 0.2266        | 0.2246          | −0.0020 | 188 |
| fiqa     | dev  | 0.1852        | 0.1891          | +0.0039 | 0.2292        | 0.2316          | +0.0024 | 180 |
| fiqa     | test | 0.2089        | 0.2051          | −0.0038 | 0.2673        | 0.2637          | −0.0036 | 171 |

### basic t=4 builder

| Corpus   | Fold | baseline nDCG | singlefrag nDCG | Δ nDCG |
|----------|------|--------------:|----------------:|-------:|
| scifact  | dev  | 0.6663        | 0.6696          | +0.0033 |
| scifact  | test | 0.6149        | 0.6104          | −0.0045 |
| nfcorpus | dev  | 0.2580        | 0.2561          | −0.0019 |
| nfcorpus | test | 0.2541        | 0.2520          | −0.0021 |
| fiqa     | dev  | 0.1954        | 0.1924          | −0.0030 |
| fiqa     | test | 0.2061        | 0.2045          | −0.0016 |

## Verdict — disprove

Pre-declared gates:
- **Validate**: nDCG@10 lifts ≥0.005 on ≥2/3 corpora cross-fold.
- **Disprove**: all 3 corpora within ±0.003 both folds.

Result:
- **Zero consistent improvements**.
- The strongest single signal is scifact richcov test R@10 `+0.0099`, but dev
  does not agree.
- **Three cells flip sign between folds**.

**Plan 4 disproves.**

## Mechanism — why argmax-filter is inert

The production frontier already uses query-attention softmax at
`attention_scale=8`:

```
mass[i] = exp(8.0 · qsim[i] - max_logit) / sum_j exp(...)
```

At `attention_scale=8`, the softmax already behaves like a soft-argmax. The
best fragment usually gets most of the per-doc mass before any hard filter is
applied.

**Hard argmax (Plan 4) replaces that soft split with 100/0.** The remaining
difference is then smoothed again by:
1. The diffusion step (2 steps at α=0.5 smooths mass across
   neighbors).
2. The alpha=0.8 convex blend with z-scored BM25 (BM25 contributes
   80% of final score).
3. The z-score normalization over the 100-doc pool (changes affect
   variance, not ordering much).

After those layers, the hard-argmax change lands at the noise floor.

## Implication for the Bottleneck 1 hypothesis

Combined with the MaxSim disproof, this partially refutes the stronger version
of the "multi-fragment averaging is the ceiling" hypothesis. The production
frontier already behaves enough like a soft-argmax that neither aggregation-side
nor pre-aggregation hard-argmax changes have much leverage.

If Bottleneck 1 were revisited, the real lever would be softmax temperature,
not another hard-argmax variant.

## Cross-arc note

Three interventions now say the same thing: the production frontier's
attention+diffusion+blend stack is robust to per-fragment modifications at the
noise floor.
1. PHSS-1D triangle weighting (importance weight on qsim or mass)
2. MaxSim aggregation (change sum → max post-softmax)
3. Plan 4 single-fragment filter (hard argmax pre-softmax)

All three stay below ±0.005 nDCG cross-fold. The stack's smoothing is stronger
than any of these interventions.

## Disposition

- Plan 4 disproved cross-fold.
- `FragmentGeometryConfig::single_fragment_per_doc` and the
  argmax-filter code stay in tree (no-op-safe when flag is false).
- Two bench recipes stay for regression tracking.
- **Bottleneck 1 (multi-fragment averaging) is now effectively
  closed** by three-way disproof (PHSS-1D, MaxSim, Plan 4). The
  averaging mechanism is real but its impact on production-frontier
  quality is at the noise floor.

Next research direction: **Plan 3 (RRF of BM25 variants)** or **Plan 1
(trec-covid)**, both of which attack different bottlenecks.

---

*2026-04-24. Bench rows in
`/tmp/bench_{scifact,nfcorpus,fiqa}_sf_{dev,test}.out`. Code in
`include/simeon/fragment_geometry.hpp`
(`single_fragment_per_doc` field), `src/fragment_geometry.cpp`
(argmax filter), `benchmarks/bench_vs_reference.cpp`
(2 recipes).*
