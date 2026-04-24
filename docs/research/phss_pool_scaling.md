# PHSS pool-size scaling — Phase C results

## Experiment

Closes the "test larger pools" question from `phss_results.md`. Runs
`LargestGapApprox` at `pool_size ∈ {100, 200, 500}` on both builders across
the three BEIR-3 fixtures.

Hypothesis under test: at larger pools the geometry pool draws from
deeper BM25 ranks, so fragment topology may diverge more from `knn=8`
and the data-driven scale selection may capture richer structure.

Pre-declared gates:
- **Validate** if any pool > 100 lifts any corpus by ≥0.003 nDCG@10
  over the pool=100 frontier at <100 ms/query.
- **Disprove** if quality is flat or regresses on ≥2/3 corpora within
  the latency budget.

## Results

### Basic builder (t=4)

| Corpus   | pool=100 nDCG/QPS | pool=200 nDCG/QPS | pool=500 nDCG/QPS |
|----------|------------------:|------------------:|------------------:|
| scifact  | 0.6149 / 287      | 0.6095 / 73       | 0.6074 / 11       |
| nfcorpus | 0.2542 / 355      | 0.2511 / 91       | 0.2517 / 14       |
| fiqa     | 0.2061 / 298      | 0.2025 / 80       | 0.2014 / 13       |

### Richcov builder (t=8) — production frontier

| Corpus   | pool=100 nDCG/QPS | pool=200 nDCG/QPS    | pool=500 nDCG/QPS |
|----------|------------------:|---------------------:|------------------:|
| scifact  | 0.6188 / 108      | 0.6134 / 27          | 0.6073 / 4        |
| nfcorpus | 0.2544 / 128      | 0.2563 / 32          | **0.2573 / 5**    |
| fiqa     | **0.2089** / 123  | 0.2049 / 32          | 0.2038 / 5        |

### R@100 — recall at depth panel

| Corpus   | pool=100 R@100 | pool=200 R@100 | pool=500 R@100 |
|----------|---------------:|---------------:|---------------:|
| scifact  | 0.8736         | 0.8746 (+0.001)| 0.8746 (+0.001)|
| nfcorpus | 0.1991         | 0.2007 (+0.002)| **0.2113 (+0.012)** |
| fiqa     | 0.4672         | 0.4719 (+0.005)| 0.4676 (+0.000)|

(richcov rows; basic builder R@100 follows the same pattern.)

## Verdict — pool=100 stays the production optimum

**Disprove gate met**: 2/3 corpora regress nDCG@10 at pool=200 and
pool=500 (richcov):

- scifact richcov: −0.0054 at 200, **−0.0115 at 500**.
- fiqa richcov: −0.0040 at 200, −0.0051 at 500.
- nfcorpus richcov: +0.0019 at 200, +0.0029 at 500 (only positive corpus).

**Validate gate not met**: nfcorpus pool=500 reaches +0.0029 nDCG@10
which is just below the +0.003 threshold, AND lands at 5 QPS (~200
ms/query) — twice the <100 ms latency budget.

The "larger pools capture richer topological structure" hypothesis is
**disproved on this fixture set**. PHSS quality plateaus at `pool=100`.

## Subfinding — nfcorpus has real R@100 upside at depth

While the nDCG gate fails, nfcorpus pool=500 richcov shows a meaningful
**R@100 lift of +0.0122** over pool=100. This is the largest R@100 gain from
any topology lever in the fragment-geometry arc and matches two earlier
findings:

1. nfcorpus has been the strongest corpus for every topology lever
   (richmmr +0.0064 nDCG@10, geometric +0.0027, rich +0.0057 — see
   `transport_results.md`).
2. The medical/long-doc nature of the corpus (avg_dl 233.8 — see
   `ltd_results.md`) gives fragment neighborhoods more semantic
   structure than short-doc fixtures.

This is real recall-at-depth signal, but not a default because latency is too
high.

## Subfinding — scifect/fiqa pool=200 R@100 lifts are noise-grade

scifact +0.001 R@100 at pool=200/500 and fiqa +0.005 R@100 at pool=500
basic are within run-to-run noise. The R@100 effect is real only on
nfcorpus and only at deep pools.

## Latency cost model

Predicted vs measured (basic builder, scifact reference):

| pool | predicted phss_select µs (from phss_profile.md) | measured QPS | measured ms/query |
|------|-------------------------------------------------:|-------------:|------------------:|
| 100  | 1500 (approx)                                    | 287          | 3.5               |
| 200  | ~6000                                            | 73           | 13.7              |
| 500  | ~37500                                           | 11           | 91                |

Measured ms/query scales sub-quadratically with N, consistent with
O(N² log N) sort dominance plus fixed BM25 / gather / blend overheads.

Even with `LargestGapApprox`, pool=500 richcov sits on the wrong side of the
latency wall and still misses the quality validate gate.

## Closing the open question

Per `phss_results.md` next-move list, item 2 ("test on larger pools")
is now closed:

- **Closed: pool=100 is the optimum** for the production frontier.
- **Closed: PHSS quality plateaus at pool=100** for nDCG@10 / R@10.
- **Open: pool=500 richcov on nfcorpus has +0.0122 R@100 signal** that
  could be useful as a recall-targeted recipe in future work; deferred
  because (a) it costs 25× over baseline at pool=100, (b) it only
  benefits one of three corpora, and (c) the simeon-vs-MiniLM R@100
  gap on nfcorpus is only −0.064 — pool=500 closes ~19% of that one
  corpus's recall gap at unrealistic latency.

The remaining open moves from `phss_results.md` are now:

1. ~~Adaptive PHSS~~ — already done (phss_results.md lists adaptive
   rows at `phssadapt_*`).
2. ~~Larger pools~~ — closed by this experiment.
3. **1D persistent homology (cycles)** — qualitatively different
   research direction; out of scope for the current engineering
   pass. Separate research plan if pursued.

## Disposition

- Phase C complete.
- pool=100 stays the production default; bench keeps the pool=200/500
  recipes for regression tracking.
- Primary engineering target met: `phss_select_mean_us < 1500 µs` on all three
  corpora via `LargestGapApprox`.
- The pool=500 secondary target is moot because the quality result is not worth
  the latency on 2/3 corpora.

## Bench rows

All measurements from `/tmp/bench_{scifact,nfcorpus,fiqa}_phaseC.out`
(saved as part of this experiment; Phase C bench recipes ship in
`benchmarks/bench_vs_reference.cpp` for regression tracking).

---

*2026-04-23. Code in `benchmarks/bench_vs_reference.cpp` (Phase C
recipes added inline alongside the existing `phssapprox_*` rows).*
