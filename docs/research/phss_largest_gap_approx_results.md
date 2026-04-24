# PHSS — `LargestGapApprox` quality characterization (Phase B6)

## Experiment

`PhssConfig::Criterion::LargestGapApprox` skips the full edge-sort + Union-Find
pass: it sorts raw pairwise similarities and picks the largest gap directly,
instead of building the 0D persistence diagram first.

The approximation had shipped for a while, but had never been benched directly
against the heavy `LargestGap` path. This experiment does that head to head.

## Results

### Heavy `LargestGap` vs `LargestGapApprox`

#### Basic builder (t=4 fragments per doc, ~400 fragments per pool)

| Corpus   | LargestGap nDCG@10 | LargestGapApprox nDCG@10 | Δ        | LargestGap QPS | LargestGapApprox QPS | Speedup |
|----------|-------------------:|-------------------------:|---------:|---------------:|---------------------:|--------:|
| scifact  | 0.6188             | 0.6149                   | **−0.0039 ⚠** | 157.9      | 294.6                |  1.87×  |
| nfcorpus | 0.2531             | **0.2542**               | +0.0011  |        195.2   |              364.1   |  1.87×  |
| fiqa     | 0.2065             | 0.2061                   | −0.0004  |        161.3   |              276.4   |  1.71×  |

#### Richcov builder (t=8 fragments per doc, ~800 fragments per pool)

| Corpus   | LargestGap nDCG@10 | LargestGapApprox nDCG@10 | Δ        | LargestGap QPS | LargestGapApprox QPS | Speedup |
|----------|-------------------:|-------------------------:|---------:|---------------:|---------------------:|--------:|
| scifact  | 0.6167             | **0.6188**               | **+0.0021** |  58.4    | 112.6                |  1.93×  |
| nfcorpus | 0.2542             | **0.2544**               | +0.0002  |         65.2   |              127.6   |  1.96×  |
| fiqa     | 0.2089             | 0.2089                   |  0.0000  |         63.7   |              119.4   |  1.87×  |

R@100 is unchanged in every cell because the geometry leg only reorders the
already-recalled BM25 pool.

## Verdict — split decision per builder

**Richcov builder (t=8)**: approx is **strictly equal-or-better** on
all three corpora with ~2× throughput. **Promote as default.**
The headline result `phssapprox_k100_t8_richcov_gap` reaches:

- scifact: 0.6188 nDCG@10 (matches BM25, no regression)
- nfcorpus: 0.2544 nDCG@10 (+0.0023 over BM25 baseline 0.2521)
- fiqa: **0.2089 nDCG@10 (+0.0036 over BM25 baseline 0.2053)**

This preserves the prior `phss_k100_t8_richcov_gap` frontier at about 2× the
throughput.

**Basic builder (t=4)**: approx **regresses scifact by −0.0039**, which
exceeds the plan's ±0.001 promotion threshold. Approx wins nfcorpus
(+0.0011) and is neutral on fiqa, but the scifact loss disqualifies a
default swap. **Keep `LargestGap` for basic-builder configs.**

## Why approx works for richcov but not basic

The approximation replaces "largest gap in 0D death similarities" with
"largest gap in raw pairwise similarities". These coincide when the similarity
distribution is smooth enough that its raw quantiles mirror the merge-order
structure.

At richer fragment counts, the histogram is denser and the raw-gap selector
tracks the death-gap selector closely. At lower fragment counts, the noisier
distribution makes the two selectors disagree more often.

Scifact's basic-builder pools appear more sensitive to outlier-similar
fragments, while richer fragment state averages those outliers out.

## Implications for Phase B / Phase C

This B6 result hits the main engineering target:

- **Speed target**: met. Approx hits 1499 / 1202 / 1339 µs on scifact /
  nfcorpus / fiqa.
- **Quality target**: met on richcov, missed on the basic builder.
- **Disposition**: promote approx as the richcov default; keep `LargestGap` for
  basic-builder configs.

Phase B2 (radix sort) becomes lower priority because the richcov path is already
inside the target budget. It only matters again if larger pools are revisited.

Phase C (pool-size scaling) became the next move because approx made it about
2× more affordable than the heavy path.

## Next move

1. Update `bench_vs_reference.cpp` so the production-frontier richcov
   recipe defaults to `LargestGapApprox`. Keep heavy `LargestGap`
   variants in the bench for regression tracking.
2. Run Phase C: pool_size sweep on all 3 corpora, both builders,
   approx criterion. Look for any quality lift at larger pools that
   justifies the latency cost.

---

*2026-04-23. Bench rows in `/tmp/bench_{scifact,nfcorpus,fiqa}_frag.out`.
Code unchanged — measurement-only experiment.*
