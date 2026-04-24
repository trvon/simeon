# Per-corpus α sweep — Bruch & Gai 2023 cleanup — DISPROVED

## Experiment

Per `literature_synthesis.md`, the last Bottleneck 2 cleanup item was
per-corpus α tuning per Bruch & Gai 2023, *"An Analysis of Fusion Functions for
Hybrid Retrieval"*:

> "convex combination is sample efficient, requiring only a small
> set of training examples to tune its only parameter to a target
> domain."

simeon's existing blend at `fragment_geometry.cpp:864` is exactly the
convex form Bruch & Gai studied:
`final_score = α · z(BM25_pool) + (1 − α) · z(geometry_pool)`,
hard-coded at α=0.8 across all corpora.

**Hypothesis**: α should be per-corpus tuned. Sweep
α ∈ {0.50, 0.65, 0.75, 0.85, 0.95} on the production frontier
(richcov + Sum + LargestGapApprox), tune on dev fold, validate on
test fold per the methodology lesson from
`phss_maxsim_results.md:118-150`.

## Results

### Dev fold sweep (training set)

richcov + Sum + LargestGapApprox.

| Corpus   | α=0.50 | α=0.65 | α=0.75 | **α=0.80 (default)** | α=0.85 | α=0.95 | Dev winner |
|----------|-------:|-------:|-------:|---------------------:|-------:|-------:|:-----------|
| scifact  | 0.6036 | 0.6494 | 0.6684 | **0.6716**           | 0.6684 | 0.6665 | α=0.80     |
| nfcorpus | 0.2509 | 0.2583 | 0.2622 | 0.2622               | **0.2638** | 0.2628 | α=0.85     |
| fiqa     | 0.1485 | 0.1740 | 0.1802 | 0.1852               | **0.1908** | 0.1890 | α=0.85     |

Dev picks: scifect → α=0.80 (matches default), nfcorpus → α=0.85,
fiqa → α=0.85.

### Test fold sweep (held-out validation)

| Corpus   | α=0.50 | α=0.65 | α=0.75 | **α=0.80 (default)** | α=0.85 | α=0.95 | Test winner |
|----------|-------:|-------:|-------:|---------------------:|-------:|-------:|:-----------|
| scifact  | 0.5610 | 0.6054 | 0.6178 | 0.6188               | 0.6175 | **0.6210** | α=0.95     |
| nfcorpus | 0.2306 | 0.2492 | **0.2546** | 0.2544           | 0.2542 | 0.2512 | α=0.75     |
| fiqa     | 0.1675 | 0.1923 | 0.2039 | **0.2089**           | 0.2070 | 0.2063 | α=0.80     |

### Cross-fold transfer table

| Corpus   | Dev winner α | Dev nDCG | Test nDCG (dev α) | Test default α=0.80 | Transfer Δ vs default |
|----------|-------------:|---------:|------------------:|--------------------:|----------------------:|
| scifact  | 0.80         | 0.6716   | 0.6188            | 0.6188              |  0.0000 (same — consistent) |
| nfcorpus | 0.85         | 0.2638   | 0.2542            | 0.2544              | **−0.0002** (loses)   |
| fiqa     | 0.85         | 0.1908   | 0.2070            | 0.2089              | **−0.0019** (loses)   |

**Zero corpora** have a non-default α that beats α=0.80 across both folds.
Per-corpus dev tuning does **not** generalize here.

## Verdict — disprove

Pre-declared validate threshold: dev winner consistently beats
default α=0.80 on test by ≥0.003 nDCG@10.

- scifact: dev winner = default; no test transfer needed (no change).
- nfcorpus: dev winner α=0.85 transfers to test as −0.0002 vs default
  (loses).
- fiqa: dev winner α=0.85 transfers to test as −0.0019 vs default
  (loses).

**Disprove**: per-corpus α tuning fails the cross-fold validate gate on 2/3
corpora. α=0.80 stays as the default.

## Mechanism — why Bruch & Gai's "sample efficient" doesn't apply

Bruch & Gai 2023 assumes dev queries are a small sample from the same
distribution as test queries. Simeon's BEIR-3 folds are structurally less
matched than that:

| Corpus   | Test α=0.80 nDCG | Dev α=0.80 nDCG | Absolute fold gap |
|----------|------------------:|----------------:|------------------:|
| scifact  | 0.6188            | 0.6716          | +0.0528 dev easier |
| nfcorpus | 0.2544            | 0.2622          | +0.0078 dev easier |
| fiqa     | 0.2089            | 0.1852          | −0.0237 dev harder |

These fold-difficulty gaps are larger than the lift threshold itself. The dev
fold is not a small sample of the test distribution, so the sample-efficiency
claim does not transfer cleanly.

This also explains why several earlier tuning sweeps looked corpus-bound but
failed to transfer across folds.

## Implications

1. **α=0.80 stays the right universal default.**
2. **Cross-fold validation should be standard** for future ±0.005-scale lifts.
3. **Bruch-Gai-style per-domain tuning needs a same-distribution dev sample**, which BEIR-3 does not provide cleanly.

## Disposition

- α sweep complete; per-corpus tuning disproved.
- α=0.80 stays as the universal default.
- 5 new bench recipes (`phssapprox_a{0.50,0.65,0.75,0.85,0.95}_richcov`)
  stay in the bench for regression tracking — disproof rows are data,
  not noise.
- **The last cleanup item from `literature_synthesis.md` is closed.**

## BEIR-3 training-free track — saturation declared

With this α sweep disproved, the literature-grounded next-experiment list from
`literature_synthesis.md` is exhausted:

- Bottleneck 1 (multi-fragment averaging): MaxSim probe disproved
  (`phss_maxsim_results.md`).
- Bottleneck 2 (BM25-pool R@100 ceiling): per-corpus α disproved
  (this doc).
- Bottleneck 3 (per-corpus routing): subsumed by Bottleneck 2.
- Bottleneck 4 (Dictionary axis): disproved on BEIR-3
  (`ac_entity_results.md`, `textrank_title_results.md`).

**Production frontier remains `phssapprox_k100_t8_richcov_gap`**:
- scifact 0.6188 nDCG@10 (= BM25)
- nfcorpus 0.2544 (+0.0023 vs BM25)
- fiqa **0.2089** (+0.0036 vs BM25)
- α=0.80 universal blend, Sum aggregation, LargestGapApprox PHSS
  scale selection, t=8 richcov fragment builder, pool=100.

The training-free lexical retrieval ceiling is reached on the BEIR-3 fixture
set. The next increment requires either:

1. A new corpus with structurally different content (e.g., trec-covid
   for a true long-doc test of LTD; a structured-document corpus
   where AhoCorasick / TextRank fields are genuinely distinct from
   body text), or
2. Moving off training-free constraint (dense retrieval, learned
   sparse) — out of memory rule.

Either move is its own plan, not a continuation of the current
research arc.

## Cross-arc summary (for the closing record)

Closing summary:

- **Disproved universal recipes**: NQC, WIG, WSDM, adaptive RM3 K, LTD, PMI,
  concept mining, entropy fusion, df-router enrichment, MaxSim, per-corpus α.
- **Validated selective levers**: scifact-tuned router default, Step 1k clarity,
  Step 1g.1 score decay, SAB routing, fixed SDM on FiQA, PHSS LargestGap,
  PHSS approx, and RM3 for SAB-smooth.
- **Pattern**: selective, signal-conditioned levers can survive; universal
  recipes do not.

---

*2026-04-24. Bench rows in
`/tmp/bench_{scifact,nfcorpus,fiqa}_alpha_{dev,test}.out`. Code in
`benchmarks/bench_vs_reference.cpp` (5 new α-sweep recipes).
No changes to scoring, router, or fragment_geometry beyond the
existing per-recipe `alpha` parameter.*
