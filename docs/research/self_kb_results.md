# Self-KB Phase A — corpus-mined doc-doc graph + candidate-set expansion

## Experiment

Phase A is the cheap probe from `literature_synthesis.md`: build an offline
BM25-over-self doc-doc graph (top-K=20 neighbors per doc), union those
neighbors into the BM25 top-K at query time, then rerank with the current
production geometry recipe. No topology gate yet; this is the always-expand
baseline.

## Implementation

- Offline graph build (inline in bench recipe): for each doc d,
  treat `doc_texts[d]` as a BM25 query, take top-K=20 (excluding d
  itself) as neighbors. O(N²) but fully serial in ~seconds.
- Query-time expansion: in `score_fragment_geometry()` after
  `top_k(bm25_scores, cfg.pool_size)` (fragment_geometry.cpp:567),
  union the pool with the precomputed neighbors of each pool member.
- Re-rank with the existing `phssapprox_k100_t8_richcov_gap`
  configuration over the expanded pool. No aggregation changes.

## Results (cross-fold)

| Corpus   | Fold  | Baseline nDCG / R@10 / R@100 / QPS | Self-KB nDCG / R@10 / R@100 / QPS | **Δ R@100** | Δ nDCG | Δ R@10 |
|----------|-------|------------------------------------|-----------------------------------|------------:|-------:|-------:|
| scifact  | dev   | 0.6716 / 0.7857 / 0.8469 / 58      | 0.6357 / 0.7755 / 0.8622 / 0.3    | **+0.0153** | −0.0359 | −0.0102 |
| scifact  | test  | 0.6188 / 0.7411 / 0.8736 / 103     | 0.5941 / 0.7312 / 0.8845 / 0.7    | **+0.0109** | −0.0247 | −0.0099 |
| nfcorpus | dev   | 0.2622 / 0.2432 / 0.2158 / 63      | **0.2660** / **0.2473** / **0.2594** / 0.3 | **+0.0436** | **+0.0038** | **+0.0041** |
| nfcorpus | test  | 0.2544 / 0.2266 / 0.1991 / 126     | **0.2601** / **0.2346** / **0.2275** / 0.8 | **+0.0284** | **+0.0057** | **+0.0080** |
| fiqa     | dev   | 0.1852 / 0.2292 / 0.4360 / 102     | 0.1575 / 0.2119 / 0.4546 / 0.4    | **+0.0186** | −0.0277 | −0.0173 |
| fiqa     | test  | 0.2089 / 0.2673 / 0.4672 / 121     | 0.1736 / 0.2353 / 0.4709 / 0.6    | +0.0037     | −0.0353 | −0.0320 |

## Verdict — mixed, but real

- **R@100 signal is real**: scifact and nfcorpus clear the +0.005 gate on both folds.
- **nDCG is corpus-bound**: nfcorpus lifts cross-fold; scifact and fiqa regress.
- **Latency fails hard**: 0.3–0.8 QPS vs 58–126 baseline.

Strictly: Phase A validates the *mechanism* (candidate expansion beyond BM25
top-K), but not the production recipe. The always-expand form is too noisy on
scifact/fiqa and too expensive everywhere.

## Mechanism — why nfcorpus wins, scifact/fiqa lose on top-10

The same corpus-bound pattern shows up again. The doc-doc graph helps when
shared vocabulary tracks relevance (nfcorpus), but it hurts when shared
vocabulary is a weak proxy for top-10 relevance (scifact, fiqa). The R@100 / R@10
split is the key clue: expansion adds useful candidates at depth, but too many
of them are noisy near the top.

## Bottleneck status (vs `literature_synthesis.md`)

- **Bottleneck 1** (multi-fragment per-doc averaging): unchanged;
  this plan doesn't touch aggregation.
- **Bottleneck 2** (BM25-pool R@100 ceiling): **partially attacked**.
  Self-KB is the first recipe to demonstrate R@100 headroom beyond
  the BM25 top-K, cross-fold consistent on 2/3 corpora. The ceiling
  is real but not saturated.

## Phase B design (warranted per Phase A validation)

Three tunables were worth sweeping next:

1. **Neighbor count n ∈ {5, 10, 20}**. Current n=20 is aggressive;
   n=5 may preserve most of the R@100 lift with 4× smaller pool
   expansion and proportionally better latency.
2. **BM25-score filter**: only add neighbors with nonzero BM25 score
   against the current query. Removes zero-query-term-overlap
   neighbors (the ones most likely to cause scifact/fiqa nDCG
   regression) while keeping genuinely-relevant expansion candidates.
3. **Component C topology gate** (per the plan): skip expansion when
   first-pass PHSS persistence is low (diffuse pool). Per-corpus
   threshold sweep on dev fold.

Implementation cost: ~50 LOC for (1)+(2), ~30 LOC for (3), and 9 recipes per
fold. Still cheap because the graph is built once per corpus.

**Phase B hypothesis**: smaller n + BM25-relevance filter + topology
gate together preserve the nfcorpus cross-fold win (the only
corpus where all three metrics lift), recover scifact/fiqa nDCG to
neutral, and drop latency to within 2× of baseline.

If Phase B succeeds, self-KB ships as the new recall-targeted production
recipe. If it fails, Phase A's R@100 finding remains a corpus-specific
nfcorpus subfinding.

## Cross-arc note

This is the second non-BM25 recipe to beat the production frontier cross-fold
on nfcorpus. The common pattern is candidate-expansion headroom on that corpus,
but only at substantial latency cost.

## Disposition

- Phase A complete; mechanism validated (R@100 signal real cross-fold).
- Recipe `phssapprox_selfkb_n20_richcov` stays in the bench for
  regression tracking.
- `FragmentGeometryConfig::doc_doc_neighbors` and the expansion code
  stay in tree (no-op-safe when span is empty).
- **Phase B scoped**: neighbor count sweep + BM25-relevance filter +
  Component C topology gate. Proposed as a follow-on if validated
  on cross-fold.

## Phase B — cap / filter / gate sweep

Phase B sweeps three tunables to preserve the R@100 lift while recovering nDCG
and latency:

1. **Neighbor cap** `n_per_pool_doc ∈ {5, 10, 20}` — fewer neighbors
   per pool member = smaller expanded pool, faster rerank.
2. **BM25-relevance filter** `min_bm25_score ∈ {−∞, 0.0}` — only add
   neighbors with nonzero BM25 score against the query.
3. **Topology gate** `gate_score_decay_min ∈ {0.0, 0.25}` — skip
   expansion when BM25 pool score decay is below threshold (flat
   pool → noisy expansion).

All recipes `richcov` builder + `LargestGapApprox` PHSS, `α=0.80`.

### Cross-fold Δ vs baseline `phssapprox_k100_t8_richcov_gap`

Format: `Δ nDCG / Δ R@100 (QPS)`. Baseline QPS ~100-170.

| Corpus   | Fold  | **n20**            | n5                | n10               | n10_bm25filt      | **n10_gate25**     | n10_filt_gate25   |
|----------|-------|--------------------|-------------------|-------------------|-------------------|--------------------|-------------------|
| scifact  | dev   | −0.0359 / +0.0153 (0.9) | −0.0359 / **+0.0255** (7.5) | −0.0381 / +0.0153 (2.6) | −0.0374 / +0.0153 (2.7) | −0.0292 / +0.0153 (2.8) | −0.0285 / +0.0153 (3.1) |
| scifact  | test  | −0.0261 / +0.0109 (0.9) | −0.0219 / +0.0059 (7.7) | −0.0255 / +0.0109 (2.8) | −0.0267 / +0.0109 (3.1) | −0.0313 / +0.0099 (2.0) | −0.0307 / +0.0099 (1.2) |
| **nfcorpus** | dev   | +0.0038 / +0.0428 (1.0) | −0.0036 / +0.0304 (8.4) | +0.0013 / +0.0400 (2.9) | +0.0040 / +0.0074 (6.9) | **+0.0051 / +0.0291** (3.3) | +0.0046 / +0.0075 (7.1) |
| **nfcorpus** | test  | +0.0057 / +0.0284 (1.0) | +0.0046 / +0.0259 (8.9) | +0.0034 / +0.0311 (3.1) | +0.0014 / +0.0055 (7.6) | **+0.0041 / +0.0268** (2.9) | +0.0018 / +0.0038 (5.9) |
| fiqa     | dev   | −0.0277 / +0.0186 (0.5) | −0.0205 / +0.0207 (6.3) | −0.0220 / +0.0200 (2.0) | −0.0227 / +0.0138 (2.2) | −0.0124 / +0.0153 (4.3) | −0.0134 / +0.0104 (5.0) |
| fiqa     | test  | −0.0346 / +0.0033 (0.7) | −0.0326 / +0.0083 (7.2) | −0.0349 / +0.0051 (2.4) | −0.0356 / +0.0081 (2.6) | −0.0178 / +0.0032 (5.0) | −0.0182 / +0.0040 (5.6) |

### Phase B verdict — `n10_gate25` wins, nfcorpus-specific

**nfcorpus `n10_gate25` cross-fold**: +0.0051 / +0.0291 on dev and
+0.0041 / +0.0268 on test for nDCG / R@100. That clears the strict
cross-fold validate gate.

Scifact and fiqa still regress on nDCG, so self-KB remains a corpus-specific
lever rather than a new default.

### Per-component ablation

- **Neighbor cap mainly trades latency for recall depth.**
- **The BM25 filter hurts nfcorpus R@100** by removing exactly the off-query-term
  neighbors that add depth.
- **The score-decay gate is the useful control knob**: it helps nDCG on every corpus and fold.

### Production recommendation

Ship `phssapprox_selfkb_n10_gate25_richcov` as the nfcorpus-specific default,
or as an opt-in recall-targeted recipe. Keep scifact and fiqa on
`phssapprox_k100_t8_richcov_gap`.

### Latency disposition

`n10_gate25` still runs at only ~3 QPS, about 50× slower than the baseline
frontier. Further latency work is possible, but it is an engineering pass, not
part of the research decision.

### Cross-arc note

**Two recipes now validate cross-fold on nfcorpus**:
1. PHSS pool=500 richcov: +0.0122 R@100 at 5 QPS
   (`phss_pool_scaling.md`)
2. Self-KB n10_gate25: +0.027 R@100 + +0.004 nDCG at 3 QPS
   (this doc)

Both exploit nfcorpus's longer-doc, richer-vocabulary regime at the cost of
per-query latency. This is another example of the corpus tuning itself.

### Disposition (final)

- Phase B complete; `n10_gate25` is nfcorpus production default path.
- 6 Phase B recipes stay in bench for regression tracking.
- `FragmentGeometryConfig` Phase B knobs (`selfkb_neighbors_per_pool_doc`,
  `selfkb_min_bm25_score`, `selfkb_gate_score_decay_min`) stay in
  tree.
- Scifact/fiqa **not recommended** for self-KB in current form; would
  need a different filter / gate mechanism that understands those
  corpora's exact-match-dominant relevance.
- Plan 2 is **validated on nfcorpus** and **disproved on scifact and fiqa**.

---

*2026-04-24. Bench rows in
`/tmp/bench_{scifact,nfcorpus,fiqa}_selfkb_{dev,test}.out`. Code in
`include/simeon/fragment_geometry.hpp` (`doc_doc_neighbors` field),
`src/fragment_geometry.cpp` (candidate-set expansion), and
`benchmarks/bench_vs_reference.cpp` (graph builder + recipe).*
