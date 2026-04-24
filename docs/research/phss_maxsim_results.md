# MaxSim aggregation probe — mixed positive (scifact-only win)

## Experiment

Per `literature_synthesis.md`, ColBERTv2-style MaxSim was the last small,
training-free probe left on the BEIR-3 fixture set after the Dictionary axis
was closed.

This probe replaces the doc-aggregation step in
`src/fragment_geometry.cpp:858-859` with a max-pooling switch on a
new `FragmentGeometryConfig::DocAggregator { Sum, Max }` flag:

- **Sum** (default): `geom_pool[doc] += mass[fragment]` — averages
  per-fragment signal across the t fragments per doc.
- **Max**: `geom_pool[doc] = max(geom_pool[doc], mass[fragment])` —
  preserves the strongest per-fragment match per doc.

This experiment applies MaxSim to **training-free PMI fragments**, not learned
token embeddings.

Hypothesis: addresses the multi-fragment averaging mechanism
documented in `phss_1d_triangle_results.md` as the failure mode that
killed the Phase A 1D triangle weighting probe.

## Implementation

- `include/simeon/fragment_geometry.hpp`: `DocAggregator` enum on
  `FragmentGeometryConfig`. Default `Sum` (no-op vs prior behavior).
- `src/fragment_geometry.cpp:858-868`: branch on
  `cfg.doc_aggregator` at the doc-aggregation step. Max path uses
  in-place compare-and-swap.
- `benchmarks/bench_vs_reference.cpp`: 2 new recipes
  (`phssapprox_max_k100_t4` and `phssapprox_max_k100_t8_richcov`),
  paired with the existing `phssapprox_k100_t{4,8_richcov}_gap` Sum
  baselines for direct delta calculation.

Total: ~15 LOC C++.

## Results

### Head-to-head (Sum vs Max), all 3 corpora, both builders

| Corpus   | Builder | Sum nDCG / R@10 | **Max nDCG / R@10** | Δ nDCG@10 | Δ R@10  | Sum QPS / Max QPS |
|----------|---------|----------------:|--------------------:|----------:|--------:|------------------:|
| scifact  | basic   | 0.6149 / 0.7262 | 0.6115 / 0.7238     | −0.0034   | −0.0024 |    279 / 281      |
| **scifact** | **richcov** | **0.6188 / 0.7411** | **0.6216 / 0.7559** | **+0.0028** | **+0.0148** | 103 / 107 |
| nfcorpus | basic   | 0.2542 / 0.2275 | 0.2521 / 0.2270     | −0.0021   | −0.0005 |    363 / 344      |
| nfcorpus | richcov | 0.2544 / 0.2266 | 0.2525 / 0.2261     | −0.0019   | −0.0005 |    126 / 126      |
| fiqa     | basic   | 0.2061 / 0.2652 | 0.2045 / 0.2678     | −0.0016   | +0.0026 |    288 / 291      |
| fiqa     | richcov | 0.2089 / 0.2673 | 0.2068 / 0.2637     | −0.0021   | −0.0036 |    120 / 120      |

R@100 is **byte-identical** to baseline in every cell — geometry
rerank still doesn't add docs to the BM25-pool (Bottleneck 2 still open).

QPS is essentially unchanged (max-aggregation is a one-line in-place
compare; not a measurable cost).

### Headline: test-fold scifact richcov breaks the BM25 ceiling

| Recipe                                          | nDCG@10 | R@10   |
|-------------------------------------------------|--------:|-------:|
| `bm25_only`                                     | 0.6188  | 0.7411 |
| `phss_k100_t4_gap` (PHSS LargestGap heavy)      | 0.6188  | 0.7312 |
| `phssapprox_k100_t8_richcov_gap` (prior front.) | 0.6188  | 0.7411 |
| **`phssapprox_max_k100_t8_richcov`** (new)      | **0.6216** | **0.7559** |

This is the first fragment-topology row to beat BM25 on scifact, with a
meaningful +0.0148 R@10 lift.

## Verdict — initial near-miss, later disproved

Pre-declared gates:

- **Validate**: lift ≥0.003 nDCG@10 on ≥1 corpus, no regression
  >0.003 elsewhere.
- **Disprove**: all 3 within ±0.001 of baseline.

Result:

- **Validate gate just-misses**: scifact richcov +0.0028 (threshold
  +0.003). Other corpora within ±0.0021 (within tolerance).
- **Disprove gate not met**: scifact lift exceeds ±0.001.

The test-fold result looked like a corpus-specific near-miss rather than a new
default. The strongest signal was the scifact richcov R@10 lift.

## Mechanism — why scifact richcov wins, others lose

**scifact**: scientific abstracts. Each TextRank-extracted fragment
captures a focused scientific concept (one method, one finding,
one experimental detail). The "best matching fragment per doc" is a
strong relevance signal because fragments don't share topic. Sum
aggregation dilutes the strongest match by averaging it with
weaker fragments.

**nfcorpus**: medical Q&A with longer, more diffuse docs. Multiple
fragments accumulate evidence — sum is the right aggregator because
relevance is distributed across the doc. Max throws away that
distributed evidence.

**fiqa**: financial Q&A with question-style content. Many fragments
per doc are question-stem variations, none clearly "the" matching
fragment. Sum averages out the question-stem noise; Max amplifies
whichever question stem happens to overlap most with the query
(often spurious).

This pattern is the per-corpus optimum from yet another angle: the
**right aggregator depends on whether per-doc relevance is
concentrated (Max wins) or distributed (Sum wins)**.

## Interim interpretation

The initial interpretation was simple: the useful signal looked more like
*fragment focus per doc* than raw document length, so the cleanest move was to
keep `Sum` as the universal default while tracking `Max` as a possible
scifact-only lever.

## Dev-fold recheck — 2026-04-23

Move 1 above was executed. Dev-fold result inverts the test-fold story:

| Corpus   | Builder | Test Δ nDCG | Dev Δ nDCG | Consistent? |
|----------|---------|------------:|-----------:|:-----------:|
| scifact  | basic   | −0.0034     | **+0.0048** | NO (flips)  |
| scifact  | richcov | **+0.0028** | −0.0045    | NO (flips)  |
| nfcorpus | basic   | −0.0021     | −0.0023    | yes (regress) |
| nfcorpus | richcov | −0.0019     | −0.0060    | yes (regress) |
| fiqa     | basic   | −0.0016     | −0.0045    | yes (regress) |
| fiqa     | richcov | −0.0021     | **+0.0031** | NO (flips)  |

**Three cells flip sign between folds; three consistently regress; zero
consistently improve.** The test-fold scifact richcov lift was a fold artifact
at the noise floor.

The fold gaps are large enough that ±0.003 to ±0.005 single-fold deltas are not
reliable here.

**Revised verdict — MaxSim disproves at the population level.** No
(corpus, builder) combination improves consistently across folds.

Top-K MaxSim variants were skipped because the dev recheck failed. Keep `Sum`
as the default; keep `DocAggregator` in tree for future experiments.

## Cross-arc note

This recheck reinforced a broader lesson: single-fold lifts below about
±0.005 nDCG@10 are unreliable on this fixture set and should not be promoted
without both folds.

---

*2026-04-23. Bench rows in `/tmp/bench_{scifact,nfcorpus,fiqa}_max.out`.
Code in `include/simeon/fragment_geometry.hpp` (`DocAggregator` enum),
`src/fragment_geometry.cpp:858-868` (aggregation switch), and
`benchmarks/bench_vs_reference.cpp` (2 new recipes).*
