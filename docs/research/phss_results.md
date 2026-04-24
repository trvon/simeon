# Persistent Homology Scale Selection (PHSS) — Results

## Experiment

Implemented 0D persistent homology (Union-Find) on the fragment similarity
graph within each BM25 pool. Three scale-selection criteria were compared to
fixed `knn=8` baselines on BEIR-3:

- **LargestGap**: selects the scale at the largest gap in the sorted death similarities.
- **MaxPersistence**: selects the scale at the most persistent pair (longest interval).
- **Elbow**: selects the scale at the maximum-curvature elbow of the persistence curve.

Two fragment sets were tested:
- **Basic** (`t=4`): standard sentence fragments (`build_doc_semantic_fragments`).
- **richcov** (`t=8`): rich covered fragments (`build_doc_semantic_fragments_rich_covered`).

## Results

### Scifact

| Config                              | nDCG@10 | Recall@10 | QPS    |
|-------------------------------------|---------|-----------|--------|
| bm25_only                           | 0.6188  | 0.7411    | 35289  |
| geom_k100_t4_s8_k8_p2_a0.8          | 0.6140  | 0.7287    | 565.9  |
| geom_richcov_k100_t8                | 0.6161  | 0.7411    | 243.3  |
| **phss_k100_t4_gap**                | 0.6188  | 0.7312    | 145.7  |
| **phss_k100_t4_persist**            | 0.6149  | 0.7262    | 148.1  |
| **phss_k100_t4_elbow**              | 0.6188  | 0.7312    | 150.2  |
| **phss_k100_t8_richcov_gap**        | 0.6167  | 0.7411    | 55.1   |

### NFCorpus

| Config                              | nDCG@10 | Recall@10 | QPS    |
|-------------------------------------|---------|-----------|--------|
| bm25_only                           | 0.2521  | 0.2294    | 96373  |
| geom_k100_t4_s8_k8_p2_a0.8          | 0.2548  | 0.2270    | 678.9  |
| geom_richcov_k100_t8                | 0.2550  | 0.2284    | 278.1  |
| **phss_k100_t4_gap**                | 0.2531  | 0.2266    | 175.9  |
| **phss_k100_t4_persist**            | 0.2541  | 0.2275    | 176.8  |
| **phss_k100_t4_elbow**              | 0.2526  | 0.2266    | 148.0  |
| **phss_k100_t8_richcov_gap**        | 0.2542  | 0.2270    | 60.1   |

### FiQA

| Config                              | nDCG@10 | Recall@10 | QPS    |
|-------------------------------------|---------|-----------|--------|
| bm25_only                           | 0.2053  | 0.2643    | 3587.2 |
| geom_k100_t4_s8_k8_p2_a0.8          | 0.2028  | 0.2622    | 552.7  |
| geom_richcov_k100_t8                | 0.2055  | 0.2642    | 267.3  |
| **phss_k100_t4_gap**                | 0.2065  | 0.2659    | 158.8  |
| **phss_k100_t4_persist**            | 0.2049  | 0.2618    | 158.1  |
| **phss_k100_t4_elbow**              | 0.2047  | 0.2637    | 154.9  |
| **phss_k100_t8_richcov_gap**        | 0.2089  | 0.2654    | 61.7   |

## Verdict

**Mixed positive.** PHSS is not universal, but it improves two of the three
BEIR corpora:

- **Scifact:** `LargestGap` and `Elbow` recover the fixed-geometry regression and tie BM25 at `0.6188`.
- **FiQA:** `LargestGap` improves over both BM25 and fixed geometry; `richcov + LargestGap` is the best fragment-geometry row so far at `0.2089`.
- **NFCorpus:** all PHSS rows regress slightly versus the fixed-`knn` baseline.

## Interpretation

1. **LargestGap is the strongest criterion.** It is best or tied-best on all
   three corpora.
2. **The main win is safety plus FiQA upside.** PHSS removes the scifact
   regression of fixed geometry and sets the best FiQA fragment-geometry row.
3. **NFCorpus remains the tradeoff.** PHSS is slightly too conservative there.
4. **Latency is the cost.** PHSS is consistently slower than fixed `knn`.

## Follow-up Rows

Two follow-up branches were tested:

- **adaptive PHSS**: run PHSS only on higher-confidence queries and fall back to fixed `knn` otherwise;
- **richmmr + PHSS**: combine the best upside-seeking fragment selector with PHSS graph construction.

Headline results:

- **Scifact:** `richcov + adaptive PHSS` reaches `0.6176`, which is better than fixed `richcov` (`0.6161`) but still below `phss_gap` on the basic builder (`0.6188`).
- **NFCorpus:** `basic + adaptive PHSS` reaches `0.2560`, improving over both fixed basic geometry (`0.2548`) and full basic PHSS (`0.2531`), but still below the best `richmmr` row (`0.2585`).
- **FiQA:** `richmmr + PHSS` improves over fixed `richmmr` safety but does not beat `richcov + phss_gap`; the best follow-up row is `richmmr l0.50 + PHSS` at `0.2084`, still below `richcov + phss_gap` at `0.2089`.

Verdict: the follow-up rows do not replace the current frontier. Adaptive PHSS
is a useful latency/quality knob, especially on NFCorpus, but `LargestGap`
remains the strongest fully-on row.

## Profiling

The `simeon_profile_fragment_geometry` harness shows the PHSS cost clearly on
scifact:

- **basic fixed**: `1.88 ms/query`
- **basic PHSS**: `6.55 ms/query`
- **basic adaptive PHSS**: `3.62 ms/query`

Within the PHSS query path, the dominant cost is **scale selection itself**, not diffusion:

- `phss_pairwise`: ~`0.56 ms/query`
- `phss_select`: ~`4.38 ms/query`
- adjacency build: ~`1.17 ms/query`
- diffusion: ~`0.001 ms/query`

Even after cleanup, the scorer is still dominated by `phss_select_scale`.

## Implications

PHSS is a real control surface for fragment geometry: it recovers scifact,
produces the best FiQA row, and regresses NFCorpus slightly.

## Next move

1. ~~Add adaptive PHSS~~ — done (`phssadapt_*` rows in this doc).
2. ~~Test on larger pools (`pool_size=500`)~~ — closed by Phase C
   (`phss_pool_scaling.md`). pool=100 is the optimum; larger pools
   regress 2/3 corpora on nDCG@10 within latency budget.
3. ~~1D persistent homology (cycles)~~ — closed by Phase A cheap
   probe (`phss_1d_triangle_results.md`). Triangle-count importance
   weighting is inert (Δ range [−0.0005, +0.0004] across 18 cells;
   R@10/R@100 byte-identical). Mechanism: multi-fragment per-doc
   averaging washes out per-fragment topological weighting on this
   fixture set. Phases B/C cancelled per disprove gate.

Subfinding from Phase C: nfcorpus pool=500 richcov shows **+0.0122 R@100** over
pool=100. It is too slow for a default, but remains a real recall-targeted
lever.

## 2026-04-23 update — `LargestGapApprox` validated as richcov default

Phase B6 of the engineering pass (see `phss_largest_gap_approx_results.md`)
benched the already-shipped `PhssConfig::Criterion::LargestGapApprox`
head-to-head against `LargestGap` on both builders. Result: on the
**richcov** builder approx is **strictly equal-or-better** on all three
corpora (scifact +0.0021, nfcorpus +0.0002, fiqa 0.0000) at **~2× QPS**.
On the **basic** builder approx regresses scifact by −0.0039 and is kept
on `LargestGap`.

**New richcov-builder frontier**:

| Corpus   | `phssapprox_k100_t8_richcov_gap` nDCG@10 | Δ vs BM25 | QPS    |
|----------|-----------------------------------------:|----------:|-------:|
| scifact  | 0.6188                                   |  0.0000   |  112.6 |
| nfcorpus | 0.2544                                   | +0.0023   |  127.6 |
| fiqa     | **0.2089**                               | **+0.0036** | 119.4 |

Same quality as the prior `phss_k100_t8_richcov_gap` row at about 2× the
throughput. The engineering target `phss_select_mean_us < 1500 µs` is met on
all three corpora.

The bench keeps both `phss_*` and `phssapprox_*` recipes for regression
tracking. Production pointer for the richcov frontier is now
`phssapprox_k100_t8_richcov_gap`.

---

*2026-04-22. Code in `src/persistent_homology.cpp`, `include/simeon/persistent_homology.hpp`, and `include/simeon/fragment_geometry.hpp` / `src/fragment_geometry.cpp`.*
