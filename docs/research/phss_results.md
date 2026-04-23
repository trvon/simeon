# Persistent Homology Scale Selection (PHSS) — Results

## Experiment

Implemented 0D persistent homology (Union-Find) on the fragment similarity graph within each BM25 pool. Three scale-selection criteria were evaluated against fixed `knn=8` baselines on BEIR-3:

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

**Mixed positive.** PHSS is not a universal win, but it improves two of the three BEIR corpora we tested:

- **Scifact:** `LargestGap` and `Elbow` recover the fixed-geometry regression and tie BM25 at `0.6188`.
- **FiQA:** `LargestGap` improves over both BM25 and fixed geometry; `richcov + LargestGap` is the best fragment-geometry row so far at `0.2089`.
- **NFCorpus:** all PHSS rows regress slightly versus the fixed-`knn` baseline.

## Interpretation

1. **LargestGap is the strongest criterion.** It is the best or tied-best PHSS row on all three corpora.
2. **The main win is safety plus FiQA upside.** PHSS removes the scifact regression of fixed geometry and sets the best FiQA fragment-geometry score.
3. **NFCorpus remains the tradeoff.** The persistence-selected threshold is slightly too conservative there.
4. **Latency is the cost.** PHSS pays the expected O(N^2) similarity cost and is consistently slower than fixed `knn`.

## Follow-up Rows

Two follow-up branches were tested after the initial PHSS pass:

- **adaptive PHSS**: run PHSS only on higher-confidence queries and fall back to fixed `knn` otherwise;
- **richmmr + PHSS**: combine the best upside-seeking fragment selector with PHSS graph construction.

Headline results:

- **Scifact:** `richcov + adaptive PHSS` reaches `0.6176`, which is better than fixed `richcov` (`0.6161`) but still below `phss_gap` on the basic builder (`0.6188`).
- **NFCorpus:** `basic + adaptive PHSS` reaches `0.2560`, improving over both fixed basic geometry (`0.2548`) and full basic PHSS (`0.2531`), but still below the best `richmmr` row (`0.2585`).
- **FiQA:** `richmmr + PHSS` improves over fixed `richmmr` safety but does not beat `richcov + phss_gap`; the best follow-up row is `richmmr l0.50 + PHSS` at `0.2084`, still below `richcov + phss_gap` at `0.2089`.

Verdict: the follow-up rows do not replace the current frontier. Adaptive PHSS is a useful latency/quality knob, especially on NFCorpus, but `LargestGap` remains the strongest fully-on PHSS row. `richmmr + PHSS` does not beat either `richmmr` on NFCorpus or `richcov + PHSS` on FiQA.

## Profiling

The new `simeon_profile_fragment_geometry` harness shows the main PHSS cost clearly on scifact:

- **basic fixed**: `1.88 ms/query`
- **basic PHSS**: `6.55 ms/query`
- **basic adaptive PHSS**: `3.62 ms/query`

Within the PHSS query path, the dominant cost is **scale selection itself**, not diffusion:

- `phss_pairwise`: ~`0.56 ms/query`
- `phss_select`: ~`4.38 ms/query`
- adjacency build: ~`1.17 ms/query`
- diffusion: ~`0.001 ms/query`

After removing unnecessary persistence-point materialization from non-diagram PHSS runs, the scorer is still dominated by `phss_select_scale`, which is now the main optimization target.

## Implications

PHSS is a real control surface for fragment geometry:
- it recovers scifact,
- it produces the best FiQA fragment-geometry score,
- and it regresses NFCorpus slightly.

## Next move

1. Add adaptive PHSS: let the router pick the criterion or fall back to fixed `knn` based on query features.
2. Test on larger pools (`pool_size=500`) where the O(N²) cost is higher but the topological structure may diverge more from `knn=8`.
3. Explore 1D persistent homology (cycles) to capture document-level manifold structure beyond fragment clustering.

---

*2026-04-22. Code in `src/persistent_homology.cpp`, `include/simeon/persistent_homology.hpp`, and `include/simeon/fragment_geometry.hpp` / `src/fragment_geometry.cpp`.*
