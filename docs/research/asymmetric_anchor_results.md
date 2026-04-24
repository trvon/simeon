# Asymmetric Two-Stage Anchor Control — Negative Result

## Experiment

Implemented `build_doc_semantic_fragments_rich_asymmetric`: Stage 1 applies
`richcov`-style hard overlap caps for sentence fragments; Stage 2 applies MMR
selection for anchor fragments with a tunable novelty penalty.

Tested two parameter settings on BEIR-3:
- `richasym`:  anchor lambda = 0.35
- `richasym2`: anchor lambda = 0.50 (stricter novelty)

## Results

| Corpus   | BM25   | richcov | richmmr | richasym | richasym2 |
|----------|--------|---------|---------|----------|-----------|
| scifact  | 0.6188 | 0.6161  | 0.6134  | 0.6154   | 0.6150    |
| nfcorpus | 0.2521 | 0.2550  | 0.2585  | 0.2539   | —         |

## Verdict

**Negative result.** The asymmetric control surface is strictly worse than
`richcov` on both scifact and NFCorpus. MMR on anchors does not add upside
because the anchor fragments are too correlated with the already-selected
sentence fragments.

## Interpretation

The failure mode confirms that the fragment-geometry upside is not in smarter
anchor selection, but in **better fragment representations** or **better
intra-pool relations**. The current anchors are too close to the sentence
fragments for MMR to find novel signal.

## Disposition

Close this thread. `richcov` remains the safest frontier and `richmmr` remains
the best upside-seeking selector. Any next geometric/topological lever should
operate on the **intra-pool relation structure** rather than on anchor
selection.

---

*2026-04-22. Code in `benchmarks/bench_vs_reference.cpp` (`build_doc_semantic_fragments_rich_asymmetric`).*
