# Safe-base feature residual at 1,024 dimensions

## Summary

- Question: can the complementary character/word charts improve the combined
  768-dimensional hashed-IDF encoder without replacing its safer geometry?
- Instrument: `kind=feature_family_atlas_idf` now lazily retains exact combined
  base evidence and replays raw-cosine, query-z-score, rank-RRF, and selective
  residual policies from one cached workspace.
- Budget: the tested representation is a 768-float base plus a 192-character /
  64-word residual, exactly 1,024 floats per document (+33.3%). Three bounded
  IDF tables occupy 393,216 bytes.
- Quality: query-z-score fusion at residual weight 0.15 improves both nDCG@10
  and Recall@100 on all three development corpora. Its nDCG gains are only
  +0.00484 SciFact, +0.00094 NFCorpus, and +0.00079 FiQA.
- Decision: retain the instrument and the z-score point as a research
  reference, but do not promote it or open holdout. No single policy clears the
  repository's +0.005 nDCG@10 gate on every development corpus.
- Regime: corpus-adaptive and training-free. The three IDF artifacts and
  optional query-to-corpus score moments use no qrels, gradients, or pretrained
  vectors. Qrels enter evaluation only.

## Experiment identity

Exploration manifest: `experiments/embedding-feature-residual-1024.exp`

- semantic manifest: `fnv1a64:c5c977188af98144`
- SciFact fixture: `fnv1a64:fe2bdf87a26e671b`
- NFCorpus fixture: `fnv1a64:315ceecf6d8146b3`
- FiQA fixture: `fnv1a64:7faa27cd473f31d5`
- split: `dev`
- metric profile: `trec-v1`
- recorded code revision: `1f26ef6-dirty`
- SIMD tier: `neon`

The manifest is deliberately still `phase=exploration` and contains multiple
variants. The runner therefore rejects any attempt to apply it to the declared
test split.

## Safe-base residual

Let `b` be the unit-normalized combined 768-dimensional embedding and let `c`
and `w` be separately normalized family prefixes. With base/residual weight
`r` and residual character weight `a`, raw-cosine mode scores

`(1-r) cosine(bq, bd) + r[a cosine(cq, cd) + (1-a) cosine(wq, wd)]`.

This is exactly a cosine in a concatenated 1,024-dimensional block vector with
scales `sqrt(1-r)`, `sqrt(r*a)`, and `sqrt(r*(1-a))`. It is deployable with one
vector lookup and does not hide an unpriced ensemble.

Query-z-score mode standardizes the base and residual score fields separately
over all corpus documents before applying the same fixed blend. This corrects
their query-dependent scale mismatch, but has no fixed single-vector cosine
interpretation: exact corpus score moments must be observed at query time.
Rank RRF likewise operates on independently retrieved top-100 lists. These
modes are useful research upper bounds on composition, not drop-in ANN recipes.

## Transfer result

The 192/64 residual uses independent family normalization and character weight
0.75. The balanced point is query-z-score fusion with residual weight 0.15.

| Corpus | Combined 768 nDCG@10 | Residual nDCG@10 | Delta | Combined R@100 | Residual R@100 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.591125 | **0.595960** | **+0.004835** | 0.879687 | **0.883704** | +0.004017 |
| NFCorpus | 0.273382 | **0.274317** | **+0.000935** | 0.225649 | **0.225816** | +0.000167 |
| FiQA | 0.172157 | **0.172943** | **+0.000787** | 0.423768 | **0.429124** | +0.005356 |

This is the only tested operating point that improves both reported metrics on
all three corpora. It is encouraging evidence that the family chart contains a
portable residual, but it is not promotion-grade evidence: every nDCG gain is
below +0.005 except the more aggressive SciFact-only points.

Raw cosine at weight 0.20 reaches 0.598073 on SciFact (+0.006948), 0.273877 on
NFCorpus (+0.000494), and 0.173186 on FiQA (+0.001030). NFCorpus Recall@100
falls by 0.002736. Query-z-score weight 0.20 clears +0.005 only on SciFact and
also regresses NFCorpus Recall@100. More aggressive weights improve isolated
primary scores but lose recall transfer.

The earlier plain 1,024-dimensional combined endpoint scored approximately
0.595695 SciFact, 0.277862 NFCorpus, and 0.171108 FiQA. The residual z-score
point is slightly better on SciFact and FiQA but materially worse on NFCorpus;
neither geometry dominates. Simply spending 1,024 coordinates differently is
not enough to approach learned-model quality uniformly.

## Routing and rank fusion

The reusable selective policy observes only label-free quantities: calibrated
word energy, character/word top-100 overlap, and combined-base/family top-100
overlap. At 192/64 their means are:

| Corpus | Word energy | Char/word overlap | Base/family overlap |
|---|---:|---:|---:|
| SciFact | 0.2771 | 0.0536 | 0.3022 |
| NFCorpus | 0.3031 | 0.0652 | 0.2798 |
| FiQA | 0.2805 | 0.0541 | 0.4555 |

The charts are distinct, but none of the fixed intervals predicts query-level
benefit portably. The strongest conservative gate, base/family overlap at least
0.25 with raw residual weight 0.20, admits 515/809 SciFact, 162/324 NFCorpus,
and 442/500 FiQA queries. It changes nDCG@10 by +0.006278, -0.000031, and
+0.001439 respectively; NFCorpus Recall@100 still falls by 0.000858. Word-
energy and inverse-overlap gates likewise interpolate between endpoints rather
than discovering a transferable region.

Weighted RRF does not solve the calibration problem. At residual weight 0.20
it changes nDCG@10 by +0.000641 SciFact, -0.002423 NFCorpus, and +0.000055
FiQA; larger residual weights regress further. It is rejected for this path.

Every rejected selective query copies the exact base ranking rather than
reconstructing it from a truncated candidate pool. This makes fallback safety
an executable property even though admission quality remains empirical.

## Cost accounting

All values below are from the balanced query-z-score variant. The deployable
column is exactly 1,024 floats per document. Cached evidence contains complete
base, character, and word query-by-document score fields and is research-only.

| Corpus | Family workspace setup | Base artifact + encoding | Three score fields | Policy replay | Deployable vectors | Cached scores |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.455 s | 0.375 s | 0.212 s | 0.047 s | 21.2 MB | 50.3 MB |
| NFCorpus | 0.321 s | 0.268 s | 0.060 s | 0.016 s | 14.9 MB | 14.1 MB |
| FiQA | 3.021 s | 2.355 s | 2.895 s | 0.158 s | 236.1 MB | 345.8 MB |

The combined 768-dimensional vectors alone occupy 15.9 MB, 11.2 MB, and
177.1 MB respectively. Raw-cosine deployment pays the 33.3% vector increase
but does not require the score cache. Query z-normalization and RRF require a
two-stage retrieval/scoring design unless an ANN-compatible approximation is
introduced and evaluated explicitly.

## Formal boundary and decision

`formal/ExperimentContract.lean` classifies fixed residual composition as a
fixed algorithm and query-to-corpus score moments as corpus statistics. It
proves both recipes remain corpus-adaptive, rejects relevance-tuned weights,
proves 768+192+64 equals the declared 1,024-float budget, and proves that
missing or inadmissible routing evidence selects the base action.

This iteration improves the research foundation more than the shipped model:
base and residual evidence are independently measurable, expensive evidence
is cached once, policies replay cheaply, storage is exact, and negative gates
are durable results. The production `quality_hashed_idf_retrieval_config()`
remains the combined 768-dimensional preset. Holdout remains unopened.

The next useful optimization should target a pre-retrieval, ANN-compatible
calibration signal or improve the residual representation itself. Further
threshold searches over the current three observations would add selection
pressure without evidence that they encode residual benefit.
