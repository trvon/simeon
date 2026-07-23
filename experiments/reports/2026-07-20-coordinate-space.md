# Coordinate-space utilization and selective routing

## Summary

- Question: are the extra FWHT dimensions underused, and can the YAMS
  retrieval-coordinate admission formula safely route between the 768- and
  1,024-dimensional hashed-IDF charts?
- Utilization result: no coordinate-collapse problem was found. The diagonal
  variance participation ratio is 1,021.7–1,022.9 effective coordinates out of
  1,024 (99.77–99.89%), and coordinate standard-deviation CV is only
  1.65–2.38%.
- Calibration result: spherical mean-centering and diagonal standardization
  regress development nDCG@10 on every corpus. The observed mean direction is
  useful lexical structure, not a nuisance component to remove.
- Routing result: score-distortion gates produce tiny corpus-specific wins, but
  no fixed gate transfers across SciFact, NFCorpus, and FiQA. A query-only
  energy gate interpolates between the fixed endpoints without improving them.
- Decision: retain the reusable coordinate diagnostics/routing provider and
  cached replay machinery; do not promote a coordinate transform, 1,024-width
  preset, or routing gate. Holdout was not opened.
- Training regime: `corpus-adaptive`; the IDF and optional coordinate moments
  use deployment documents only. Runtime routing consumes no qrels or learned
  representation.

## Formal bridge from YAMS

The experiment adapts the separation in YAMS
[`RetrievalCoordinates.lean`](../../../../formal/topology/Yams/Topology/RetrievalCoordinates.lean):
an embedding is not assumed to be one globally faithful coordinate system;
local charts are ordered by evidence, while hard narrowing requires separately
observed distortion and uncertainty evidence. The 1,024-dimensional projection
is the reference chart and its nested 768 prefix is the local chart.

Simeon's local Lean contract mirrors the `full` / `augment` / `narrow` actions,
requires positive observation status, and proves that missing or rejected
evidence can never select `narrow`. This is deliberately a safety contract, not
a theorem that a low routing energy improves relevance.

## Reusable provider

`kind=coordinate_calibrated_idf` builds one maximum-width raw hashed-IDF/FWHT
workspace and an 8 KiB document-coordinate artifact containing the 1,024 means
and variances. Nested widths, transforms, blends, and admission thresholds then
reuse that identity. Its JSONL records contain:

- corpus anisotropy and coordinate variance utilization;
- prefix energy, top-100 overlap/Jaccard, and mean similarity distortion;
- exact ranked metrics and route action counts;
- IDF/coordinate artifact fingerprints and storage;
- transform, 768-score, 1,024-reference-score, and policy replay timings; and
- whether exact per-query score evidence came from the replay cache.

The cache holds both full-corpus score fields for research sweeps. After one
evidence build, later thresholds perform only policy selection and exact top-K.
This changed the SciFact eight-policy sweep to one score build plus roughly
15–17 ms per policy without changing any metric bit.

## Coordinate utilization

Exploration manifest `fnv1a64:45ffac3e614b052d` measures unit-sphere document
coordinates. Normalizing before estimating the centroid is essential: raw
hashed-IDF magnitude tracks document length and is not comparable to a short
query's magnitude.

| Corpus | Mean-direction / RMS norm | Effective dimensions | Effective fraction | Coordinate stddev CV |
|---|---:|---:|---:|---:|
| SciFact | 18.46% | 1,022.49 | 99.85% | 1.92% |
| NFCorpus | 20.27% | 1,021.67 | 99.77% | 2.38% |
| FiQA | 21.46% | 1,022.88 | 99.89% | 1.65% |

FWHT is doing its job: useful variance is spread almost uniformly across the
sampled rows. There is no small set of individually privileged axes to prune or
upweight. The 1,024-dimensional corpus-coordinate artifact takes 8 KiB; its
calibration pass took 1.7 ms SciFact, 1.2 ms NFCorpus, and 19.4 ms FiQA.

## Calibration ablation

| Variant nDCG@10 | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| Raw 768 | **0.591125** | 0.273382 | **0.172157** |
| Raw 1,024 | 0.595695 | **0.277862** | 0.171108 |
| Centered 768 | 0.586493 | 0.270859 | 0.161972 |
| Centered 1,024 | 0.591926 | 0.274876 | 0.162804 |
| Full-standardized 768 | 0.586771 | 0.270972 | 0.162450 |
| Full-standardized 1,024 | 0.591716 | 0.275296 | 0.164707 |

Fifty-percent variance shrinkage lands between centering and full diagonal
standardization but also regresses every corpus. Since variance is already
nearly uniform, standardization mostly perturbs useful cosine ordering.

## Chart evidence

| Corpus | 768 energy fraction | 768/1,024 top-100 overlap | Mean similarity distortion |
|---|---:|---:|---:|
| SciFact | 74.89% | 71.32% | 0.01768 |
| NFCorpus | 75.02% | 67.25% | 0.01809 |
| FiQA | 74.99% | 83.12% | 0.01502 |

Energy matches the uniform 75% expectation almost exactly in aggregate, yet
the candidate sets differ substantially. The tail is not empty; it is another
nearly uniform random sample of the weighted sketch. Whether its additional
evidence helps the first ten ranks remains query- and corpus-dependent.

## Selective routing

Routing manifest `fnv1a64:88c68837f2510e40` tests fixed endpoints, direct
score blends, distortion gates, overlap gates, and conjunctions. Per-corpus
maxima select different policies:

| Corpus | Fixed 768 | Fixed 1,024 | Best observed route | Route nDCG@10 |
|---|---:|---:|---|---:|
| SciFact | 0.591125 | 0.595695 | distortion ≤ 0.014 | 0.596190 |
| NFCorpus | 0.273382 | 0.277862 | overlap ≥ 0.75 | 0.277866 |
| FiQA | 0.172157 | 0.171108 | distortion ≤ 0.018 | 0.172562 |

Those are not transferable. The SciFact gate scores 0.277680/0.170712 on
NFCorpus/FiQA; the FiQA gate scores 0.593294/0.275406 on SciFact/NFCorpus.
Fixed 768/1,024 blends also stay within the endpoint envelope.

The query-only energy manifest `fnv1a64:32b7e18e595abcdc` tests absolute
deviation thresholds from 0.0025 through 0.08. It can decide before document
scoring and therefore has a real work interpretation: an admitted query uses
768 rather than 1,024 dot-product coordinates. It still produces no shared
quality gain. At the tightest 0.0025 gate, nDCG@10 is
0.594942/0.277646/0.171526, below the best fixed endpoint on every corpus.

## Cost and decision

One cached development evidence build costs the following exact score work:

| Corpus | 768 scoring | 1,024 reference scoring | Policy replay | Cached score evidence |
|---|---:|---:|---:|---:|
| SciFact | 241 ms | 316 ms | 15–17 ms | 33.5 MB |
| NFCorpus | 64 ms | 82 ms | 5–6 ms | 9.4 MB |
| FiQA | 3.75 s | 4.70 s | about 30 ms | 230.6 MB |

The cache is experiment-only. Production fixed-width retrieval does not need
it. Score-overlap/distortion routing must compute both charts, so it is a
safety/diagnostic mechanism rather than a latency optimization. Query-energy
routing can reduce distance evaluations for admitted queries, but retains the
4,096-byte 1,024-vector unless a separate tiered storage design is introduced.

The development evidence rejects promotion before holdout. More axis-level
weighting would fight an already uniform random projection. The better use of
the YAMS formula is to treat independently constructed feature families—word,
character, sparse lexical, PMI/fragment, and graph evidence—as overlapping
charts. Their distortions and failure modes can differ meaningfully; individual
FWHT rows cannot.
