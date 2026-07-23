# Character/word feature-family atlas

## Summary

- Question: can YAMS-style overlapping charts improve the 768-dimensional
  hashed-IDF encoder by assigning character and word evidence independent
  coordinates instead of hashing them into one sketch?
- Instrument: `kind=feature_family_atlas_idf` builds separate character/word
  IDF artifacts and maximum-width FWHT charts, enforces an exact 768-float
  deployment budget, and caches score evidence for weight replay.
- Signal: the charts are genuinely complementary. Their top-100 overlap is
  only 6–26%, and the evaluation-only candidate union adds as much as 0.021
  Recall@100 on FiQA.
- Quality: independent chart normalization improves SciFact by 0.00509 and
  FiQA by 0.00386 at their respective best points, but every split regresses
  NFCorpus. Corpus-RMS joint normalization does not repair transfer.
- Decision: retain the atlas as a reusable research instrument, but do not
  promote a feature split or open holdout. The combined 768-dimensional preset
  remains the production quality point.
- Regime: corpus-adaptive and training-free. Documents fit IDF and optional RMS
  scales; qrels are evaluation-only and never enter the representation or
  policy.

## Representation

For character prefix `c`, word prefix `w`, and fixed character weight `a`, the
independently normalized mode scores

`a cosine(cq, cd) + (1-a) cosine(wq, wd)`.

This is exactly the cosine of a concatenated vector whose normalized blocks
are scaled by `sqrt(a)` and `sqrt(1-a)`. The vector stores `char_dim +
word_dim` floats, required by the resolver to equal the declared 768-float
budget. It is therefore a deployable block embedding rather than an unpriced
ensemble.

The joint mode concatenates raw weighted prefixes and normalizes once. Its
`joint_rms` variant first divides each family by its corpus document RMS. This
preserves within-family document/query energy variation while calibrating the
average family scale without relevance labels.

Each family needs its own IDF artifact because tokenization mode belongs to the
artifact identity. The two 65,536-bucket tables occupy 262,144 bytes total,
twice the combined encoder's IDF state but independent of corpus size.

## Independent chart sweep

Exploration manifest: `fnv1a64:7ea862b9bdf49f95`. Four allocations consume
exactly 768 floats: 704/64, 640/128, 512/256, and 384/384. Several fixed score
weights replay over each allocation's cached evidence.

| Corpus | Combined 768 | Best independent split | Split nDCG@10 | Delta |
|---|---:|---|---:|---:|
| SciFact | 0.591125 | 512/256, char weight 0.75 | **0.596213** | **+0.005088** |
| NFCorpus | **0.273382** | 640/128, char weight 0.85 | 0.271800 | -0.001582 |
| FiQA | 0.172157 | 384/384, char weight 0.65 | **0.176018** | +0.003861 |

The per-corpus maxima select different allocations and weights. More
importantly, the SciFact winner falls to 0.264356 on NFCorpus, while the FiQA
winner falls to 0.262647. No fixed recipe is non-regressing across the three
development corpora.

## Chart complementarity and energy

At the 512/256 allocation:

| Corpus | Char/word top-100 overlap | Family-union Recall@100 | Combined Recall@100 | Raw char query-energy share |
|---|---:|---:|---:|---:|
| SciFact | 12.28% | 0.877215 | **0.879687** | 97.76% |
| NFCorpus | 12.02% | **0.236514** | 0.225649 | 97.54% |
| FiQA | 22.49% | **0.443641** | 0.423768 | 97.87% |

Low overlap establishes that these are meaningful feature charts rather than
random views of one projection. NFCorpus and FiQA have candidate headroom, but
a fixed cosine mixture does not consistently move those extra relevant
documents into the first ten ranks. SciFact's combined encoder already recalls
more relevant documents than the family union at this allocation; its gain is
an ordering effect.

Raw joint normalization is poorly conditioned for a portable weight: character
n-grams carry roughly 98% of prefix energy at 512/256. This comes from feature
multiplicity, not dimension collapse. The earlier coordinate audit found
nearly uniform variance within a chart; this experiment finds strongly unequal
scale *between* semantically constructed charts.

## RMS calibration

Exploration manifest: `fnv1a64:b53258063048177d`. `joint_rms` estimates two
document-prefix RMS values for each dimension allocation and rescales the raw
blocks before joint cosine normalization.

| Corpus | Combined 768 | Best RMS split | Split nDCG@10 | Delta |
|---|---:|---|---:|---:|
| SciFact | 0.591125 | 512/256, char weight 0.80 | 0.592587 | +0.001462 |
| NFCorpus | **0.273382** | 640/128, char weight 0.80 | 0.268492 | -0.004891 |
| FiQA | 0.172157 | 640/128, char weight 0.65 | **0.177082** | +0.004925 |

RMS calibration moves the best points but does not make a weight transferable.
It also reduces Recall@100 materially on NFCorpus. The varying raw family ratio
contains useful document-specific lexical structure; treating all deviations
from the corpus RMS as nuisance removes part of that signal.

## Research cost

The deployable representation remains exactly 768 floats per document. The
research workspace retains two maximum-width 768-float document charts so it
can replay allocations, plus two full query-by-corpus score fields for the
current allocation.

| Corpus | Workspace setup | Raw workspace vectors | Cached scores | Deployable vectors |
|---|---:|---:|---:|---:|
| SciFact | 0.44 s | 31.8 MB | 33.5 MB | 15.9 MB |
| NFCorpus | 0.32 s | 22.3 MB | 9.4 MB | 11.2 MB |
| FiQA | 3.02 s | 354.1 MB | 230.6 MB | 177.1 MB |

The cache is bounded to one dimension allocation. Consecutive weight variants
reuse it; changing the split recomputes scores without retaining every matrix.

## Formal boundary and next direction

`formal/ExperimentContract.lean` now classifies fixed feature partitioning and
fixed score fusion as algorithmic evidence, and family RMS as corpus
statistics. It proves the RMS atlas is corpus-adaptive rather than
artifact-free, rejects relevance-tuned runtime weights under the
corpus-adaptive regime, and proves that a declared allocation cannot hide
ensemble storage beyond its dimension budget.

The negative transfer result narrows the next useful direction. Another static
character/word mixture is unlikely to help. Future YAMS-style routing should
keep the combined chart as the safe base and admit a family residual only from
query-observable evidence with an explicit fallback; the family-union recall
shows candidate value, while this sweep shows that unconditional replacement
is unsafe.
