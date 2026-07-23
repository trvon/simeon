# Reusable experimentation

Simeon's research path now has a stable contract: a versioned manifest defines
the hypothesis and variants, a fixture fingerprint identifies the data, and a
JSONL result records the resolved configuration, metrics, timing, memory
estimate, SIMD tier, and source revision. New retrieval ideas should extend
this instrument instead of creating another hard-coded sweep.

The first provider is exact retrieval over `simeon::Encoder`. A second provider,
`kind=wsdm_idf_fusion`, replays candidate and score fusion variants over the
frozen six-leg WSDM workbench. A third,
`kind=coordinate_calibrated_idf`, audits maximum-width coordinate utilization
and replays nested chart transforms/routing policies. A fourth,
`kind=feature_family_atlas_idf`, partitions the fixed vector budget into
independent feature-family charts and replays their normalization/weighting.
The shared layer owns
manifest parsing, fixture loading, fingerprints, ranking metrics, and result
serialization, so future PMI, BM25, fragment, or routing providers can reuse
the same controls.

## Run an experiment

Build with benchmarks enabled (the default), then run the exploration manifest
against a dev fixture:

```sh
meson setup build
meson compile -C build simeon_embedding_experiment
mkdir -p experiments/results
./build/benchmarks/simeon_embedding_experiment \
  experiments/embedding-foundation.exp \
  --fixture /path/to/scifact-fixture \
  --split dev \
  --code-revision "$(git describe --always --dirty)" \
  > experiments/results/scifact-dev.jsonl
```

Use `--variant <name>` for an isolated rerun and
`--document-block-size <count>` to trade temporary memory for batching. Block
size is an execution parameter only: tests assert that it cannot change scores.
The manifest value is used unless the CLI overrides it. The scorer retains the
exact deterministic top 100 because the current metric profile has no cutoff
beyond Recall@100; it does not materialize the full query-by-corpus score matrix.

The fixture format is deliberately small and model-independent:

```text
corpus.tsv              doc_id<TAB>text
queries_dev.tsv         query_id<TAB>text
qrels_dev.tsv           query_id<TAB>doc_id<TAB>positive_integer_grade
queries.tsv             unsuffixed test queries
qrels.tsv               unsuffixed test qrels
```

Reference-model vectors are not part of this contract. A trained model may be
reported as an external evaluation baseline, but it must never become an input
to a Simeon variant that is labelled training-free.

## Manifest contract

The `.exp` syntax is a small, dependency-free `key=value` format. Comments begin
with `#`; values may be quoted. Top-level metadata and variant parameters are
included in the semantic manifest fingerprint, so whitespace and key order do
not change identity but any experimental choice does.

```ini
schema = 1
name = representation-study
metric_profile = trec-v1
training_regime = artifact-free
phase = exploration
selection_split = dev
holdout_split = test
primary_metric = ndcg_at_10
document_block_size = 128

[variant.char-32k]
kind = encoder
ngram_mode = char
sketch_dim = 32768
projection = none
```

Encoder manifests reject unknown parameters. This is intentional: a typo must
fail rather than silently fall back to a default. Result records contain the
fully resolved config, including defaults and seeds. `kind=encoder` is permitted
in either `artifact-free` or `corpus-adaptive` manifests. The
`kind=hashed_idf_encoder` provider additionally builds a label-free document-
frequency artifact from `corpus.tsv` and therefore requires
`training_regime=corpus-adaptive`.

Supported encoder parameters are:

- `ngram_mode`: `char`, `word`, or `char_word`
- `char_ngram_scope`: legacy whole `text` or marker-aware `word_bounded`
- `ngram_min`, `ngram_max`, `sketch_dim`, and `output_dim`
- `hash`: `splitmix64`, `xxhash64`, `crc32`, or `mixed_tabulation`
- `hash_seed` and `projection_seed` (decimal or `0x` hexadecimal)
- `text_normalization`: `none` or locale-independent `ascii_lower`
- `feature_weighting`: `raw` or experimental collision-safe pre-sketch `sqrt_tf`
- `sketch_weighting`: `raw` or deterministic fixed-point `signed_sqrt`
- `projection`: `none`, `achlioptas`, `dense_gaussian`, `very_sparse`,
  `sparse_jl`, or `fwht`
- `l2_normalize`, `matryoshka`, `matryoshka_decay`, and `sparse_jl_eps`

The hashed-IDF provider accepts the same encoder parameters plus:

- `idf_hash_dim`: bounded document-frequency table width
- `idf_scope`: `all`, `char`, or `word`

Its result config records the artifact fingerprint, scope, document count,
table width, serialized storage, and build time. Hash collisions conservatively
increase observed document frequency, so they can suppress a feature but never
spuriously make it rarer.

The `wsdm_idf_fusion` provider accepts that full hashed-IDF encoder identity
plus three strict parameters:

- `idf_weight`: fixed convex z-fusion contribution in `[0, 1]`; the remaining
  mass preserves the WSDM(SAB)/WSDM(Atire) 0.6/0.4 ratio
- `idf_candidates`: whether the IDF top-K extends the frozen six-leg union pool
- `pool_per_leg`: positive candidate depth for every base and optional IDF leg

All fusion variants in one manifest must share `pool_per_leg` and their exact
IDF encoder/artifact identity. The corpus workspace is built once, caches only
per-query candidate evidence, and cheaply replays weights. Result records
separate candidate recall from ranked Recall@100 and report the IDF artifact,
full document-vector bytes, cached evidence, corpus preparation, IDF encoding,
dense scoring, and fusion time.

The `coordinate_calibrated_idf` provider accepts a maximum-width FWHT
hashed-IDF encoder plus:

- `retrieval_dim`: nested prefix width, no larger than `output_dim`
- `coordinate_transform`: `none`, `center`, or `standardize`
- `variance_shrinkage`: diagonal variance mixing in `[0, 1]`
- `min_variance_ratio`: positive floor relative to mean coordinate variance
- `coordinate_policy`: `fixed`, `blend`, `selective`, or `selective_energy`
- `full_weight`: maximum-width score contribution for blend/fallback
- `min_chart_overlap` and `max_chart_distortion`: score-evidence admission gates
- `max_energy_deviation`: query-only deviation from the uniform prefix-energy
  expectation

It estimates coordinate moments on unit-normalized corpus vectors so document
length cannot masquerade as a direction. Results report variance-effective
dimension, anisotropy, coordinate-scale CV, prefix energy, top-100 overlap,
similarity distortion, route counts, and all artifact/memory/timing costs.
Exact per-query prefix/reference scores are cached only inside the research
workspace; subsequent thresholds replay without re-encoding or re-scoring.
Variants must share the full encoder/IDF identity. Coordinate moments and IDF
are corpus statistics, so this provider requires `corpus-adaptive`.

The `feature_family_atlas_idf` provider resolves one strict `char_word`
hashed-IDF/FWHT identity, then derives character-only and word-only encoders
with separate compatible IDF artifacts. It adds:

- `char_dim` and `word_dim`: nested chart widths;
- `storage_budget_dim`: required to equal `char_dim + word_dim`;
- `char_weight`: character block/score weight in `[0, 1]`; and
- `family_normalization`: `independent`, `joint`, or `joint_rms`.

`independent` normalizes both prefixes separately and takes their convex score
sum. `joint` concatenates square-root-weighted raw prefixes and performs one
cosine normalization, preserving query/document family-energy ratios.
`joint_rms` first scales each chart by its corpus document RMS, a label-free
corpus statistic intended to make the fixed weight portable. Every mode has an
exact single-vector interpretation using exactly `storage_budget_dim` floats.

The workspace retains maximum-width family vectors and caches exact score
fields for one dimension allocation; consecutive weight variants replay only
fusion and top-K. Results expose per-family metrics and a clearly marked
evaluation-only top-100 union recall diagnostic. Qrels never enter chart
construction, RMS calibration, or fusion. The two IDF tables and optional RMS
statistics make this provider `corpus-adaptive`, not artifact-free.

The same provider can preserve the combined encoder as a safe base and price a
family chart as an additive residual. Its additional strict parameters are:

- `family_policy`: `family_only`, `base_only`, `residual_blend`, or
  `selective`;
- `residual_score_normalization`: `raw_cosine`, `query_zscore`, or `rank_rrf`;
- `residual_weight` and the positive `residual_rrf_k`; and
- inclusive admission intervals `min_`/`max_word_energy`,
  `min_`/`max_family_overlap`, and `min_`/`max_base_family_overlap`.

`base_only` lazily constructs the exact combined `output_dim` hashed-IDF chart.
Residual policies add `storage_budget_dim` family coordinates, so a 768 base
plus a 192/64 residual is reported as 1,024 floats per document. Raw-cosine
fusion has an exact concatenated-vector interpretation. Per-query z-score
fusion instead observes both complete query-to-corpus score distributions, and
rank RRF observes their top-100 rankings; neither is a drop-in single-vector
ANN metric. Their full score fields are therefore charged as research cache,
and result timings separate base scoring, family scoring, and policy replay.

`selective` admits a residual only when every label-free query/chart observation
falls inside its declared interval. A rejected query copies the base ranking
exactly. This is a safe fallback property, not a quality claim: thresholds
explored across qrels remain evaluation choices and cannot become shipped
defaults without the normal cross-corpus selection and frozen-holdout process.
Three independent IDF tables—combined, character, and word—are fingerprinted
and charged whenever the residual path is active.

BPE and PMI require corpus-owned artifacts and therefore belong in separate
providers with explicit artifact fingerprints; they are not hidden behind the
artifact-free encoder provider.

## Holdout discipline

When `phase`, `selection_split`, and `holdout_split` are present, the runner
enforces them:

1. `phase=exploration` can run the selection split but is rejected on holdout.
2. Select one variant using only dev results and record the exploration result's
   `provenance.manifest` value.
3. Copy that one variant to a new manifest, set `phase=frozen`, and add
   `selection_manifest=<the exploration fingerprint>`.
4. The frozen manifest may then run the holdout. It is rejected if it contains
   more than one variant or lacks selection lineage.

The frozen manifest has its own fingerprint. Together, its
`selection_manifest` metadata and result provenance form a machine-readable
chain from exploration to the single holdout read.

## Result contract

Each variant emits one `simeon.experiment.result.v1` JSON object. Stable fields
include:

- experiment, variant, kind, and all manifest metadata;
- semantic manifest and fixture fingerprints;
- fixture name, split, code revision, and active SIMD tier;
- fully resolved encoder configuration and effective output dimension;
- query/document/qrel counts;
- nDCG@10, P@10, Recall@10, Recall@100, and MRR@10;
- artifact-build, encoding, exact-scoring, and evaluation timings;
- retrieval depth, artifact bytes, and an estimated peak working-set size.

`trec-v1` is the only metric profile today. Like `trec_eval`, it uses the qrel
value directly as the nDCG@10 gain, a fixed denominator of ten for P@10, all
positive qrels as the recall denominator, and document index as the
deterministic equal-score tie-break.
Historical numbers in `research.md` came from older drivers and are not assumed
comparable unless their metric semantics and fixture fingerprints match.

JSONL output is ignored by Git. Durable experiment reports should cite the
result files' manifest and fixture fingerprints plus the code revision rather
than copying a floating “best score” into prose.

## Evidence and formal checks

Paperbridge and Lean fit beside the runner rather than inside the scoring path:

- Use Paperbridge to build a reproducible evidence packet for a hypothesis, then
  record its content identifier as manifest metadata such as
  `evidence_snapshot=paperbridge:<id>`. Because all metadata is fingerprinted
  and copied into results, the literature basis travels with every run.
- Use Lean for structural obligations: metric bounds, fusion-weight constraints,
  deterministic tie rules, and the exploration-to-holdout state transition.
  Record a checked module/revision as `formal_contract=lean:<module>@<rev>`.
- Keep empirical claims empirical. A proof can show that an evaluator or fusion
  rule satisfies its specification; it cannot prove that a representation will
  improve retrieval quality on unseen corpora.

The C++ contract tests remain the executable runtime boundary.
`formal/ExperimentContract.lean` additionally checks the training-regime
source restrictions and proves that a permitted holdout manifest is frozen,
has selection lineage, and contains one variant. It also prices the tested
768+192+64 residual as exactly 1,024 floats and proves that missing or rejected
residual-routing evidence selects the base action. Meson runs the module when
Lean is available, without introducing a Lean dependency into the runtime
library. Metric-bound proofs remain a later extension.

## The training-free model space

“Training-free” is a boundary, not a single algorithm. We use three explicit
regimes so quality gains do not quietly weaken the claim:

| Regime | Allowed state | Examples |
|---|---|---|
| `artifact-free` | Fixed algorithms, dimensions, and seeds; no corpus fitting | hashed n-grams, fixed random projections, deterministic normalization |
| `corpus-adaptive` | Unlabelled deployment-corpus statistics with a reproducible builder and fingerprint | hashed IDF, BM25 statistics, BPE merges, PMI/SVD rows, pseudo-relevance feedback |
| `evaluation-only` | Labels or pretrained models, never consumed by a shipped training-free path | qrels, MiniLM reference rankings, dev-set model selection |

PQ codebooks, label-selected fusion weights, and learned routers must be named
as fitted artifacts; they cannot be folded into `artifact-free`. A component
can still be useful, but the result must state the correct regime.

Within those boundaries, reusable experiments should cover these axes:

| Axis | Current surface | Next reusable provider or ablation |
|---|---|---|
| Token units | whole-text or word-bounded char, word, char+word, BPE | Unicode normalization and field-aware units |
| Feature weighting | raw, signed-sqrt or pre-hash sqrt-TF; fixed weights; corpus-adaptive hashed IDF | entropy weighting and asymmetric length normalization |
| Representation | hash sketch, random projection, PMI rows | random indexing and corpus-adaptive distributional variants |
| Aggregation | global sum; semantic fragments | sublinear pooling, query-conditioned but label-free aggregation |
| Geometry | cosine, fragment MaxSim/PHSS, CSLS; coordinate utilization diagnostics | feature-family local charts; axis centering/standardization rejected |
| Candidate generation | BM25 variants and RRF | provider registry with explicit pool provenance and oracle recall |
| Adaptation | corpus recipes, PRF, query router | unsupervised routing rules separated from label-fitted policies |
| Compression | projection, BF16, PQ; frozen word-bounded 8k FWHT-to-384 compact preset | quality/latency/memory Pareto manifests with artifact identity |

The immediate loop should stay narrow: establish artifact-free encoder
baselines, add one weighting or tokenization mechanism at a time, then promote
only effects that transfer across fixtures under the frozen-holdout workflow.
NUMEN can remain a source of hypotheses, but it no longer needs to define the
shape of the experiment system.

The current artifact-free compact Pareto point is `compact_retrieval_config()`. Its dev
selection, one-shot holdout, complexity comparison, and negative neighboring
ablations are recorded in
[`2026-07-20-compact-fwht.md`](../experiments/reports/2026-07-20-compact-fwht.md)
and the subsequent
[`2026-07-20-word-boundaries.md`](../experiments/reports/2026-07-20-word-boundaries.md).
The preset is an artifact-free fixed recipe, not a corpus-fitted model.

The promoted corpus-adaptive point is
`compact_hashed_idf_retrieval_config()`, backed by a 65,536-bucket (128 KiB)
artifact learned from deployment documents only. Its selection, capacity/scope
ablations, frozen holdout, complexity, and learned-reference gap are recorded
in [`2026-07-20-hashed-idf.md`](../experiments/reports/2026-07-20-hashed-idf.md).

The dimension-scaling manifest keeps that artifact and its 8,192-bucket sketch
fixed while sweeping 384, 512, 768, and 1,024 FWHT coordinates. The frozen
quality point is `quality_hashed_idf_retrieval_config()` at 768 dimensions;
384 remains the compact default. Selection, one-read holdout results, and the
storage/scoring tradeoff are recorded in
[`2026-07-20-hashed-idf-dimensions.md`](../experiments/reports/2026-07-20-hashed-idf-dimensions.md).

The subsequent fusion study retained the provider but did not promote its
selected recipe: candidate+score IDF fusion improved all three frozen holdouts,
but every nDCG@10 gain remained below the ±0.005 promotion gate. The complete
candidate-versus-score attribution is recorded in
[`2026-07-20-hashed-idf-fusion.md`](../experiments/reports/2026-07-20-hashed-idf-fusion.md).

The coordinate-space study then treated 768 and 1,024 dimensions as nested
retrieval charts under the YAMS admission discipline. It found 99.77–99.89%
variance-effective coordinate use, rejected centering/standardization on all
three dev corpora, and found no routing threshold that transferred. Holdout was
not opened. The diagnostics and cached policy provider remain available; see
[`2026-07-20-coordinate-space.md`](../experiments/reports/2026-07-20-coordinate-space.md).

## Extending the instrument

A new experiment kind should reuse `benchmarks/experiment_support.*` and add
only three pieces: a strict parameter resolver, a scorer producing per-query
`Ranking` rows, and kind-specific resolved configuration in the result. Keep
fixture parsing, metric semantics, fingerprints, split policy, and JSON schema
shared. Add contract tests before wiring the new provider into a CLI.
