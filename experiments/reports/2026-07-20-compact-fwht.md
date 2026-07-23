# Compact FWHT retrieval optimization

## Summary

- Baseline: lowercase character 3–5 grams, 4,096-bucket count sketch,
  Achlioptas projection to 384 dimensions.
- Selected preset: lowercase character 3–5 grams plus word tokens,
  8,192-bucket count sketch, fixed FWHT projection to 384 dimensions.
- Outcome: nDCG@10 improves on every dev and holdout fixture; median dev
  encoding time improves by 3.7–4.9x with unchanged vector storage and scoring
  dimensionality.
- Promotion: `compact_retrieval_config()` exposes the frozen recipe while
  legacy `EncoderConfig` defaults remain unchanged for embedding compatibility.
- Training regime: `artifact-free`; no corpus statistics or learned artifact.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Evidence and design

Paperbridge resolved Weinberger et al., _Feature Hashing for Large Scale
Multitask Learning_ (DOI `10.1145/1553374.1553516`) and Ailon and Chazelle,
_The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors_
(DOI `10.1137/060673096`). These works motivate fixed signed hashing and a fast
structured projection. They do not imply a retrieval gain; selection and
validation below are empirical.

The old projection applies about `4096 * 384 / 3` integer additions per vector
after tokenization. The selected path hashes into twice as many buckets to
reduce the first-stage collision rate, then applies an 8,192-point signed
Walsh–Hadamard transform and samples 384 fixed rows. Its projection work is
`O(S log S + D)` instead of `O(S * D)` for sketch width `S` and output width
`D`. Output vectors remain 384 floats, so exact scoring stays
`O(Q * N * 384)` and scorer working memory is unchanged. The encoder's sketch
scratch grows by 16 KiB; the structured projection also avoids the large
per-row sparse Achlioptas index cache.

## Development selection

The selected exploration manifest is `fnv1a64:07c2df7e37a8ac82`. Fixture
fingerprints are SciFact `fnv1a64:fe2bdf87a26e671b`, NFCorpus
`fnv1a64:315ceecf6d8146b3`, and FiQA `fnv1a64:7faa27cd473f31d5`.

| Corpus | 4k Achlioptas nDCG@10 | 8k FWHT nDCG@10 | Relative delta | Median encoding speedup |
|---|---:|---:|---:|---:|
| SciFact | 0.393494 | 0.399826 | +1.61% | 3.96x |
| NFCorpus | 0.175300 | 0.179504 | +2.40% | 3.73x |
| FiQA | 0.093749 | 0.096574 | +3.01% | 4.92x |

Timings are medians of three sequential query-plus-document encoding runs. The
direct 384-bucket sketch and the broader projection/n-gram ablations remain in
`embedding-compact-sketch.exp` and `embedding-compact-geometry.exp`; they were
mixed across corpora and were not selected. The frozen manifest records the
single selected variant and its dev lineage.

## Frozen holdout

Candidate manifest `fnv1a64:ed50fcdf674e9aa1` was compared with the previously
frozen lowercase baseline `fnv1a64:0102edd5a6bd6e61` exactly once per test
fixture.

| Corpus | Test fingerprint | Baseline nDCG@10 | Frozen FWHT nDCG@10 | Relative delta | Encoding speedup |
|---|---|---:|---:|---:|---:|
| SciFact | `fnv1a64:cdc576493e9c566a` | 0.365665 | 0.400024 | +9.40% | 3.80x |
| NFCorpus | `fnv1a64:a99992f39ee60a0b` | 0.175816 | 0.183135 | +4.16% | 3.77x |
| FiQA | `fnv1a64:a8fd6ec5bfc2f5fe` | 0.088133 | 0.089828 | +1.92% | 4.99x |

Recall@100 changes from 0.6770 to 0.6579 on SciFact, 0.1678 to 0.1675 on
NFCorpus, and 0.2127 to 0.2406 on FiQA. The preset is therefore a primary-metric
and encoding-throughput promotion, not a claim of strict dominance on every
secondary metric.

## Verification

- `compact_retrieval_config()` is covered by exact field, output-width,
  normalization, and deterministic case-folding tests.
- The experiment runner records complete configs, fingerprints, timings, SIMD
  tier, selection lineage, and exact top-100 metrics.
- Exploration manifests cannot read holdout; frozen manifests require one
  variant and a selection fingerprint in both C++ tests and
  `formal/ExperimentContract.lean`.
- Residual risks: three public IR corpora do not establish universal transfer;
  FiQA's holdout gain is modest; embeddings remain SIMD-tier deterministic
  rather than guaranteed byte-identical across architectures.

