# Signed-square-root sketch weighting

## Summary

- Scope analyzed: artifact-free feature weighting after ASCII normalization.
- Stack detected: C++20, Meson, and the manifest-driven exact retrieval runner.
- Baseline: raw signed counts in the selected lowercase char and char+word encoders.
- Hypothesis: sublinear counts reduce document repetition bias without corpus fitting.
- Patch status: implemented; selected only for the wide sketch; compact use rejected.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.
- Holdout status: unread.

## Evidence context

Paperbridge surfaced two relevant but non-identical precedents. Chisholm and
Kolda study term-weighting formulas for vector-space information retrieval
(DOI `10.2172/5698`). Sánchez et al. describe coordinate-wise signed power
normalization with exponent one-half for pooled high-dimensional vectors and
connect it to repeated-feature “burstiness” (Paperbridge paper
`17bef2198888`, DOI `10.1007/s11263-013-0636-x`). These sources motivate the
axis; they do not establish that a post-hash transform will work in Simeon.
That claim is determined by the controlled dev results below, including the
compact counterexample.

## Findings

### Wide, low-collision sketch

- Location: `EncoderConfig::sketch_weighting` and `Encoder::Impl::encode_one`.
- Previous pattern: signed bucket counts grow linearly with repeated features.
- Change: apply `sign(x) * floor(1024 * sqrt(abs(x)))` to each populated sketch bucket before projection or normalization.
- Complexity before: `O(T + S)` encoding for `T` emitted tokens and sketch dimension `S` (the sketch is already initialized and emitted as an `S`-wide vector).
- Complexity after: `O(T + S)` with one additional constant-time transform per nonzero bucket and `O(1)` extra memory.
- Determinism: a hardware square-root estimate is corrected against integer division, producing the exact integer floor before the existing projection path.
- Training regime: `artifact-free`; no corpus statistics, labels, or learned artifacts are consumed.

| Corpus | Raw wide nDCG@10 | Signed-sqrt wide | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| SciFact | 0.502608 | 0.577643 | +0.075035 | +14.93% |
| NFCorpus | 0.232064 | 0.271658 | +0.039594 | +17.06% |
| FiQA | 0.142925 | 0.177338 | +0.034412 | +24.08% |

The comparison is manifest `fnv1a64:698fecd345d5f936`; fixture fingerprints remain SciFact `fnv1a64:fe2bdf87a26e671b`, NFCorpus `fnv1a64:315ceecf6d8146b3`, and FiQA `fnv1a64:7faa27cd473f31d5`.

On a sequential SciFact rerun, query-plus-document encoding rose from 171,846 to 309,745 microseconds (1.80x), while end-to-end exact retrieval rose about 2.1% because wide-vector scoring dominates. Output size and estimated working memory are unchanged.

### Compact, collision-heavy sketch

The same transform must not be treated as a universal default. A nonlinear operation after hashing does not commute with feature collision and signed cancellation. At 4,096 sketch dimensions followed by a 384-dimensional projection, it regressed every development corpus:

| Corpus | Raw compact nDCG@10 | Signed-sqrt compact | Relative delta |
|---|---:|---:|---:|
| SciFact | 0.393494 | 0.367193 | -6.68% |
| NFCorpus | 0.175300 | 0.148572 | -15.25% |
| FiQA | 0.093749 | 0.079710 | -14.97% |

The mode therefore remains opt-in. The selected wide configuration is recorded in `embedding-sqrt-weighting-frozen.exp`; compact retrieval remains on raw counts plus ASCII lowercase. The proposed feature-level follow-up was subsequently implemented and rejected: it removed the collision-ordering issue but regressed compact nDCG@10 even further, showing that repeated-feature magnitude itself carries useful lexical evidence. See `2026-07-20-feature-sqrt.md`.

## Verification

- Focused tests cover the exact fixed-point transform, repeated features, projection determinism, manifest resolution, and result serialization.
- Raw remains the default, preserving existing embeddings.
- PMI rejects non-raw sketch weighting rather than silently ignoring it.
- The three-corpus dev matrix covers both the wide and compact spaces.
- Paperbridge identifiers are copied into the frozen manifest's fingerprinted metadata.
- `formal/ExperimentContract.lean` checks the regime and holdout-state obligations; C++ contract tests separately check the runner implementation.
- Residual risks: wide results still require a one-shot holdout; post-sketch weighting remains collision-sensitive; encoding overhead is data dependent.
