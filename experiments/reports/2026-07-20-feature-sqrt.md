# Pre-hash square-root TF weighting

## Summary

- Hypothesis: applying `sqrt(tf)` to distinct feature identities before signed
  bucket aggregation would preserve the wide power-normalization gain without
  amplifying compact sketch collisions.
- Result: rejected for promotion. It regresses all compact dev fixtures and is
  dominated by the cheaper post-sketch transform in the wide space.
- Training regime: `artifact-free`; the mode consumes no corpus statistics,
  labels, or learned state.
- Holdout status: unread. A rejected exploration variant was not frozen.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Method

`FeatureWeighting::SqrtTf` counts each distinct 64-bit feature identity before
count-sketch bucket aggregation. It materializes
`floor(1024 * sqrt(tf))` with an exact integer-square-root correction, then
applies the feature's fixed count-sketch sign and bucket. Character and word
features retain their existing 1.0 and 0.5 relative weights. Applying both
feature-level and post-sketch weighting is rejected rather than silently
composing two nonlinear transforms.

For `T` emitted tokens, `U` distinct feature identities, and sketch width `S`,
raw encoding uses `O(T + S)` time and `O(S)` scratch. Feature-level weighting
uses expected `O(T + U + S)` time and `O(U + S)` scratch because it maintains a
hash table of per-feature counts. In these fixtures it increased compact
encoding time by roughly 1.4–1.6x and wide encoding time by roughly 2.0–2.3x.

## Development results

The controlled comparison is manifest `fnv1a64:a2a444ab78f71b8c` on SciFact
`fnv1a64:fe2bdf87a26e671b`, NFCorpus `fnv1a64:315ceecf6d8146b3`, and FiQA
`fnv1a64:7faa27cd473f31d5`.

At 4,096 sketch dimensions followed by a 384-dimensional Achlioptas projection,
pre-hash weighting loses more than the previously rejected post-sketch mode:

| Corpus | Raw compact nDCG@10 | Pre-hash sqrt-TF | Relative delta |
|---|---:|---:|---:|
| SciFact | 0.393494 | 0.345105 | -12.30% |
| NFCorpus | 0.175300 | 0.134719 | -23.15% |
| FiQA | 0.093749 | 0.066291 | -29.29% |

In the 32,768-dimensional char+word space, it approximately reproduces the
post-sketch signed-square-root effect but does not justify its added state:

| Corpus | Post-sketch sqrt nDCG@10 | Pre-hash sqrt-TF | Relative delta |
|---|---:|---:|---:|
| SciFact | 0.577643 | 0.583905 | +1.08% |
| NFCorpus | 0.271658 | 0.271085 | -0.21% |
| FiQA | 0.177338 | 0.176478 | -0.48% |

The collision-ordering explanation was therefore incomplete. Moving the
nonlinearity before bucket aggregation fixes the mathematical interaction with
bucket collision, but removing repeated-feature magnitude also removes useful
lexical evidence. The implementation remains an explicit research ablation so
the negative result is replayable; raw remains the default and the mode is not
part of the compact preset.

After the 8,192-bucket char+word FWHT geometry was selected, a separate
interaction check (`fnv1a64:1802c0fa7661ef03`) compared its raw and pre-hash
variants without touching holdout. Feature sqrt-TF again regressed nDCG@10:
0.399826 to 0.355563 on SciFact (-11.07%), 0.179504 to 0.128498 on NFCorpus
(-28.41%), and 0.096574 to 0.086311 on FiQA (-10.63%). This rules out the old
projection as the sole cause of the compact failure.

## Verification

- Unit tests distinguish pre-hash and post-sketch behavior under a deliberate
  bucket collision and check that the nonlinear modes cannot be combined.
- Manifest resolution and JSON result serialization record the weighting stage.
- PMI rejects non-raw feature weighting rather than ignoring it.
- No test split was accessed for this rejected variant.
