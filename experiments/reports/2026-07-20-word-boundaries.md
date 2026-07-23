# Word-bounded character n-gram optimization

## Summary

- Baseline: the frozen lowercase whole-text character 3–5 gram plus word-token,
  8,192-bucket FWHT-to-384 compact preset.
- Selected preset: the same recipe with marker-aware per-word character grams.
- Outcome: nDCG@10 improves on all three development and holdout fixtures;
  recall@100 improves on every holdout. FiQA holdout nDCG is effectively flat.
- Cost: unchanged 384-float output and scoring complexity; median development
  encoding time decreases by 0.9–2.7%.
- Training regime: `artifact-free`; no corpus statistic, fitted vocabulary,
  learned weight, or pretrained artifact is used.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Evidence and semantics

Paperbridge resolved McNamee and Mayfield, _Character N-Gram Tokenization for
European Language Text Retrieval_ (DOI `10.1023/B:INRT.0000009441.78971.BE`),
and Bojanowski et al., _Enriching Word Vectors with Subword Information_ (DOI
`10.1162/tacl_a_00051`, cached paper `d21c204f070c`). They motivate character
subwords and explicit boundary symbols; they do not establish Simeon's
retrieval result, which is measured below. Simeon adopts only fixed
preprocessing, not Bojanowski et al.'s learned word vectors.

`CharNGramScope::WordBounded` splits on ASCII punctuation and whitespace,
preserves every high-bit byte so UTF-8 sequences are not discarded, wraps each
word in `<` and `>`, and emits the configured widths independently per word.
For example, word-bounded trigrams for `ab cd` are `<ab`, `ab>`, `<cd`, and
`cd>`; no gram crosses the space. The generic default remains the legacy
whole-text mode for stored-embedding compatibility.

Tokenization remains `O(L * W)` for input byte length `L` and `W` configured
gram widths, with `O(M)` reusable scratch for maximum word byte length `M`.
The new path usually submits fewer n-grams because it removes cross-boundary
windows. Sketching and FWHT work remain `O(F + S log S + D)`, where `F` is the
emitted feature count, `S = 8192`, and `D = 384`. Exact scoring remains
`O(Q * N * D)`, and stored vectors remain 384 floats.

Lean's reusable experiment contract now classifies ASCII folding, boundary
n-grams, fixed hashing, and fixed projection as fixed algorithms and proves the
composed compact recipe is permitted by `artifact-free`. The same contract
rejects a recipe containing corpus statistics, relevance-tuned weights, or a
pretrained encoder from that regime.

## Development selection

Exploration manifest `fnv1a64:2e149b28b12816e6` compared the frozen whole-text
baseline against word-bounded char+word widths 3–5 and 3–6 and char-only widths
3–5 and 3–6. Fixture fingerprints are SciFact
`fnv1a64:fe2bdf87a26e671b`, NFCorpus `fnv1a64:315ceecf6d8146b3`, and FiQA
`fnv1a64:7faa27cd473f31d5`. Word-bounded char+word 3–5 was the unambiguous
winner; every neighboring candidate was weaker on all three primary scores.

| Corpus | Whole-text nDCG@10 | Word-bounded nDCG@10 | Relative delta | Recall@100 before | Recall@100 after |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.399826 | 0.445894 | +11.52% | 0.691862 | 0.764998 |
| NFCorpus | 0.179504 | 0.197035 | +9.77% | 0.152282 | 0.175106 |
| FiQA | 0.096574 | 0.117932 | +22.12% | 0.254013 | 0.293665 |

Timings are medians of three sequential query-plus-document encoding runs.

| Corpus | Whole-text encode (µs) | Word-bounded encode (µs) | Relative time |
|---|---:|---:|---:|
| SciFact | 176,359 | 174,786 | -0.89% |
| NFCorpus | 125,312 | 122,761 | -2.04% |
| FiQA | 1,264,769 | 1,230,951 | -2.67% |

## Frozen holdout

Frozen manifest `fnv1a64:d5cd0d0053250ee8`, linked to the exploration
fingerprint and containing only the selected variant, was read exactly once on
each test fixture. The baseline is the prior frozen FWHT preset.

| Corpus | Test fingerprint | Whole-text nDCG@10 | Word-bounded nDCG@10 | Relative delta | Recall@100 before | Recall@100 after |
|---|---|---:|---:|---:|---:|---:|
| SciFact | `fnv1a64:cdc576493e9c566a` | 0.400024 | 0.440093 | +10.02% | 0.657889 | 0.735333 |
| NFCorpus | `fnv1a64:a99992f39ee60a0b` | 0.183135 | 0.198537 | +8.41% | 0.167501 | 0.183156 |
| FiQA | `fnv1a64:a8fd6ec5bfc2f5fe` | 0.089828 | 0.089862 | +0.04% | 0.240572 | 0.270579 |

The result transfers on the declared primary metric and materially raises
deep recall. FiQA's nDCG difference is too small to interpret as a meaningful
ranking improvement, so promotion rests on cross-fixture non-regression,
stronger SciFact/NFCorpus results, and the all-fixture recall improvement.

## Verification and residual risk

- Unit tests cover exact boundary markers, absence of cross-word grams,
  punctuation-only input, UTF-8 byte preservation, encoder propagation,
  manifest parsing/serialization, and the frozen preset fields.
- The strict runner records resolved scope, dimensions, seeds, fingerprints,
  timings, metrics, and selection lineage; its phase gate prevented exploration
  manifests from reading holdout.
- `formal/ExperimentContract.lean` checks both artifact-free composition and
  the one-variant frozen-holdout rule.
- Residual risks: ASCII boundaries are not full Unicode word segmentation;
  byte-level n-grams can split inside a multi-byte code point while still
  preserving the bytes; three public corpora do not prove universal transfer;
  and SIMD-tier reduction order remains part of embedding identity.
