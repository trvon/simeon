# Hashed-IDF dimension scaling

## Summary

- Baseline: the promoted 384-dimensional, 65,536-bucket all-feature hashed-IDF
  encoder.
- Development sweep: 384, 512, 768, and 1,024 output dimensions with identical
  lexical features, IDF artifact, 8,192-bucket sketch, FWHT seed, and exact
  scorer.
- Selected recipe: 768 dimensions, the last point with consistent nDCG@10 gains
  on all three development corpora.
- Outcome: frozen holdout nDCG@10 improves over 384 dimensions by 5.13% on
  SciFact, 4.59% on NFCorpus, and 8.97% on FiQA; recall@100 improves everywhere.
- Cost: vectors grow from 1,536 to 3,072 bytes. Median exact-scoring time grows
  by 77–97%, while encoding changes by no more than 1.2% because the fixed
  8,192-point FWHT dominates projection work.
- Training regime: unchanged `corpus-adaptive`; widening a fixed projection
  consumes no new corpus evidence, labels, or pretrained representation.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Controlled mechanism

The experiment changes `output_dim` only. Each wider FWHT projection uses the
same deterministic sign diagonal and partial Fisher–Yates row order, so its
coordinates extend the narrower sample. L2 normalization removes the global
`1/sqrt(output_dim)` scale. The comparison therefore asks how much retrieval
information survives as more coordinates of the same transformed 8,192-bucket
weighted sketch are retained.

The exploration manifest is `fnv1a64:1b3b849a41590986`. Fixture fingerprints
are SciFact `fnv1a64:fe2bdf87a26e671b`, NFCorpus
`fnv1a64:315ceecf6d8146b3`, and FiQA `fnv1a64:7faa27cd473f31d5`.

## Development selection

| Output dimensions | SciFact nDCG@10 | NFCorpus nDCG@10 | FiQA nDCG@10 | Bytes/vector |
|---:|---:|---:|---:|---:|
| 384 | 0.572972 | 0.253910 | 0.157384 | 1,536 |
| 512 | 0.578905 | 0.260253 | 0.164261 | 2,048 |
| **768** | **0.591125** | **0.273382** | **0.172157** | **3,072** |
| 1,024 | 0.595695 | 0.277862 | 0.171108 | 4,096 |

Relative to 384 dimensions, 768 improves development nDCG@10 by 3.17%, 7.67%,
and 9.39%. Moving from 768 to 1,024 yields only another 0.77% on SciFact and
1.64% on NFCorpus while regressing FiQA by 0.61%. Recall@100 continues to rise
at 1,024, so those coordinates are not empty, but they are not the consistent
top-rank quality/cost point.

## Runtime and storage

Median-of-three sequential development runs:

| Corpus | 384 score ms | 768 score ms | 1,024 score ms | 768 vs 384 | 1,024 vs 384 | 768 encode delta |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 91.3 | 164.5 | 233.8 | +80.2% | +156.2% | +0.8% |
| NFCorpus | 26.9 | 47.6 | 67.2 | +76.5% | +149.3% | -0.4% |
| FiQA | 523.1 | 1,032.7 | 1,503.9 | +97.4% | +187.5% | +1.2% |

Persistent float-vector storage scales exactly with dimension: 768 is 2x the
384-dimensional baseline and 1,024 is 2.67x. The experiment runner's bounded
document block keeps transient memory below a full corpus matrix, but a stored
production index still pays the full per-vector cost. Artifact size remains
128 KiB at every width.

## Frozen holdout

The single selected 768-dimensional variant was frozen in manifest
`fnv1a64:e0a066885031718d`, with lineage to the exploration manifest above.
Each holdout fixture was read once.

| Corpus | Test fingerprint | 384 nDCG@10 | 768 nDCG@10 | Relative delta | Recall@100 384 | Recall@100 768 |
|---|---|---:|---:|---:|---:|---:|
| SciFact | `fnv1a64:cdc576493e9c566a` | 0.553510 | 0.581911 | +5.13% | 0.848889 | 0.885889 |
| NFCorpus | `fnv1a64:a99992f39ee60a0b` | 0.264164 | 0.276286 | +4.59% | 0.209313 | 0.228680 |
| FiQA | `fnv1a64:a8fd6ec5bfc2f5fe` | 0.141598 | 0.154296 | +8.97% | 0.404254 | 0.433026 |

Against the repository's frozen `all-MiniLM-L6-v2` reference, the new
standalone gaps are 0.0632 on SciFact, 0.0404 on NFCorpus, and 0.2144 on FiQA.
The wider projection closes 31.0%, 23.1%, and 5.6% of the respective gap that
remained after the 384-dimensional IDF promotion. It does not establish parity
with current learned SOTA, and comparing a 768-dimensional Simeon vector with a
384-dimensional MiniLM vector must include the 2x storage difference.

## Promotion and next question

`quality_hashed_idf_retrieval_config()` freezes the 768-dimensional recipe as
an opt-in quality preset. `compact_hashed_idf_retrieval_config()` remains at 384
dimensions, and no existing embedding identity changes implicitly. Lean proves
that the fixed wider geometry remains a corpus-adaptive hashed-IDF recipe and
checks `768 <= 8192` for the projection geometry.

The next controlled optimization is not additional standalone width: returns
are already flattening by 1,024. It is whether the 768-dimensional signal adds
complementary candidates or ranking evidence to the existing learning-free
fusion stack, ideally followed by compression/quantization if the wider vector
is retained in production.
