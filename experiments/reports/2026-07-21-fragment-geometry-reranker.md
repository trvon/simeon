# Fixed-hash fragment reranker: PHSS versus MaxSim

Date: 2026-07-21

## Contract

- Exploration manifest: `reranker-fragment-geometry.exp`
- Exploration fingerprint: `fnv1a64:bc96efb80fff9f5b`
- Frozen manifest: `reranker-fragment-maxsim-frozen.exp`
- Frozen fingerprint: `fnv1a64:cae1543166a1afb0`
- Code revision: `3361107-dirty`
- SIMD tier: NEON
- Encoder: pinned `simeon-v1-384` fixed hash
- Candidate pool: BM25 top 100
- Query repeats: 5 for latency; rankings are deterministic

The graph and MaxSim arms use the same encoder, rich fragment builder, float32
storage, candidate pool, alpha, and four fragments per document. The isolated
difference is PHSS graph diffusion versus outer MaxSim scoring.

## Selection results

| Corpus | BM25 nDCG@10 | PHSS nDCG@10 | PHSS delta | MaxSim nDCG@10 | MaxSim delta | PHSS mean us | MaxSim mean us |
|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 0.666985 | 0.672705 | +0.005720 | 0.676663 | +0.009678 | 1187 | 306 |
| NFCorpus | 0.266897 | 0.266424 | -0.000473 | 0.269672 | +0.002775 | 1176 | 360 |
| FiQA | 0.237843 | 0.238110 | +0.000267 | 0.239706 | +0.001863 | 2269 | 796 |

MaxSim dominated PHSS on all three selection corpora. The PHSS survivor graph
was used on 97.9% to 100% of queries, so the result is not caused by a dormant
graph path.

## Frozen holdout

| Corpus | BM25 nDCG@10 | MaxSim nDCG@10 | Delta | BM25 MRR@10 | MaxSim MRR@10 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.662213 | 0.670034 | +0.007821 | 0.630243 | 0.637783 | +0.007540 |
| NFCorpus | 0.306195 | 0.307028 | +0.000833 | 0.513815 | 0.515584 | +0.001769 |
| FiQA | 0.237515 | 0.236894 | -0.000622 | 0.295076 | 0.293422 | -0.001654 |

Candidate recall was exactly unchanged within each corpus: 0.882556 SciFact,
0.235954 NFCorpus, and 0.508214 FiQA. This experiment measures ordering inside
the lexical pool, not candidate expansion.

## Decision

Do not promote PHSS graph diffusion as a general reranker. Keep fixed-hash
fragments and MaxSim available as default-off, corpus-gated experimental ranking
evidence. The PHSS survivor graph may still be studied as a relation producer,
but that requires edge-level stability, representation, and saturation evidence;
aggregate rerank quality does not discharge those obligations.
