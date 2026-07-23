# Hashed-IDF fusion complementarity

## Summary

- Baseline: the frozen six-leg union pool ranked by
  `0.6*z(WSDM-SAB) + 0.4*z(WSDM-Atire)`.
- Candidate: the promoted 768-dimensional, 65,536-bucket hashed-IDF encoder as
  an optional top-100 candidate leg and z-normalized score leg.
- Development selection: candidate+score fusion at IDF weight 0.30; the
  remaining 0.70 preserves the baseline WSDM 0.6/0.4 ratio.
- Frozen result: positive on all three holdouts, but nDCG@10 gains of only
  +0.0032/+0.0016/+0.0029 are below the ±0.005 promotion gate.
- Decision: retain the reusable experiment provider and manifests; do not add a
  production preset or change the published learning-free fusion frontier.
- Training regime: `corpus-adaptive`; no qrels, pretrained representations, or
  relevance-fitted runtime state enter a variant.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Reusable provider

`kind=wsdm_idf_fusion` reconstructs the exact production workbench pool from
Atire, WSDM(Atire), SAB, WSDM(SAB), PMI rich-covered fragment geometry, and
five-BM25 RRF. It builds the corpus assets once, adds the IDF top-K only when
requested, caches raw WSDM/IDF scores over the resulting per-query evidence,
and replays all manifest weights without rebuilding the corpus.

The provider accepts only `idf_weight`, `idf_candidates`, `pool_per_leg`, and a
fully explicit hashed-IDF encoder configuration. Variants in one manifest must
share the pool and encoder identity. Its JSONL records include semantic
manifest/fixture fingerprints, artifact fingerprint, candidate recall, ranked
metrics, marginal IDF timings, document-vector bytes, and cached evidence size.
Result lines are flushed at record boundaries so progress stderr cannot corrupt
captured JSONL under a PTY.

The baseline reproduces the historical development anchors exactly to the
reported precision: 0.6950 SciFact, 0.2977 NFCorpus, and 0.2480 FiQA.

## Development attribution

Exploration manifest `fnv1a64:1e0dfa1068456354` sweeps score-only and combined
candidate+score weights from 0.05 through 0.60.

| Variant nDCG@10 | SciFact | NFCorpus | FiQA |
|---|---:|---:|---:|
| Frozen WSDM baseline | 0.695006 | 0.297708 | 0.248044 |
| IDF candidates only | 0.693187 | 0.298651 | 0.248603 |
| IDF score only, w=0.30 | 0.697892 | 0.309460 | 0.256103 |
| Combined, w=0.20 | 0.699132 | 0.310157 | 0.255214 |
| **Combined, w=0.30** | **0.700217** | **0.310713** | **0.256002** |
| Combined, w=0.40 | 0.698412 | 0.311978 | 0.255901 |

Corpus-specific maxima differ: SciFact selects combined 0.30, NFCorpus selects
score-only 0.40 at 0.312135, and FiQA selects score-only 0.50 at 0.259237.
Those choices are not transferable fixed recipes. Combined 0.30 is the only
shared row that improves nDCG@10 by more than 0.005 on all three corpora.

The mechanism attribution is also clear. Candidate-only changes are neutral or
negative at the top ten. Score-only supplies most of the nDCG gain. Adding IDF
candidates raises the development candidate recall and Recall@100:

| Corpus | Candidate recall baseline → IDF | Recall@100 baseline → selected |
|---|---:|---:|
| SciFact | 0.9355 → 0.9448 | 0.9306 → 0.9333 |
| NFCorpus | 0.2652 → 0.2892 | 0.2488 → 0.2596 |
| FiQA | 0.5454 → 0.5783 | 0.5012 → 0.5279 |

Thus the new leg retrieves relevant documents the six-leg pool misses, but the
fixed WSDM/IDF score blend only partially converts that headroom into nDCG@10.

## Frozen holdout

The selected row was frozen alone in manifest `fnv1a64:97540b89ad10cd0f`, with
lineage to the exploration manifest. Each test fixture was read once.

| Corpus | Test fingerprint | WSDM baseline | Selected fusion | Absolute delta | Candidate recall |
|---|---|---:|---:|---:|---:|
| SciFact | `fnv1a64:cdc576493e9c566a` | 0.6885 | 0.691701 | +0.0032 | 0.9553 |
| NFCorpus | `fnv1a64:a99992f39ee60a0b` | 0.3220 | 0.323588 | +0.0016 | 0.3009 |
| FiQA | `fnv1a64:a8fd6ec5bfc2f5fe` | 0.2512 | 0.254068 | +0.0029 | 0.5993 |

Every direction transfers, but none clears the promotion threshold. The result
also stays below fused-feedback PRF on SciFact/NFCorpus (0.6990/0.3261). FiQA
sets a new observed row above 0.2512, but +0.0029 is within the gate and does not
justify changing the frozen production recipe.

## Marginal cost

The existing PMI fragment preparation dominates total workspace construction.
The additional IDF work on development data was:

| Corpus | Artifact build | Document encode | Query encode + dense score | Float document vectors |
|---|---:|---:|---:|---:|
| SciFact | 0.168 s | 0.194 s | 0.227 s | 15.9 MB |
| NFCorpus | 0.123 s | 0.140 s | 0.051 s | 11.2 MB |
| FiQA | 0.905 s | 1.415 s | 1.732 s | 177.1 MB |

The artifact itself remains 128 KiB. Exact full-corpus IDF scoring and
768-float document storage are the material deployment costs; cached sweep
evidence is research-only and measured separately in result JSON.

## Conclusion

The experiment falsifies the simple assumption that the large standalone IDF
gain will add linearly to a mature fusion stack. IDF contributes useful recall
and modest independent ordering evidence, but the fixed blend is already near a
lexical precision plateau. The next optimization should target how added
candidates are reranked—using a new training-free interaction signal—or reduce
the cost of the 768-dimensional leg through quantization. Merely increasing its
fixed fusion weight is exhausted by the 0.05–0.60 curve.
