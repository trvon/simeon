# SOTA comparison anchors for BEIR-style corpora

_Date: 2026-05-03._  This note compares the latest local Simeon/YAMS profile
against public BEIR-style numbers. Treat this as **orientation**, not a strict
leaderboard submission: implementations, analyzers, qrel splits, title handling,
and evaluation wrappers differ across sources.

## Sources used

- BEIR package / original benchmark table: https://pypi.org/project/beir/0.0.12/
- OpenSearch neural sparse benchmark table: https://opensearch.org/blog/improving-document-retrieval-with-sparse-semantic-encoders/
- Jina-ColBERT-v2 model card: https://huggingface.co/jinaai/jina-colbert-v2
- OpenMatch ANCE-Tele model docs: https://openmatch.readthedocs.io/en/latest/models/ance-tele_coco-base_msmarco_qry-psg-encoder.html
- OpenSearch agentic search relevance table: https://opensearch.org/blog/evaluating-agentic-search-in-opensearch/
- Local profile artifacts: `/tmp/simeon_profile_20260503_175035/core` and
  `/tmp/simeon_profile_20260503_175035/diagnostics`.

## Public comparison numbers, nDCG@10

| System / source | Training category | TREC-COVID | NFCorpus | FiQA | ArguAna | SciFact |
|---|---|---:|---:|---:|---:|---:|
| BEIR package BM25 | non-neural / unsupervised lexical | 0.616 | 0.294 | — | 0.441 | — |
| BEIR package SBERT | pretrained dense, zero-shot | 0.461 | 0.233 | 0.223 | 0.415 | — |
| OpenSearch BM25 | non-neural / lexical | 0.688 | 0.327 | 0.254 | 0.472 | 0.690 |
| OpenSearch dense TAS-B | trained dense | 0.481 | 0.319 | 0.300 | 0.427 | 0.643 |
| OpenSearch hybrid dense+BM25 | trained dense + lexical | 0.698 | 0.335 | 0.322 | 0.378 | 0.672 |
| OpenSearch neural sparse bi-encoder | trained sparse neural | 0.771 | 0.360 | 0.376 | 0.508 | 0.723 |
| OpenSearch neural sparse doc-only | trained doc encoder + lexical query path | 0.707 | 0.352 | 0.344 | 0.461 | 0.716 |
| Jina-ColBERT-v2 | trained late interaction | 0.834 | 0.346 | 0.408 | 0.366 | 0.678 |
| ColBERTv2.0 model-card baseline | trained late interaction | 0.726 | 0.337 | 0.354 | 0.465 | 0.689 |
| ANCE-Tele / CoCondenser | MS MARCO-trained dense | 0.774 | 0.344 | 0.290 | 0.456 | 0.710 |
| OpenSearch agentic lexical | prompted/agentic lexical | 0.645 | 0.353 | 0.194 | 0.283 | 0.676 |
| OpenSearch agentic hybrid | prompted/agentic hybrid | 0.680 | 0.367 | 0.262 | 0.361 | 0.695 |

## Local profile numbers, nDCG@10

| Local row | TREC-COVID test | NFCorpus test | FiQA test | ArguAna test | SciFact test |
|---|---:|---:|---:|---:|---:|
| Simeon BM25 current fixture | 0.5649 | 0.2521 | 0.2053 | 0.3293 | 0.6188 |
| Oracle@100 over Simeon BM25 pool | 0.9613 | 0.4355 | 0.5122 | 0.9269 | 0.8755 |
| Oracle@500 over Simeon BM25 pool | 0.9928 | 0.5446 | 0.6608 | 0.9701 | 0.9224 |
| ArguAna topic-stem adapter diagnostic | — | — | — | 0.7694 | — |
| ArguAna argument-point diagnostic | — | — | — | 1.0000 | — |

## Interpretation by corpus

### ArguAna

Public trained/non-trained numbers cluster around 0.36–0.51 nDCG@10 depending on
system and evaluator. The gated ArguAna topic-stem diagnostic reaches 0.7694,
and the schema diagnostics reach 1.0, but those are **not** product recipes.
They prove the missing axis is corpus structure. This supports the YAMS
`CorpusAdapter` direction: relation/field adapters can beat generic trained
similarity models when the structure is explicit and deterministic.

### FiQA and NFCorpus

The best public anchors here are neural sparse / late-interaction systems:
FiQA roughly 0.38–0.41 and NFCorpus roughly 0.35–0.36. Our Oracle@500 is much
higher (FiQA 0.6608, NFCorpus 0.5446), while our current BM25 floor is lower.
This says the next useful work is candidate-pool expansion plus discriminative
reranking, not another small encoder tweak.

### SciFact

Public trained sparse/dense anchors sit near 0.69–0.72. Our BM25 fixture already
lands at 0.6188, Oracle@100 is 0.8755, and Oracle@500 is 0.9224. This is a good
candidate for a lightweight claim/evidence adapter or title/abstract field-aware
scorer.

### TREC-COVID

Public strong systems range from ~0.70 to ~0.83. Our BM25 floor is lower than
many published BM25 runs, but Oracle@100/500 is very high (0.9613/0.9928). The
runway is almost entirely ranking/discrimination over already-recalled docs.

## Caveats

1. The local fixture uses fixed MiniLM reference embeddings and explicit dev/test
   splits generated for Simeon research. Some public BEIR numbers use only the
   official test split, different analyzers, title inclusion, or ElasticSearch
   settings.
2. Oracle rows are upper bounds over a BM25 candidate pool, not runnable systems.
3. ArguAna schema diagnostics intentionally use fixture structure and should not
   be marketed as universal retrieval.
4. “Training-free” here means no target qrel training and no learned model
   inference. Public “zero-shot” dense/sparse systems are usually trained on
   MS MARCO or other corpora and evaluated out-of-domain.

## Experimental target implied by SOTA comparison

To be competitive with strong public trained systems while staying training-free:

- **ArguAna:** productize structure adapters, not similarity. Target >0.50 first,
  then >0.70 with non-schema-leaking relation structure.
- **FiQA/NFCorpus:** expand candidate pools and add cheap discriminators. Target
  public neural-sparse band: FiQA ~0.38, NFCorpus ~0.36.
- **SciFact/TREC-COVID:** title/abstract/evidence adapters and field-aware BM25F.
  Target public strong sparse band: SciFact ~0.72, TREC-COVID ~0.77–0.83.
