# Phase LXIII: Contrastive Scoring + Query-Type Router

## Setup

Two ideas were combined:

1. **Contrastive debate scorer**
   - build a centroid from BM25 top documents in simeon space
   - reward documents far from that centroid
   - intended use: counterargument retrieval

2. **Query-type router**
   - debate: low entropy, very long query
   - expansion-friendly: high entropy or extreme `avg_idf`
   - fact-claim: low entropy, short query
   - default: BM25 + lead blend

## Results

| System | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| contrastive router | 0.3104 | 0.2006 | 0.2773 | 0.6003 | 0.6300 | 0.4037 |
| `observed_ordering_gated_ensemble` | 0.3494 | 0.2048 | 0.2661 | 0.6044 | 0.6185 | 0.4086 |
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | **0.4108** |

## Failure mode

The query-type routing idea is mostly correct; the debate scorer is wrong.

Counterarguments in ArguAna are not anti-topic outliers. They are usually
topically close to the original claim. Therefore:

- far from the BM25 centroid => often irrelevant
- near the BM25 centroid => includes both support and opposition

Distance from a topical centroid is not a stance signal.

## Interpretation

This phase rejects the hypothesis:

```text
counterargument ≈ topic-neighbor with opposite embedding direction
```

What survives is the routing insight:

- TREC-COVID and NFCorpus are expansion-friendly
- SciFact prefers the safe lexical branch
- ArguAna requires explicit structural or stance evidence, not contrastive
  topical distance

## Status relative to later work

- Phase LXV provides the correct ArguAna signal: pair-ID structure
- Phase LXXI provides the later best universal router: entropy+length hard
  selection

So this phase is best read as a partial success in **query regime detection** and
a failure of the proposed debate scorer.
