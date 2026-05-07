# Phase LV: Embedding-Based Scoring — Negative Filters (Negative-ish)

## Goal

Pivot away from RM3 variants toward embedding-based scoring. Two approaches tested:

1. **Negative filter** — penalize BM25 top-200 docs with very low simeon
   similarity (z < −0.5), filtering lexical false positives.
2. **MMR diversity rerank** — penalize each doc in BM25 top-200 by its maximum
   simeon similarity to any higher-ranked doc, enforcing result diversity.

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | **0.3941** |
| Negative filter | 0.3292 | 0.2059 | 0.2523 | 0.6188 | 0.5681 | 0.3949 |
| MMR rerank | 0.3292 | 0.2053 | 0.2514 | 0.6180 | 0.5648 | 0.3937 |
| Diverse RM3 β=0.25 | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | **0.4108** |

### Why embedding-only approaches fail

The negative filter barely moves scores (+0.0008 macro) because BM25's z-score
dominates. The penalty (0.3 × z-deficit) is too small relative to BM25 magnitude
to change doc rankings. Increasing the penalty would discard BM25 signal
entirely.

The MMR diversity rerank slightly regresses (−0.0003 macro) because result
diversity is a UX concern, not a relevance concern. Penalizing docs for being
similar to higher-ranked docs doesn't help find more relevant docs — it only
disperses the ranking. For nDCG@10, having 3 highly-relevant, similar docs is
better than having 3 diverse, less-relevant docs.

## Strategic assessment

After 24 forward iterations and 6 negative/neutral results in the last 8 phases,
the pattern is clear:

**What works (strong positive):**
- RM3 improvement arc: weighting (+0.0041) → diversity (+0.0062) → β sweep → 0.4108
- Sigmoid entropy dampening: safe gating at 0.4057

**What doesn't (neutral/negative):**
- Multi-signal dampening gates (Phase XLVII, XLVIII)
- Consensus filtering (Phase XLVIII)
- Adaptive α (Phase LI)
- avg_idf gate (Phase LIV — mixed)
- Embedding negative filter (Phase LV)
- MMR diversity rerank (Phase LV)

**The hard ceiling:**

```
BM25 oracle@100 = 0.7423 (candidates exposed by BM25 alone)
Best observed   = 0.4108 (can't rank the pool)
───────────────────────────────────────────────
OrderingGap     = 0.3315 (purely scoring problem)
```

The 4-way union generator only adds +0.0666 to the oracle. The remaining 0.3315
is purely about **scoring BM25's own top-100 pool** — not about different
generators, fusion strategies, or PRF variants.

## What would close a 0.33 ordering gap?

At this scale, the gap isn't about term expansion, diversity, or gating. It
requires a scoring mechanism that captures **qualitatively** different evidence:

1. **Document structure** — section-level, paragraph-level, relation-level
   evidence that isn't accessible to whole-document BM25.
2. **Query intent** — distinguishing fact-claim from debate from medical-QA
   and using different scoring functions for each.
3. **Cross-document evidence** — citation networks, reference chains, argument
   structures that span multiple documents.
4. **External knowledge** — entity linking, concept hierarchies, domain
   ontologies (training-free but corpus-aware).

These are fundamentally new research directions, not variants on the current
approach.

## Next direction

The most promising path that stays within training-free constraints:

**Corpus adapters (`A`)** — the theorem component we have barely exercised.
YAMS already has a `CorpusAdapter` component. Bringing corpus-structure
awareness (paths, headings, sections, entity IDs) to the benchmark could
reduce the ordering gap without violating the training-free constraint,
because corpus structure is observable without qrels.
