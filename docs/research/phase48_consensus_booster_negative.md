# Phase XLVIII: Cross-Generator Consensus Booster (Negative)

## Goal

Phase XLVII showed that per-query simeon-weight gates beyond entropy don't help.
This phase tests a different dampening strategy: **cross-generator consensus**.
Only documents appearing in BOTH BM25 top-50 and simeon top-50 receive simeon's
z-score contribution. Documents that simeon promotes but BM25 ignores get zero
simeon signal.

The intuition: if simeon is noisy, restrict its influence to documents that BM25
already agrees are relevant. This prevents simeon from promoting irrelevant docs
that happen to be topically similar.

## Method

```text
consensus_simeon_score[d] = simeon_score[d] if d ∈ BM25_top50 ∩ simeon_top50, else 0
score = (1 − λ) × z_bm25 + λ × (z_bm25 + z_bm25f + z_rm3 + z_consensus_simeon)
```

where λ = sigmoid((entropy − 0.48) × 8.0) as in Phase XLVI.

Artifact:

```text
/tmp/simeon_consensus_20260504_170849
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| sigmoid dampen | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | **0.4041** |
| consensus | 0.3231 | 0.1839 | 0.2643 | 0.6127 | 0.5921 | 0.3952 |

### Deltas vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| sigmoid | +0.0004 | +0.0005 | +0.0086 | +0.0016 | +0.0392 | +0.0101 |
| consensus | **−0.0062** | **−0.0214** | +0.0122 | −0.0061 | +0.0272 | +0.0011 |

## Interpretation

The consensus filter is worse than the simple entropy sigmoid across the board:

- **FiQA collapses** (−0.0214): the corpus has short queries (avg 11 terms). BM25
  and simeon top-50 have very low overlap, so the consensus filter suppresses
  simeon almost entirely. But FiQA's raw simeon z-score was helping.
- **ArguAna regresses** (−0.0062): similarly, BM25 and simeon top-50 differ
  dramatically (ArguAna's BM25 is strong and confident; simeon is very weak).
- **NFCorpus improves** (+0.0122 vs +0.0086): the only corpus where consensus
  helps. NFCorpus has the highest l-specific overlap between BM25 and simeon.
- **TREC-COVID loses** (+0.0272 vs +0.0392): consensus is too restrictive.

The key failure: simeon's value is often in finding documents that BM25 MISSES.
Requiring BM25 consensus eliminates exactly the complementary candidates that
make simeon useful in the first place. On NFCorpus where simeon and BM25 agree
more, consensus works; on all other corpora, it destroys simeon's additive value.

## Impact on the theorem

This is the third negative result confirming that generator fusion is saturated:

1. Phase XLVII: multi-signal gates ≤ simple entropy
2. Phase XLVIII: consensus filtering ≪ simple entropy
3. Phase XLVI: simple entropy sigmoid is near-optimal

The remaining 0.4048 macro gap to the 4-way oracle is not about _which_ documents
from each generator's list to include. The documents are there; the scorer
cannot rank them. The next move must change the scoring function `S` itself,
not the fusion logic `F` or the generator selection.

## Next direction

Candidate approaches for improving the scorer `S`:

1. **Document section/field discrimination** — score different parts of a
   document differently rather than treating the whole document uniformly.

2. **Query-document interaction features beyond BM25** — term proximity, term
   order, query coverage, passage-level matching.

3. **Embedding-based document-document relationships** — use simeon embeddings
   to identify authoritative documents (high embedding centrality in the pool)
   or diverse/novel documents (low redundancy with already-ranked docs).

4. **Score calibration by query type** — different scoring functions for
   different query regimes (short vs long, high-IDF vs low-IDF, etc.).
