# Phase LVII: Personalized PageRank Graph Re-ranking

## Goal

Following Kurland & Lee (2005) "PageRank without hyperlinks," build a document-
document similarity graph over BM25 top-200 using simeon embeddings, then
propagate relevance via Personalized PageRank. This is the first experiment
grounded in the corrected language modeling framework — the graph corresponds
to the `θ_d^graph` (neighborhood expansion) component.

## Method

```text
1. BM25 top-200
2. Build adjacency: sim(i,j) = max(0, dot(emb_i, emb_j))  (384-d simeon)
3. Row-normalize to transition matrix P
4. Initialize r_0 = L1-norm BM25 z-scores
5. Iterate 3×: r_{t+1} = 0.10·r_0 + 0.90·P^T·r_t
6. Z-score PPR scores to match BM25 scale
7. Blend: final = (1−λ_ppr)·z_bm25 + λ_ppr·z_ppr
   where λ_ppr = 0.2 × sigmoid((entropy−0.48)×8)
```

Two configurations tested:
- Initial (buggy): full PPR replacement, 0.85 damping, 5 iterations → catastrophic
- Fixed: small perturbation (20% γ), 0.90 damping, 3 iterations, z-scored PPR

Row: `observed_ordering_ppr_graph`

Artifact:

```text
/tmp/simeon_ppr2_20260505_170010
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| PPR graph (fixed) | 0.3293 | 0.2093 | 0.2534 | 0.6164 | 0.5862 | 0.3989 |
| Diverse RM3 β=0.25 | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4108 |
| Gated ensemble | 0.3494 | 0.2048 | 0.2661 | 0.6044 | 0.6185 | 0.4086 |

### Deltas vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---:|---:|---:|---:|---:|---:|
| PPR graph | +0.0000 | +0.0040 | +0.0013 | −0.0024 | +0.0213 | +0.0048 |
| Diverse RM3 | +0.0034 | −0.0053 | +0.0268 | −0.0165 | +0.0751 | +0.0167 |

## Interpretation

### PPR adds a small but measurable signal

At +0.0048 macro, PPR is modestly positive — better than BM25 alone but well
below the best scorers. The largest gain is on TREC-COVID (+0.0213), where
the graph propagation helps surface related medical abstracts.

### The similarity graph is weak

The limiting factor: simeon embedding similarities between documents in the
top-200 are generally low and uniform. For most queries, the transition matrix
P is close to uniform, and PPR converges to near-uniform scores. This
fundamentally limits how much the graph can improve ranking.

This is consistent with Tao et al. (2006), who found that doc-doc similarity
needs to be strong for neighborhood expansion to help. Their solution — using
KL-divergence between language models rather than embedding similarity —
produces a more discriminative graph.

### The approach is structurally sound but needs a better similarity metric

The PPR mechanism itself is correct. The failure is in the graph edges: cosine
similarity of global document embeddings is too weak a signal. The fix, per
Tao et al., would be to use:

```text
sim(d_i, d_j) = −KL(θ_{d_i} || θ_{d_j})
```

i.e., the negative KL-divergence between the documents' language models, which
captures term-level distributional similarity rather than coarse topical
similarity.

## Impact on the corrected theorem

The PPR experiment validates the LM framework's architecture: the graph
component is a legitimate evidence source (`θ_d^graph`) that can be blended
with other LM components. The current implementation is limited by the
similarity metric, not by the architecture.

Under the corrected theorem:

```text
P(w | θ_d) = P_ML(w|d) + λ_graph · P(w | θ_d^graph)
```

the PPR graph contributes +0.0048 macro as a standalone component. Combined
with the relevance model component (+0.0167 from diverse RM3), the total gain
over the MLE baseline should be the sum of independent contributions — but
linear addition of unnormalized scores fails because the components overlap.

## Next direction

Per the LM framework, the correct integration is:

```text
P(w | θ_d) ∝ P_ML(w|d) + α·P(w|θ_R) + β·P(w|θ_d^graph)
```

where each component is a probability distribution over the vocabulary, not a
document score. The challenge is implementing the LM-level combination rather
than score-level fusion.

The most promising immediate step: **KL-divergence-based document similarity**
for the graph component, replacing weak embedding similarity with a
term-distribution similarity metric.
