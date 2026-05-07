# Phase LVIII: LM-inspired Weighted Mixture (Inert)

## Goal

The corrected theorem frames retrieval as Language Model interpolation:

```text
P(w | θ_d) = Σ_c λ_c · P_c(w | d)
Score(d,q) = Σ_{w∈q} log P(w | θ_d)
```

This phase tests whether changing the per-component WEIGHTS in a score-level
fusion approximates the term-level LM interpolation. Two experiments:

1. **Convex combination** (LM mixture): `w_bm25 = 1/(1+3λ)`, others `λ/(1+3λ)`
2. **Fixed weights**: body=0.40 or 0.55, aux/expansion/embedding = 0.20 or 0.15

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| LM mixture (convex) | 0.3298 | 0.2051 | 0.2637 | 0.6204 | 0.6093 | 0.4057 |
| LM weighted (0.4/0.2) | 0.3297 | 0.2074 | 0.2642 | 0.6190 | 0.6075 | 0.4056 |
| LM safe (0.55/0.15) | 0.3282 | 0.2079 | 0.2610 | 0.6263 | 0.5965 | 0.4040 |
| Sigmoid dampening | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | 0.4041 |
| Diverse RM3 alone | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4108 |

All weighted variants are within ±0.002 of each other and of the sigmoid
dampening. **Changing the per-component weights has negligible effect.**

## Why the LM formulation doesn't change results

The LM mixture is mathematically equivalent to the sigmoid dampening:

```text
LM_mixture = sigmoid_dampening / (1 + 3λ)
```

Since ranking is invariant to multiplicative constants per query, the two
produce identical rankings. The "convex combination" is just the sigmoid
divided by a constant.

The fixed-weight variants just multiply the z-scored components by constants.
Since z-scores are normalized (mean 0, std 1), changing the weights is
equivalent to changing the relative influence of each component. But the
components are highly correlated (BM25 and diverse RM3 both use BM25 term
frequencies), so the weight choice barely moves the ranking.

## The actual gap: score-level vs term-level

The LM framework requires **term-level probability interpolation**:

```text
P(w | θ_d) = λ_bm25 · P_ML(w|d_body)
           + λ_struct · P_ML(w|d_aux)
           + λ_rm · P_RM(w|d, Q)
           + λ_emb · P_emb(w|d)        // not trivially available
Score(d,q) = Σ_{w∈q∪R} log P(w | θ_d)  // sum over expanded query
```

What we actually compute is **document-level score addition**:

```text
Score(d,q) = z(BM25_body(d,q)) + z(BM25_aux(d,q)) + z(RM3(d,q)) + z(simeon(d,q))
```

These are fundamentally different architectures:

| Aspect | Term-level LM | Score-level fusion |
|---|---|---|
| Blending | Per query term, before summation | Per document, after summation |
| Weight semantics | Probability mixture weights | Arbitrary scale factors |
| Expansion | Contributes to P(w\|θ_d) for w∈R | Separate document score |
| Constraint | Σ λ = 1 (proper mixture) | No constraint |
| Firepower | Proper query likelihood | Linear combination of whole-doc scores |

The term-level approach can capture **per-term interactions**: an expansion term
gets high weight in the RM component but zero in the body component, and the LM
interpolation naturally handles this. The score-level approach treats each
component as a black box and adds them.

## Why we can't implement term-level LM in the current benchmark

The current codebase's scoring API is `Bm25Index::score(query, out_scores)` which
produces `float[]` per-document scores. There is no API for:

1. Per-document, per-term probability estimation
2. Per-term interpolation between multiple LM components
3. Query likelihood computation from per-term probabilities

The `score_weighted_hashes` API comes closest — it scores with a weighted set
of term hashes. But it only blends original query terms with expansion terms
(α blending), not multiple document-level language model components.

## Implementation path for proper LM interpolation

To actually implement the LM framework, the benchmark would need:

1. **Per-term probability access**: API to compute `P_ML(w|d)` for each
   (query term, document) pair.
2. **LM interpolation layer**: compute `P(w|θ_d) = Σ_c λ_c · P_c(w|d)` for
   each term.
3. **Score aggregation**: `Score(d,q) = Σ log P(w|θ_d)`.

Step 1 requires adding a new method to `Bm25Index` that returns per-term
contributions rather than aggregated scores. This is a library change, not
just a benchmark row.

## Conclusion

Score-level fusion is saturated. The remaining ordering gap (0.3315) requires
term-level probability interpolation. The LM framework provides the right
theory, but implementing it requires new library infrastructure. See
[`theorem_correction_lm_framework.md`](theorem_correction_lm_framework.md).
