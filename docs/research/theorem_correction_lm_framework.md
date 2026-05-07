# Theorem Correction: Language-Model Formulation

## Incorrect target

The older framing treated retrieval as minimizing distance to an external
answer-key oracle:

```text
Gap_M(T; C,Q,R) = M(O_M(C,Q,R)) - M(T(C,Q))
```

This is diagnostically useful but not an executable retrieval objective.

## Correct object

For a query `q` and document `d`, the training-free retrieval object is a
document language model `θ_d`:

```text
Score(d, q) = log P(q | θ_d)
            = Σ_{w ∈ q} log P(w | θ_d)
```

Equivalent KL form:

```text
Score(d, q) = - KL(θ_q || θ_d)
```

The feasible search space is the family of language models estimable without
relevance labels.

## Component decomposition

Write the document model as:

```text
P(w | θ_d)
  = λ_mle    P_ML(w|d)
  + λ_col    P(w|C)
  + λ_struct P(w|θ_d^struct)
  + λ_graph  P(w|θ_d^graph)
  + λ_rm     P(w|θ_R)
```

with `Σ λ = 1`.

Interpretation of the components in current simeon terms:

| LM term | Training-free estimator | Row / implementation |
|---|---|---|
| `P_ML(w|d)` | lexical MLE approximation | `bm25_only` |
| `P(w|C)` | collection statistics | BM25 IDF / smoothing side |
| `P(w|θ_d^struct)` | field / lead evidence | `bm25f_lead64` |
| `P(w|θ_d^graph)` | neighborhood smoothing | `ppr_graph` |
| `P(w|θ_R)` | pseudo-relevance feedback | `observed_ordering_rm3_diverse_k10_b0.25` |

## Query-dependent selection

The best universal row is not a single global interpolation. It is a
query-dependent family:

```text
θ_d(q) = θ_d^struct               if H_BM25@10(q) < 0.05 and |q| > 30
       = θ_d^mle + θ_R^diverse    otherwise
```

Observed row: `observed_ordering_entropy_length_router = 0.4141`.

This is better described as **configuration selection over feasible language
models** than as a fixed mixture.

## Why the oracle is not the target

The external oracle `O_M` remains useful only as a benchmark yardstick.

It is not executable because:

1. it sorts by labels, not by text
2. it ignores whether the signal is representable in the training-free document model
3. it can reward task-specific relations, such as counterargument structure in
   ArguAna, that generic text similarity does not encode

Hence:

```text
M(T(C,Q)) ≤ M(O_M(C,Q,R))
```

but the research goal is to enlarge or choose the best feasible `θ_d(q)`, not to
"implement the oracle."

## Mapping observed rows

| Row | LM interpretation | nDCG@10 |
|---|---|---:|
| `bm25_only` | lexical MLE baseline | 0.3941 |
| `observed_ordering_rm3_diverse_k10_b0.25` | `θ_d^mle + θ_R` | 0.4108 |
| `observed_ordering_entropy_length_router` | query-dependent select between `θ_d^struct` and `θ_d^mle + θ_R` | **0.4141** |
| `observed_ordering_gated_ensemble` | earlier query-dependent selector with BM25 middle branch | 0.4086 |

## Consequences

1. `F` is not arbitrary fusion; it is **query-dependent model selection**.
2. `A` matters because some tasks require a different representation, not just a
   different weight vector.
3. The correct next universal move is better QPP-style selection, not softer
   blending.
4. The correct ArguAna move is better argument / stance structure, not generic
   lexical polishing.

## Literature alignment

- QPP / difficulty prediction: Cronen-Townsend et al. 2002; Shtok et al. 2012
- selective expansion: Amati et al. 2004
- query-dependent configuration ranking: Deveaud et al. 2019
- verbose-query handling: Paik and Oard 2014; Roitman 2018
- expansion alternative to RM3: Xu and Croft 2000

## Next experiments under this formulation

1. Add QPP-style gates over clarity, SCQ, NQC, WIG, and score variance only if
   they improve on `observed_ordering_entropy_length_router`.
2. Treat verbose-query handling as a separate branch of the model family rather
   than a small threshold perturbation.
3. Treat argument / stance adapters as representation changes, not as ordinary
   scorer improvements.
