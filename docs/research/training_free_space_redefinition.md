# Redefining the Training-Free Retrieval Space

## Object of study

A training-free retrieval system is a tuple:

```text
T = (G, A, S, F, B)
```

where:

- `G`: candidate generator family
- `A`: corpus-adapter family
- `S`: scorer family
- `F`: fusion / routing family
- `B`: resource budget

The oracle is not a component of `T`. It is an external evaluation object.

## Objective

For metric `M`, corpus `C`, queries `Q`, and relevance relation `R`:

```text
Gap_M(T; C,Q,R) = M(O_M(C,Q,R)) - M(T(C,Q))
```

`R` is never available at inference time. Therefore the feasible optimum under
budget `B` is:

```text
T*_B = argmax_{T ∈ TrainingFreeSpace(B)} M(T(C,Q))
```

with `O_M` used only for diagnostics.

## Inequality chain

```text
M(T(C,Q)) ≤ M(O_M(C,Q,R))
```

For a fixed generator family `G` and depth `K`:

```text
M(BestObserved_G@K)
  ≤ M(BestFeasible_G@K)
  ≤ M(O_M restricted to P_G@K)
  ≤ M(O_M)
```

`O_M restricted to P_G@K` is a counterfactual pool oracle, not a callable
retrieval system.

## Gap terms

```text
Gap_M(T) = ExposureGap + OrderingGap + RoutingApproxGap + BudgetGap + residual
```

Definitions:

- **Exposure gap:** relevant material is not in the exposed pool
- **Ordering gap:** relevant material is present but ranked poorly
- **Routing approximation gap:** the oracle can choose the right branch, but the
  real router must infer it from qrel-free evidence
- **Structural gap:** the missing signal is relation / field / stance / graph
  structure rather than generic similarity

## Current measured floor

```text
bm25_only                                0.3941
observed_ordering_entropy_length_router  0.4141
adapter-aware ensemble                   0.4634
oracle_bm25_pool_k100                    0.7423
oracle_union_4way_k100                   0.8089
```

Interpretation:

- Universal text-only progress remains in the `0.41x` range.
- The universal gap is still large relative to recent gains.
- Adapter gains dominate universal scorer / router gains.

## Current best evidence for each gap term

### Exposure gap

Union oracles improve over BM25 pool oracles, so candidate generation remains a
real variable. BM25 is not the theorem target.

### Ordering gap

BM25 pool oracles remain far above constructive rows on ArguAna and TREC-COVID,
so ranking quality remains unresolved even when relevant documents are exposed.

### Routing approximation gap

The best current constructive reduction is
`observed_ordering_entropy_length_router`.

Its rule is:

```text
if H_BM25@10 < 0.05 and n_terms > 30:
    LeadFieldStrategy
else:
    Rm3DiverseStrategy
```

This is better modeled as **query-performance prediction** than as generic
fusion. The relevant literature line is: query difficulty, selective expansion,
and query-dependent configuration ranking.

### Structural gap

ArguAna is the strongest witness. Its residual gap is better described as an
**argument-retrieval / stance-retrieval mismatch** than as a generic scoring
failure. The best evidence comes from:

- pair-ID adapter gains
- argument-retrieval literature
- stance-aware retrieval literature

## Research rule implied by the corrected space

Do not ask whether a new method "closes the oracle."

Ask instead:

```text
which gap term dominates on this corpus, and what is the smallest
training-free change to G, A, S, or F that reduces it under budget B?
```

## Current priority order

1. Better QPP-style hard routing over the existing universal strategies
2. Better argument / stance adapters for structural corpora
3. Better generators where exposure is still the dominant limiter

Detailed empirical history remains in the phase notes; this file is the compact
framework statement.
