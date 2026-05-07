# Phase XXXVIII: Oracle-External Theorem

## Correction

The oracle is not an implementation variable. It is the answer key used to
measure the distance between a realizable training-free system and the ideal
ranking induced by relevance judgments.

A realizable system is:

```text
T = (G, A, S, F, B)
```

where `G` generates candidates, `A` adapts observable corpus structure, `S`
scores query-document evidence, `F` fuses or routes scores, and `B` fixes the
budget.

The oracle is external:

```text
O_M(C, Q, R)
```

where `C` is the corpus, `Q` the query set, `R` the relevance relation/qrels,
and `M` the evaluation metric. For nDCG, this is the ideal relevance-sorted list
used to normalize discounted cumulative gain.

Therefore the target is not:

```text
T contains Oracle(...)
```

but:

```text
minimize Gap_M(T; C,Q,R) = M(O_M(C,Q,R)) - M(T(C,Q))
```

under the constraint that `T` never observes `R` at inference time.

## Paper-backed foundations

This framing matches classical IR theory:

1. **Ideal ranking / metric oracle.** Järvelin and Kekäläinen define cumulative
   gain measures relative to an ideal ranked list under graded relevance. That is
   the evaluation oracle: an external normalization target, not an algorithmic
   component.
2. **Optimal ranking principle.** Robertson's Probability Ranking Principle says
   ranking by probability of relevance is optimal only under specific
   assumptions. This gives the theoretical target: approximate the relevance
   ordering without observing relevance labels at inference.
3. **Model family approximations.** Robertson and Zaragoza's probabilistic
   relevance framework grounds BM25/BM25F as estimators inside a retrieval model,
   not as ceilings. BM25 is one approximation to relevance evidence.
4. **Query-performance/risk prediction.** Cronen-Townsend, Zhou, and Croft's
   query performance prediction work justifies measuring when a retrieval method
   is likely to fail before seeing qrels. This is the right theoretical support
   for a general router.
5. **Fusion as approximation.** Reciprocal rank fusion and related data-fusion
   work provide model-agnostic ways to combine rank evidence. Fusion is a
   realizable approximation to the oracle ordering, not a substitute for the
   oracle.
6. **Pseudo-relevance feedback.** Relevance-model/RM3 work treats top retrieved
   documents as noisy evidence for query expansion. This supports candidate-pool
   interventions while keeping the true relevance relation external.

## Gap decomposition from theorem to system

Let `P_T(q)` be the candidate pool exposed by system `T` for query `q`, and let
`I_R(q)` be the ideal metric-optimal ranking under relevance relation `R`.

The full measured gap is:

```text
Gap_M(T) = M(I_R) - M(rank_T)
```

We decompose this using diagnostic counterfactuals, not callable oracle modules:

### 1. Candidate-exposure gap

Relevant documents are missing from the exposed pool.

```text
ExposureGap_K(T) = M(I_R) - M(I_R restricted to P_T@K)
```

System response:

- improve `G`: BM25F, RM3, graph/path/title generators, fragment generators
- improve `A`: corpus/language adapters that expose observable structure
- increase or reallocate budget `B`

### 2. Ordering gap inside exposed candidates

Relevant documents are present but misordered.

```text
OrderingGap_K(T) = M(I_R restricted to P_T@K) - M(rank_T over P_T@K)
```

System response:

- improve `S`: field scores, discourse/cue scores, section/fragment scores
- improve score calibration and evidence normalization

### 3. General routing/fusion approximation gap

A diagnostic answer-key router can tell which generator/fusion would have won,
but a real router must infer that without qrels.

```text
RoutingApproxGap(T) = M(best_answer_key_choice over candidates)
                      - M(F_general(q, generator_diagnostics))
```

System response:

- improve qrel-free query performance prediction features
- use risk-aware fallback instead of corpus-specific rules
- validate by pooled-dev→pooled-test and leave-one-corpus-out transfer

### 4. Budget gap

The better approximation exists in the search space but violates latency, memory,
or index-build constraints.

```text
BudgetGap_B = M(best feasible at larger B) - M(best feasible at B)
```

System response:

- cascades
- early exit
- approximate candidate unions
- compact indexes

## Correct theorem statement

For a fixed metric `M`, corpus/query distribution `D`, and budget `B`, define the
training-free feasible family:

```text
F_TF(B) = { T=(G,A,S,F,B) : T uses no relevance labels at inference }
```

Then:

```text
T*_B = argmin_{T in F_TF(B)} E_{(C,Q,R)~D}[ M(O_M(C,Q,R)) - M(T(C,Q)) ]
```

The oracle `O_M` appears only in the objective/evaluation. It is never in `T`.

Our empirical proof program is to estimate which term dominates the observed
shortfall:

```text
Gap_M(T)
  = ExposureGap
  + OrderingGap
  + RoutingApproxGap
  + BudgetGap
  + residual/error
```

and then show that each implementation change reduces one named term without
using `R` at inference.

## Immediate consequence for the current experiments

The earlier `Oracle_BM25@K` and `Oracle_union@K` rows should be described as
**diagnostic projections**:

- `Oracle_BM25@K` measures exposure/order headroom in a BM25 candidate pool.
- `Oracle_union@K` measures how much additional headroom a broader generator
  family exposes.
- Winner dumps measure the gap between answer-key generator choice and qrel-free
  router choice.

They are not system rows and should not be written as if the system calls an
oracle.

## Sources found with Paperbridge

- Järvelin, Kalervo and Kekäläinen, Jaana. 2002. “Cumulated gain-based evaluation
  of IR techniques.” ACM TOIS. DOI: `10.1145/582415.582418`.
- Kekäläinen, Jaana and Järvelin, Kalervo. 2002. “Using graded relevance
  assessments in IR evaluation.” JASIST. DOI: `10.1002/asi.10137`.
- Robertson, S. E. 1977. “The Probability Ranking Principle in IR.” Journal of
  Documentation. DOI: `10.1108/eb026647`.
- Robertson, Stephen and Zaragoza, Hugo. 2009. “The Probabilistic Relevance
  Framework: BM25 and Beyond.” Foundations and Trends in IR. DOI:
  `10.1561/1500000019`.
- Cronen-Townsend, Steve; Zhou, Yun; Croft, W. Bruce. 2002. “Predicting query
  performance.” SIGIR. DOI: `10.1145/564376.564429`.
- Cormack, Gordon V.; Clarke, Charles L. A.; Buettcher, Stefan. 2009.
  “Reciprocal rank fusion outperforms condorcet and individual rank learning
  methods.” SIGIR. DOI: `10.1145/1571941.1572114`.
- Lavrenko, Victor and Croft, W. Bruce. 2001. “Relevance based language models.”
  SIGIR. DOI: `10.1145/383952.383972`.
