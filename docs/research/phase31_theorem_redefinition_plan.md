# Phase XXXI — theorem redefinition plan

## Decision

The previous theorem named the target incorrectly. The measured object is useful,
but narrower than claimed:

```text
Oracle_BM25@K = upper bound for reranking BM25 top-K
```

It is not the full training-free retrieval ceiling.

## Correct goal

Prove and improve within this broader space:

```text
TrainingFreeSpace = Generator × Adapter × Scorer × Fusion × Budget
```

The new objective is to measure generator-family oracles and union oracles, then
move `BestObserved_TF` toward the best measured oracle under a fixed budget.

## Benchmark work needed

### Keep

- `--core-only`: minimal BM25-pool lemma harness.
- `SIMEON_ARGUANA_DIAGNOSTICS=1`: reproduction-only structural diagnostics.

### Add next

1. `--oracle-family` or explicit rows for generator-family oracles.
2. `oracle_bm25f_fields_k{50,100,200,500}`
   - candidate pool generated from field-aware BM25F over title/body/proxy fields.
3. `oracle_bm25_expanded_k{50,100,200,500}`
   - candidate pool from RM3/query-expanded BM25.
4. `oracle_union_bm25_bm25f_k{50,100,200,500}`
   - union candidates, qrel-sorted upper bound.
5. Later: graph/path/metadata oracle families once fixtures expose those fields.

## Reporting format

For each corpus/fold, report:

```text
BestObserved_G
Oracle_G@K
Oracle_union@K
recall_gap = Oracle_union@K - Oracle_BM25@K
discrimination_gap = Oracle_G@K - BestObserved_G
```

## Success criteria

We are closer to proving the theorem when we can say, per corpus:

- which generator family dominates candidate recall;
- which adapter/scorer family dominates discrimination;
- whether the remaining gap is budget-bound or structure-bound.

Only after that can we claim meaningful progress toward `R*_TF`.


## Phase XXXII update

Implemented `--generator-slices` with a first BM25F/TextRank generator and a
BM25∪BM25F union oracle. The union oracle lifts `Oracle@100` on every corpus/fold
(+0.005 to +0.036), so the hypothesis that `G` is a limiting variable is already
confirmed. BM25 remains a useful diagnostic slice, but not the objective.
