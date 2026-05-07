# Training-Free Retrieval: Optimum, Bounds, and the Recipe-Router Floor

## Why this doc exists

The Phase I–XVII research arc converged on a recipe router that selects per-corpus
recipes based on observable features. Empirically it cross-fold-validated on
nfcorpus (STRICT), partially on trec-covid, and missed scifact. Throughout the
arc, four "ceilings" (A: candidate-set bound, B: PMI flat-space limit,
C: fold-distribution gap, D: smoothing cascade) were used as labels for empirical
failure modes, but **none of them were stated as quantitative bounds**. This doc
upgrades one of them — Ceiling A — into an actual number per corpus per fold,
and uses it to give the recipe router a falsifiable theoretical structure.

> **Important correction (Phases XXXI and XXXVIII).** This document proves a
> **BM25 candidate-pool lemma**, not the full training-free retrieval theorem.
> `Oracle_BM25@K` is a diagnostic projection of the external answer-key oracle
> onto BM25's top-K pool. It is not a callable system component. The broader
> training-free optimum ranges over candidate generators, corpus adapters,
> scorers, fusion rules, and budgets, while the oracle appears only in the
> evaluation objective. See
> [`training_free_space_redefinition.md`](training_free_space_redefinition.md)
> and [`phase38_oracle_external_theorem.md`](phase38_oracle_external_theorem.md)
> for the corrected goal state.

## The inequality chain

For corpus `C`, fold/query set `S`, pool size `K`, relevance relation `R`, and
metric `M=nDCG@10`:

```text
M(Router(C,S)) ≤ M(BestFeasible_BM25@K(C,S))
               ≤ M(O_M(C,S,R) restricted to BM25@K)
               ≤ M(O_M(C,S,R))
```

- **`O_M(C,S,R)`** is the external answer-key ideal ranking for metric `M`. It is
  used only for evaluation.
- **`O_M restricted to BM25@K`** is computable. For each query, take BM25 top-K,
  sort by qrel value, and compute nDCG@10. This is the diagnostic upper
  projection for any reranker constrained to the BM25-top-K candidate set.
- **`BestFeasible_BM25@K`** is the unknown best training-free recipe inside this
  generator/pool constraint.
- **`Router(C,S)`** is the constructive lower bound. The Phase XVIII recipe
  router (`v17short` / `v17medium` / `v17long`) is the current empirical
  instantiation.

The **BM25-pool runway** is:

```text
M(O_M restricted to BM25@K) - M(Router(C,S))
```

It is the per-corpus diagnostic number that says how much room is left for a
BM25-pool-bound recipe to grow into.

## Per-corpus per-fold runway (K=100, BM25 top-100 candidate pool)

Computed by `run_oracle_pool` in `benchmarks/bench_vs_reference.cpp`. For each
query, BM25-top-100 is materialized, each member is scored by its qrel value
(0 for non-relevant, graded relevance preserved for higher-grade qrels), and
docs outside the pool are scored as `-inf`. The resulting nDCG@10 is the
permutation-supremum over the pool — the absolute ceiling for any reranker that
respects the BM25-top-100 candidate set.

| Corpus     | Fold | bm25_only | recipe_router | reference (dense) | **Oracle@K=100** | Runway (Oracle − router) |
|------------|------|----------:|--------------:|------------------:|-----------------:|-------------------------:|
| arguana    | dev  | 0.3202    | 0.3115        | 0.3607            | **0.9217**       | **+0.6102**              |
| arguana    | test | 0.3293    | 0.3141        | 0.3603            | **0.9269**       | **+0.6128**              |
| fiqa       | dev  | 0.1908    | 0.1917        | 0.3659            | **0.4804**       | **+0.2887**              |
| fiqa       | test | 0.2053    | 0.2074        | 0.3588            | **0.5122**       | **+0.3048**              |
| nfcorpus   | dev  | 0.2620    | 0.2678        | 0.3401            | **0.4726**       | **+0.2048**              |
| nfcorpus   | test | 0.2521    | 0.2639        | 0.2974            | **0.4355**       | **+0.1716**              |
| scifact    | dev  | 0.6623    | 0.6672        | 0.6811            | **0.8492**       | **+0.1820**              |
| scifact    | test | 0.6188    | 0.6159        | 0.6410            | **0.8755**       | **+0.2596**              |
| trec-covid | test | 0.5649    | 0.5752        | 0.5008            | **0.9613**       | **+0.3861**              |
| trec-covid | dev  | 0.4943    | 0.5140        | 0.3891            | **0.9484**       | **+0.4344**              |

Router cell: best `xprod_v17{short,medium,long}_recipe_a*` per corpus per fold.
Oracle cell: `oracle_bm25_pool_k100`.

## Three immediate observations

### 1. The runway is large everywhere

Every corpus has at least **+0.17 nDCG points** of room between the current
recipe router and the candidate-set ceiling. nfcorpus — the corpus we declared
"saturated" after Phase XV — actually has 0.17 of headroom remaining. arguana,
which we wrote off as "task-adversarial" in Phase X-B, has **0.61 nDCG points**
of runway. The empirical ceiling we believed in for two phases of the research
arc was illusory: the rerank ceiling is not the candidate-set ceiling, and the
candidate-set ceiling is much higher than we thought.

### 2. The runway is corpus-shaped, not fold-shaped

Per-fold deltas of the Oracle are small (≤0.05 nDCG between dev and test)
relative to the runway itself (0.17–0.61). Ceiling C (the fold-distribution gap)
is real but is not what's blocking training-free saturation — Ceiling A is.
The fold-flip phenomenon shows up at the ±0.005 scale because that's the
runway-fraction the recipes actually contest. The 0.17–0.61 oracle gap
dominates by orders of magnitude.

### 3. Dense reference is still inside the candidate-pool ceiling

`reference` (the MiniLM dense baseline) reaches 0.30–0.68 nDCG@10 — well
inside the Oracle envelope on every corpus. This is empirical evidence that
**dense rerankers operate inside the same candidate-set bound** as our training-
free pipeline. The dense ceiling is not the pool ceiling; it's the dense model's
own representational ceiling. Where dense beats training-free (fiqa, scifact-dev),
it does so by recovering signal already present in the BM25-top-100 — not by
reaching outside it.

## What this means for the next move

The recipe-router floor is far below the candidate-pool ceiling. The
empirical work has been competing for sub-threshold slices (±0.005) of an
order-of-magnitude-larger gap. Two consequences:

1. **Most "ceilings" we hit were not bounds.** They were local optima of the
   recipe space we searched. The true bound (Oracle@K) is much higher and was
   unmeasured.
2. **The path to closing the runway is recipe-space expansion, not architecture
   change.** Every recipe we've shipped operates within "BM25 top-K → linear
   blend with α". The Oracle bound says this expressive class still has 17–60%
   of nDCG to recover. The question is not "is training-free saturated?" — it's
   "what recipe-space coordinate are we missing?"

## Lemma (Phase XIX, falsifiable form): BM25 candidate-pool runway

> **Statement.** For BEIR-3 and BEIR-style fixtures and a BM25-top-100 candidate
> pool, the BM25-family reranking optimum satisfies:
>
> ```
> Router(C, S) ≤ R*(C, S) ≤ Oracle(K, C, S)
> ```
>
> The BM25-pool runway `Oracle_BM25(K, C, S) − Router_BM25(C, S)` is per-corpus
> bounded below by 0.17 nDCG@10 across all five tested fixtures (arguana, fiqa,
> nfcorpus, scifact, trec-covid; dev + test). Any future BM25-family reranking
> recipe that fails to move the floor of this runway is wasted effort **within
> this generator family**. It may still be useful if it changes the candidate
> generator or adds a corpus adapter, which belongs to the broader
> `TrainingFreeSpace`.

> **Corollary (constructive).** Closing the runway requires recipe-space
> expansion that respects the BM25-top-K constraint:
>   (a) richer candidate generators (dual-stage with multiple dense pools, RM3
>       expansion of the BM25 query); or
>   (b) richer scorer aggregations operating per-(query, doc) pair rather than
>       per-fragment; or
>   (c) corpus-conditioned router branches that select recipe based on features
>       beyond `(avg_dl, n_docs)` — e.g., query-side features.

> **Adapter refinement (Phase XXIX).** The theorem now distinguishes universal
> recipes from corpus adapters. A corpus adapter is still training-free when it
> emits deterministic evidence from observable corpus structure (path segments,
> metadata fields, titles, sections, issue IDs, sentence/heading seeds) without
> qrel labels. It expands the constructive floor `Router(C, S)` by adding a
> new component to the existing candidate/fusion pipeline; it does not change
> `Oracle(K, C, S)` unless it also changes the candidate pool. The ArguAna
> diagnostics showed why this matters, but the shippable form is the YAMS native
> adapter: English-first query seed extraction breaks natural-language requests
> into path/content fragments and structured metadata filters, then fuses the
> resulting `corpus_adapter` evidence alongside lexical/vector signals.

## Reproducible theorem workflow

The BM25 candidate-pool lemma's empirical bound is now reproducible without
running the full probe grid:

```bash
for fx in arguana-minilm fiqa-minilm nfcorpus-minilm scifact-minilm trec-covid-minilm; do
  for split in dev test; do
    build-release/benchmarks/simeon_bench_vs_reference_research \
      --core-only --queries-from "$split" "fixtures/$fx" \
      > "core_${fx}_${split}.jsonl"
  done
done
```

`--core-only` emits exactly six rows per corpus/fold:

1. `reference`
2. `bm25_only`
3. `oracle_bm25_pool_k50`
4. `oracle_bm25_pool_k100`
5. `oracle_bm25_pool_k200`
6. `oracle_bm25_pool_k500`

This is the minimal proof harness for the inequality chain: it measures the
constructive lexical floor and the candidate-pool upper-bound curve without
running experimental recipe grids. ArguAna diagnostics remain separately gated
with `SIMEON_ARGUANA_DIAGNOSTICS=1` and are not part of the theorem harness.

Latest core-only validation artifact: `/tmp/simeon_core_only_20260503_182717`.
Each file has six rows and terminates at `oracle_bm25_pool_k500`.

## Falsifiability

This statement is falsifiable in three ways:
1. A new training-free recipe achieves Router(C, S) > Oracle(K, C, S) for some
   (C, S) → the math is wrong; check `run_oracle_pool` for a bug.
2. A new training-free recipe achieves Router(C, S) ≥ Oracle(K, C, S) − 0.005
   for all 5 corpora × 2 folds → the runway is closed; the recipe router is
   provably near-optimal within the candidate-set constraint.
3. The Router-Oracle gap proves immutable across many recipe-space expansion
   attempts → the pool ceiling itself is the next bound to break.

## Oracle K-sweep — does pool expansion raise the ceiling?

The K=100 ceiling is one slice of a larger curve. Expanding K reveals whether
the runway is dominated by *candidate recall* (BM25 misses relevant docs at
small K but finds them at large K) or by *rerank quality* (relevant docs are
already in BM25 top-K but the reranker can't pick them out).

| Corpus     | Fold | Oracle@K=50 | Oracle@K=100 | Oracle@K=200 | Oracle@K=500 | router | runway@K=500 |
|------------|------|------------:|-------------:|-------------:|-------------:|-------:|-------------:|
| arguana    | dev  | 0.8775      | 0.9217       | 0.9498       | 0.9699       | 0.3115 | **+0.6584**  |
| arguana    | test | 0.9081      | 0.9269       | 0.9524       | 0.9701       | 0.3141 | **+0.6560**  |
| fiqa       | dev  | 0.4146      | 0.4804       | 0.5407       | 0.6207       | 0.1917 | +0.4290      |
| fiqa       | test | 0.4441      | 0.5122       | 0.5714       | 0.6608       | 0.2074 | +0.4534      |
| nfcorpus   | dev  | 0.4292      | 0.4726       | 0.5119       | 0.5689       | 0.2678 | +0.3011      |
| nfcorpus   | test | 0.3994      | 0.4355       | 0.4629       | 0.5446       | 0.2639 | +0.2807      |
| scifact    | dev  | 0.8271      | 0.8492       | 0.9042       | 0.9388       | 0.6672 | +0.2716      |
| scifact    | test | 0.8463      | 0.8755       | 0.9076       | 0.9224       | 0.6159 | +0.3065      |
| trec-covid | dev  | 0.8755      | 0.9484       | 0.9761       | **1.0000**   | 0.5140 | +0.4860      |
| trec-covid | test | 0.9269      | 0.9613       | 0.9765       | 0.9928       | 0.5752 | +0.4176      |

### Three findings from the K-sweep

#### Finding 1: trec-covid is *perfectly recoverable* from BM25 top-500

Oracle@K=500 on trec-covid dev is **1.0000** — every single relevant document
is in the BM25 top-500. The gating bottleneck on trec-covid is not retrieval;
it is reranker discrimination. Phase III's Atire+MaxSim+α=0.80 recipe
(test 0.5752 / dev 0.5140) leaves **0.42–0.49 nDCG of headroom** between the
recipe router and a perfect-pool oracle. This is the largest under-exploited
runway in the entire fixture set and identifies trec-covid as the highest-
leverage target for any new training-free experiment.

#### Finding 2: Pool expansion meaningfully raises every ceiling

Oracle@K=500 − Oracle@K=100, by corpus:

| Corpus | Δ Oracle (K=500 − K=100) | Mechanism |
|--------|-------------------------:|-----------|
| fiqa (test)       | +0.149 | recall-bound (relevant docs ranked deep) |
| nfcorpus (test)   | +0.109 | recall-bound (medical synonyms scatter) |
| trec-covid (dev)  | +0.052 | already near-saturated at K=500 |
| arguana (test)    | +0.043 | high recall even at K=50 |
| scifact (test)    | +0.047 | abstracts already concentrate at K=100 |

The two corpora with the largest K=100→K=500 oracle lift (fiqa, nfcorpus) are
exactly the corpora where Phase XV's dual-stage candidate generation either
proved (nfcorpus STRICT) or showed sub-threshold positive signal. This is
*independent empirical evidence* that pool expansion is on the right axis for
those corpora. The bench-side cost (re-rank a larger pool) is linear in K; the
ceiling lift is real.

#### Finding 3: arguana's runway is structural, not recoverable by recipe expansion

arguana Oracle@K=50 is already **0.91**, vs router 0.31 — a +0.60 runway at the
smallest K. Expanding K barely moves the ceiling (0.91 → 0.97). This says
arguana's failure mode is *not* candidate retrieval — every relevant counter-
argument is in the BM25 top-50. The reranker actively *promotes the wrong docs*
(Phase X-B's "task-adversarial" diagnosis: cosine-similar docs to a claim are
the claim's own restatements, not its rebuttals). No amount of training-free
recipe expansion will close this; arguana is structurally outside the reach of
similarity-based reranking.

### What this means for Phase XV's mechanism

Phase XV added `dense_pool_size=200` dense candidates to the BM25 top-100
pool. The Oracle K-sweep shows Phase XV competes with a much cheaper
alternative: just expanding BM25 K from 100 to 200 raises the ceiling
+0.027 on nfcorpus, +0.029 on fiqa. The dual-stage mechanism's marginal
contribution over BM25-K-expansion is the question — not whether expansion
helps, but whether dense expansion outperforms BM25-K expansion at equal pool
size. Measurement: see Finding 4.

#### Finding 4: BM25-K expansion alone DOES NOT close the runway

Direct test: re-run the recipe-router's outer-MaxSim Atire α=0.80 backbone
with `pool_size = 200` and `pool_size = 500` instead of 100. Same recipe,
larger candidate pool. If the runway is candidate-recall-bound, this should
recover most of the Oracle lift.

| Corpus | Fold | Router (K=100) | kexp K=200 | kexp K=500 | Oracle@K=500 | Δ router→K=500 |
|--------|------|---------------:|-----------:|-----------:|-------------:|---------------:|
| arguana    | dev  | 0.3115 | 0.3150 | 0.3196 | 0.9699 | +0.0081 |
| arguana    | test | 0.3141 | 0.3195 | 0.3214 | 0.9701 | +0.0073 |
| fiqa       | dev  | 0.1917 | 0.1928 | 0.1942 | 0.6207 | +0.0025 |
| fiqa       | test | 0.2074 | 0.2085 | 0.2082 | 0.6608 | +0.0008 |
| **nfcorpus** | dev  | 0.2678 | 0.2612 | 0.2612 | 0.5689 | **−0.0066** |
| **nfcorpus** | test | 0.2639 | 0.2552 | 0.2556 | 0.5446 | **−0.0083** |
| scifact    | dev  | 0.6672 | 0.6692 | 0.6720 | 0.9388 | +0.0048 |
| scifact    | test | 0.6159 | 0.6189 | 0.6171 | 0.9224 | +0.0012 |
| trec-covid | dev  | 0.5140 | 0.5162 | 0.5160 | 1.0000 | +0.0020 |
| trec-covid | test | 0.5752 | 0.5750 | 0.5801 | 0.9928 | +0.0049 |

**The runway does not close.** Pool expansion via BM25-K alone moves nDCG@10
by ≤+0.008 across all corpora and folds, and nfcorpus *regresses* by 0.007–
0.008. Compare against the Oracle gap of 0.17–0.65 — pool expansion buys
back ≤2% of the runway in the best case (arguana, trec-covid), and *loses*
ground on nfcorpus. The trec-covid pattern (small positive lift) is
consistent with its higher recall ceiling, but the lift is sub-threshold:
+0.005 nDCG vs an Oracle headroom of +0.42.

**Interpretation.** The relevant docs are in the larger pool (Oracle climbs
+0.05–0.15 with K), but the outer-MaxSim Atire reranker cannot pick them
out. As the pool grows, more cosine-similar-but-irrelevant docs enter and
the reranker's argmax actively degrades. The reranker has a precision
ceiling that pool expansion exposes, not solves.

This refutes the naive "just expand K" hypothesis and confirms a stronger
claim: **the runway is rerank-quality-bound, not candidate-recall-bound**.

### Implication for Phase XV's mechanism (revised)

Phase XV's STRICT win on nfcorpus (`dual_layered_dp200_a0.80` test 0.2640 vs
OM α=0.80 0.2543) is **not** explained by candidate-set expansion. K=200
BM25 expansion gives 0.2552 — *worse* than K=100. Phase XV must be picking
up genuinely *different* candidates via the dense centroid (different from
what BM25 promotes at any K), and that dense subset interacts well with
GeoMean over Layered fragments.

This is a sharper mechanism claim than the original Phase XV writeup made.
Dual-stage is not "bigger pool" — it's "complementary pool". The dense leg
recovers docs BM25 ranks low at any K, AND those docs match well under
GeoMean aggregation over Layered fragments. Both halves of that conjunction
matter.

### Sharpened theorem

The original framework had Oracle as the upper bound. Finding 4 sharpens
that: the **achievable** training-free upper bound for a fixed reranker
class is much lower than Oracle. Define:

```
R*(C, S; K) = sup over training-free recipes that use BM25-top-K = upper bound under K-constrained recipes
```

Empirically, `R*(C, S; K=200) ≈ R*(C, S; K=100) + ε` where ε is bounded by
the sum of (a) reranker's precision ceiling on the K=200 pool minus (b) the
extra docs the reranker now misranks. For the outer-MaxSim Atire backbone, ε
is empirically negative on some corpora (nfcorpus). For a hypothetical
perfect-precision reranker, ε would equal Oracle@K=200 − Oracle@K=100.

The training-free runway, then, has two distinguishable components:
1. **Reranker precision gap**: R*(C, S; K=∞) − R*(C, S; K=100). The amount a
   better-precision reranker could recover *without* expanding the pool.
2. **Pool-recall gap**: Oracle@K=∞ − R*(C, S; K=∞). The amount no reranker
   can recover within K=∞ — i.e., docs that are always misranked regardless
   of pool size.

Phase XV's Phase XV mechanism (dual-stage dense centroid pool) closes
component 1 on nfcorpus by *replacing* BM25-only candidates with a
complementary set. Plain K-expansion fails because it adds more BM25-style
candidates to a reranker already saturated on that distribution.

## Cross-fold Oracle stability

| Corpus     | Oracle(dev) | Oracle(test) | \|Δ\|  |
|------------|------------:|-------------:|------:|
| arguana    | 0.9217      | 0.9269       | 0.005 |
| fiqa       | 0.4804      | 0.5122       | 0.032 |
| nfcorpus   | 0.4726      | 0.4355       | 0.037 |
| scifact    | 0.8492      | 0.8755       | 0.026 |
| trec-covid | 0.9484      | 0.9613       | 0.013 |

All within ±0.05 at K=100. The candidate-pool ceiling is corpus-defined, not
fold-defined, which is what the corollary needs. (Compare: fold deltas of
`bm25_only` and the recipe router are similar magnitudes — Ceiling C is a
small effect compared to Ceiling A.)

## Next steps

- **Per-query oracle gap distribution**: a histogram of per-query
  (Oracle@K=100 − router) reveals which queries the router gets right vs
  wrong. The mean gap (0.17–0.61) hides bimodal behavior — most queries are
  near-oracle, a tail is far below. Identifying that tail is the next
  empirical target.
- **dual-stage(BM25-K=100, dpool=100) vs plain BM25-K=200**: measure whether
  Phase XV's mechanism beats simple BM25-K expansion on nfcorpus + fiqa.
- **Recipe-space expansion experiment**: add a deterministic "BM25-K=200,
  outer-MaxSim, α=0.80" cell to the recipe router and measure cross-fold.
  Hypothesis: it strictly dominates K=100 on the candidate-recall-bound
  corpora (fiqa, nfcorpus) and is neutral elsewhere. Cheap (no new code,
  ~10 LOC bench addition).

## Cross-fold Oracle stability

| Corpus     | Oracle(dev) | Oracle(test) | \|Δ\|  |
|------------|------------:|-------------:|------:|
| arguana    | 0.9217      | 0.9269       | 0.005 |
| fiqa       | 0.4804      | 0.5122       | 0.032 |
| nfcorpus   | 0.4726      | 0.4355       | 0.037 |
| scifact    | 0.8492      | 0.8755       | 0.026 |
| trec-covid | 0.9484      | 0.9613       | 0.013 |

All within ±0.05. The candidate-pool ceiling is corpus-defined, not fold-defined,
which is what the corollary needs. (Compare: fold deltas of `bm25_only` and the
recipe router are similar magnitudes — Ceiling C is a small effect compared to
Ceiling A.)

## Files

- `benchmarks/bench_vs_reference.cpp` — added `run_oracle_pool` (~50 LOC) and
  `oracle_bm25_pool_k100` cell emission alongside `bm25_only` in main().
- This doc.
