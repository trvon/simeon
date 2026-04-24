# Query router design

Per-query selection between simeon's BM25 variants and cascade configurations,
using cheap corpus-statistic predictors and static thresholds chosen from the
scifact ablation in `docs/benchmarks.md`.

## Why route at all

The Step 1c ablation produced configurations that win on different query
classes:

| Recipe                                  | Best for                                 |
|-----------------------------------------|------------------------------------------|
| `Bm25Atire`                             | Rare-term, high-IDF queries              |
| `Bm25SabSmooth` (γ=5)                   | Morphologically-rich, OOV-tolerant       |
| `CascadeLinearAlpha` (α=0.75, pool=500) | Multi-term semantic intent               |

A single global recipe leaves quality on the table. The router picks per query
and avoids invoking the cascade when the lexical route is already enough.

## Predictors

All pre-retrieval predictors use only the BM25 index's existing term statistics.

| Predictor                       | Definition                                                           | Source                                  |
|---------------------------------|----------------------------------------------------------------------|-----------------------------------------|
| `oov_rate`                      | fraction of query terms with `df = 0`                                | He & Ounis 2003 (CIKM, AvIDF derivation)|
| `avg_idf`                       | mean of `log(N/df(t))` over present terms (skip OOV)                 | Cronen-Townsend & Croft 2002, SIGIR     |
| `max_idf`                       | max IDF over present terms                                           | He & Ounis 2003                         |
| `min_idf`                       | min IDF over present terms (Step 1f); 0 when every term is OOV       | He & Ounis 2003 (IDF distribution family)|
| `idf_stddev`                    | population stddev of IDF over present terms (Step 1f); 0 when n≤1    | Cronen-Townsend & Croft 2002, SIGIR     |
| `n_terms`                       | tokenized query length (post-stop, post-lowercase)                   | Mothe & Tanguy 2005, SIGIR PARM         |
| `avg_term_chars`                | mean character length of query terms                                 | proxy for morphological complexity      |

These are all standard pre-retrieval predictors in the Carmel & Yom-Tov sense.

### Post-retrieval-lite predictors (Step 1g.1)

The cascade route already computes a BM25 top-K pool, so Step 1g.1 feeds a few
pool-derived signals back into the router via
`QueryRouter::features_with_pool()`. These are cheaper than full post-retrieval
predictors because they only inspect the BM25 top-K (default `K=50`).

| Predictor                | Definition                                                              | Source                                  |
|--------------------------|-------------------------------------------------------------------------|-----------------------------------------|
| `score_decay_rate`       | `(score@1 − score@10) / score@1` over pools[0]; clamped to [0, 1]       | Hauff et al. 2008 ("WIG-lite")          |
| `score_normalized_var`   | population var ÷ mean of top-K pools[0] BM25 scores                     | Cronen-Townsend & Croft 2002 (clarity proxy) |
| `top_k_score_entropy`    | Shannon entropy (nats) of softmax(top-K pools[0] scores)                | Yom-Tov & Carmel 2010                   |
| `pool_overlap_jaccard`   | Jaccard of top-K doc-id sets across pools[0] vs pools[1]                | novel — disagreement-based difficulty (cf. Aslam & Pavlu 2007 ECIR) |

Convention: `pools[0]=Atire`, `pools[1]=SAB`, so `pool_overlap_jaccard`
measures Atire-vs-SAB disagreement. Defaults are no-signal values, so the
matching AND-gates stay no-ops unless `features_with_pool()` is used.

### Step 1k pre-retrieval predictors (Pass E — SCQ and simplified clarity)

Two more pre-retrieval features from the Carmel-Yom-Tov catalog, both computed
from collection statistics with no first-pass scoring.

| Predictor            | Definition                                                      | Source                              |
|----------------------|-----------------------------------------------------------------|-------------------------------------|
| `scq_sum`            | `Σ_t (1 + log(tf_C(t))) · idf(t)` over present query terms      | Zhao, Scholer, Tsegay 2008 ECIR     |
| `simplified_clarity` | `Σ_t p(t|Q) · log(p(t|Q) / p(t|C))` over distinct present terms | Cronen-Townsend & Croft 2002 SIGIR  |

`scq_sum` rewards terms that are both rare and collection-supported;
`simplified_clarity` is the KL divergence of the query unigram distribution
from the background LM. Both feed Atire-route AND-gates that default to pass.

#### Step 1k tuning result (Pass E, 2026-04-20, three-corpus)

Pass E grid: `atire_min_scq ∈ {0, 5, 10, 20}` × `atire_max_clarity ∈ {∞, 3.0,
5.0, 8.0}` = 16 rows, Step 1f Pass A winners held constant otherwise.

- **scifact** (null result): every SCQ floor is a no-op (all Atire-route
  queries already clear it) and every clarity ceiling either ties
  `router_default` (clar=∞) or regresses (clar≤8.0 cuts off 0.3–2.4 nDCG
  points). Defaults unchanged.
- **NFCorpus** (win): `passE_scq0_clar3.0` hits 0.298 nDCG@10 — ties MiniLM
  (0.297) and `bm25_sab_smooth_gamma5` alone (0.298), **+2.7 points over
  scifact-tuned `router_default`** (0.270). Any `clar ≤ 5.0` ceiling
  produces the same score; the SCQ floor is insensitive on NFCorpus too.
  Mechanism: on NFCorpus the clarity ceiling demotes Atire on queries
  whose unigram distribution is close to the background LM — exactly the
  queries where SAB-smooth's n-gram blend wins. The gate behaves like an
  implicit per-corpus switch that picks SAB-smooth for medical-morphology
  queries without per-corpus tuning.
- **FiQA** (modest lift): `passE_scq0_clar3.0` hits 0.208 nDCG@10 vs
  `router_default_4096_768` 0.202 — **+0.006** on the 444-query test
  fold. `clar ≤ 5.0` produces the same 0.208; `clar ≤ 8.0` is weaker
  (0.204). SCQ floor is a no-op across all tested settings — every FiQA
  high-IDF query clears the floor. The clarity lift is smaller than
  NFCorpus (+0.028) because FiQA's paraphrase-heavy queries aren't as
  neatly bimodal in clarity space, but the gate is still net-positive
  and does not regress any test row.

Takeaway: these gates should stay opt-in. `atire_max_clarity=3.0` helps on
NFCorpus and FiQA, and is neutral on scifact. `atire_min_scq` stays at `0.0`.

## Routing rules

Decision tree, ordered first-match-wins. Thresholds are scifact-derived
defaults; callers override via `RouterConfig`.

```
1.  if oov_rate > router.oov_threshold (default 0)
        → Bm25SabSmooth
        # SAB's char-n-gram backoff is the only recipe that surfaces docs
        # for OOV terms. Even a single OOV term in the query justifies
        # SAB-smooth — the in-corpus terms still get the smooth blend, and
        # the OOV term is recovered by the n-gram side index.

2.  if avg_idf > router.high_idf_threshold (default 3.0)
        AND n_terms >= router.atire_min_terms (default 0)
        AND min_idf >= router.atire_min_idf_floor (default 0)
        AND pool_overlap_jaccard <= router.atire_max_pool_jaccard (default 1.0)
        AND score_decay_rate >= router.atire_min_score_decay (default 0.0)
        AND scq_sum >= router.atire_min_scq (default 0.0)
        AND simplified_clarity <= router.atire_max_clarity (default +inf)
        → Bm25Atire
        # Rare-term-heavy queries usually want exact match, and all extra
        # Atire gates default to no-ops.

3.  if n_terms >= router.cascade_min_terms (default 4)
        AND avg_idf <= router.cascade_max_idf (default 5.0)
        → CascadeLinearAlpha
        # Multi-term, moderate-IDF queries benefit most from the semantic leg.

4.  default → Bm25SabSmooth
        # Best general-purpose standalone scorer in the ablation.
```

"First-match-wins" keeps the router training-free and matches how the ablations
were measured.

## Threshold derivation

The defaults above come from the scifact ablation. They are not corpus-
universal; tune per corpus by:

1. Pick a held-out fold of queries with relevance judgments.
2. Score each query under all recipes.
3. For each predictor, find thresholds that maximize the per-query
   `argmax_recipe nDCG@10` agreement with the routing rule.

The bench provides this loop directly: `simeon_bench_vs_reference
--queries-from {test,dev} --router-per-query <path>` emits a Pass A
threshold sweep (`router_grid_4096_768_passA_*` rows) and an oracle row
(`router_oracle_4096_768`) that scores the per-query argmax over all three
recipes — that's the upper bound any pre-retrieval router can achieve at
`(pool_size=500, alpha=0.75)`.

### scifact tuning result

The Pass A sweep on scifact shows that `high_idf_threshold` is the only knob
that materially moves the metric. Lowering it from `6.0` to `3.0` unlocks the
Atire route and lifts the result:

| Config (variants of default thresholds)         | nDCG@10 | R@10  | R@100 | MRR@10 |
|-------------------------------------------------|--------:|------:|------:|-------:|
| `router_default_4096_768` (idf=6.0)             |   0.640 | 0.763 | 0.889 |  0.609 |
| `passA_oov0.00_idf3_*` (idf=3.0)                | **0.654** | 0.768 | 0.892 |  0.626 |
| `router_oracle_4096_768` (per-query argmax)     |   0.713 | 0.834 | 0.931 |  0.684 |

The oracle reaches **0.713 nDCG@10** with route mix `60 / 230 / 10`. The tuned
threshold closes about 20% of the static→oracle gap.

`cascade_alpha` and `pool_size` did not move the scifact metric in Pass B.

For corpora other than scifact, repeat the sweep on a held-out dev fold
before reporting test numbers, per the workflow in [build.md](../build.md).

### Step 1f tuning result (negative)

Step 1f adds two cheap pre-retrieval predictors (`min_idf`, `idf_stddev`) and
two extra AND-gates on the Atire route so the rule can subdivide the Atire/SAB
region without any new index structure.

Sweep on the dev fold (98 queries, scifact `--dev-fraction 0.33`) shows
modest lift over the Pass A winner:

| Config                                              | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768` (Pass A winner)           |   0.684 | 0.799 | 0.903 |
| `passC_ant14_amif0.0` (best Pass C on dev)          |   0.694 | 0.824 | 0.918 |
| `passC_ant0_amif1.5`                                |   0.688 | 0.824 | 0.918 |
| `router_oracle_4096_768` (dev oracle)               |   0.759 | 0.878 | 0.929 |

Re-evaluating those same dev winners on the held-out test fold (202
queries) erases the lift:

| Config                                              | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768` (Pass A winner)           |   0.640 | 0.753 | 0.886 |
| `passC_ant14_amif0.0` (dev winner)                  |   0.630 | 0.745 | 0.879 |
| `passC_ant0_amif1.5`                                |   0.629 | 0.743 | 0.880 |
| `router_oracle_4096_768` (test oracle)              |   0.691 | 0.812 | 0.932 |

The Step 1f dev winner does not hold on test. Cheap pre-retrieval signals over
`df` alone are not enough to close the Atire↔SAB gap on scifact, so the new
knobs stay available but default to pass-through.

### Step 1g.1 tuning result

Step 1g.1 follows from the Step 1f null result: `df`-only predictors cannot
reliably separate Atire-best from SAB-best queries, so the next probe uses a
few post-retrieval-lite signals from the already-computed BM25 top-K pool.

Pass D sweep on the dev fold (98 queries) shows essentially-flat dev lift,
yet picks up a real test lift:

| Config (dev fold)                                   | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768` (Pass A winner)           |   0.684 | 0.799 | 0.903 |
| `passC_ant14_amif0.0` (Step 1f best on dev)         |   0.694 | 0.824 | 0.918 |
| `passD_jac0.7_dec0.3` (best Pass D)                 |   0.683 | 0.809 | 0.903 |
| `passD_jac0.7_dec0.6`                               |   0.688 | 0.824 | 0.918 |
| `router_oracle_4096_768` (dev oracle)               |   0.759 | 0.878 | 0.929 |

| Config (test fold, 202 queries)                     | nDCG@10 | R@10  | R@100 |
|-----------------------------------------------------|--------:|------:|------:|
| `router_default_4096_768` (Pass A winner)           |   0.640 | 0.753 | 0.886 |
| `passC_ant14_amif0.0` (Step 1f dev winner)          |   0.630 | 0.745 | 0.879 |
| `passD_jac0.7_dec0.3` (best Pass D on test)         | **0.645** | 0.763 | 0.896 |
| `passD_jac0.7_dec0.6` (dev winner)                  |   0.619 | 0.738 | 0.880 |
| `router_oracle_4096_768` (test oracle)              |   0.691 | 0.812 | 0.932 |

Headline: **Pass D `jac0.7_dec0.3` beats Pass A by +0.0046 nDCG@10 / +0.010
R@10 / +0.010 R@100 on test**. That closes roughly 9% of the remaining
router→oracle gap.

The `pool_overlap_jaccard` knob does *not* move the metric on scifact
(every `jac` value at fixed `dec` ties); the lift comes entirely from
`score_decay_rate`. This is consistent with route-mix telemetry: the
Atire-vs-SAB top-50 Jaccard on scifact is high (Atire and SAB postings
overlap heavily on word-only tokens), so Jaccard-based routing has little
signal here. Corpora with stronger morphological variation should see
larger Jaccard separation and may benefit from non-default
`atire_max_pool_jaccard`.

Route-mix shift from the decay floor (test fold, 202 queries):

| Config                                          | Atire | SAB | Cascade |
|-------------------------------------------------|------:|----:|--------:|
| `router_default_4096_768`                       |    88 |  57 |      57 |
| `passD_jac0.7_dec0.3`                           |    75 |  57 |      70 |
| `router_oracle_4096_768`                        |    45 | 152 |       5 |

The decay floor helps mostly by moving some over-routed Atire queries to the
less-damaging cascade path. It improves scifact modestly, but still does not
solve the Atire↔SAB ambiguity. Defaults remain pass-through.

Subsequent simeon work pivots to orthogonal recall/quality levers rather than
more router-rule tuning.

## What this is not

- **Not a learned classifier.** No supervised training.
- **Not query expansion.** Predictors observe the query; they do not modify it.
- **Not a reranker.** The router picks the recipe before retrieval.
- **Not a scorer.** Re-scorers compose inside the chosen recipe.
- **Not a fusion-α picker.** `entropy_alpha` is a separate fusion primitive.

## API surface

```cpp
namespace simeon {

struct RouterConfig {
    float oov_threshold = 0.0f;            // (>) → SAB
    float high_idf_threshold = 3.0f;       // (>) → Atire (scifact-tuned default)
    std::uint32_t atire_min_terms = 0u;    // (>=) Step 1f AND-gate on Atire
    float atire_min_idf_floor = 0.0f;      // (>=) Step 1f AND-gate on Atire
    float atire_max_pool_jaccard = 1.0f;   // (<=) Step 1g.1 AND-gate on Atire
    float atire_min_score_decay = 0.0f;    // (>=) Step 1g.1 AND-gate on Atire
    std::uint32_t cascade_min_terms = 4;   // (>=) → cascade
    float cascade_max_idf = 5.0f;          // (<=) → cascade
};

enum class Recipe : std::uint8_t {
    Bm25Atire,
    Bm25SabSmooth,
    CascadeLinearAlpha,
};

struct QueryFeatures {
    float oov_rate = 0.0f;
    float avg_idf = 0.0f;
    float max_idf = 0.0f;
    float min_idf = 0.0f;             // Step 1f
    float idf_stddev = 0.0f;          // Step 1f
    std::uint32_t n_terms = 0;
    float avg_term_chars = 0.0f;
    // Step 1g.1 post-retrieval-lite predictors. Filled only by
    // features_with_pool(); features() leaves them at no-signal defaults
    // (decay=0, var=0, entropy=0, jaccard=1) so the matching AND-gates
    // in choose() are no-ops by default.
    float score_decay_rate = 0.0f;
    float score_normalized_var = 0.0f;
    float top_k_score_entropy = 0.0f;
    float pool_overlap_jaccard = 1.0f;
};

class QueryRouter {
public:
    QueryRouter(const Bm25Index& idx, RouterConfig cfg = {}) noexcept;
    QueryFeatures features(std::string_view query) const;
    QueryFeatures features_with_pool(
        std::string_view query,
        std::span<const Bm25Index* const> pools,
        std::uint32_t k = 50) const;
    Recipe choose(std::string_view query) const;
    Recipe choose(const QueryFeatures& f) const noexcept;
};

}  // namespace simeon
```

`features_with_pool()` runs one `Bm25Index::score()` per pool to compute the
post-retrieval-lite fields. Convention: `pools[0]` drives decay/var/entropy,
`pools[1]` (if present) drives `pool_overlap_jaccard` against `pools[0]`.
Empty span / null pools / `k == 0` make the call a strict no-op (returns
`features(query)`).

The router holds a const reference to a finalized `Bm25Index` (which already
carries the per-term IDF cache after `finalize()`). It does not own the
index — caller controls lifetime.

## Corpus transfer 2026-04 (Step 1i, FiQA)

Scifact-tuned thresholds transferred to BEIR FiQA (57,638 docs / 444 test
queries, finance, semantic-paraphrase-heavy) without retuning. Results:

| Config                                              | scifact nDCG@10 | FiQA nDCG@10 |
|-----------------------------------------------------|-----------------|--------------|
| `router_default_4096_768` (scifact-tuned)           |      0.654      |    0.202     |
| `bm25_atire` (baseline)                             |      0.633      |    0.205     |
| `bm25_pool500_linear_alpha075`                      |      0.638      |    0.211     |
| `router_oracle_4096_768` (per-query argmax, ceiling) |      0.713      |    0.244     |

Three observations:

- **The scifact `high_idf_threshold = 3.0` default is corpus-specific.** On
  FiQA it effectively collapses back toward BM25.
- **The best cascade α flips on FiQA.** A cross-corpus default needs a
  per-corpus sweep.
- **The oracle gap widens.** Cheap pre-retrieval features resolve less of the
  per-query ambiguity on FiQA.

Full FiQA headline rows are summarized in [benchmarks.md](benchmarks.md).

## Corpus transfer 2026-04 — three-corpus (Step 1j, NFCorpus)

Third fixture added: BEIR NFCorpus (3,633 docs / 224 test queries, medical abstracts). Same `router_default` config, no retuning.

| Config                                              | scifact | FiQA  | NFCorpus |
|-----------------------------------------------------|--------:|------:|---------:|
| MiniLM-L6 reference                                 |   0.641 | 0.359 |    0.297 |
| `bm25_atire` (baseline)                             |   0.619 | 0.205 |    0.252 |
| `bm25_sab_smooth_gamma5` (novel BM25 alone)         |   0.612 | 0.198 |  **0.298** |
| `router_default_4096_768` (scifact-tuned)           |   0.640 | 0.202 |    0.270 |
| `router_oracle_4096_768`                            |   0.691 | 0.244 |    0.327 |

Four observations:

- **SAB-smooth is the portable winner on morphology-heavy corpora.**
- **Scifact-tuned router thresholds do not saturate cross-corpus.**
- **The scifact cascade headline is corpus-specific.**
- **A per-corpus tuner is the natural follow-on.**

## References

- Carmel, D. & Yom-Tov, E. (2010). *Estimating the Query Difficulty for
  Information Retrieval*. Synthesis Lectures on Information Concepts,
  Retrieval, and Services 2(1).
- Cronen-Townsend, S., Zhou, Y., & Croft, W.B. (2002). "Predicting Query
  Performance." SIGIR 2002.
- He, B. & Ounis, I. (2003). "A Study of Parameter Tuning for Term Frequency
  Normalization." CIKM 2003 (introduces AvIDF and related pre-retrieval
  predictors).
- Mothe, J. & Tanguy, L. (2005). "Linguistic features to predict query
  difficulty." SIGIR Workshop on Predicting Query Difficulty.
- Hauff, C., Murdock, V., & Baeza-Yates, R. (2008). "Improved query
  difficulty prediction for the web." CIKM 2008.
- Aslam, J.A. & Pavlu, V. (2007). "Query Hardness Estimation Using
  Jensen-Shannon Divergence Among Multiple Scoring Functions." ECIR 2007
  (motivates the pool-disagreement form of `pool_overlap_jaccard`).
- Metzler, D. & Croft, W.B. (2005). "A Markov Random Field Model for Term
  Dependencies." SIGIR 2005 (Step 1l SDM — composes with routing as the
  unigram-leg variant inside the recipe the router picks).
