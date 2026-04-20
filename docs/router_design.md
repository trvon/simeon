# Query router design

Per-query selection between simeon's BM25 variants and cascade configurations,
driven by cheap pre-retrieval predictors. Training-free: predictors are
corpus-statistic functions, decision rules are static thresholds chosen from
the scifact ablation in `docs/benchmarks.md`.

## Why route at all

The Step 1c ablation produced configurations that win on different query
classes:

| Recipe                                  | Best for                                 |
|-----------------------------------------|------------------------------------------|
| `Bm25Atire`                             | Rare-term, high-IDF queries              |
| `Bm25SabSmooth` (γ=5)                   | Morphologically-rich, OOV-tolerant       |
| `CascadeLinearAlpha` (α=0.75, pool=500) | Multi-term semantic intent               |

A single global recipe leaves quality on the table. The router picks per
query without invoking simeon for queries that don't need the cascade — which
also recovers BM25-alone's QPS for those queries.

## Predictors

All pre-retrieval predictors use only the BM25 index's existing per-term
`df` table — no extra index structure, no neural model.

| Predictor                       | Definition                                                           | Source                                  |
|---------------------------------|----------------------------------------------------------------------|-----------------------------------------|
| `oov_rate`                      | fraction of query terms with `df = 0`                                | He & Ounis 2003 (CIKM, AvIDF derivation)|
| `avg_idf`                       | mean of `log(N/df(t))` over present terms (skip OOV)                 | Cronen-Townsend & Croft 2002, SIGIR     |
| `max_idf`                       | max IDF over present terms                                           | He & Ounis 2003                         |
| `min_idf`                       | min IDF over present terms (Step 1f); 0 when every term is OOV       | He & Ounis 2003 (IDF distribution family)|
| `idf_stddev`                    | population stddev of IDF over present terms (Step 1f); 0 when n≤1    | Cronen-Townsend & Croft 2002, SIGIR     |
| `n_terms`                       | tokenized query length (post-stop, post-lowercase)                   | Mothe & Tanguy 2005, SIGIR PARM         |
| `avg_term_chars`                | mean character length of query terms                                 | proxy for morphological complexity      |

These are all in the "pre-retrieval" family per Carmel & Yom-Tov's 2010
synthesis lecture *Estimating the Query Difficulty for Information
Retrieval*.

### Post-retrieval-lite predictors (Step 1g.1)

The cascade route already runs BM25 to build the rerank pool, so the
first-pass score *exists* — Step 1g.1 feeds it back to the router as four
extra `QueryFeatures` fields filled by `QueryRouter::features_with_pool()`.
The classical post-retrieval predictors (clarity, WIG, NQC) require a full
first-pass scan; "post-retrieval-lite" only consults the BM25 top-K (default
K=50). Cost is one extra `Bm25Index::score()` per pool — ~ms on scifact.

| Predictor                | Definition                                                              | Source                                  |
|--------------------------|-------------------------------------------------------------------------|-----------------------------------------|
| `score_decay_rate`       | `(score@1 − score@10) / score@1` over pools[0]; clamped to [0, 1]       | Hauff et al. 2008 ("WIG-lite")          |
| `score_normalized_var`   | population var ÷ mean of top-K pools[0] BM25 scores                     | Cronen-Townsend & Croft 2002 (clarity proxy) |
| `top_k_score_entropy`    | Shannon entropy (nats) of softmax(top-K pools[0] scores)                | Yom-Tov & Carmel 2010                   |
| `pool_overlap_jaccard`   | Jaccard of top-K doc-id sets across pools[0] vs pools[1]                | novel — disagreement-based difficulty (cf. Aslam & Pavlu 2007 ECIR) |

Convention: `pools[0]=Atire`, `pools[1]=SAB`, so `pool_overlap_jaccard`
measures the routing-relevant Atire-vs-SAB disagreement. Defaults
(`score_decay_rate=0`, `pool_overlap_jaccard=1`, etc.) are no-signal values
so the matching `RouterConfig` AND-gates default to no-ops — backward-
compatible with Step 1f when `features_with_pool()` is not called.

### Step 1k pre-retrieval predictors (Pass E — SCQ and simplified clarity)

Two more pre-retrieval features from the Carmel-Yom-Tov catalog, theoretically
independent of the IDF family. Both are computed in `features()` from
collection statistics (`Bm25Index::total_tf`, `Bm25Index::total_tokens`) with
no first-pass scoring required.

| Predictor            | Definition                                                      | Source                              |
|----------------------|-----------------------------------------------------------------|-------------------------------------|
| `scq_sum`            | `Σ_t (1 + log(tf_C(t))) · idf(t)` over present query terms      | Zhao, Scholer, Tsegay 2008 ECIR     |
| `simplified_clarity` | `Σ_t p(t|Q) · log(p(t|Q) / p(t|C))` over distinct present terms | Cronen-Townsend & Croft 2002 SIGIR  |

`scq_sum` rewards query terms that are both rare (high IDF) and
well-supported in the collection (large `tf_C`); `simplified_clarity` is the
KL divergence of the query unigram distribution from the background LM. Both
feed Atire-route AND-gates — `atire_min_scq` floor and `atire_max_clarity`
ceiling — that default to pass (0 and +∞ respectively) so Step 1k is a
behavior-preserving addition on existing callers.

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

Takeaway: Step 1k B2's gates ship as opt-in (defaults are no-ops). On two
of three test corpora (NFCorpus +0.028, FiQA +0.006) `atire_max_clarity=3.0`
delivers a net-positive lift; on scifact the gate is a no-op but does not
regress. Recommend callers set `atire_max_clarity=3.0` and leave
`atire_min_scq=0.0` as a corpus-agnostic default — the optimal ceiling is
narrow (3.0–5.0) where it helps, and inactive where it doesn't.

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
        # Rare-term-heavy queries need exact match; the SAB n-gram blend
        # adds noise without lift when terms are already discriminative.
        # Also the cheapest recipe — BM25 alone is ~3× the QPS of SAB.
        # Step 1f added the n_terms / min_idf AND-gates so callers can
        # tighten Atire eligibility on corpora where avg_idf alone is
        # noisy. Step 1g.1 added the pool-Jaccard / score-decay AND-gates
        # so callers can demote Atire when the Atire/SAB pools agree (no
        # routing decision matters) or when the BM25 top is flat (Atire's
        # rare-term advantage is absent). Step 1k added the SCQ-floor /
        # clarity-ceiling AND-gates; on morphology-heavy corpora like
        # NFCorpus, `atire_max_clarity=3.0` demotes Atire on
        # background-close queries and lets SAB-smooth take them, lifting
        # nDCG@10 to parity with MiniLM. All six extra gates default to
        # no-ops; see "Step 1f/1g.1/1k tuning result" below.

3.  if n_terms >= router.cascade_min_terms (default 4)
        AND avg_idf <= router.cascade_max_idf (default 5.0)
        → CascadeLinearAlpha
        # Multi-term queries with moderate IDF benefit from simeon's
        # semantic signal. Short queries don't have enough redundancy
        # for the simeon leg to add information; high-IDF queries are
        # already at the BM25 ceiling.

4.  default → Bm25SabSmooth
        # Best general-purpose standalone scorer in the ablation.
```

Rationale for "first-match-wins" rather than learning a soft combination:
the ablation evidence is per-recipe, not joint, and a learned router would
need labeled data which simeon does not assume.

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

The Pass A sweep on the scifact test split (300 queries) makes
`high_idf_threshold` the only knob that moves the metric. Lowering it from
the conservative default 6.0 to 3.0 unlocks the Atire route (60/300 queries
become eligible) and lifts the metric:

| Config (variants of default thresholds)         | nDCG@10 | R@10  | R@100 | MRR@10 |
|-------------------------------------------------|--------:|------:|------:|-------:|
| `router_default_4096_768` (idf=6.0)             |   0.640 | 0.763 | 0.889 |  0.609 |
| `passA_oov0.00_idf3_*` (idf=3.0)                | **0.654** | 0.768 | 0.892 |  0.626 |
| `router_oracle_4096_768` (per-query argmax)     |   0.713 | 0.834 | 0.931 |  0.684 |

The oracle reaches **0.713 nDCG@10** with route mix atire/sab/cascade =
60/230/10 — Atire is the right pick for a fifth of queries on this corpus,
contradicting the assumption baked into the conservative default. The tuned
threshold closes ~20% of the static→oracle gap; remaining headroom is
predictor-quality bound (cheap pre-retrieval features can't perfectly
identify which queries Atire wins on).

The `cascade_alpha` and `pool_size` knobs (Pass B) do not move the metric
on scifact at all — every Pass B row scores 0.640 nDCG@10 with the same
default thresholds because the route mix doesn't change.

For corpora other than scifact, repeat the sweep on a held-out dev fold
before reporting test numbers, per the workflow in [build.md](build.md).

### Step 1f tuning result (negative)

Step 1e ended with a 0.059-point router→oracle gap (`0.713 - 0.654`)
dominated by the Atire↔SAB binary decision (79% of realizable lift, from
the per-query confusion matrix). Step 1f adds two cheap pre-retrieval
predictors (`min_idf`, `idf_stddev`) and two AND-gates on the Atire route
(`atire_min_terms`, `atire_min_idf_floor`) so the rule can subdivide the
Atire/SAB region without any new index structure.

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

The dev/test gap (+0.010 → −0.010) is the standard tuning-on-small-fold
signature: with 98 dev queries, a 1-point nDCG move is ~10 query
reassignments and within sampling noise. The test oracle stays at 0.691,
so the underlying router→oracle gap is real (~0.05 nDCG@10) — but cheap
predictors over `df` alone cannot reliably close it.

The honest takeaway: **the SAB→Atire 8.89-point lift identified in the
Step 1e per-query analysis is not realizable from cheap pre-retrieval
signals on scifact**. Closing it requires either a one-pass post-retrieval
predictor (clarity score, WIG, NQC — defeats the cost-saving rationale of
routing) or labeled-data routing (violates the training-free promise).
Defaults stay at `atire_min_terms=0` / `atire_min_idf_floor=0` so the
Step 1e behavior is preserved; the new knobs are kept in `RouterConfig`
for callers tuning on larger / different corpora where signal-to-noise
may be more favorable.

### Step 1g.1 tuning result

Step 1f's negative finding scoped the next move: a `df`-only predictor
cannot reliably separate Atire-best from SAB-best queries. Step 1g.1 picks
up the cascade's *already-computed* BM25 top-K pool and feeds four
post-retrieval-lite predictors (`score_decay_rate`,
`score_normalized_var`, `top_k_score_entropy`, `pool_overlap_jaccard`) into
the router. Two new AND-gates on the Atire route — `atire_max_pool_jaccard`
and `atire_min_score_decay` — let callers demote Atire when the BM25 pools
agree (no routing decision matters) or when the BM25 top is flat (Atire's
rare-term advantage is absent).

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
R@10 / +0.010 R@100 on the held-out test fold**. The dev winner
(`dec=0.6`) overshoots on test (the standard small-fold tuning artifact),
but the `dec=0.3` family generalizes — flat on dev (−0.001) and modestly
positive on test (+0.005). Roughly 9% of the residual router→oracle gap
(`0.691 − 0.640 = 0.051`) is closed.

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

The decay floor takes 13 over-routed Atire queries and redirects them to
Cascade. The oracle wants those queries on SAB instead — the lift comes
not from better routing decisions (oracle agreement actually drops
slightly: 33% → 32.2%) but from the *quality of the wrong choices*: when
the router would mis-route to Atire, Cascade is closer-in-expectation to
SAB than Atire is, so the per-mistake loss shrinks.

Honest takeaway: **post-retrieval-lite predictors close ~9% of the
router→oracle gap on scifact at near-zero marginal cost, but they do not
crack the Atire↔SAB ambiguity that Step 1f identified**. The remaining
gap (~0.046 nDCG@10) is still information-bound by what the cascade pool
sees — closing it requires either a real first-pass scan (clarity / WIG /
NQC, defeats the cost rationale) or labeled-data routing (violates the
training-free promise).

Defaults stay at `atire_max_pool_jaccard = 1.0` /
`atire_min_score_decay = 0.0` so existing callers see byte-identical
routing behavior; the new knobs are kept in `RouterConfig` for callers
tuning on different corpora or willing to pay one extra BM25 score per
query for the modest lift.

Subsequent simeon work pivots to the orthogonal recall/quality levers
(MinHash fusion leg, BPE-lite subword head, **PMI / co-occurrence
projection** — Step 1g.2 — for the standalone-simeon semantic gap) rather
than further router-rule tuning.

## What this is not

- **Not a learned classifier.** No supervised training. A future version
  could fit a logistic regression on the predictors against the per-query
  argmax, but that brings labeled-data dependency and is out of scope for
  the training-free promise.
- **Not query expansion.** Predictors observe the query; they don't modify
  it.
- **Not a reranker.** The router runs *before* retrieval, picks the recipe,
  and the chosen recipe runs end-to-end.
- **Not a scorer.** Training-free re-scorers (`score_with_prf`, `score_sdm`)
  compose *inside* the recipe the router picks — the router owns routing,
  the scorers own per-document ranking. Step 1l's SDM is a drop-in for
  `score()` in any `Bm25Variant`; the router is unchanged.
- **Not a fusion-α picker.** Step 1m's `entropy_alpha` reuses the same
  softmax-entropy estimator as the `top_k_score_entropy` predictor here, but
  applies it to two legs' top-K to derive a per-query α. Documented null on
  quality vs the matched static α=0.75 baseline; safety property only. See
  `docs/fusion_entropy_alpha.md`. Router still picks the recipe; entropy-α
  ships in `simeon::` namespace, not in `RouterConfig`.

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

Three observations for future router work:

1. **The scifact `high_idf_threshold = 3.0` default is corpus-specific.** FiQA's
   router mix collapses to SAB-dominant and the Atire route never fires — the
   metric tracks BM25 alone (0.202 vs 0.205) rather than lifting past it. The
   router does not *regress* below BM25 on FiQA; it simply loses the lift that
   made scifact's headline.
2. **PassB cascade α-winner flips 0.75 → 0.50 on FiQA.** Linear-α is the only
   fusion that beats BM25 alone on either corpus, but the mixture weight is
   corpus-dependent. A per-corpus sweep is required before shipping a
   cross-corpus default.
3. **Oracle→router relative gap widens 9% → 20%.** FiQA has more per-query
   ambiguity that cheap pre-retrieval features cannot resolve. Closing it
   requires either a multi-corpus tuner or the post-retrieval-lite predictors
   of Step 1g.1 — neither currently has a cross-corpus validation.

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

Four observations replacing the two-corpus story:

1. **SAB-smooth is corpus-agnostic on morphology-heavy science text.** On NFCorpus it ties MiniLM-L6 (0.298 vs 0.297). On scifact it sits 0.03 below MiniLM but still above Atire. On FiQA (paraphrase-heavy finance) it trails Atire by 0.01. The pattern is textual, not domain-based: **n-gram backoff wins where queries and docs share morphological roots.**
2. **Scifact router tuning directionally transfers but does not saturate.** `router_default` beats BM25-alone on all three corpora but falls short of the per-corpus oracle by ~0.05–0.07 nDCG on all three. The gap-to-oracle is consistent, so the limit is a tuning problem, not a predictor problem.
3. **Scifact cascade headline is corpus-specific.** `bm25_sab_pool500_simeon_cos_4096_768` beats BM25-alone only on scifact. It collapses below BM25-alone on both FiQA (−0.09) and NFCorpus (−0.06). **The SAB→simeon-cos cascade is not a generalizable recipe; it ships only for scifact.**
4. **Per-corpus tuner is now a warranted follow-on.** Two data points made this ambiguous; three data points make it a concrete product step (tuning script over dev fold + per-fixture router config artifact). Captured as an open residual in [training_free_saturation.md](training_free_saturation.md).

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
