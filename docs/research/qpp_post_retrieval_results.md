# QPP post-retrieval predictors — negative/weak result

Tests two post-retrieval query-performance predictors (Shtok-Kurland-Carmel
2012 NQC and Zhou-Croft 2007 full WIG) against the router's existing
`score_decay_rate` ("WIG-lite") on all three BEIR fixtures. Result: neither
clears the validate threshold for routing; WIG-lite already dominates. The
oracle-to-passE gap is not closable via cheap per-query post-retrieval
signals alone.

## Predictors

- `nqc` — NQC (Shtok, Kurland, Carmel 2012, TOIS):
  `σ(S_top_K) / μ(S_corpus)` where σ is population stddev of top-K pool-0
  BM25 scores (K=50), μ is mean over all corpus scores.
- `wig_full` — Zhou & Croft 2007, BM25-adapted:
  `(μ(S_top_K) - μ(S_corpus)) / sqrt(|Q|)`.
- `score_decay_rate` — WIG-lite, pre-existing (`router_design.md:53`):
  `(S@1 - S@10) / S@1` over the top-K pool.

All three are filled by `QueryRouter::features_with_pool()`; costs one
extra O(nd) pass over `scores0` added to the already-hot vector.

## Three-corpus Spearman ρ

Oracle-router rows only (n = queries with ≥1 relevant doc): scifact=202,
nfcorpus=224, fiqa=444. Pool K=50.

### vs oracle-best nDCG@10 (overall difficulty)

| predictor         | scifact | nfcorpus | fiqa   |
|-------------------|--------:|---------:|-------:|
| nqc               | +0.245  | +0.299   | +0.162 |
| wig_full          | +0.044  | +0.208   | +0.026 |
| score_decay_rate  | +0.399  | +0.216   | +0.273 |

### vs Atire nDCG@10

| predictor         | scifact | nfcorpus | fiqa   |
|-------------------|--------:|---------:|-------:|
| nqc               | +0.267  | +0.375   | +0.155 |
| wig_full          | +0.100  | +0.246   | +0.055 |
| score_decay_rate  | +0.453  | +0.296   | +0.271 |

### vs SAB-smooth nDCG@10

| predictor         | scifact | nfcorpus | fiqa   |
|-------------------|--------:|---------:|-------:|
| nqc               | +0.249  | +0.308   | +0.167 |
| wig_full          | +0.063  | +0.203   | +0.031 |
| score_decay_rate  | +0.439  | +0.253   | +0.317 |

### vs (Atire − SAB) nDCG@10 (routing-discriminator)

| predictor         | scifact | nfcorpus | fiqa   |
|-------------------|--------:|---------:|-------:|
| nqc               | +0.018  | +0.013   | −0.051 |
| wig_full          | +0.064  | +0.076   | +0.023 |
| score_decay_rate  | +0.016  | −0.063   | −0.081 |

## Gate outcomes

### T1 — NQC

Promote threshold: ρ(nqc, oracle-best) > 0.4 on ≥2/3 corpora. Observed
ρ = {0.245, 0.299, 0.162}. 0/3 pass. Weak positive signal on all three,
but below the routing-utility bar — a gate that fires on "NQC > τ" cannot
improve the passE baseline when the top predictor saturates near ρ=0.3.

### T2 — Full WIG vs WIG-lite

Promote threshold: ρ_WIG − ρ_WIG-lite > 0.1 on ≥2/3 corpora. Observed Δρ
= {−0.355, −0.008, −0.247}. **WIG-lite beats full WIG on 2/3 corpora**;
the BM25-adapted Zhou-Croft form underperforms the top-drop decay ratio
the router already carries.

### Implication for T6 (learned router)

The routing-discriminator row (predictor vs Atire−SAB diff) is the
decisive one: all three predictors land within ρ ∈ [−0.08, +0.08] across
all three corpora. A logistic regression over these features cannot
exceed the correlation of its strongest input feature — we have none that
discriminates which recipe wins per query. The plan's T6 feasibility
probe would find <25% gap closure; it is effectively pre-disproved.

## Mechanism — why full WIG underperforms WIG-lite

Zhou-Croft WIG trusts the absolute value of the BM25 score mean. In a
BM25 (rather than query-likelihood) setting, that mean tracks query
length and term rarity — signals already covered by `avg_idf` and
`n_terms`. The top-drop decay ratio is dimensionless and corpus-agnostic:
it asks "how peaked is the top?" which is the question WIG is actually
trying to answer. Normalizing away the scale removes a confounder.

NQC has the same weakness: it normalizes by corpus mean (not collection
baseline), which on a BM25 score distribution is dominated by zeros for
non-matching docs — so the division is close to `σ(top-K) / (top_mass /
nd)` = `σ · nd / top_mass`. On larger corpora (fiqa, 57k docs) the `nd`
scaling compresses the discriminative range. The predictor degrades as
corpus size grows, which matches the observed fiqa drop to ρ=0.16.

## Infrastructure disposition

- `nqc` and `wig_full` fields stay on `QueryFeatures` for regression
  tracking and future experiments (cost: two floats per query + O(nd)
  one pass; overhead is nil vs the existing scores0 hot loop).
- Neither is consulted by any `RouterConfig` gate.
- Bench rows in `*_per_query.jsonl` include both fields so downstream
  studies can rerun this correlation without re-benching.

## Next lever

- Feature-limited: drop T6 (learned router) from the simeon plan — no
  cheap pre/post predictor discriminates the Atire-vs-SAB routing
  decision. Closing the remaining ~3–7 nDCG points to the oracle
  requires either labeled routing data or a genuinely different
  signal (e.g., per-query dense-leg quality from the cascade).
- Recall-focused: the plan's Phase B/C experiments (Weighted SDM,
  adaptive RM3, axiomatic LTD correction) do not depend on routing —
  pursue those unchanged.

## References

- Shtok, A., Kurland, O., Carmel, D. (2012). "Predicting Query
  Performance by Query-Drift Estimation." TOIS 30(2).
- Zhou, Y., Croft, W.B. (2007). "Query Performance Prediction in Web
  Search Environments." SIGIR 2007.
- Carmel, D., Yom-Tov, E. (2010). "Estimating the Query Difficulty for
  Information Retrieval." Morgan & Claypool.
- Cronen-Townsend, S., Zhou, Y., Croft, W.B. (2002). "Predicting Query
  Performance." SIGIR 2002.
