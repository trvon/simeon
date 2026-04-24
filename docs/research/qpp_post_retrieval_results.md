# QPP post-retrieval predictors — negative/weak result

Tests two post-retrieval query-performance predictors (NQC and full WIG)
against the router's existing `score_decay_rate` ("WIG-lite") on all three
BEIR fixtures. Neither clears the validate threshold; WIG-lite already
dominates.

## Predictors

- `nqc` — NQC (Shtok, Kurland, Carmel 2012, TOIS):
  `σ(S_top_K) / μ(S_corpus)` where σ is population stddev of top-K pool-0
  BM25 scores (K=50), μ is mean over all corpus scores.
- `wig_full` — Zhou & Croft 2007, BM25-adapted:
  `(μ(S_top_K) - μ(S_corpus)) / sqrt(|Q|)`.
- `score_decay_rate` — WIG-lite, pre-existing (`router_design.md:53`):
  `(S@1 - S@10) / S@1` over the top-K pool.

All three are filled by `QueryRouter::features_with_pool()` and cost one extra
O(nd) pass over `scores0`.

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
ρ = {0.245, 0.299, 0.162}. Weak signal, below the routing-utility bar.

### T2 — Full WIG vs WIG-lite

Promote threshold: ρ_WIG − ρ_WIG-lite > 0.1 on ≥2/3 corpora. Observed Δρ
= {−0.355, −0.008, −0.247}. **WIG-lite beats full WIG on 2/3 corpora**.

### Implication for T6 (learned router)

The decisive row is the routing-discriminator one: all three predictors land
within ρ ∈ [−0.08, +0.08]. No cheap feature here can reliably predict which of
Atire or SAB wins per query, so T6 is effectively pre-disproved.

## Mechanism — why full WIG underperforms WIG-lite

Zhou-Croft WIG trusts the absolute value of the BM25 score mean, which in this
setting mostly tracks query length and rarity. `score_decay_rate` instead asks
the cleaner question: how peaked is the top of the ranking?

NQC has a related weakness: corpus-mean normalization on sparse BM25 scores is
dominated by zeros, so its discriminative range compresses as corpus size grows.

## Infrastructure disposition

- `nqc` and `wig_full` fields stay on `QueryFeatures` for regression
  tracking and future experiments (cost: two floats per query + O(nd)
  one pass; overhead is nil vs the existing scores0 hot loop).
- Neither is consulted by any `RouterConfig` gate.
- Bench rows in `*_per_query.jsonl` include both fields so downstream
  studies can rerun this correlation without re-benching.

## Next lever

- Drop T6 (learned router) from the near-term plan: no cheap pre/post predictor
  here discriminates the Atire-vs-SAB decision.
- The remaining Phase B/C experiments are orthogonal to routing and can proceed unchanged.

## References

- Shtok, A., Kurland, O., Carmel, D. (2012). "Predicting Query
  Performance by Query-Drift Estimation." TOIS 30(2).
- Zhou, Y., Croft, W.B. (2007). "Query Performance Prediction in Web
  Search Environments." SIGIR 2007.
- Carmel, D., Yom-Tov, E. (2010). "Estimating the Query Difficulty for
  Information Retrieval." Morgan & Claypool.
- Cronen-Townsend, S., Zhou, Y., Croft, W.B. (2002). "Predicting Query
  Performance." SIGIR 2002.
