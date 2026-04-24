# Weighted SDM (Bendersky-Croft 2010) — negative result

Tests the training-free reduction of Bendersky-Croft 2010's WSDM:
per-bigram λ scales with the bigram's IDF, normalized to query mean so
WSDM(β=0) recovers fixed SDM byte-identically. Best β across the sweep gives
only +0.0015 on scifact and +0.0009 on nfcorpus; canonical β=1.0 regresses on
FiQA.

## Math

Per-bigram weight (BigramIdfNorm strategy):

    w_b = (idf_b / mean_query_bigram_idf)^β
    λ_b = λ_base * w_b

with mean computed over the present (non-OOV) bigrams in the query.
β=0 disables weighting (recovers fixed SDM); β=1 is canonical
Bendersky-Croft IDF reweighting; β>1 amplifies discrimination.

## Three-corpus measurement (nDCG@10)

All rows: same word-bigram index as fixed SDM, λ_u/λ_o/λ_uw =
0.85/0.10/0.05. β=0 row is the byte-identity sanity check against the
existing `bm25_atire_sdm_l0.85_0.10_0.05` row. SAB row scored once at
canonical β=1.0.

| Corpus   | SDM (Atire) | WSDM β=0   | WSDM β=0.5 | WSDM β=1.0 | WSDM β=1.5 | Δ best  |
|----------|------------:|-----------:|-----------:|-----------:|-----------:|--------:|
| scifact  | 0.6121      | 0.6121 ✓   | 0.6126     | 0.6121     | **0.6136** | +0.0015 |
| nfcorpus | 0.2529      | 0.2529 ✓   | 0.2533     | 0.2536     | **0.2538** | +0.0009 |
| fiqa     | **0.2115**  | 0.2115 ✓   | 0.2095     | 0.2078 ⚠   | 0.2089     | −0.0020 |

| Corpus   | SDM (SAB)   | WSDM β=1.0 | Δ       |
|----------|------------:|-----------:|--------:|
| scifact  | 0.6119      | 0.6131     | +0.0012 |
| nfcorpus | 0.2981      | 0.2981     | +0.0000 |
| fiqa     | 0.2055      | 0.2046     | −0.0009 |

R@100 panel — included because the BC-2010 paper frames its win as
recall-driven on long-query corpora (FiQA matches that profile):

| Corpus   | SDM (Atire) R@100 | WSDM β=1.0 | Δ       |
|----------|------------------:|-----------:|--------:|
| scifact  | 0.8835            | 0.8835     | +0.0000 |
| nfcorpus | 0.1991            | 0.1991     | +0.0000 |
| fiqa     | 0.4679            | 0.4687     | +0.0008 |

The promote target was +0.010 nDCG@10 on FiQA. Observed deltas stay inside the
disprove band, and canonical β=1.0 is an outright regression on FiQA. T3 is
disproved.

## Mechanism — why per-bigram IDF reweighting doesn't lift

Three factors explain the null result:

### 1. The fixed-SDM bigram leg already captures the "rare bigram is
informative" signal via BM25 on bigram postings

BM25 already gives the bigram leg an IDF-weighted contribution. WSDM then
reweights by another function of the same IDF, which overemphasizes outlier-rare
bigrams rather than the load-bearing ones.

### 2. Mean normalization is fragile on short queries

Mean normalization is fragile on short queries. Once OOV bigrams are removed,
the remaining `(idf/mean)^β` range is usually too small to help and mainly
amplifies noise on outliers.

### 3. FiQA's win for fixed SDM came from generic financial bigrams,
not rare ones

FiQA's fixed-SDM lift came from moderate-IDF financial phrases, not extreme-IDF
bigrams. WSDM downweights those and overweights the generic question-stem
fragments that should matter less.

This is another corpus-bound, not query-bound, weighting result.

## Subfinding — β monotonicity differs by corpus

| Corpus   | β=0    | β=0.5  | β=1.0  | β=1.5  |
|----------|-------:|-------:|-------:|-------:|
| scifact  | 0.6121 | 0.6126 | 0.6121 | 0.6136 |
| nfcorpus | 0.2529 | 0.2533 | 0.2536 | 0.2538 |
| fiqa     | 0.2115 | 0.2095 | 0.2078 | 0.2089 |

nfcorpus is the only corpus with a monotone β response, and its best lift is
still only +0.0009. The three corpora show three different response shapes, so
there is no useful default β.

## Infrastructure disposition

- `Bm25Index::score_wsdm()` and `WeightedSdmConfig` ship in
  `simeon/bm25.hpp` for downstream callers and future experiments.
- Cost: one extra bigram-IDF prepass per query (O(|q|) hash lookups +
  one `powf` per present bigram); negligible vs the existing scoring
  loop.
- Bench rows (`bm25_atire_wsdm_b{0.0, 0.5, 1.0, 1.5}`,
  `bm25_sab_smooth_wsdm_b1.0`) stay in the three `*_full.jsonl`
  result files for regression tracking.
- Not invoked by `run_router_cascade()` or any default router recipe.
- β=0 sanity row (`bm25_atire_wsdm_b0.0`) byte-matches
  `bm25_atire_sdm_l0.85_0.10_0.05` on all 3 corpora — the dispatch
  back to `score_sdm()` for β=0 is verified by the data.

## Next lever

The full Bendersky-Croft 2010 WSDM uses linear-regressed weights over
~6 features ({tf_collection, df, length-norm, Wikipedia entity
indicator, …}); the IDF-only reduction tested here is §6.4 of the
paper, the "no external resource" baseline. The full model needs a
held-out tuner. Per existing memory rule (no learned-routing
investment after Phase A), this is **not** the next experiment.

The natural follow-up is **adaptive SDM bigram-leg gating** rather than more
per-bigram weighting.

## References

- Bendersky, M. & Croft, W.B. (2010). "Learning Concept Importance
  Using a Weighted Dependence Model." WSDM 2010.
- Metzler, D. & Croft, W.B. (2005). "A Markov Random Field Model for
  Term Dependencies." SIGIR 2005.
- See also `docs/sdm_results.md` (fixed SDM baseline) and
  `docs/rm3_adaptive_results.md` (T4 — same "corpus-bound, not
  query-bound" finding).
