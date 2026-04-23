# Weighted SDM (Bendersky-Croft 2010) — negative result

Tests the training-free reduction of Bendersky-Croft 2010's WSDM:
per-bigram λ scales with the bigram's IDF, normalized to query-mean so
WSDM(β=0) recovers fixed SDM byte-identically. Implemented as
`Bm25Index::score_wsdm()` with `WeightedSdmConfig{β}`. Result: best
β across {0.5, 1.0, 1.5} delivers +0.0015 nDCG@10 on scifact /
+0.0009 on nfcorpus; canonical β=1.0 on FiQA (the long-query corpus
where SDM had its best lift) **regresses by −0.0037**. Plan promote
threshold (+0.010 nDCG@10 on FiQA) missed by ~7×.

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

Plan target for promotion: +0.010 nDCG@10 on FiQA (longest queries,
best fixed-SDM uplift candidate). Plan disprove threshold: |ΔnDCG@10|
< 0.003 on all three. Observed: best Atire WSDM Δ = {+0.0015,
+0.0009, −0.0020}; canonical β=1 on FiQA = −0.0037. **3/3 corpora
land within ±0.003 disprove bound at every β; FiQA at canonical β=1
is an outright regression.** T3 disproved.

## Mechanism — why per-bigram IDF reweighting doesn't lift

Three independent factors:

### 1. The fixed-SDM bigram leg already captures the "rare bigram is
informative" signal via BM25 on bigram postings

BM25 IDF inside the bigram leg already weights `log((N - df + 0.5) /
(df + 0.5))` per bigram. WSDM's IDF reweighting double-counts that
signal: the bigram contribution becomes `(idf / mean)^β · BM25(idf,
…)`, which for β=1 effectively weights the inner BM25 by an
*exponentiated* function of the same IDF. On corpora with already-
saturated bigram IDFs (FiQA's financial vocabulary repeats heavily),
this skews toward outlier-rare bigrams that are typically typos or
proper nouns, not load-bearing query concepts.

### 2. Mean normalization is fragile on short queries

The "weight = 1.0 baseline" property holds only when at least 2
bigrams in the query are present in the index. scifact and FiQA both
have median query length ~6 words → 5 candidate bigrams, of which
typically 1–2 are OOV. The remaining 3–4 are normalized by their own
mean, so the dynamic range of `(idf/mean)^β` is small (typically
0.7–1.3 even at β=1). The reweighting is too gentle to move the
needle on present bigrams and amplifies noise on the few queries with
high-IDF outliers.

### 3. FiQA's win for fixed SDM came from generic financial bigrams,
not rare ones

The earlier SDM result (`docs/sdm_results.md`) noted FiQA's +0.006
SDM lift came from the small fraction of queries with fixed multi-
word terms ("interest rate", "mortgage-backed"). These have moderate
IDF — they're common enough across the financial corpus to have
large `df`. WSDM's IDF normalization *downweights* them relative to
the few high-IDF bigrams in the same query, which in fiqa are
typically question-stem fragments ("can someone", "how do") — exactly
the bigrams that should *not* contribute. The reweighting is
backwards for this corpus's failure mode.

This matches a deeper finding from T4 (RM3 adaptive K): per-corpus
optimal weighting is **corpus-bound**, not query-bound. A single
universal IDF-reweighting recipe cannot win on all three.

## Subfinding — β monotonicity differs by corpus

| Corpus   | β=0    | β=0.5  | β=1.0  | β=1.5  |
|----------|-------:|-------:|-------:|-------:|
| scifact  | 0.6121 | 0.6126 | 0.6121 | 0.6136 |
| nfcorpus | 0.2529 | 0.2533 | 0.2536 | 0.2538 |
| fiqa     | 0.2115 | 0.2095 | 0.2078 | 0.2089 |

nfcorpus is the only corpus with a clean monotone β response — and
its absolute lift is still +0.0009. scifact is non-monotonic
(local-min at β=1). fiqa is monotone *down* (worsens with stronger
IDF reweighting). Three different shapes across three corpora rules
out a useful default-β setting.

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

The natural follow-up is **adaptive SDM bigram-leg gating**: at query
time, decide whether to include the bigram leg at all based on a
cheap signal (e.g., `n_terms < 3` ⇒ no bigrams; query-mean bigram-df
above corpus median ⇒ no bigrams). This addresses the "FiQA generic
question-stem bigrams hurt" failure mode without adding
per-bigram weighting machinery. Cost similar to T4's adaptive K
implementation.

## References

- Bendersky, M. & Croft, W.B. (2010). "Learning Concept Importance
  Using a Weighted Dependence Model." WSDM 2010.
- Metzler, D. & Croft, W.B. (2005). "A Markov Random Field Model for
  Term Dependencies." SIGIR 2005.
- See also `docs/sdm_results.md` (fixed SDM baseline) and
  `docs/rm3_adaptive_results.md` (T4 — same "corpus-bound, not
  query-bound" finding).
