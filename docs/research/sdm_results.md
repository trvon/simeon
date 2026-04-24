# SDM (Sequential Dependence Model) — negative result

Training-free adjacent-term co-occurrence scoring via Metzler & Croft 2005.
Implemented as a composable rescoring path on `Bm25Index` (`score_sdm()`),
gated behind `Bm25Config::build_word_bigrams = false` by default.

## Math

    score(q, d) = λ_u · Σ_{t ∈ q}       BM25(t, d)
                + λ_o · Σ_{(a,b) adj q} BM25(a_b_ordered, d)
                + λ_uw · Σ_{(a,b) adj q} BM25(a_b_unordered_w, d)

Metzler's fixed defaults: `(λ_u, λ_o, λ_uw) = (0.85, 0.10, 0.05)`, unordered
window `w = 8`. Unigram leg dispatches through the chosen `Bm25Variant`
(Atire, SAB-smooth, …); both bigram legs always score with Atire form and
word-length `dl`.

## Three-corpus measurement

| Corpus   | Unigram base | SDM (Metzler) | Δ       | SDM (0.90/0.05/0.05) | Δ       |
|----------|-------------:|--------------:|--------:|---------------------:|--------:|
| scifact  | Atire 0.619  | 0.612         | −0.007  | 0.619                | +0.000  |
| FiQA     | Atire 0.205  | 0.212         | **+0.006** | 0.208             | +0.003  |
| NFCorpus | Atire 0.252  | 0.253         | +0.001  | 0.254                | +0.002  |
| scifact  | SAB-sm 0.612 | 0.612         | +0.000  | —                    | —       |
| FiQA     | SAB-sm 0.198 | 0.206         | +0.008  | —                    | —       |
| NFCorpus | SAB-sm 0.298 | 0.298         | +0.000  | —                    | —       |

Plan target for promotion: FiQA Atire-SDM lift ≥ +0.010. Measured +0.006, so
the row stays below threshold. The ordered-bigram leg is real, but too small to
justify promotion.

## Mechanism — why FiQA undershoots

SDM rewards adjacency of fixed multi-word terms. FiQA's failure mode is more
often paraphrase and reordering, so SDM only helps on the minority of queries
that contain genuinely fixed phrases.

Scifact and NFCorpus undershoot for the opposite reason: SAB-smooth already
captures much of the useful signal that those bigrams carry, so SDM mostly
double-counts it.

## Infrastructure disposition

- `score_sdm()` and the bigram postings build stay in the codebase.
- `Bm25Config::build_word_bigrams` defaults `false`; existing callers are
  byte-identical to pre-Step-1l behavior.
- Not wired into `run_router_cascade()` or the default router recipe.
- Bench rows (`bm25_atire_sdm_l0.85_0.10_0.05`,
  `bm25_sab_smooth_sdm_l0.85_0.10_0.05`, `bm25_atire_sdm_l0.90_0.05_0.05`)
  stay in the three `*_full.jsonl` result files for regression tracking.

## Next lever

Weighted Sequential Dependence (Bendersky & Croft 2010) is the natural follow-up
if a corpus ever shows a stronger fixed-SDM lift.

## References

- Metzler, D. & Croft, W.B. (2005). "A Markov Random Field Model for Term
  Dependencies." SIGIR 2005.
- Bendersky, M. & Croft, W.B. (2010). "Discovering Key Concepts in Verbose
  Queries." SIGIR 2008 / *WSDM follow-ons.* (Deferred follow-on reference.)
