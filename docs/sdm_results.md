# SDM (Sequential Dependence Model) — negative result

Training-free adjacent-term co-occurrence scoring via Metzler & Croft 2005.
Implemented as a composable re-scorer on `Bm25Index` (`score_sdm()`) with
parallel ordered/unordered word-bigram postings, gated behind
`Bm25Config::build_word_bigrams = false` (default off; zero runtime cost
for callers that do not opt in).

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

Plan target for promotion: FiQA Atire-SDM lift ≥ +0.010. Measured +0.006 —
below threshold. Metzler's defaults beat the unigram-heavier weights on
FiQA (+0.006 vs +0.003), confirming the ordered-bigram leg is
load-bearing; the magnitude is just too small to justify cascade/router
promotion.

## Mechanism — why FiQA undershoots

SDM rewards query-document **adjacency** of fixed multi-word terms. FiQA's
semantic-paraphrase failure mode is different: a query "cash flow
statement" against a doc that says "statement of operating cash flows"
lifts on unigram IDF sharing, not on bigram adjacency (the words are
reordered and split by prepositions). The wins SDM does deliver on FiQA
come from the small fraction of queries that contain a genuinely fixed
term ("interest rate", "mortgage-backed"). That subset isn't large enough
to move the corpus-wide nDCG past the promote threshold.

Scifact and NFCorpus undershoot for the opposite reason: scientific
abstracts *do* contain multi-word terminology ("T cell", "cystic
fibrosis"), but SAB-smooth's char-n-gram backoff already captures the
morphological signal these bigrams carry. Adding bigrams on top of SAB
double-counts the same information; SAB-SDM on scifact ties SAB-smooth
exactly.

## Infrastructure disposition

- `score_sdm()` and the bigram postings build stay in the codebase.
- `Bm25Config::build_word_bigrams` defaults `false`; existing callers are
  byte-identical to pre-Step-1l behavior.
- Not wired into `run_router_cascade()` or the default router recipe.
- Bench rows (`bm25_atire_sdm_l0.85_0.10_0.05`,
  `bm25_sab_smooth_sdm_l0.85_0.10_0.05`, `bm25_atire_sdm_l0.90_0.05_0.05`)
  stay in the three `*_full.jsonl` result files for regression tracking.

## Next lever

Weighted Sequential Dependence (Bendersky & Croft 2010, WSDM) replaces
fixed λ with per-bigram corpus-statistic weights. Requires a tuner —
deferred unless a corpus lands where fixed SDM shows a promote-threshold
lift and the weight-sensitivity row disagrees with Metzler's defaults.

## References

- Metzler, D. & Croft, W.B. (2005). "A Markov Random Field Model for Term
  Dependencies." SIGIR 2005.
- Bendersky, M. & Croft, W.B. (2010). "Discovering Key Concepts in Verbose
  Queries." SIGIR 2008 / *WSDM follow-ons.* (Deferred follow-on reference.)
