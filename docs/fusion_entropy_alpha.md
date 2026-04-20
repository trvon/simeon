# Entropy-weighted runtime fusion α — negative result with safety property

Per-query α for linear-α fusion derived from each leg's score-distribution
entropy. Strict generalization of fixed α: when both legs have equal
softmax-confidence the formula returns 0.5 exactly, and when pool-overlap
Jaccard exceeds the agreement threshold the weighting collapses to equal
blend. No hyperparameter tuning, no labeled data, no per-corpus fit.

Implementation lives in `src/fusion.cpp`
(`entropy_alpha`, `linear_alpha_entropy_fuse`); the public API is in
`include/simeon/fusion.hpp`.

## Math

    H_i           = − Σ_k softmax(top_K_scores_i)_k · log softmax(top_K_scores_i)_k
    confidence_i  = 1 − H_i / log(K)                          ∈ [0, 1]
    α(q)          = confidence_A / (confidence_A + confidence_B)
    fused(d)      = α(q) · z(scores_A(d)) + (1 − α(q)) · z(scores_B(d))

`z` is per-pool z-score normalization, identical to the static `linear_alpha`
fusion path. `K = 50` matches the `top_k_score_entropy` predictor used by
`QueryRouter`. When `pool_overlap_jaccard ≥ agreement_threshold` (default
0.8) the function returns α = 0.5 — at high inter-leg agreement, equal
weighting is the strictly safer choice. When both confidences round to zero
(both legs uniform on top-K) the function returns 0.5 rather than 0/0.

## Three-corpus measurement

Pool size 500, simeon cosine width 4096→768, BM25 either Atire (top block) or
SAB-smooth (bottom block). The matched static-α baseline is
`bm25_pool500_linear_alpha075_4096_768` (Atire) — the strongest fixed-α
configuration shipped before Step 1m.

| Corpus   | Pool        | Pure cosine | Static α=0.75 | Entropy-α | Δ vs static-α | Δ vs pure cosine |
|----------|-------------|------------:|--------------:|----------:|--------------:|-----------------:|
| scifact  | Atire-500   | 0.411       | 0.625         | 0.619     | −0.006        | **+0.208**       |
| FiQA     | Atire-500   | 0.112       | 0.211         | 0.205     | −0.006        | **+0.093**       |
| NFCorpus | Atire-500   | 0.178       | 0.261         | 0.252     | −0.010        | **+0.074**       |
| scifact  | SAB-500     | 0.414       | —             | 0.612     | —             | **+0.198**       |
| FiQA     | SAB-500     | 0.117       | —             | 0.198     | —             | **+0.081**       |
| NFCorpus | SAB-500     | 0.190       | —             | 0.298     | —             | **+0.108**       |

Plan target for promotion: FiQA Δ vs static-α ≥ +0.005. Measured −0.006 —
below threshold on all three corpora. Step 1m closes as a documented null
result against the matched fixed-α baseline.

## Mechanism — why entropy-α undershoots fixed α=0.75 here

Entropy-α derives weight from how peaked each leg's top-K is. On these three
corpora the cosine head's top-K is consistently flatter than BM25's — the
sketch+projection signal disperses mass across more pool entries. The
formula reads that as low confidence and shifts α upward toward 1.0
(BM25-only). At the corpus mean, α(q) ≈ 0.95–1.0, so the fused score is
nearly the BM25-only ranking — which matches the `bm25_only` row to two
decimals (0.619 vs 0.619 on scifact; 0.205 vs 0.205 on FiQA; 0.252 vs 0.252
on NFCorpus).

Static α=0.75 wins by *less* aggressively trusting BM25, leaving 25% room for
the cosine leg's contribution on a few queries where it actually helps. The
entropy estimator is too quick to discount cosine because flatness ≠
unhelpfulness — a cosine leg with a long thin tail of correct rerankings
still looks high-entropy on the top-50, even when the right answer is
ranked there.

## Safety property — entropy-α as a regression rail

The other half of the picture: against the *pure cosine* baseline (no BM25
weight), entropy-α delivers +0.07 to +0.21 nDCG on all three corpora, in
both pool variants. When the dense leg would catastrophically regress on its
own (e.g. SAB-pool + pure cosine on scifact = 0.41 vs SAB alone 0.61),
entropy-α automatically discovers that BM25 is the more confident source
and defers to it. Static α requires a human to know which value won't
regress; entropy-α is *self-correcting* against a bad fusion partner.

This is the property that justifies leaving the infrastructure shipped:

- A caller building an arbitrary linear-α cascade with an untested dense
  source no longer has to tune α on a dev set to avoid regression below the
  BM25 baseline. Entropy-α gives a "no-worse-than-best-leg" floor.
- Composes with the existing `linear_alpha` path — the matched static-α
  call site can swap in `linear_alpha_entropy_fuse` without API changes.
- Zero hyperparameter footprint: no λ, no per-corpus fit, no labeled data.

## Infrastructure disposition

- `entropy_alpha()` and `linear_alpha_entropy_fuse()` ship in `simeon::`
  namespace; opt-in. Callers using `rrf_fuse` / inline static-α blends are
  byte-identical to pre-Step-1m behavior.
- `RerankMode::EntropyAlpha` added to the bench harness; rows
  `bm25_pool500_entropy_alpha_4096_768` and
  `bm25_sab_pool500_entropy_alpha_4096_768` stay in the three corpus
  `*_full.jsonl` files for regression tracking.
- Not wired into `QueryRouter::choose()` recipes. The router still picks
  cascade vs single-scorer; if and when a future recipe pairs a tuned dense
  leg with BM25 at unknown α, this is the safer fusion call.

## Next lever

The undershoot mechanism suggests the formula needs *correlation-aware*
confidence, not just per-leg entropy. Concretely: if cosine's flatness
correlates with BM25's flatness on the same query, both legs are uncertain
and the equal-weight blend is right; if cosine is flat *while* BM25 is
peaked, BM25 is the right answer (current behavior); but if cosine is
peaked *while* BM25 is flat, the current formula overweights BM25 because
the absolute confidence comparison ignores BM25's own reliability. A
joint-confidence formula (e.g. mutual-information-weighted α, or
Aslam-Pavlu-style JS divergence between the two top-K distributions used as
an *agreement* term rather than a confidence term) is the next research
target. Deferred unless a dense leg lands where the asymmetric-flatness
case dominates.

## References

- Aslam, J. A. & Pavlu, V. (2007). "Query Hardness Estimation Using Jensen-
  Shannon Divergence Among Multiple Scoring Functions." ECIR 2007.
  (Adjacent: uses JS divergence between scoring functions as a difficulty
  signal, not as a fusion weight.)
- Cronen-Townsend, S. & Croft, W. B. (2002). "Predicting Query Performance."
  SIGIR 2002. (The clarity score that simeon's `simplified_clarity`
  predictor approximates.)
- Cormack, G. V., Clarke, C. L. A. & Büttcher, S. (2009). "Reciprocal Rank
  Fusion outperforms Condorcet and Individual Rank Learning Methods."
  SIGIR 2009. (The fixed-rank-aggregation baseline; entropy-α is the
  score-distribution analog.)
