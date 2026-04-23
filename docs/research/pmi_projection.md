# PMI / co-occurrence projection

Training-free word embeddings via shifted positive PMI, factored with
randomized SVD. Caller-owned seed corpus, deterministic given
`(seed_corpus, PmiConfig)`.

## Math

For a seed corpus tokenized into words, let `c(a)` be the unigram count of
token `a` and `c(a,b)` be the sliding-window co-occurrence count at radius
`window_size`. With total count `D = Σ_a c(a)` and window mass
`W = Σ_{a,b} c(a,b)`, the shifted positive PMI matrix is

    SPPMI[a, b] = max( log( c(a,b) · W / (c(a) · c(b)) ) − log(k), 0 )

where `k = exp(shift_log_k)` (set `shift_log_k = 0` for vanilla PPMI).

Row embeddings come from a truncated randomized SVD of the sparse symmetric
SPPMI matrix `M` (Halko, Martinsson & Tropp 2011, Algorithm 4.1 + power
iteration):

1. Draw Gaussian test matrix `Ω ∈ R^{n × (r + p)}` from a seeded PRNG.
2. `Y = M Ω` (sparse × dense matmul).
3. Repeat `svd_iters` times: `Y = M (M^T Y)` with Gram–Schmidt
   reorthonormalization between each product.
4. `Y = Q R`; form `C = Q^T M Q` (small, `(r+p) × (r+p)`).
5. Jacobi eigendecomposition `C = V Λ V^T`. Keep top `r` by `|λ|`.
6. Row embeddings `U = Q V_k · sqrt( max(Λ_k, 0) )`.

`M` is symmetric (co-occurrence is symmetric), so this reduces to the
eigendecomposition branch of Halko §5 rather than the general SVD.

Levy & Goldberg (NeurIPS 2014) proved word2vec-SGNS is implicit
factorization of this same SPPMI matrix — the factorization is closed-form;
only the SGD optimizer is “trained.” A direct SPPMI + randomized SVD
pipeline is a strict subset of word2vec: same math, deterministic optimizer.

## Encoder integration

When `EncoderConfig::pmi_rows != nullptr`, the encoder bypasses the sketch
and projection pipeline entirely. Document encoding:

1. Tokenize to lowercased words (word-only, no char n-grams or BPE — PMI
   rows exist at word granularity only).
2. For each in-vocab token, sum the rank-`r` PMI row into the output. OOV
   tokens are skipped; an all-OOV document produces the zero vector.
3. Apply matryoshka weights if set.
4. L2 normalize.

`output_dim` is overridden to equal `pmi_rows->dim()`. `projection` must be
`None`; other configs throw `std::invalid_argument`.

## Seed-corpus provenance

1. **In-corpus (`_incorpus`)** — seed corpus is the evaluation corpus
   itself. Leakage-positive; use only as a sanity ceiling, not a headline.
2. **Held-out fold** — learn PMI on one split, evaluate on the other.
3. **External corpus** — learn PMI from an unrelated corpus.

Every benchmark row should name its provenance tier in the config string.

## Honest failure mode on scifact

Even with the leakage-positive `_incorpus` ceiling, unigram static PMI
undershoots the target quality band on scifact and makes the cascade reranker
worse, not better. The gap appears to come from two limits:

- **Contextual vs static.** MiniLM-L6 produces context-sensitive token
  representations; PMI rows are static per-type.
- **Multi-word terminology.** Scientific relevance often lives in phrases
  (“T cell”, “reverse transcriptase”), while the current learner is unigram.

Decision: keep PMI as a documented experimental head, but do not present it as
a production-quality improvement on scifact.

## References

- Levy, Goldberg 2014. *Neural Word Embedding as Implicit Matrix
  Factorization.* NeurIPS.
- Halko, Martinsson, Tropp 2011. *Finding Structure with Randomness:
  Probabilistic Algorithms for Constructing Approximate Matrix
  Decompositions.* SIAM Review 53(2). arXiv:0909.4061.
