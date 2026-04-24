# Latent Concept Model — negative result against all three corpora

Step 1n: corpus-PMI word-bigram concept mining with PMI-weighted concept BM25
linearly blended into the base BM25 variant. Training-free (closed-form PMI,
no labeled data, no dev-fold tuner). Intended to attack FiQA's paraphrase
gap (−0.147 vs MiniLM) by rewarding documents matching high-coherence
multi-word terms from the query.

Implementation lives in `src/concept_mining.cpp` (`mine_concepts`,
`ConceptIndex::score`, `score_bm25_with_concepts`); public API is in
`include/simeon/concept_mining.hpp`.

## Math

    PMI(a, b)           = log( P(a, b) / (P(a) P(b)) )              // natural log
    P(a, b)             = bigram.total_tf / total_bigrams_across_corpus
    P(a), P(b)          = unigram.total_tf / total_tokens

    concepts            = { bigrams c  :  total_tf(c) ≥ min_ttf
                                         ∧ PMI(c) ≥ pmi_floor
                                         ∧ both components occur in the unigram index }

    concept_score(q, d) = Σ_c ∈ q.bigrams ∩ concepts
                            PMI(c) · BM25_atire(tf(c, d), dl_bigram(d),
                                                idf(c), k1, b, avg_dl_bigram)

    fused(q, d)         = BM25_base(q, d) + concept_weight · concept_score(q, d)

Defaults: `min_ttf=5`, `pmi_floor=2.0` (nats), `concept_weight=0.5`, BM25
saturation `k1=1.2`, `b=0.75`. Initial ship is bigrams only; trigrams are
deferred to a follow-on step.

## Three-corpus measurement

Concept mining runs on the same corpus used to evaluate (in-corpus config),
re-using each BM25 index's unigram `total_tf` for PMI denominators. Two base
variants (Atire and SAB-smooth `γ=5`) × two concept weights for Atire (0.50,
0.25) × SAB-smooth at 0.50.

| Corpus   | Base            | Weight | nDCG@10 | Δ vs base |
|----------|-----------------|-------:|--------:|----------:|
| scifact  | Atire 0.619     |   0.50 |   0.510 |   **−0.108** |
| scifact  | Atire 0.619     |   0.25 |   0.564 |   **−0.055** |
| scifact  | SAB-smooth 0.612 |   0.50 |   0.614 |   +0.002    |
| NFCorpus | Atire 0.252     |   0.50 |   0.242 |   **−0.011** |
| NFCorpus | Atire 0.252     |   0.25 |   0.245 |   **−0.007** |
| NFCorpus | SAB-smooth 0.298 |   0.50 |   0.295 |   **−0.003** |
| FiQA     | Atire 0.205     |   0.50 |   0.138 |   **−0.068** |
| FiQA     | Atire 0.205     |   0.25 |   0.167 |   **−0.039** |
| FiQA     | SAB-smooth 0.198 |   0.50 |   0.182 |   **−0.016** |

Plan pre-declared thresholds:

- **Promote gate**: FiQA lift ≥ +0.010 nDCG@10 over the base variant.
- **Regression gate**: scifact / NFCorpus drop < 0.005 nDCG@10.

All six FiQA × {Atire, SAB-smooth} × {0.50, 0.25} cells regress. Atire
scifact / NFCorpus / FiQA all violate the 0.005 regression ceiling. The
SAB-smooth base stays inside the regression gate on scifact (+0.002) and
NFCorpus (−0.003) but breaks it on FiQA (−0.016); promote gate (FiQA lift
≥ +0.010) is missed on every config.

Step 1n closes as a documented null result. The infrastructure ships opt-in in
`simeon::`, with no router integration or default-on path.

## Mechanism — why concept expansion regresses the Atire base

The concept score magnitude significantly exceeds the base BM25 magnitude. At
`concept_weight=0.5`, one matched concept can rival the whole base score, so a
few matched concepts can reorder the top-100 around exact-phrase hits.

Exact-phrase reward is the wrong signal on scifact, where relevant abstracts
often paraphrase the claim. SAB-smooth already captures much of the useful
morphological signal, so the concept leg either duplicates it or amplifies the
wrong phrase matches, especially on FiQA.

## Why the fix isn't just "lower the weight"

Lowering the blend weight is not enough. The mismatch is structural: base BM25
and PMI-weighted concept BM25 do not live on the same scale, and the number of
matched concepts varies too much across queries for one fixed scalar to work.

Training-free fixes worth exploring in a follow-on step:

1. **Z-score normalize concept scores per-query** before blending, matching
   `linear_alpha_fuse`'s treatment of dense cosine scores. Would make the
   blend invariant to the number of matched concepts.
2. **Average concept score by match count** (divide by `|q.bigrams ∩ concepts|`)
   so the concept leg's per-query magnitude is bounded independently of
   query length and match density.
3. **Corpus-wide PMI max normalization**: precompute `max_pmi` across all
   mined concepts; divide PMI by `max_pmi` so the weight PMI ∈ [0, 1].

Each is a small calibration change, but none is justified yet by the current
three-corpus evidence.

## Infrastructure disposition

- `mine_concepts()`, `ConceptIndex`, `score_bm25_with_concepts()` ship in
  `simeon::` namespace; opt-in. `bm25.hpp` does not include
  `concept_mining.hpp`, so callers that never call the concept API are
  byte-identical to pre-Step-1n behavior.
- Bench rows `bm25_atire_concepts_l0.50`, `bm25_atire_concepts_l0.25`,
  `bm25_sab_smooth_concepts_l0.50` stay in the three-corpus `*_full.jsonl`
  files for regression tracking.
- Not wired into `QueryRouter::choose()` because the regression gate is clearly violated.

## References

- Bendersky, M. & Croft, W. B. (2008). "Discovering Key Concepts in Verbose
  Queries." SIGIR 2008. (Original latent-concept-model formulation; used a
  supervised term-weighting head that we omit as "training-free.")
- Bendersky, M., Metzler, D. & Croft, W. B. (2010). "Learning Concept
  Importance Using a Weighted Dependence Model." WSDM 2010. (Per-concept
  λ from labeled data; declined here per training-free contract.)
- Church, K. & Hanks, P. (1990). "Word Association Norms, Mutual
  Information, and Lexicography." Computational Linguistics. (Foundational
  PMI definition used in `mine_concepts()`.)
