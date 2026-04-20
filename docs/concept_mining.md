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

Step 1n closes as a documented null result. Infrastructure ships in
`simeon::` namespace with no routing wire-up and no default-on flag, so
existing callers are byte-identical to pre-Step-1n behavior.

## Mechanism — why concept expansion regresses the Atire base

The concept score magnitude significantly exceeds the base BM25 magnitude.
A single matched bigram contributes `PMI × BM25_bigram_term`, typically 5–25
score units (PMI 2–10 nats × BM25 term 1–5). Base BM25 for a 4-word query is
typically 10–30. At `concept_weight=0.5`, one matched concept can rival the
entire base score; a query with 3+ matched concepts can reorder the top-100
around exact-phrase hits.

Exact-phrase reward is the wrong signal on scifact: queries are scientific
claims that correct-answer papers routinely paraphrase. Docs containing the
*exact phrase* in the query are often not the most relevant — rewarding them
outweighs the paraphrase-tolerant base BM25.

SAB-smooth's char-n-gram backoff already rewards morphological variants of
query terms, which subsumes most of the high-PMI bigrams a human would flag
as "concepts." Adding a second exact-phrase signal on top either replays the
same match (neutral on scifact: +0.002), or double-counts (mild regression
on NFCorpus: −0.003, and moderate regression on FiQA: −0.016). On FiQA,
the finance-Q&A register makes this worse: question bigrams rarely co-occur
verbatim in paraphrased answers, so concepts fire on spurious matches more
often than on the paraphrase the query intended to express.

## Why the fix isn't just "lower the weight"

On scifact with Atire, `weight=0.50 → −0.108`, `weight=0.25 → −0.055`. A
weight of 0.05 would plausibly zero out the regression but also zero out
any hypothetical FiQA lift. The scale mismatch is structural: base BM25 and
PMI-weighted concept BM25 don't live on the same scale, and the per-query
number of matched concepts is highly variable (some queries match 0, some
match 10+). No fixed scalar weight normalizes across queries.

Training-free fixes worth exploring in a follow-on step:

1. **Z-score normalize concept scores per-query** before blending, matching
   `linear_alpha_fuse`'s treatment of dense cosine scores. Would make the
   blend invariant to the number of matched concepts.
2. **Average concept score by match count** (divide by `|q.bigrams ∩ concepts|`)
   so the concept leg's per-query magnitude is bounded independently of
   query length and match density.
3. **Corpus-wide PMI max normalization**: precompute `max_pmi` across all
   mined concepts; divide PMI by `max_pmi` so the weight PMI ∈ [0, 1].

Each is a small re-ship (tens of LOC) but doesn't change the infrastructure
under test — it calibrates the blend. Deferred until we have evidence that
calibration would close the FiQA gap.

## Infrastructure disposition

- `mine_concepts()`, `ConceptIndex`, `score_bm25_with_concepts()` ship in
  `simeon::` namespace; opt-in. `bm25.hpp` does not include
  `concept_mining.hpp`, so callers that never call the concept API are
  byte-identical to pre-Step-1n behavior.
- Bench rows `bm25_atire_concepts_l0.50`, `bm25_atire_concepts_l0.25`,
  `bm25_sab_smooth_concepts_l0.50` stay in the three-corpus `*_full.jsonl`
  files for regression tracking.
- Not wired into `QueryRouter::choose()` recipes. The pre-declared regression
  gate is violated; shipping a routed version would need at least one of
  the training-free calibrations above.

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
