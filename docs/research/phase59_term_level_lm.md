# Phase LIX: Term-Level LM Interpolation

## Goal

Implement proper term-level Language Model interpolation — the architecture
difference between the LM framework and score-level fusion. Added
`Bm25Index::score_lm_interpolation()` which accesses both body and aux
postings per term, then blended evidence via a custom ScoreFn.

## Library change

Added `Bm25Index::score_lm_interpolation(body_terms, aux_terms, out_scores)`:

```cpp
score(d) = Σ_{t∈body_terms} body_w(t) × contrib_body(t,d)
         + Σ_{t∈aux_terms}  aux_w(t)  × contrib_aux(t,d)
```

where `contrib_body` uses body postings/doc lengths and `contrib_aux` uses
aux postings/doc lengths. Both use Atire-style BM25 scoring.

## Benchmark row

`observed_ordering_term_lm` — builds body and aux weighted terms from the
original query plus diverse RM3 expansion, then calls `score_lm_interpolation`.

Artifact:

```text
/tmp/simeon_termlm_20260505_191439
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| Term-level LM | 0.3293 | 0.2063 | 0.2639 | 0.6192 | 0.5701 | 0.3978 |
| Diverse RM3 | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4108 |
| Sigmoid 4-way | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | 0.4041 |

Term-level LM improves over BM25 (+0.0037) but falls well short of the best
score-level rows. TREC-COVID is the clearest failure: +0.0052 vs +0.0751 for
diverse RM3.

## Why term-level loses RM3's advantage

The score-level fusion computes:

```text
score = z_bm25 + z_bm25f + z_diverse_rm3 + z_simeon
```

Each component is a full-document score, z-normalized for scale. The diverse
RM3 component carries its full expansion signal through scoring + z-scoring.

The term-level LM computes:

```text
For each expansion term t with RM weight w_rm(t):
    body_contrib += w_rm * lam * w_rm(t) * tf_body(t,d) * idf(t)
For each original term t:
    body_contrib += (1 − w_rm − w_aux) * tf_body(t,d) * idf(t)
    aux_contrib  += w_aux * tf_aux(t,d) * idf(t)
score = Σ body_contrib + Σ aux_contrib
```

The expansion terms contribute at weight `w_rm * lam * w_rm(t)` — two levels
of scaling (λ gate × expansion weight). Original terms contribute at
`(1 − w_rm − w_aux)` — close to 1.0. The expansion signal is heavily diluted.

## When term-level LM would beat score-level

Term-level interpolation has a structural advantage only when:

1. **The aux field has complementary content** — terms that appear in aux but
   NOT in body (e.g., titles, metadata, captions) can be scored separately.

2. **The expansion terms are qualitatively different from query terms** — if
   expansion introduces entirely new vocabulary, term-level blending gives each
   new term equal standing.

Neither condition holds in the current BEIR fixtures:
- The aux field (lead-64 or textrank title) is a SUBSET of the body — every
  aux term is also a body term
- RM3 expansion terms are BM25-derived and highly correlated with body terms

With body-only corpora and BM25-derived expansions, term-level = score-level
up to a scaling factor. The architectural advantage is latent.

## Library impact

The `score_lm_interpolation` method is now available in `Bm25Index` for
applications with genuinely complementary aux fields (e.g., YAMS with path
fragments, documents with title/body separation, multi-field indexes).

## Next direction

The term-level architecture is correct but can't show its value on body-only
BEIR fixtures. The remaining 0.33 ordering gap must be addressed through:

1. **Richer document representation** — complementary fields that the term-level
   interpolator can exploit
2. **Qualitatively different expansion mechanisms** — not just term re-weighting
   from BM25 top-K
3. **Corpus adapters** — YAMS-style path/entity/metadata field extraction
