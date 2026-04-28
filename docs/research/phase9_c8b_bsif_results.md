# Phase IX Results: C8b Bigram Hadamard SIF Fragment Encoding

## Experiment

Addresses Ceiling B (flat PMI bag-of-words space). Extends C8a (SIF unigram weighting) with
Mitchell-Lapata multiplicative composition: adjacent token pairs contribute an element-wise
Hadamard product to the fragment vector:

```
fragment_vec = Σᵢ idf(wᵢ)·pmi(wᵢ)                    [SIF unigram terms]
             + Σᵢ bigram_weight·√(idf(wᵢ)·idf(wᵢ₊₁))·(pmi(wᵢ) ⊙ pmi(wᵢ₊₁))   [bigram terms]
L2-normalized
```

The Hadamard product `pmi(a) ⊙ pmi(b)` captures the intersection of dimensions where both
words have strong co-occurrence signal — encoding relational context without a POS tagger
or grammar parser. Approximates the multiplicative composition of Mitchell & Lapata (2010,
CL), which outperforms additive composition on subject-verb and modifier-noun tasks.

Implementation: `src/fragment_geometry.cpp` — `encode_bsif_weighted()` +
`build_doc_semantic_fragments_richcov_bsif()`. Builder is structurally identical to C8a
but uses `encode_bsif_weighted()` (unigram SIF + Hadamard bigrams). Config header:
`include/simeon/fragment_geometry.hpp` — `build_doc_semantic_fragments_richcov_bsif`.
Falls back to standard richcov on non-PMI encoders.

Tested as C1+C8b (outer MaxSim + BSIF fragments) and PHSS+C8b on trec-covid — the validated
corpus from Phase III. Reference: Mitchell & Lapata (2010), doi:10.1111/j.1551-6709.2010.01106.x.

## Alpha Sweep Results (richcov t8, trec-covid)

### Test Fold (32 queries)

| Config | nDCG@10 | Δ vs BM25 | Δ vs OM α=0.80 | Δ vs SIF α=0.80 |
|--------|---------|-----------|----------------|-----------------|
| BM25 baseline | 0.5649 | — | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5752 | +0.0103 | — | — |
| SIF+MaxSim α=0.80 (Phase VI) | 0.5721 | +0.0072 | −0.0031 | — |
| BSIF+MaxSim α=0.50 | 0.5636 | −0.0013 | −0.0116 | +0.0025 |
| BSIF+MaxSim α=0.65 | 0.5802 | +0.0153 | +0.0050 | +0.0051 |
| BSIF+MaxSim α=0.80 | 0.5710 | +0.0061 | −0.0042 | −0.0011 |
| BSIF+MaxSim α=0.90 | 0.5639 | −0.0010 | −0.0113 | +0.0019 |
| BSIF+MaxSim α=0.95 | 0.5648 | −0.0001 | −0.0104 | −0.0035 |
| PHSS+BSIF α=0.80 | 0.5805 | +0.0156 | +0.0053 | +0.0000 |

### Dev Fold (17 queries)

| Config | nDCG@10 | Δ vs BM25 | Δ vs OM α=0.80 | Δ vs SIF α=0.80 |
|--------|---------|-----------|----------------|-----------------|
| BM25 baseline | 0.4943 | — | — | — |
| Outer MaxSim α=0.80 (Phase III) | 0.5025 | +0.0082 | — | — |
| SIF+MaxSim α=0.80 (Phase VI) | 0.4902 | −0.0041 | −0.0123 | — |
| BSIF+MaxSim α=0.50 | 0.4657 | −0.0286 | −0.0368 | +0.0015 |
| BSIF+MaxSim α=0.65 | 0.4943 | +0.0000 | −0.0082 | +0.0097 |
| BSIF+MaxSim α=0.80 | 0.4933 | −0.0010 | −0.0092 | +0.0031 |
| BSIF+MaxSim α=0.90 | 0.4950 | +0.0007 | −0.0075 | −0.0014 |
| BSIF+MaxSim α=0.95 | 0.4940 | −0.0003 | −0.0085 | −0.0059 |
| PHSS+BSIF α=0.80 | 0.4946 | +0.0003 | −0.0079 | +0.0101 |

## Cross-Fold Assessment

| Config | Δtest vs OM | Δdev vs OM | Direction | Verdict |
|--------|-------------|------------|-----------|---------|
| BSIF+MaxSim α=0.65 | +0.0050 | −0.0082 | FLIP | DISPROVED |
| BSIF+MaxSim α=0.80 | −0.0042 | −0.0092 | consistent negative | DISPROVED |
| **PHSS+BSIF α=0.80** | **+0.0053** | **−0.0079** | **FLIP** | **DISPROVED** |

Cross-fold threshold: ±0.005 nDCG@10 per fold.

## Scifact Sanity (test fold)

BSIF is consistently ~0.0018 below SIF at every alpha on scifact. Expected — scifact
abstracts are too short for bigram co-occurrence statistics; the Hadamard products
add noise rather than signal on 2-5 sentence abstracts.

| Config | scifact test | Δ vs BM25 | Δ vs SIF |
|--------|-------------|-----------|---------|
| BM25 baseline | 0.6188 | — | — |
| SIF + MaxSim α=0.80 | 0.6183 | −0.0005 | — |
| SIF + MaxSim α=0.90 | 0.6219 | +0.0031 | — |
| BSIF + MaxSim α=0.80 | 0.6165 | −0.0023 | −0.0018 |
| BSIF + MaxSim α=0.90 | 0.6201 | +0.0013 | −0.0018 |
| BSIF + PHSS α=0.80 | 0.6127 | −0.0061 | — |

## Verdict

**DISPROVED.** BSIF inherits C8a's failure: all variants sign-flip or are consistently
negative on cross-fold validation.

PHSS+BSIF showed +0.0053 on the test fold (same as PHSS+SIF — the Hadamard bigrams add
nothing), but collapsed to −0.0079 on dev. The test fold result is a fold artifact.

BSIF+OM α=0.65 showed +0.0050 test / −0.0082 dev — another sign flip. The apparent
improvement at this unusual alpha is also fold-specific.

Compared to C8a (SIF alone): BSIF is marginally better on dev fold in some cells
(e.g., PHSS+BSIF dev −0.0079 vs PHSS+SIF dev −0.0180) but both are unambiguously
negative and don't cross the validation threshold.

## Mechanism

The Hadamard bigram product `pmi(a) ⊙ pmi(b)` retains only dimensions where both tokens
have strong co-occurrence signal. For "COVID-19 increases mortality":
- `pmi("increases") ⊙ pmi("mortality")` → medical outcome dimensions

However, this fails for the same reason C8a fails: the PMI space on trec-covid is
already saturated at the content-word level. Hadamard products cannot create new
separability when the PMI matrix itself lacks the relational structure to encode
verb-argument relationships at rank 128. The vocabulary overlap between
subject-verb-object triples in COVID abstracts maps to a near-flat PMI subspace.

**Key observation**: PHSS+BSIF and PHSS+SIF produce identical test-fold scores (0.5805).
This confirms the bigram Hadamard terms add no new signal beyond SIF unigrams — the
PMI space simply does not encode bigram co-occurrence structure at this rank.

## Next

Both C8a and C8b DISPROVED. Ceiling B confirmed absolute for weighted PMI composition.

The Mitchell-Lapata multiplicative operator requires a vector space where relational
structure is encoded in individual dimensions. PMI SPPMI at rank 128 does not provide
this — co-occurrence dimensions are not semantically interpretable at this granularity.

The DisCoCat direction (C9) would require a fundamentally better base embedding (e.g.,
fastText or GloVe trained specifically on the corpus) rather than PMI-SPPMI, making it
a training-dependent approach despite the training-free composition step.
