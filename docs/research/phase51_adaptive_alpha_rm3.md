# Phase LI: Adaptive α Weighted RM3

## Goal

Phase XLIX's weighted RM3 uses a fixed α=0.5 for all queries, which regresses
SciFact (−0.0154 vs BM25). This phase makes α adaptive: α = 0.5 ×
sigmoid((entropy − 0.48) × 8.0), so RM3 expansion is naturally suppressed on
low-entropy queries (SciFact, ArguAna) without a hard gate.

The intuition: BM25 entropy proxies query difficulty. On easy queries where BM25
is confident, stick closer to the original query (α≈0). On hard queries where
BM25 is uncertain, expand aggressively (α≈0.5).

## Method

```text
α = 0.5 × sigmoid((bm25_entropy10 − 0.48) × 8.0)
θ'(w) = (1−α) × p_Q(w) + α × p_R(w)
```

where `p_R` is the simeon-weighted relevance model from Phase XLIX.

Per-corpus mean α:

| Corpus | Entropy | α |
|---|---:|---:|
| ArguAna | 0.0006 | 0.010 |
| SciFact | 0.21 | 0.052 |
| FiQA | 0.45 | 0.220 |
| NFCorpus | 0.57 | 0.337 |
| TREC-COVID | 0.68 | 0.416 |

The row is `observed_ordering_rm3_weighted_adaptive_alpha`.

Artifact:

```text
/tmp/simeon_adapt_20260505_081110
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| Weighted RM3 (α=0.5) | 0.3323 | 0.2034 | 0.2782 | 0.6034 | 0.6059 | **0.4046** |
| Adaptive α | 0.3295 | 0.2042 | 0.2711 | 0.6101 | 0.5826 | 0.3995 |
| Sigmoid 4-way | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | 0.4041 |

### Adaptive vs fixed α

| Corpus | Fixed α=0.5 | Adaptive α | Δ |
|---|---:|---:|---:|
| ArguAna | 0.3323 | 0.3295 | **−0.0028** |
| FiQA | 0.2034 | 0.2042 | +0.0008 |
| NFCorpus | 0.2782 | 0.2711 | **−0.0071** |
| SciFact | 0.6034 | 0.6101 | **+0.0067** |
| TREC-COVID | 0.6059 | 0.5826 | **−0.0233** |
| **Macro** | **0.4046** | **0.3995** | **−0.0051** |

## Interpretation

The adaptive α **helps SciFact** (+0.0067) by suppressing expansion on
low-entropy fact-claim queries. But it **costs too much elsewhere**:

- TREC-COVID −0.0233: the entropy signal is too aggressive at suppressing
  expansion on medical queries. TREC-COVID has the highest entropy (0.68) and
  should get full α=0.5, but the sigmoid at center 0.48 only gives α=0.416.

- NFCorpus −0.0071: similar issue. NFCorpus entropy (0.57) gets α=0.337
  instead of 0.5.

- ArguAna −0.0028: the weighted RM3's ArguAna gain (+0.0030 over BM25) is
  lost when α is reduced to 0.010.

The core tension: **entropy cannot simultaneously suppress expansion on SciFact
(where it helps) and preserve expansion on TREC-COVID/NFCorpus (where it's
also needed).** The sigmoid's steepness controls the tradeoff between these
regimes, but no single center and steepness can satisfy both.

### Why entropy fails as a scalar gate for α

| Corpus | Needs α | Entropy | Problem |
|---|---|---|---|
| SciFact | ≈0.0 | 0.21 | Entropy is low → suppress ✓ |
| TREC-COVID | ≈0.5 | 0.68 | Entropy is high → expand ✓ |
| NFCorpus | ≈0.5 | 0.57 | Entropy is moderate → partial |
| ArguAna | ≈0.5 | 0.0006 | Entropy is very low → suppress ✗ |

ArguAna is the failure case: weighted RM3 helps ArguAna (+0.0030 over BM25)
despite its near-zero entropy. But any entropy-based gate suppresses it.

## Impact on the theorem

The adaptive α experiment confirms that **entropy alone cannot decompose the
"should we expand?" question.** Weighted RM3 helps on some low-entropy queries
(ArguAna) and hurts on others (SciFact). The helpfulness of expansion depends
on the query's RELATION to the document collection, not just BM25's internal
score distribution.

## Strategic assessment

After three phases of scoring experiments (XLIX, L, LI), the picture is clear:

| Row | Macro | SciFact Δ | Type |
|---|---|---:|---:|---|
| Weighted RM3 | 0.4046 | −0.0154 | Best raw quality |
| Sigmoid 4-way | 0.4041 | +0.0016 | Best safety |
| Adaptive α RM3 | 0.3995 | −0.0087 | Compromise, worse |

The training-free scorer space appears saturated at ≈0.405 macro for this
generator set. The remaining 0.4043 gap to the oracle likely requires:

1. **A richer query-document interaction model** — beyond BM25 variants and
   linear score combination.
2. **Diversity-aware expansion** — pseudo-relevance terms selected for coverage
   rather than similarity.
3. **Expanded generator set** — generators that access qualitatively different
   evidence (metadata, citation, entity, relation).
