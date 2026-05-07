# Phase LX: Pseudo-Label Likelihood Ratio + KS-Weighted Scoring

## Goal

Use BM25 top-10 as pseudo-positive and bottom-100 as pseudo-negative to drive
per-generator scoring weights. Two formulations:

1. **Log-likelihood ratio per generator** — estimate P(score | pos) / P(score | neg)
   per doc per generator, sum evidence. Result: catastrophically weak (0.0167).
2. **KS-weighted linear combination** — weight each generator by its Kolmogorov-
   Smirnov separation between pseudo-positive and pseudo-negative scores.

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| KS-weighted PLR | 0.3439 | 0.2104 | 0.2646 | 0.6129 | 0.5669 | 0.3997 |
| Diverse RM3 | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4108 |
| Gated ensemble | 0.3494 | 0.2048 | 0.2661 | 0.6044 | 0.6185 | 0.4086 |

### Δ vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---:|---:|---:|---:|---:|---:|
| KS-weighted | +0.0146 | +0.0051 | +0.0125 | −0.0059 | +0.0020 | +0.0057 |
| Diverse RM3 | +0.0034 | −0.0053 | +0.0268 | −0.0165 | +0.0751 | +0.0167 |

## Why KS-weighted works where it does

The ArguAna improvement (+0.0146) comes from BM25F lead-64 being a strong
separator of top vs bottom on debate posts: the lead-64 text is structurally
different between relevant counterarguments and irrelevant claims.

FiQA (+0.0051 over BM25) benefits from the simeon embedding and RM3 providing
orthogonal separation signals — neither dominates, but the KS-weighted
combination slightly improves on BM25 alone.

## Why it fails on TREC-COVID

The KS distances are equalizing: on TREC-COVID, all 4 generators have similar
separation between top-10 and bottom-100 (the corpora have many medical
abstracts about COVID, so even bottom-ranked docs have similar BM25 scores).
The KS weights approach ≈0.25 each, which is just equal-weight z-fusion.

Diverse RM3's TREC-COVID advantage (+0.0751 over BM25) comes from its expansion
mechanism, not from better per-document score weighting. The KS approach cannot
replicate this because it only reweights existing per-doc scores.

## Impact on the thesis

The pseudo-label approach confirms that **the remaining ordering gap is not
about weight optimization**. After 30 forward iterations:

| Category | Best approach | Macro | Ceiling |
|---|---|---|---|
| Fusion optimization | Sigmoid dampening | 0.4041 | 0.4057 |
| Scorer improvement | Diverse RM3 | 0.4108 | — |
| Composite ensemble | Gated 3-way | 0.4086 | — |
| Term-level LM | score_lm_interpolation | 0.3978 | Needs richer docs |
| Pseudo-label scoring | KS-weighted | 0.3997 | BM25 biased |

All approaches converge to 0.40–0.41. The 0.33 gap to the BM25 oracle remains
unaddressed. The structural finding: **relevance on these corpora requires
evidence that BM25 and its linear derivatives cannot provide.**

## End of the current research arc

The 30-phase investigation has exhaustively tested:

- **Generators (G):** BM25, BM25F, RM3, simeon embedding — solo and in union
- **Scorers (S):** z-score fusion, weighted fusion, PRF variants (weighted,
  diverse, MMR), term-level LM, pseudo-label likelihood ratio, graph PPR
- **Fusion (F):** RRF, static weighted z, sigmoid dampening, convex combination,
  entropy gating, KS-weighted combination
- **Routing (F(q)):** entropy gate, avg_idf gate, multi-signal gates, ensemble

No approach has broken past 0.411 macro. The BM25 oracle@100 is 0.742.

The gap is structural, not algorithmic. Closing it requires qualitatively
different evidence — document structure, entity extraction, cross-document
relations, or domain knowledge — that the current body-only BEIR fixtures
do not expose.
