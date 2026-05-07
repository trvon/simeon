# Paperbridge Synthesis + Next Directions

## Validated anchors

| Area | Paper | Use in our argument |
|---|---|---|
| QPP | Cronen-Townsend, Zhou, Croft (2002), *Predicting Query Performance* | Foundational anchor for query-level effectiveness prediction without qrels |
| Selective expansion | Amati, Carpineto, Romano (2004), *Query Difficulty, Robustness, and Selective Application of Query Expansion* | Precedent for gating expansion rather than applying it globally |
| Score-shape QPP | Shtok et al. (2012), *Predicting Query Performance by Query-Drift Estimation* | Supports score-distribution shape as a drift / difficulty signal |
| Adaptive config selection | Deveaud et al. (2019), *Learning to Adaptively Rank Document Retrieval System Configurations* | Strongest supervised analogue of our query-dependent hard router |
| QPP taxonomy | Carmel and Yom-Tov (2010); Faggioli et al. (2023) | Places entropy, clarity, SCQ, and related features in a standard QPP framework |
| Verbose queries | Gupta and Bendersky (2015); Paik and Oard (2014) | Supports treating long queries as a separate retrieval regime |
| Verbose-query prediction | Roitman (2018), *Passage-Level Information* | Supports passage / lead-aware handling for long queries |
| Argument retrieval | Ashley and Walker (2013); Dumani et al. (2020); Bergmann et al. (2020) | Supports the claim that ArguAna is an argument-retrieval mismatch, not ordinary topical retrieval |
| Stance-aware retrieval | Kiesel et al. (2021), *Image Retrieval for Arguments Using Stance-Aware Query Expansion* | Contemporary evidence that stance-aware expansion changes retrieval behavior |
| Expansion alternative | Xu and Croft (2000), *Local Context Analysis* | Non-RM3 expansion path based on co-occurrence rather than document weighting |

## Claims supported by the literature

### 1. The best universal result is a routing result

The current universal winner is `observed_ordering_entropy_length_router`
(`0.4141`), not a pure expansion row. The correct literature frame is therefore
**QPP and query-dependent configuration selection**, not only query expansion.

### 2. Query length is a regime split, not a cosmetic feature

The winning router keeps BM25 entropy as the main signal and uses `n_terms > 30`
as the ArguAna separator. The verbose-query papers support exactly this move:

- long queries contain many extraneous terms
- passage / lead evidence is often more appropriate than short-query scoring

### 3. ArguAna is an argument-retrieval mismatch

The structural-gap story is now literature-backed:

- ordinary IR does not retrieve arguments as arguments
- claims, premises, and stance relations are retrieval primitives
- stance-aware expansion is a real retrieval behavior, not a metaphor

This is the right frame for why structural adapters dominate universal lexical
polishing on ArguAna.

### 4. Keyphrase adapters are secondary, not primary

Keyphrases, RAKE, YAKE, TextRank, and entity-style signals remain plausible,
but they no longer carry the main theory story. They are now best viewed as
secondary adapter evidence for TREC-COVID, FiQA, and NFCorpus.

## Minimal research program implied by the synthesis

| Phase | Question | Minimal design |
|---|---|---|
| LXXII | Can a QPP-style hard router beat entropy+length? | Keep hard selection; test clarity, SCQ, NQC, WIG, and variance only as extra gates over `observed_ordering_entropy_length_router` |
| LXXIII | Can we approximate counterargument structure without leakage? | Treat ArguAna as argument retrieval; search for observable stance / claim / premise signals and fuse as adapter evidence |
| LXXIV | Do query-salience adapters help entity-heavy corpora? | Compare RAKE / TextRank / YAKE query phrases against local-context-analysis-style co-occurrence evidence |

## Priority order

1. **Primary universal line:** QPP-guided hard routing
2. **Primary structural line:** argument / stance adapters
3. **Secondary lexical line:** keyphrase / local-context adapters

This ordering matches both the benchmark numbers and the corrected theorem: the
largest remaining universal gains are more likely to come from better
configuration selection, while the largest ArguAna gains come from better task
matching.
