# Paperbridge Synthesis + Next Directions

## Validated anchors

| Area | Paper | Use in our argument |
|---|---|---|
| QPP | Cronen-Townsend, Zhou, Croft (2002), *Predicting Query Performance* | Foundational anchor for query-level effectiveness prediction without qrels |
| Selective expansion | Amati, Carpineto, Romano (2004), *Query Difficulty, Robustness, and Selective Application of Query Expansion* | Precedent for gating expansion rather than applying it globally |
| Score-shape QPP | Shtok et al. (2012), *Predicting Query Performance by Query-Drift Estimation* | Supports score-distribution shape as a drift / difficulty signal |
| Adaptive config selection | Deveaud et al. (2019), *Learning to Adaptively Rank Document Retrieval System Configurations* | Strongest supervised analogue of our query-dependent hard router |
| QPP taxonomy | Carmel and Yom-Tov (2010); Faggioli et al. (2023) | Places entropy, clarity, SCQ, and related features in a standard QPP framework |
| Verbose queries | Gupta and Bendersky (2015); Paik and Oard (2014); Di Buccio et al. (2014) | Supports treating long queries as a separate retrieval regime and detecting them explicitly |
| Verbose-query term selection | Park and Croft (2010); Bendersky, Metzler, and Croft (2011) | Supports moving from mere routing to query-side term ranking / concept weighting |
| Verbose-query prediction | Roitman (2018), *Passage-Level Information* | Supports passage / lead-aware handling for long queries |
| Argument retrieval | Ashley and Walker (2013); Dumani et al. (2020); Bergmann et al. (2020) | Supports the claim that ArguAna is an argument-retrieval mismatch, not ordinary topical retrieval |
| Counterargument retrieval | Wachsmuth, Syed, and Stein (2018), *Retrieval of the Best Counterargument without Prior Topic Knowledge* | Strong evidence that counterargument retrieval needs simultaneous similarity and dissimilarity, not just topical match |
| Argument search systems | Wachsmuth et al. (2017), *Building an Argument Search Engine for the Web* | Supports treating arguments as a distinct search object with its own acquisition/mining/ranking pipeline |
| Refutation structure | Jo et al. (2020), *Detecting Attackable Sentences in Arguments* | Suggests sentence-level attackability as a possible structural signal for counterargument-focused retrieval |
| Stance-aware retrieval | Kiesel et al. (2021), *Image Retrieval for Arguments Using Stance-Aware Query Expansion* | Contemporary evidence that stance-aware expansion changes retrieval behavior |
| Expansion alternative | Xu and Croft (2000), *Local Context Analysis* | Non-RM3 expansion path based on co-occurrence rather than document weighting |
| Document expansion | Tao et al. (2006), *Language model information retrieval with document expansion* | Candidate/document representation expansion path when routing uplift plateaus |

## Claims supported by the literature

### 1. The best universal result is still a routing result, but the latest uplift is small

The current universal winner is `observed_ordering_entropy_length_qpp_router`
(`0.4169`), not a pure expansion row. The correct literature frame is therefore
**QPP and query-dependent configuration selection**, not only query expansion.

However, the latest gain over the previous universal winner is only `+0.0028`.
That is large enough to validate the QPP direction, but the subsequent
ArguAna text-pair adapter ensemble moves macro to `0.4745`, and the
claim/premise refinement moves it to `0.4780`, confirming that the
larger remaining gains come from structural branches rather than more router
micro-tuning.

### 2. Query length is a regime split, but literature suggests moving upstream to query shaping

The winning router keeps BM25 entropy as the main signal and uses `n_terms > 30`
as the ArguAna separator. The verbose-query papers support exactly this move:

- long queries contain many extraneous terms
- passage / lead evidence is often more appropriate than short-query scoring

The new papers strengthen the next step beyond routing-only treatment:

- detect verbose queries explicitly
- rank or prune verbose-query terms
- weight concepts or dependencies inside the query itself

### 3. ArguAna is an argument-retrieval and counterargument-retrieval mismatch

The structural-gap story is now literature-backed:

- ordinary IR does not retrieve arguments as arguments
- claims, premises, and stance relations are retrieval primitives
- stance-aware expansion is a real retrieval behavior, not a metaphor
- best-counterargument retrieval needs simultaneous similarity and dissimilarity,
  not just topic distance
- attackability is a plausible structural cue, but naive attackability-weighting
  regressed our current adapter branch

This is the right frame for why structural adapters dominate universal lexical
polishing on ArguAna.

### 4. Generator / representation changes are now more promising than extra gates

The QPP hard-router refinement succeeded, but only marginally. The claim/premise
adapter refinement then produced the next real lift. Together with the papers on
verbose-query term weighting, counterargument retrieval, and document expansion,
this points to a stronger class of interventions than "one more gate":

- change the query representation
- change the argument representation
- change the document representation or candidate set

### 5. Keyphrase adapters are secondary, not primary

Keyphrases, RAKE, YAKE, TextRank, and entity-style signals remain plausible,
but they no longer carry the main theory story. They are now best viewed as
secondary adapter evidence for TREC-COVID, FiQA, and NFCorpus.

## Minimal research program implied by the synthesis

| Phase | Question | Minimal design |
|---|---|---|
| LXXIII | Can query shaping beat router-only gains on long or noisy queries? | Detect verbose queries, then test term ranking / concept weighting / dependency-based pruning before retrieval |
| LXXV | Can we approximate counterargument structure without leakage on non-ArguAna corpora? | Transfer claim/premise-aware structural evidence beyond ArguAna-specific page neighborhoods |
| LXXVI | Can document or candidate expansion beat the current universal ceiling? | Compare local-context analysis and document expansion against RM3-diverse on corpora where exposure or representation still dominates |
| LXXVII | Do query-salience adapters help entity-heavy corpora? | Compare RAKE / TextRank / YAKE query phrases against local-context-analysis-style co-occurrence evidence |

## Priority order

1. **Primary structural line:** argument / stance / attackability adapters
2. **Primary query line:** verbose-query shaping rather than more gate tuning
3. **Primary representation line:** document / local-context expansion where routing uplift plateaus
4. **Secondary universal line:** more QPP only if it promises more than the current `+0.0028`
5. **Secondary lexical line:** keyphrase / entity adapters

This ordering matches both the benchmark numbers and the corrected theorem: the
largest remaining gains come from better task matching or representation changes,
while additional routing polish remains second-order.
