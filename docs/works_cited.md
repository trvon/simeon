# Works cited

This is the compact prior-work list for the ideas used directly in simeon's
implementation, benchmarks, and design notes. It is intentionally selective:
these are the papers and books a downstream reader is most likely to need when
understanding the library's design or citing its evaluation context.

## Core retrieval context

- **Sharma (2026), _NUMEN: Non-neural embeddings that match transformer embeddings on retrieval_.**
  Prior art for the training-free retrieval framing that simeon reimplements and
  stress-tests. The benchmark and router docs in this repo should be cited for
  simeon's own measured behavior rather than treating this repository as an
  independent validation of NUMEN's headline claims.
- **Khattab and Zaharia (2020), _ColBERT_.** DOI: [10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)
  Reference point for late-interaction retrieval and the broader multi-vector
  retrieval framing used when positioning simeon's dense/lexical hybrid path.


## Evaluation oracles and ranking theory

- **Järvelin and Kekäläinen (2002), _Cumulated gain-based evaluation of IR techniques_.** DOI: [10.1145/582415.582418](https://doi.org/10.1145/582415.582418)
  Source for cumulative gain / discounted cumulative gain framing and the ideal
  ranked list used by nDCG-style evaluation. In simeon's theorem notes, this is
  the basis for treating the oracle as an external answer-key evaluator rather
  than a retrieval component.
- **Kekäläinen and Järvelin (2002), _Using graded relevance assessments in IR evaluation_.** DOI: [10.1002/asi.10137](https://doi.org/10.1002/asi.10137)
  Background for graded relevance judgments and why evaluation should credit
  highly relevant documents differently from marginally relevant ones.
- **Robertson (1977), _The Probability Ranking Principle in IR_.** DOI: [10.1108/eb026647](https://doi.org/10.1108/eb026647)
  Classical statement of the probability ranking principle: optimal ranking is
  tied to relevance probability under assumptions. Used to separate the ideal
  relevance ordering from realizable approximations.
- **Robertson and Zaragoza (2009), _The Probabilistic Relevance Framework: BM25 and Beyond_.** DOI: [10.1561/1500000019](https://doi.org/10.1561/1500000019)
  Theoretical context for BM25, BM25F, relevance feedback, and probabilistic
  relevance modeling. Supports treating BM25/BM25F as generator/scorer variables
  inside the system, not as the global ceiling.

## Fusion and feedback

- **Cormack, Clarke, and Buettcher (2009), _Reciprocal rank fusion outperforms condorcet and individual rank learning methods_.** DOI: [10.1145/1571941.1572114](https://doi.org/10.1145/1571941.1572114)
  Basis for RRF-style rank fusion rows and for treating fusion as a realizable
  approximation to the external ideal ranking.
- **Lavrenko and Croft (2001), _Relevance based language models_.** DOI: [10.1145/383952.383972](https://doi.org/10.1145/383952.383972)
  Foundation for relevance models and pseudo-relevance feedback. Used by the RM3
  generator-slice experiments as a training-free candidate-pool intervention.

## Compression and representation

- **Jégou, Douze, and Schmid (2011), _Product Quantization for Nearest Neighbor Search_.** DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
  Basis for the PQ/ADC compression path.
- **Li, Hastie, and Church (2006), _Very sparse random projections_.** DOI: [10.1145/1150402.1150436](https://doi.org/10.1145/1150402.1150436)
  Reference for the very-sparse projection option.
- **Bhatt et al. (2022), _Matryoshka Representation Learning_.** DOI: [10.52202/068431-2192](https://doi.org/10.52202/068431-2192)
  Motivation for the nested-prefix representation interface; simeon uses a
  training-free analog rather than the learned method itself.

## Routing and query-difficulty estimation

- **Carmel and Yom-Tov (2010), _Estimating the Query Difficulty for Information Retrieval_.** DOI: [10.1007/978-3-031-02272-2](https://doi.org/10.1007/978-3-031-02272-2)
  Background for the pre-retrieval query-difficulty family used by the router.
- **Cronen-Townsend, Zhou, and Croft (2002), _Predicting query performance_.** DOI: [10.1145/564376.564429](https://doi.org/10.1145/564376.564429)
  Reference for predictor families reused in the router notes.

## Language modeling, graph methods, and query expansion

- **Ponte and Croft (1998), _A language modeling approach to information retrieval_.** DOI: [10.1145/290941.291008](https://doi.org/10.1145/290941.291008)
  Foundation for the language modeling IR framework. Establishes `P(q|θ_d)` as
  the retrieval score.
- **Zhai and Lafferty (2001/2004), _A study of smoothing methods for language models applied to ad hoc information retrieval_.** DOI: [10.1145/984321.984322](https://doi.org/10.1145/984321.984322)
  Two-stage smoothing: Dirichlet for verbose queries, Jelinek-Mercer for short
  queries. Explains why ArguAna (long debate posts) and SciFact (fact claims)
  need different scoring approaches.
- **Lafferty and Zhai (2001), _Document language models, query models, and risk minimization for information retrieval_.** DOI: [10.1145/383952.383970](https://doi.org/10.1145/383952.383970)
  Risk minimization framework connecting LM estimation to retrieval.
- **Song and Croft (1999), _A general language model for information retrieval_.** DOI: [10.1145/319950.320022](https://doi.org/10.1145/319950.320022)
  Good-Turing and curve-fitting smoothing for LM-based retrieval.
- **Lavrenko and Croft (2001/2003), _Relevance Models in Information Retrieval_.** DOI: [10.1007/978-94-017-0171-6_2](https://doi.org/10.1007/978-94-017-0171-6_2)
  Relevance model framework for pseudo-relevance feedback. RM3 is a special
  case; the full framework provides a richer relevance language model estimate.
- **Kurland and Lee (2005), _PageRank without hyperlinks: Structural re-ranking_.** DOI: [10.1145/1076034.1076087](https://doi.org/10.1145/1076034.1076087)
  Builds a doc-doc similarity graph from initial retrieval results, then applies
  PageRank for authority-based re-ranking. Grounds the graph methods used in
  the simeon PPR experiments.
- **Kurland and Lee (2006), _Respect my authority! HITS without hyperlinks_.** DOI: [10.1145/1148170.1148188](https://doi.org/10.1145/1148170.1148188)
  Extends the graph-based re-ranking idea to HITS (hub/authority). Authority
  scores from inter-document similarity.
- **Metzler and Croft (2005), _A Markov random field model for term dependencies_.** DOI: [10.1145/1076034.1076115](https://doi.org/10.1145/1076034.1076115)
  Markov Random Field (MRF) framework for modeling term dependencies (proximity,
  order, co-occurrence) in retrieval. Generalizes the independence assumption
  in standard language models.
- **Metzler and Croft (2007), _Latent concept expansion using Markov random fields_.** DOI: [10.1145/1277741.1277796](https://doi.org/10.1145/1277741.1277796)
  Extends the MRF model with latent concept nodes, enabling concept-based query
  expansion beyond surface term matching.
- **Xu and Croft (1996/2000), _Query expansion using local and global document analysis_ and _Improving the effectiveness of information retrieval with local context analysis_.** DOI: [10.1145/333135.333138](https://doi.org/10.1145/333135.333138)
  Local context analysis: selects expansion terms by co-occurrence with query
  terms in top-ranked documents. More robust than relevance-model feedback on
  some corpora.
- **Tao, Wang, Mei, and Zhai (2006), _Language model information retrieval with document expansion_.** DOI: [10.3115/1220835.1220887](https://doi.org/10.3115/1220835.1220887)
  Probabilistic neighborhood document expansion: smooth each document's LM
  with similar documents' LMs. Training-free and corpus-level.
- **Bendersky and Kurland (2008/2010), _Utilizing passage-based language models for document retrieval_.** DOI: [10.1007/s10791-009-9118-8](https://doi.org/10.1007/s10791-009-9118-8)
  Passage-level language model estimation for ad hoc retrieval. Combines
  document-level and passage-level evidence.
- **Vechtomova (2006), _Noun phrases in interactive query expansion and document ranking_.** DOI: [10.1007/s10791-006-6390-8](https://doi.org/10.1007/s10791-006-6390-8)
  Query representation matters: noun-phrase structure can outperform plain
  bag-of-words expansion/ranking on the right query classes.
- **Efron (2013), _Query representation for cross-temporal information retrieval_.** DOI: [10.1145/2484028.2484054](https://doi.org/10.1145/2484028.2484054)
  Query representation should vary with the retrieval regime and collection
  dynamics; one fixed query view is not generally optimal.
- **Xu and Benaroch (2005), _Information Retrieval with a Hybrid Automatic Query Expansion and Data Fusion Procedure_.** DOI: [10.1023/B:INRT.0000048496.31867.62](https://doi.org/10.1023/B:INRT.0000048496.31867.62)
  Hybrid procedures that combine expansion and fusion/configuration selection
  can outperform either family alone.
- **Kimura and Araki (2006), _Query Expansion for Contextual Question Using Genetic Algorithms_.** DOI: [10.1007/11880592_48](https://doi.org/10.1007/11880592_48)
  Genetic algorithms are useful as an **offline search procedure** over query
  expansion choices; they optimize over a retrieval design space rather than
  becoming part of the deployed retrieval function itself.
- **Dinçer, Macdonald, and Ounis (2016), _Risk-Sensitive Evaluation and Learning to Rank using Multiple Baselines_.** DOI: [10.1145/2911451.2911511](https://doi.org/10.1145/2911451.2911511)
  Risk-sensitive IR evaluation: the system ranking depends on which baseline- or
  risk-aware objective is optimized, not only on average effectiveness.
- **Cañamares, Castells, and Moffat (2020), _Offline evaluation options for recommender systems_.** DOI: [10.1007/s10791-020-09371-3](https://doi.org/10.1007/s10791-020-09371-3)
  Offline evaluation objectives encode deployment priorities; changing the
  objective can change which system is preferred even under the same data.
- **Armstrong, Moffat, Webber, and Zobel (2009), _Improvements that don't add up_.** DOI: [10.1145/1645953.1646031](https://doi.org/10.1145/1645953.1646031)
  Retrieval gains are not safely compositional: improvements that look positive
  in isolation or under one composition can fail to transfer unchanged once the
  full system is materialized.
- **Carterette (2007), _Robust test collections for retrieval evaluation_.** DOI: [10.1145/1277741.1277754](https://doi.org/10.1145/1277741.1277754)
  Robustness is a first-class evaluation concern, not just a secondary summary;
  different test-collection constructions can legitimately favor different
  systems.
- **Scholer, Kelly, and Carterette (2016), _Information retrieval evaluation using test collections_.** DOI: [10.1007/s10791-016-9281-7](https://doi.org/10.1007/s10791-016-9281-7)
  Test-collection evaluation is the standard validation frame for IR, but its
  conclusions are conditional on the collection, judgments, and metric.
- **Sanderson (2010), _Test Collection Based Evaluation of Information Retrieval Systems_.** DOI: [10.1561/1500000009](https://doi.org/10.1561/1500000009)
  Broad survey of why exact benchmark/test-collection evaluation remains the
  authoritative way to state retrieval effectiveness claims.
- **Bellogín, Castells, and Cantador (2017), _Statistical biases in Information Retrieval metrics for recommender systems_.** DOI: [10.1007/s10791-017-9312-z](https://doi.org/10.1007/s10791-017-9312-z)
  Metric design can bias system comparisons; optimizer scores and deployment
  validation scores should not be treated as interchangeable without checking.

## Theorem framework (revised)

- Cite the **language modeling framework** with the papers above, especially
  Ponte & Croft 1998 and Zhai & Lafferty 2004.
- Cite the **corrected theorem** framing with
  [docs/research/theorem_correction_lm_framework.md](research/theorem_correction_lm_framework.md).
- Cite the **graph-based re-ranking experiments** with
  [docs/research/phase57_ppr_graph_rerank.md](research/phase57_ppr_graph_rerank.md).

## How to cite this repo responsibly

- Cite **the software** with [`CITATION.cff`](../CITATION.cff).
- Cite **the benchmark tables and negative findings** with
  [docs/research/benchmarks.md](research/benchmarks.md), [docs/research/router_design.md](research/router_design.md), and
  [docs/research/pmi_projection.md](research/pmi_projection.md).
- Cite **the oracle-external theorem framing** with
  [docs/research/phase38_oracle_external_theorem.md](research/phase38_oracle_external_theorem.md) and
  [docs/research/training_free_space_redefinition.md](research/training_free_space_redefinition.md).
- Cite the **upstream papers above** when discussing the underlying prior work.
