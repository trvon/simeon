# Research notes

This file summarizes what the research benchmark infrastructure (`bench_vs_reference`
and `bench_vs_reference_research`) probes, what each sweep found, and which
components were promoted to the production surface.

Build with `-Denable_research=true` to enable research-only components (GloVe loader,
ArguAna adapters, SelfAssessRouter, Rm3DiverseStrategy) and the reference-comparison
benchmark binaries.

## Research benchmark sweeps

All sweeps run against an external fixture (corpus + queries + qrels + reference
embeddings). The fixture format is documented in [reference_fixture.md](reference_fixture.md).
Both the production bench (`bench_vs_reference`) and the research bench
(`bench_vs_reference_research`) consume the same fixture layout.

### Production surface (shipped router recipes)

The default `bench_vs_reference` binary keeps the stable comparison surface:
- **BM25 variants** (Atire, AtireLTD, SAB-smooth, BM25+, BM25L, DPH, PL2, DCM, Layered, LayeredW)
- **BM25 ↔ simeon RRF fusion** at pool sizes 100/300/500 and α ∈ {0.65, 0.75, 0.85, 0.95}
- **Fragment-geometry rerank** (richcov, PHSS approx, MaxSim SPLATE)
- **Query router** (pre-retrieval predictors → Atire vs SAB vs CascadeLinearAlpha)
- **Quality router** (short vs medium vs long semantic query → Bm25Only vs FragmentRichCovPhssApprox vs FragmentRichCovPhssApproxMax)
- **Step 1g.1 post-retrieval gates** (pool Jaccard, score decay) and **Step 1k B2 pre-retrieval gates** (SCQ, clarity)
- **BM25-variant RRF** (`score_bm25_variants_rrf`) and entropy-weighted linear-α fusion (`linear_alpha_entropy_fuse`)
- **RM3 pseudo-relevance feedback** (`score_with_prf`)
- **Concept mining** (`mine_concepts` + BM25-concept blend)
- **Corpus-class recipe routing** (AtireLTD for long-doc corpora, Atire for short-doc)

### Research sweeps (opt-in, `bench_vs_reference_research`)

| Sweep flag | Description | Outcome |
|---|---|---|
| `--aux-from {textrank,ac}` | BM25F with synthetic aux fields (TextRank titles, Aho-Corasick entities) | Marginal; not shipped |
| `--softmatch-only` | PMI-neighbor query expansion (corpus-derived, training-free) | Neutral-to-negative; not shipped |
| `--transport-only` | Phrase/concept transport via SDM bigrams + mined concepts + z-scored BM25 pool fusion | Mixed; concept leg occasionally helped, phrase leg did not |
| `--graph-only` | Personalized-PageRank-style graph reranking over query-phrase-doc graphs | Negative; not shipped |
| `--cluster-only` | TextRank-fragment cluster rescoring by query-cover and containment mass | Neutral; not shipped |
| `--fragment-only` | PMI-encoded fragment graphs with semantic edges + lexical bridge edges + geometric query-centered neighborhoods + DocScorer cross-product grid | Fragment geometry (richcov) promoted to production; geometric neighborhoods and hybrid bridge edges were not |

### Key research findings

1. **Fragment geometry with rich coverage (richcov) is the strongest training-free dense signal.**
   PHSS approximate scale selection (LargestGapApprox) is equal-or-better than the exact LargestGap
   path at ~2× throughput. Shipped as `build_doc_semantic_fragments_rich_covered` +
   `FragmentGeometryConfig::use_phss = true` with `PhssConfig::Criterion::LargestGapApprox`.

2. **BM25 variants matter, but corpus-specifically.**
   AtireLTD lifts long-doc corpora (avg_dl > 250). Atire is safest for short-doc.
   BM25-variant RRF fusion helps NFCorpus but regresses on SciFact/FiQA.
   Use `recommend_recipe_by_avg_dl()` as a starting point.

3. **The pre-retrieval router (QueryRouter) works across the three shipped fixtures.**
   Step 1g.1 post-retrieval gates (pool Jaccard, score decay) and Step 1k B2 pre-retrieval
   gates (SCQ, clarity) are the recommended config for production.

4. **Graph-based re-ranking (PPR, cluster) did not transfer.**
   The PPR graph experiments (phrase-doc, doc-doc edges, damping 0.70-0.85) and cluster-topology
   experiments were negative across all three fixtures. These paths were not promoted.

5. **RM3 PRF is corpus-sensitive.**
   It helps NFCorpus but stays flat or regresses on SciFact/FiQA. Keep opt-in and
   tune per corpus.

6. **Concept mining is fragile but occasionally useful.**
   Works on FiQA (structured product descriptions) but the PMI-floor threshold
   needs per-corpus tuning.

7. **Double-array Aho-Corasick was rejected.**
   The dense 256-col goto table outperforms the double-array variant in throughput
   despite larger memory footprint. The DA prototype was removed (commit 7faf23e+).

## Components gated behind `enable_research`

When `-Denable_research=false` (production default), the following are excluded:

| Component | Files | Reason |
|---|---|---|
| GloVe/fastText loader | `glove_embeddings.hpp/cpp` | Only used by research benchmarks |
| SelfAssessRouter | `retrieval_strategy.hpp/cpp` | Only used by research benchmarks |
| Rm3DiverseStrategy | `retrieval_strategy.hpp/cpp` | Only used by research benchmarks |
| ArguanaAdapter | `corpus_adapter.hpp/cpp` | Corpus-specific; never used in production |
| ArguanaTextPairAdapter | `corpus_adapter.hpp/cpp` | Corpus-specific; never used in production |
| `bench_vs_reference` | `benchmarks/bench_vs_reference.cpp` | Requires GloVe + SelfAssessRouter |
| `bench_vs_reference_research` | `benchmarks/bench_vs_reference_research.cpp` | Research-only sweeps |

## Negative results and dead ends

These were investigated and did not produce consistent gains:

- **PMI soft-match query expansion**: did not generalize across corpora.
- **SDM/WSDM bigram legs**: helped some corpora modestly but not enough for default routing.
- **PPR graph reranking**: negative; the back-propagation overestimated authority.
- **Cluster-topology rescoring**: neutral; the fragment-containment signal was too noisy.
- **TextRank/Aho-Corasick aux fields for BM25F**: marginal; the synthetic titles/entities
  didn't carry enough discriminative signal.
- **Double-array Aho-Corasick trie**: slower than dense 256-col goto table despite
  theoretical memory advantages; the dense table is bandwidth-bound, not capacity-bound,
  on the tested dictionary sizes.

For detailed per-sweep results including nDCG tables, run:
```sh
./build/benchmarks/simeon_bench_vs_reference fixtures/<dataset> > results.jsonl
./build/benchmarks/simeon_bench_vs_reference_research fixtures/<dataset> --<flag> > research.jsonl
```

## Works cited

See [works_cited.md](works_cited.md) for the paper-level citations backing each
component and experiment in this library.
