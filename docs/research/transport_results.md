# Phrase/document transport results

Experiment: move beyond unigram neighbor expansion by scoring a **sparse phrase transport leg** inside a BM25 top-K pool. The transport leg is built from existing simeon primitives:

- ordered and unordered SDM bigram legs (`score_sdm()` with `lambda_unigram=0`)
- optional PMI-mined concept postings (`ConceptIndex::score()`)
- pool-restricted z-score fusion with BM25

This is a training-free proxy for stronger transport structure: query mass is no longer pushed only onto isolated term neighbors, but onto phrase nodes that then connect to candidate documents.

Implementation shipped in `benchmarks/bench_vs_reference.cpp` behind `--transport-only`.

## Protocol

- Base scorer: BM25 Atire.
- Candidate pool: BM25 top-K, `K ∈ {100, 500}`.
- Transport legs:
  - ordered bigrams only
  - ordered + unordered bigrams
  - concepts only
  - ordered + unordered + concepts
- Fusion: `alpha * z(BM25_pool) + (1-alpha) * z(transport_pool)`.
- Query-side calibration:
  - divide phrase leg by adjacent query-bigram count
  - divide concept leg by matched query-concept count
- Extra gate trial:
  - disable the transport leg when its positive-mass coverage in the pool is too diffuse (`gate0.2`, `gate0.5`)

## Results

| Corpus | BM25 | Best FiQA-seeking row | Δ | Best safe row | Δ |
|--------|-----:|----------------------:|--:|--------------:|--:|
| scifact | 0.6188 | 0.6115 (`phrase k100 a0.8`) | -0.0073 | 0.6162 (`phrase k100 a0.8 gate0.5`) | -0.0026 |
| nfcorpus | 0.2521 | 0.2531 (`ordered k100 a0.8`) | +0.0010 | 0.2511 (`phrase k100 a0.8 gate0.2`) | -0.0010 |
| fiqa | 0.2053 | 0.2101 (`phrase k100 a0.8`) | +0.0048 | 0.2055 (`phrase k100 a0.8 gate0.5`) | +0.0002 |

## Verdict

**Near-miss, not yet a promote.**

What worked:

1. **Phrase transport is materially better than unigram soft matching.**
   The best row (`ordered+unordered`, `k=100`, `alpha=0.8`) lifts FiQA from `0.2053 -> 0.2101` (`+0.0048`), much closer to the predeclared `+0.005` bar than any prior portable transport attempt.
2. **Concept nodes still hurt.**
   Even after per-query normalization, adding mined concepts regresses all three corpora relative to the phrase-only transport leg.

What failed:

1. The best FiQA row regresses scifact by `-0.0073`, so it fails the cross-corpus regression gate.
2. Diffuse-mass gating (`gate0.2`, `gate0.5`) protects scifact, but it also removes almost all of the FiQA lift.

## Interpretation

This experiment refines the theory:

- **Flat token transport is too weak.**
  That was the soft-match result.
- **Phrase structure does add real signal.**
  The FiQA gain is the first portable transport-like lift that is clearly above noise.
- **But phrase transport alone is still too local and corpus-bound.**
  On scifact, the same mechanism overweights phrase adjacency where paraphrase and reordered terminology matter more.

So the next mathematical step is not “more concept weight” or “another scalar blend.” It is a **higher-level transport graph**:

1. **document-anchored transport** (`query -> top-K docs -> phrase nodes -> docs`), or
2. **fragment-level transport** (sentence / passage nodes) so the walk has more structure than adjacent bigrams.

Those are the closest sparse, portable analogues to position-sensitive / hierarchical OT that remain plausible in simeon's current architecture.

## Concept rework probes

After inspecting the concept failures, two reworks were tested:

1. **doc-anchored concept transport** — seed concept mass from BM25 top-K docs, then project it back into the pool.
2. **filtered exact concepts** — only allow concepts when both concept terms are contentful query terms; bound PMI before fusion.

| Corpus | BM25 | raw concepts | filtered concepts | doc-anchored concepts |
|--------|-----:|-------------:|------------------:|----------------------:|
| scifact | 0.6188 | 0.6021 | 0.6104 | 0.6127 |
| nfcorpus | 0.2521 | 0.2517 | 0.2520 | 0.2520 |
| fiqa | 0.2053 | 0.2040 | 0.2028 | 0.2025 |

What this means:

- **Generic-phrase suppression helps.** Filtering concepts by requiring both concept terms to be contentful query terms recovers much of the scifact damage (`0.6021 -> 0.6104`).
- **But the deeper problem remains.** Neither filtered exact concepts nor doc-anchored concept transport beats plain BM25 on FiQA, which is exactly where concepts were supposed to help most.

Updated conclusion: the concept failure is **partly a usage problem** (generic scaffolding phrases like `how do`, `risk of`, `associated with` were being rewarded), but **not only** a usage problem. Even after suppressing those, the concept leg still behaves too much like lexical phrase reuse and does not add enough paraphrase reach to justify its cost.

## Graph transport follow-up

The next hypothesis was that the phrase near-miss needed a stronger topology, not another direct phrase bonus. A pool-restricted personalized-PageRank reranker was added behind `--graph-only`:

- one query node
- query-adjacent phrase nodes
- BM25 top-K document nodes
- optional doc-doc overlap edges inside the pool

Rows tested:

- `bm25_graph_phrase_ppr_k100_d0.85_a0.8`
- `bm25_graph_phrase_docdoc_ppr_k100_d0.85_a0.8`
- `bm25_graph_phrase_ppr_k300_d0.85_a0.8`
- `bm25_graph_phrase_docdoc_ppr_k100_d0.70_a0.8`

| Corpus | BM25 | Best graph row | Δ | Best prior phrase row | Δ |
|--------|-----:|---------------:|--:|----------------------:|--:|
| scifact | 0.6188 | 0.6125 (`phrase_ppr k300 d0.85`) | -0.0063 | 0.6115 (`phrase k100 a0.8`) | -0.0073 |
| nfcorpus | 0.2521 | 0.2512 (`phrase_ppr k100 d0.85`) | -0.0009 | 0.2531 (`ordered k100 a0.8`) | +0.0010 |
| fiqa | 0.2053 | 0.2072 (`phrase_docdoc_ppr k100 d0.70`) | +0.0019 | 0.2101 (`phrase k100 a0.8`) | +0.0048 |

### Verdict

**Graph transport also disproves in its first form.**

What changed:

1. Adding doc-doc edges and lower damping makes FiQA slightly safer than pure phrase PPR (`0.2072` vs `0.2024`), so topology is not completely inert.
2. But the graph never beats the earlier direct phrase row, and it never reaches the predeclared `+0.005` promote bar on any corpus.

What this says about the theory:

- The current graph is still too close to **adjacent-phrase reuse**.
- Diffusion over query bigrams plus pool-local doc overlap mostly spreads the same local lexical evidence; it does not create new support paths strong enough to offset noise on scifact.
- The remaining promising graph move is a **document-anchored or fragment-anchored walk** (`query -> docs -> phrases/fragments -> docs`), where first-pass BM25 docs seed the walk directly and sentence/passage structure can mediate support. That is closer to hierarchical transport than the current query-bigram PPR.

## Cluster topology follow-up

The next topology pass replaced query-bigram nodes with **document-anchored fragment clusters**. Each document contributes its top TextRank fragments; each fragment becomes a sparse high-IDF signature; fragments inside the BM25 pool are greedily clustered by weighted overlap; documents are rescored by cluster query-cover plus fragment containment.

Rows tested:

- `bm25_cluster_cover_k100_o0.35_q0.35_a0.8`
- `bm25_cluster_cover_k100_o0.50_q0.35_a0.8`
- `bm25_cluster_cover_k100_o0.35_q0.20_a0.8`
- `bm25_cluster_cover_k300_o0.35_q0.35_a0.8`

| Corpus | BM25 | Best cluster row | Δ |
|--------|-----:|-----------------:|--:|
| scifact | 0.6188 | 0.6155 (`k100 o0.35 q0.20`) | -0.0033 |
| nfcorpus | 0.2521 | 0.2494 (`k100 o0.50 q0.35`) | -0.0027 |
| fiqa | 0.2053 | 0.2025 (`k100 o0.50 q0.35`) | -0.0028 |

### Verdict

**Cluster topology also disproves in the first pass.**

What this adds:

1. The damage is more uniform than the earlier phrase graph: the best scifact row is only `-0.0033`, but FiQA and NFCorpus both move negative too.
2. Tightening the overlap threshold (`o0.50`) helps slightly on FiQA versus the looser cluster row, but it still stays below BM25.

What this says about the failure mode:

- The current cluster cover behaves like a **salience/intersection filter**, not like a real semantic neighborhood.
- Using sparse high-IDF fragment signatures plus greedy overlap clustering mostly groups fragments that already share rare lexical anchors; it does not bridge alternate lexical realizations.
- So the topology got better, but the semantic substrate did not. The missing piece is not “more clustering,” it is a better **intra-pool relation** than raw lexical overlap.

Updated next move:

- keep the document-/fragment-anchored shape;
- replace raw sparse-overlap clustering with a richer relation, e.g.:
  - query-seeded `doc -> fragment -> fragment -> doc` walks,
  - fragment nodes connected by shared concept/phrase neighborhoods rather than direct token overlap,
  - or a lightweight vectorized fragment signature if portability constraints allow an in-corpus, training-free embedding.

## Semantic fragment graph follow-up

The next pass kept the document-/fragment-anchored shape but replaced lexical overlap with **in-corpus PMI fragment vectors**. Each document contributes its top TextRank fragments; fragments are encoded with training-free PMI embeddings learned from the fixture corpus; query-seeded mass diffuses through a fragment-to-fragment semantic graph inside the BM25 pool.

Rows tested:

- `bm25_fragment_graph_k100_d0.85_q0.20_f0.35_a0.8`
- `bm25_fragment_graph_k100_d0.70_q0.20_f0.35_a0.8`
- `bm25_fragment_graph_k100_d0.85_q0.10_f0.20_a0.8`
- `bm25_fragment_graph_k300_d0.85_q0.20_f0.35_a0.8`

| Corpus | BM25 | Best fragment row | Δ |
|--------|-----:|------------------:|--:|
| scifact | 0.6188 | 0.6191 (`k300 d0.85 q0.20 f0.35`) | +0.0003 |
| nfcorpus | 0.2521 | 0.2525 (`k100 d0.85 q0.10 f0.20`) | +0.0004 |
| fiqa | 0.2053 | 0.2055 (`k100 d0.70 q0.20 f0.35`) | +0.0002 |

### Verdict

**Semantic fragment graphs are much safer, but still effectively inert.**

What changed:

1. The large negative regressions from lexical phrase/cluster topology mostly disappear.
2. The best rows are now tiny positive nudges (`+0.0002` to `+0.0004`) instead of material regressions.

What this says about the theory:

- The **semantic substrate matters**: replacing lexical overlap with PMI fragment similarity removes most of the harm.
- But PMI fragment neighborhoods are still too weak to create a meaningful reranking signal on top of BM25.
- So the problem is no longer just topology. The walk is now safer, but the fragment relation lacks enough discriminative semantic separation to produce a real top-10 lift.

Updated next move:

- keep the semantic fragment graph shape;
- strengthen the state space rather than the diffusion math:
  - sentence/passage fragments beyond top-TextRank only,
  - concept-mediated fragment edges on top of PMI similarity,
  - or a stronger in-corpus fragment representation than summed PMI rows.

## Hybrid fragment graph follow-up

The next pass tested whether the safe-but-inert PMI fragment graph needed a light lexical scaffold rather than a new topology. The fragment graph was extended with:

- up to 6 TextRank fragments per document,
- lexical bridge mass in query-to-fragment seeding,
- lexical bridge mass on fragment-to-fragment edges via weighted signature overlap.

Rows tested:

- `bm25_fragment_hybrid_k100_t6_d0.70_q0.20_f0.35_b0.20_a0.8`
- `bm25_fragment_hybrid_k100_t6_d0.85_q0.10_f0.20_b0.35_a0.8`
- `bm25_fragment_hybrid_k300_t6_d0.70_q0.20_f0.35_b0.20_a0.8`

| Corpus | BM25 | Best hybrid row | Δ |
|--------|-----:|----------------:|--:|
| scifact | 0.6188 | 0.6157 (`k100 t6 d0.85 q0.10 f0.20 b0.35`) | -0.0031 |
| nfcorpus | 0.2521 | 0.2546 (`k100 t6 d0.85 q0.10 f0.20 b0.35`) | +0.0025 |
| fiqa | 0.2053 | 0.2034 (`k100 t6 d0.70 q0.20 f0.35 b0.20`) | -0.0019 |

### Verdict

**The lexical scaffold reintroduces the old failure mode.**

What changed:

1. NFCorpus improves the most so far for fragment graphs (`+0.0025`), which means the hybrid graph is not empty.
2. But scifact and fiqa both move negative again, so the scaffold trades safety for corpus-local lexical gain.

What this says about the theory:

- PMI-only fragment similarity was weak but stable.
- Adding lexical bridge edges makes the graph sharper, but also pulls it back toward the same corpus-bound lexical bias that hurt phrase transport and overlap clustering.
- So the next missing piece is probably **better semantic fragmentation / better fragment representations**, not more lexical reinforcement.

Updated next move:

- preserve the PMI fragment-graph backbone;
- strengthen fragment state without lexical backsliding:
  - sentence/passage coverage beyond top-TextRank selection,
  - richer in-corpus fragment encoders,
  - or concept-style bridges only when they are semantically filtered rather than raw overlap-driven.

## Geometric fragment graph follow-up

After reviewing the literature on local metric similarity, manifold-style ranking, graph diffusion in vector spaces, and attention-like soft neighborhood weighting, the next pass tested a **query-centered geometric neighborhood** over PMI fragment vectors.

Implementation:

- whiten fragment vectors by the local BM25-pool covariance diagonal (a cheap local Mahalanobis approximation),
- seed fragment mass with a softmax over query-to-fragment similarity,
- propagate through a local fragment kNN graph using softmax edge weights.

Rows tested:

- `bm25_fragment_geom_k100_t4_s8_k8_p2_a0.8`
- `bm25_fragment_geom_k100_t4_s4_k16_p2_a0.8`
- `bm25_fragment_geom_k100_t6_s8_k8_p3_a0.8`

| Corpus | BM25 | Best geometric row | Δ |
|--------|-----:|-------------------:|--:|
| scifact | 0.6188 | 0.6140 (`k100 t4 s8 k8 p2`) | -0.0048 |
| nfcorpus | 0.2521 | 0.2548 (`k100 t4 s8 k8 p2`) | +0.0027 |
| fiqa | 0.2053 | 0.2053 (`k100 t6 s8 k8 p3`) | +0.0000 |

### Verdict

**Geometric neighborhoods produce a real corpus-specific signal, but still fail the cross-corpus gate.**

What changed:

1. NFCorpus gets the best gain so far for a non-lexicalized fragment graph (`+0.0027`).
2. FiQA stays roughly neutral instead of collapsing.
3. Scifact still regresses materially, so the geometric neighborhood remains corpus-sensitive.

What this says about the theory:

- The literature intuition was directionally right: **query-centered soft geometry is stronger than plain PMI diffusion**.
- But the local geometric kernel is still not robust across corpora; on scifact it likely amplifies dense neighborhood structure that does not correspond to relevance.
- So cluster/graph shape is not fully exhausted, but any next step should adapt the geometry by corpus/query regime rather than assuming one neighborhood shape fits all three.

Updated next move:

- keep the geometric fragment backbone as the strongest non-lexical topology so far on nfcorpus;
- add **query- or corpus-adaptive neighborhood sharpness** rather than a fixed geometric kernel;
- or gate geometry to the regimes where it behaves more like nfcorpus and less like scifact.

## Adaptive geometric fragment graph follow-up

The next pass tested the obvious follow-on from the fixed-geometry result:
**adapt the geometric kernel per query** instead of using one global setting.

Implementation:

- start from the same PMI fragment backbone and local whitening as the fixed
  geometric pass,
- estimate query regime from cheap lexical signals already available in-tree:
  - query `avg_idf`,
  - BM25 top-pool score decay,
- map that regime to query-specific geometry parameters:
  - BM25-vs-geometry blend weight,
  - attention sharpness,
  - kNN width,
  - propagation depth.

Rows tested:

- `bm25_fragment_geom_adapt_k100_t4_a0.70_0.98_s4_10_k4_16_p1_3`
- `bm25_fragment_geom_adapt_k100_t6_a0.65_0.95_s3_8_k8_20_p2_4`

| Corpus | BM25 | Best fixed geometry | Δ | Best adaptive geometry | Δ |
|--------|-----:|--------------------:|--:|-----------------------:|--:|
| scifact | 0.6188 | 0.6140 (`k100 t4 s8 k8 p2`) | -0.0048 | 0.6078 (`t4 a0.70_0.98 s4_10 k4_16 p1_3`) | -0.0110 |
| nfcorpus | 0.2521 | 0.2548 (`k100 t4 s8 k8 p2`) | +0.0027 | 0.2557 (`t6 a0.65_0.95 s3_8 k8_20 p2_4`) | +0.0036 |
| fiqa | 0.2053 | 0.2053 (`k100 t6 s8 k8 p3`) | +0.0000 | 0.2024 (`t4 a0.70_0.98 s4_10 k4_16 p1_3`) | -0.0029 |

### Verdict

**This first adaptive/gated geometry attempt disproves.**

What changed:

1. NFCorpus improves again and sets a new best fragment-geometry gain (`+0.0036`).
2. But scifact gets materially worse than the fixed-geometry pass, and fiqa
   turns clearly negative.
3. The adaptive heuristics also add runtime cost without buying cross-corpus
   safety.

What this says about the theory:

- The broad direction was right — **geometry strength really is regime-sensitive**.
- But the chosen regime signals (`avg_idf` + BM25 score decay) are not aligned
  enough with the fragment-geometry failure mode to gate it safely.
- In practice, this heuristic mostly sharpened the NFCorpus win while
  over-correcting on the corpora where geometry was already fragile.

Updated next move:

- stop treating **simple BM25-side gating** as the likely fix for fragment geometry;
- if this family continues, prefer:
  - richer fragment representations,
  - geometry-specific regime signals,
  - or explicit structured-document fixtures where fragment neighborhoods are
    more meaningful than on short body-only BEIR docs.

## Rich fragment vs geometry-signal ablation

The next pass tested the two strongest remaining hypotheses side by side:

1. **richer fragment representations** — keep the geometry fixed, but enrich the
   fragment state with multi-scale anchors:
   - top TextRank sentence fragments,
   - a centroid fragment built from those sentence vectors,
   - a whole-document anchor fragment;
2. **geometry-specific regime signals** — keep the baseline fragments, but adapt
   the geometry from fragment-space evidence rather than BM25-side lexical
   signals:
   - query-to-fragment similarity peak,
   - query-to-fragment top-gap,
   - query-attention entropy;
3. **combined** — richer fragment state plus geometry-native control.

Rows tested:

- `bm25_fragment_geom_rich_k100_t8_s8_k8_p2_a0.8`
- `bm25_fragment_geom_gsig_k100_t4_a0.65_0.98_s3_10_k4_16_p1_3`
- `bm25_fragment_geom_rich_gsig_k100_t8_a0.65_0.98_s3_10_k4_16_p1_3`

| Corpus | BM25 | Best fixed geometry | Δ | Best new ablation row | Δ |
|--------|-----:|--------------------:|--:|----------------------:|--:|
| scifact | 0.6188 | 0.6140 (`k100 t4 s8 k8 p2`) | -0.0048 | 0.6128 (`rich_gsig`) | -0.0060 |
| nfcorpus | 0.2521 | 0.2548 (`k100 t4 s8 k8 p2`) | +0.0027 | 0.2578 (`rich`) | +0.0057 |
| fiqa | 0.2053 | 0.2053 (`k100 t6 s8 k8 p3`) | +0.0000 | 0.2060 (`rich_gsig`) | +0.0007 |

### Verdict

**Richer fragment representations help; geometry-native control does not rescue cross-corpus robustness yet.**

What changed:

1. The **rich fragment** row sets the best fragment-geometry result so far on
   NFCorpus (`+0.0057`), finally clearing the original `+0.005` lift bar on at
   least one corpus.
2. The rich rows also produce the first positive FiQA fragment-geometry result
   above the earlier neutral floor (`0.2060` best, `+0.0007`).
3. Geometry-specific gating by itself (`gsig`) is weaker than the fixed
   geometry baseline on all three corpora.
4. The combined row is a bit safer than rich-only on scifact, but still
   regresses materially and does not beat the fixed-geometry scifact row.

What this says about the theory:

- **State quality matters more than control quality right now.** The biggest
  gain came from giving the graph better fragment objects, not from changing
  the gating logic.
- The fragment geometry is still **corpus-sensitive**, but the richer
  multi-scale state shifts the tradeoff in the right direction on the corpora
  where fixed geometry had signal.
- Geometry-native confidence signals are directionally cleaner than lexical
  gating, but they are not yet calibrated enough to solve the scifact failure
  mode.

Updated next move:

- make **richer fragment representations** the leading continuation of this
  family;
- treat geometry-signal gating as a secondary control path until it can help
  scifact instead of only modulating already-positive regimes;
- if we continue after this, prefer:
  - stronger multi-scale fragment construction,
  - fragment dedup / coverage controls,
  - or structured-document fixtures where richer fragment anchors should pay
    off more clearly than on short body-only BEIR docs.

## Rich-fragment coverage / dedup follow-up

The next pass tested whether the rich-fragment gains were partly coming from
**useful multi-scale coverage** and partly being dragged down by **redundant
anchors**. The builder was changed to greedily suppress fragments whose sparse
signature overlapped too strongly with already-kept fragments:

- sentence fragments kept only when overlap with prior kept fragments stayed
  below `0.60`,
- centroid / whole-doc anchor fragments kept only when overlap stayed below
  `0.80`.

Rows tested:

- `bm25_fragment_geom_richcov_k100_t8_o0.60_0.80_s8_k8_p2_a0.8`
- `bm25_fragment_geom_richcov_gsig_k100_t8_o0.60_0.80_a0.65_0.98_s3_10_k4_16_p1_3`

| Corpus | BM25 | Best prior rich row | Δ | Best coverage-controlled row | Δ |
|--------|-----:|--------------------:|--:|-----------------------------:|--:|
| scifact | 0.6188 | 0.6128 (`rich_gsig`) | -0.0060 | 0.6161 (`richcov`) | -0.0027 |
| nfcorpus | 0.2521 | 0.2578 (`rich`) | +0.0057 | 0.2550 (`richcov`) | +0.0029 |
| fiqa | 0.2053 | 0.2060 (`rich_gsig`) | +0.0007 | 0.2064 (`richcov_gsig`) | +0.0011 |

### Verdict

**Coverage/dedup controls improve robustness, but trade away part of the rich-fragment gain.**

What changed:

1. Scifact improves sharply relative to the earlier rich rows (`-0.0060 -> -0.0027`).
2. FiQA gets its best fragment-geometry result so far (`+0.0011`).
3. NFCorpus remains positive, but gives back roughly half of the rich-fragment
   lift (`+0.0057 -> +0.0029`).

What this says about the theory:

- The rich-fragment win was not just "more fragments is better" — **coverage and
  redundancy control matter**.
- Redundant anchors appear to amplify the bad regime on scifact.
- But overly aggressive dedup also suppresses some of the productive dense
  neighborhood structure on nfcorpus.

Updated next move:

- keep **rich fragment state** as the main direction;
- treat **coverage / dedup** as the best current safety lever for scifact-like
  corpora;
- next continuation should tune that tradeoff more precisely, e.g.:
  - separate sentence-fragment vs anchor-fragment budgets,
  - confidence-weighted anchor retention,
  - or corpus/query-adaptive dedup thresholds driven by fragment-space signals
    rather than lexical heuristics.

## Budgeted rich-fragment follow-up

The next pass tested a stricter control surface on top of the coverage rows:

- keep at most **4 sentence fragments**,
- keep at most **1 anchor fragment**,
- require anchor novelty of at least `0.15`,
- retain the same sentence/anchor overlap caps (`0.60 / 0.80`).

Rows tested:

- `bm25_fragment_geom_richbud_k100_t6_s4_a1_n0.15_o0.60_0.80_s8_k8_p2_a0.8`
- `bm25_fragment_geom_richbud_gsig_k100_t6_s4_a1_n0.15_o0.60_0.80_a0.65_0.98_s3_10_k4_16_p1_3`

| Corpus | BM25 | Best prior coverage row | Δ | Best budgeted row | Δ |
|--------|-----:|------------------------:|--:|------------------:|--:|
| scifact | 0.6188 | 0.6161 (`richcov`) | -0.0027 | 0.6113 (`richbud`) | -0.0075 |
| nfcorpus | 0.2521 | 0.2550 (`richcov`) | +0.0029 | 0.2533 (`richbud_gsig`) | +0.0012 |
| fiqa | 0.2053 | 0.2064 (`richcov_gsig`) | +0.0011 | 0.2042 (`richbud_gsig`) | -0.0011 |

### Verdict

**This first sentence/anchor budgeting pass disproves.**

What changed:

1. It is noticeably faster than the richer coverage rows because fewer fragment
   nodes survive into the geometry step.
2. But the quality tradeoff is wrong: scifact gets worse again, nfcorpus loses
   most of the retained upside, and fiqa turns negative.

What this says about the theory:

- The current richcov row was not simply "too many fragments." A hard budget of
  `4 sentence + 1 anchor` throws away useful state rather than just removing
  noise.
- The anchor problem is likely subtler than a global count cap; **which anchor**
  survives appears more important than just how many.
- So the next useful control surface is probably **confidence-weighted anchor
  retention** or softer, asymmetric budgets rather than a hard global cap.

Updated next move:

- keep the **richcov** rows as the active best frontier;
- if continuing, prefer:
  - confidence-weighted anchor retention,
  - softer anchor-only budgets,
  - or adaptive dedup thresholds before revisiting sentence-count caps.

## MMR-style rich-fragment selection follow-up

The next pass replaced the hard sentence/anchor cap with a **soft MMR-style
selector** over the same rich-fragment candidate set:

- start from the rich builder's top TextRank sentence fragments plus centroid and
  whole-doc anchors,
- assign each candidate a simple salience prior (earlier sentences > centroid >
  whole-doc anchor),
- greedily keep the fragment with the best `salience - lambda * redundancy`
  score,
- apply the same sentence / anchor overlap caps (`0.60 / 0.80`) as soft
  penalties rather than a hard count budget,
- stop when the best remaining candidate falls below a per-type minimum score.

Rows tested:

- `bm25_fragment_geom_richmmr_k100_t8_l0.35_m0.30_0.15_o0.60_0.80_s8_k8_p2_a0.8`
- `bm25_fragment_geom_richmmr_k100_t8_l0.50_m0.24_0.12_o0.60_0.80_s8_k8_p2_a0.8`

| Corpus | BM25 | Best prior coverage row | Δ | Best MMR row | Δ |
|--------|-----:|------------------------:|--:|-------------:|--:|
| scifact | 0.6188 | 0.6161 (`richcov`) | -0.0027 | 0.6141 (`richmmr l0.50`) | -0.0047 |
| nfcorpus | 0.2521 | 0.2550 (`richcov`) | +0.0029 | 0.2585 (`richmmr l0.35`) | +0.0064 |
| fiqa | 0.2053 | 0.2055 (`richcov`) | +0.0002 | 0.2079 (`richmmr l0.35`) | +0.0026 |

### Verdict

**This is the strongest soft-control near-miss so far, but it still does not
clear the scifact safety bar.**

What changed:

1. The balanced MMR row (`lambda=0.35`) is the first post-`richcov` control
   surface that clearly improves **both** nfcorpus and fiqa at once.
2. The novelty-heavier row (`lambda=0.50`) recovers some scifact safety relative
   to the balanced row, but it gives back part of the nfcorpus / fiqa upside.
3. Both rows still outperform the earlier hard budget on every corpus, which is
   direct evidence that **soft novelty-aware selection is better than blunt
   count caps** for this fragment family.

What this says about the theory:

- The literature-backed intuition was right: the missing control surface is more
  like **MMR / soft redundancy suppression** than a global fragment budget.
- Rich-fragment state still has upside left in it: the balanced MMR row pushes
  nfcorpus to `+0.0064` and fiqa to `+0.0026` over BM25 without reintroducing
  the severe regressions from earlier lexical graph variants.
- But scifact still prefers the stricter `richcov` safety profile, so the
  remaining problem is likely **asymmetric control**, especially around anchors
  and late-selected fragments, rather than one global MMR setting.

Updated next move:

- keep **richcov** as the safest active frontier and **richmmr** as the best
  upside-seeking soft selector;
- if continuing, prefer:
  - anchor-specific MMR penalties or anchor-only minimum-score tightening,
  - sentence-MMR plus stricter anchor retention,
  - or a two-stage selector that keeps `richcov` sentence safety but lets MMR
    choose among the remaining anchor candidates.

---

## Asymmetric two-stage anchor control

*Negative result — detailed in [asymmetric_anchor_results.md](asymmetric_anchor_results.md).*

The follow-up implemented exactly the two-stage selector suggested above: `richcov`-style hard overlap caps for sentence fragments, plus MMR selection for anchor fragments (centroid + whole-doc) with tunable novelty penalty. Two parameter settings were tested (`richasym` with lambda=0.35 and `richasym2` with lambda=0.50).

| Corpus   | BM25   | richcov | richmmr | richasym | richasym2 |
|----------|--------|---------|---------|----------|-----------|
| scifact  | 0.6188 | 0.6161  | 0.6134  | 0.6154   | 0.6150    |
| nfcorpus | 0.2521 | 0.2550  | 0.2585  | 0.2539   | —         |

Verdict: strictly worse than `richcov` on both corpora. Anchor fragments are too correlated with sentence fragments for MMR to find novel signal. The hard overlap caps on sentences are already the optimal safety lever. This thread is closed.

## Persistent Homology Scale Selection (PHSS)

*Mixed positive — detailed in [phss_results.md](phss_results.md).*

The next geometric/topological lever replaced the fixed `knn=8` with a data-driven scale from 0D persistent homology on the fragment similarity graph. Three criteria (LargestGap, MaxPersistence, Elbow) were tested on both basic and `richcov` fragment sets across BEIR-3.

| Corpus   | Baseline geom (knn=8) | PHSS gap (t=4) | PHSS richcov gap |
|----------|----------------------:|---------------:|-----------------:|
| scifact  | 0.6140                | **0.6188**     | 0.6167           |
| nfcorpus | 0.2548                | 0.2531         | 0.2542           |
| fiqa     | 0.2028                | **0.2065**     | **0.2089**       |

Verdict: PHSS is a real lever. `LargestGap` is the strongest criterion. It recovers the scifact regression, gives the best FiQA fragment-geometry score so far, and regresses NFCorpus slightly. Follow-up rows show that adaptive PHSS is a useful latency/quality knob, but it does not replace the full PHSS frontier, and `richmmr + PHSS` does not beat either the best `richmmr` or `richcov + PHSS` rows. Profiling shows the dominant PHSS cost is scale selection itself, not diffusion. The next step is larger-pool PHSS or a cheaper scale-selection approximation.
