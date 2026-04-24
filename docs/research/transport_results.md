# Phrase/document transport results

Goal: move beyond unigram neighbor expansion by scoring a **sparse phrase transport leg** inside a BM25 top-K pool. The transport leg is built from existing simeon primitives:

- ordered and unordered SDM bigram legs (`score_sdm()` with `lambda_unigram=0`)
- optional PMI-mined concept postings (`ConceptIndex::score()`)
- pool-restricted z-score fusion with BM25

This is a training-free proxy for stronger transport structure: query mass
flows through phrase nodes rather than only through isolated term neighbors.

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

Key points:

- **Phrase transport is materially better than unigram soft matching.** The
  best row lifts FiQA by `+0.0048`, closer to the `+0.005` bar than any prior
  portable transport attempt.
- **Concept nodes still hurt.** Adding mined concepts regresses all three
  corpora relative to phrase-only transport.
- **The cross-corpus gate still fails.** The best FiQA row regresses scifact by
  `-0.0073`, and diffuse-mass gating removes most of the FiQA gain.

## Interpretation

- flat token transport is too weak
- phrase structure adds real signal
- phrase transport is still too local and corpus-bound

That points to a higher-level transport graph rather than more concept weight or
another scalar blend.

## Concept rework probes

After inspecting the concept failures, two reworks were tested:

1. **doc-anchored concept transport** — seed concept mass from BM25 top-K docs, then project it back into the pool.
2. **filtered exact concepts** — only allow concepts when both concept terms are contentful query terms; bound PMI before fusion.

| Corpus | BM25 | raw concepts | filtered concepts | doc-anchored concepts |
|--------|-----:|-------------:|------------------:|----------------------:|
| scifact | 0.6188 | 0.6021 | 0.6104 | 0.6127 |
| nfcorpus | 0.2521 | 0.2517 | 0.2520 | 0.2520 |
| fiqa | 0.2053 | 0.2040 | 0.2028 | 0.2025 |

Takeaway:

- filtering generic concepts helps scifact safety
- the deeper problem remains: even filtered or doc-anchored concepts do not beat
  BM25 on FiQA

So concept failure is partly a usage problem, but mostly a weak-signal problem.

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

Takeaway:

- topology is not inert, but the graph never beats the earlier direct phrase row
- diffusion over query bigrams plus pool-local doc overlap mostly spreads the
  same lexical evidence
- the next plausible move remains a document- or fragment-anchored walk

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

Takeaway:

- the cluster cover behaves like a salience/intersection filter, not a semantic neighborhood
- overlap clustering mostly groups fragments that already share rare lexical anchors
- the missing piece is a better intra-pool relation than raw lexical overlap

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

Takeaway:

- replacing lexical overlap with PMI similarity removes most of the harm
- PMI fragment neighborhoods are still too weak to create a real reranking lift
- the next lever is stronger fragment state, not more diffusion math

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

Takeaway:

- the hybrid graph is not empty: nfcorpus improves the most so far for a
  fragment graph (`+0.0025`)
- but scifact and fiqa regress again, so the lexical scaffold trades safety for
  corpus-local lexical gain
- the next missing piece is better fragment state, not more lexical reinforcement

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

Takeaway:

- query-centered soft geometry is stronger than plain PMI diffusion
- the local geometric kernel is still corpus-sensitive
- if this family continues, the next lever is adaptive sharpness or gating, not
  another fixed neighborhood shape

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

Takeaway:

- geometry strength really is regime-sensitive
- the chosen BM25-side signals were not aligned enough with the fragment-geometry
  failure mode to gate it safely
- richer fragment representations remain the cleaner next direction

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

Takeaway:

- **state quality matters more than control quality**
- richer fragment representations are the first fragment-geometry branch to
  clear the +0.005 bar on nfcorpus
- geometry-native control remains secondary until it can help scifact rather
  than only modulate already-positive regimes

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

Takeaway:

- the rich-fragment win was not just "more fragments is better"; redundancy
  control matters
- coverage/dedup is the best current safety lever for scifact-like corpora
- but overly aggressive dedup suppresses part of the nfcorpus upside

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

Takeaway:

- hard budgets are faster but not better
- `4 sentence + 1 anchor` throws away useful state rather than just noise
- the next control surface would need softer or asymmetric anchor retention

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

Takeaway:

- soft novelty-aware selection is better than blunt count caps
- `richmmr` is the best upside-seeking soft selector so far
- scifact still prefers the stricter `richcov` safety profile, so the remaining
  problem is asymmetric control rather than one global MMR setting

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
