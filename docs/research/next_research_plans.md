# Four research plans to address the BEIR-3 training-free ceiling

## Context

After the PHSS and follow-up ablation cycle, `training_free_saturation.md`
lands at a simple conclusion: BEIR-3 has no remaining within-scope
universal-recipe levers in the training-free regime.

The remaining bottlenecks from `literature_synthesis.md` are:

1. **Multi-fragment per-doc averaging** washes per-fragment signal
   (mechanism that killed Phase A 1D triangle, MaxSim only got
   single-fold lift that disconfirmed on dev).
2. **BM25-pool determinism caps R@100** (geometry rerank only
   re-orders; can't add candidates).
3. **Per-corpus optimal recipes** (recurring meta-finding;
   per-corpus α tuning disconfirmed by cross-fold).
4. **Dictionary/Structural axis** (closed on BEIR-3 for body-only
   docs; open for structured-doc corpora).

Production frontier locked at `phssapprox_k100_t8_richcov_gap`:
scifact 0.6188 (= BM25), nfcorpus 0.2544 (+0.0023), fiqa 0.2089
(+0.0036), ~120 QPS.

These four plans avoid already-disproved directions, stay training-free, and
carry explicit cross-fold validate/disprove gates.

---

## Plan 1 — trec-covid as 4th BEIR corpus (long-doc regime test)

### Hypothesis

The BEIR-3 fixture set lacks a true long-doc corpus. Avg_dl across the three is
scifact 214.6, nfcorpus 233.8, fiqa 132.9, so fiqa is actually the shortest.
trec-covid (avg_dl ~290) would let several already-implemented but
corpus-mismatched levers run in the regime they were designed for:

- LTD α<1 helped R@100 on long-doc scifact / nfcorpus
  (+0.0049 / +0.0029); trec-covid would test the prediction at the
  actual long end.
- PHSS pool=500 richcov on nfcorpus produced +0.0122 R@100 (the only
  positive Phase C cell, but at 5 QPS); trec-covid may show the lift
  is a long-doc property, not a nfcorpus quirk.
- Topology track winners (MaxSim on focused-content scifact) and
  losers (MaxSim on Q&A fiqa) suggest a 4th corpus would test the
  generality of the per-corpus aggregator finding.

### Validate / disprove

- **Validate** if ≥2 already-implemented variants (LTD, PHSS pool=500,
  MaxSim, RM3) reach their predicted lifts on trec-covid AND don't
  regress more than ±0.005 nDCG@10 on the existing 3 corpora.
  Outcome: per-corpus router can ship the long-doc-specific recipes.
- **Disprove** if trec-covid behaves like an averaging of the existing
  3 (corpus-bound finding extends but doesn't surface new winners).
  Outcome: declare 4-corpus training-free saturation; recommend
  off-fixture work.

### Cost

~1-2 days fixture engineering + 1 day bench:

- **Corpus**: TREC-COVID round 5 from BEIR (171k docs, ~8M tokens).
- **Queries + qrels**: BEIR ships these directly.
- **Reference.bin**: encode 171k docs with sentence-transformers
  MiniLM-L6 (one-shot Python preprocess).
- **Dev/test split**: BEIR provides round-1 / round-2 / round-3 as
  natural folds; pick one as dev, one as test.
- **Bench**: re-run all existing recipes — they Just Work since the
  geometry pipeline is corpus-agnostic.

### Risk

- graded qrels need a compatibility check before benchmarking
- 171k docs is larger than the current fixtures, but still manageable

### Bottleneck addressed

**Bottleneck 3** directly, and indirectly Bottlenecks 1 and 2 in a true
long-doc regime.

---

## Plan 2 — Corpus-as-self-KB: offline doc-doc graph + query-time reflection

*Revised 2026-04-24: replaced the original external-KB plan with a
corpus-self-tuning version. External KBs add license, freshness, and scope
problems that the corpus itself can avoid.*

### Hypothesis

Every existing lever computes structure inside a per-query top-K pool. **No
prior simeon experiment has computed corpus-global structure once at index time
and reused it as a query-time prior.** That is the genuinely unexplored axis.

The "self-KB": for each doc, precompute the top-K most-similar docs
in the corpus (by BM25-over-self or PMI-fragment cosine). At query
time, expand the BM25 candidate pool by unioning in the precomputed
neighbors of the top-K BM25 hits. The "reflection" step is a topology-
confidence gate (re-using PHSS persistence) that decides per-query
whether to apply the expansion.

### Why this is genuinely new

- **vs RM3**: term expansion becomes doc-level expansion.
- **vs PHSS**: per-query topology becomes corpus-precomputed topology.
- **vs concept mining / AC fields**: this is candidate-set expansion, not a new scoring leg.
- **vs PHSS pool=500**: expansion is precomputed offline instead of paid per query.

### Implementation

Three components:

**Component A — Offline doc-doc graph** (one-shot at index build).
For each doc, compute top-K (≈20) most similar docs. Three similarity
candidates in priority order:

1. **BM25-over-self**: D as the query, BM25-retrieve corpus.
   O(N²) but reuses shipped `Bm25Index::score`. Tractable for 5k–60k.
2. **PMI-fragment cosine**: doc encoded as average of richcov
   fragments via existing `Encoder` + PMI. Reuses
   `build_doc_semantic_fragments_rich_covered()`.
3. **Hybrid**: union of (1) and (2).

Persisted as sparse adjacency: ~400 KB (scifact) to ~4.5 MB (fiqa).

**Component B — Query-time reflection (candidate-set expansion).**
First-pass BM25 top-K → fetch precomputed neighbors of each top-K
doc → union to expanded set (typically 200-400 unique docs after
dedup) → re-rank with the existing geometry pipeline.

**Component C — Topology confidence gate.** Use PHSS LargestGap
persistence on first-pass BM25 top-K. When persistence is high (top-K
has coherent cluster structure), expand. When low, return BM25
ranking directly (avoids noise expansion on diffuse pools). Sweep
per-corpus threshold on dev fold.

### Phasing

- **Phase A (cheap probe, ~1-2 days)**: Component A with BM25-over-
  self + Component B (always-expand baseline). If R@100 lifts ≥0.005
  on ≥1 corpus, advance.
- **Phase B (~1 day)**: Add Component C gate; sweep persistence
  threshold on dev fold; cross-fold validate per
  `feedback_cross_fold_validation` rule.
- **Phase C (~1 day, gated)**: Try Component A variants 2 and 3.
  Promote per-corpus best to production.

### Validate / disprove

- **Validate**: ≥0.005 R@100 lift on ≥2/3 corpora at <2× current
  query latency, cross-fold consistent.
- **Disprove**: R@100 lift <0.005 on ≥2 corpora OR nDCG@10 regression
  >0.005 on any corpus → BM25 top-K and corpus-mined neighborhoods
  are too correlated; remaining moves are off-fixture (Plan 1
  trec-covid) or off-training-free (dense retrieval).

### Cost

~3-5 days across 3 phases:

- ~150 LOC: new `simeon/doc_doc_graph.{hpp,cpp}` (offline builder +
  sparse adjacency persistence).
- ~50 LOC: extend `FragmentGeometryConfig` + modify BM25-pool gather
  step in `fragment_geometry.cpp` to union with neighbor expansion
  when gate passes.
- ~30 LOC bench wiring (3-6 new recipes).
- Cross-fold bench on existing 3 corpora.

### Risk

- BM25-over-self neighbors may simply be too correlated with BM25 retrieval
- the PHSS gate may fail to discriminate
- offline graph build is still a real index-time cost on larger corpora

### Bottleneck addressed

**Bottleneck 2** (BM25-pool R@100 ceiling) directly via candidate-set
expansion, addressed at the offline-precomputation layer rather than
the per-query rerank layer. Indirectly addresses Bottleneck 4
(Dictionary axis) by surfacing corpus-mined doc-level structure as
a curation-equivalent signal.

---

## Plan 3 — Pool diversification via RRF fusion of BM25 variants

### Hypothesis

Bottleneck 2 is real and unaddressed by any geometry rerank. The current pool is
a single BM25 variant top-K. Different BM25 variants retrieve different doc
sets because their term weighting and length normalization differ.

RRF over multiple BM25-variant pools produces the **union** of those top-K
sets, so it guarantees more candidate diversity than any single variant.

This is a **first-pass** intervention that lifts the candidate ceiling all
current rerankers operate within.

### Implementation

1. Build pools at K=100 per BM25 variant: `{Atire, BM25+, BM25L,
   DLH13, PL2, DPH, SAB-smooth}`. Already all shipped.
2. RRF-fuse the 7 pools → diverse candidate set of size ≤ 7 × 100
   = 700 (with overlap, typically ~250-400 unique docs).
3. Re-rank the diverse pool with the production
   `phssapprox_k100_t8_richcov_gap` recipe (fragment geometry on
   the union pool; pool_size effectively expands to RRF set size).
4. Bench R@100, R@10, nDCG@10 vs single-variant Atire pool=100.

### Validate / disprove

- **Validate** if R@100 improves by ≥0.01 on ≥2/3 corpora **and**
  nDCG@10 doesn't regress by more than 0.003 on any corpus.
  Outcome: ship as new production default if latency cost is <2x.
- **Disprove** if R@100 lift is <0.005 on every corpus (BM25
  variants' top-K already converge at K=100 on these fixtures →
  diversification adds no candidates).

### Cost

~2-3 days:

- ~50 LOC bench wiring for RRF-fused pool builder (reuses existing
  `rrf_fuse` from `fusion.cpp`).
- 7x BM25 builds at index time (~2x current build time; one-shot
  per corpus).
- 7x BM25 retrievals per query at query time + RRF fusion cost
  (estimated 3-5x current QPS reduction; still 25-40 QPS on
  scifact richcov).
- Cross-fold bench on existing 3 + trec-covid if Plan 1 lands.

### Risk

- BM25 variants may have correlated top-K sets (R@100 is mostly
  determined by which 100 docs share rare query terms, regardless
  of weighting). Easy to disprove cheaply.
- Fragment geometry cost scales with pool size; may need to keep
  pool=100 from RRF and accept smaller diversity gain.

### Bottleneck addressed

**Bottleneck 2** (BM25-pool R@100 ceiling) directly. The only
plan in this set that attacks the candidate-set bound rather than
the rerank operator.

---

## Plan 4 — Single-fragment-per-doc builder (break the averaging math)

### Hypothesis

Bottleneck 1 was disproved at the **aggregation** step (MaxSim), but the
underlying issue may still be the t=8 fragments-per-doc structure. A builder
that emits **one query-time-selected fragment per doc** breaks the averaging
math at the source.

This is distinct from MaxSim:
- **MaxSim**: keep 8 fragments per doc, change aggregation.
- **Plan 4**: keep 8 candidates but select 1 at query time.

This is a query-time fragment-pruning approach. The fragment set stays fixed at
index time; the doc just contributes the single fragment with highest
`dot(query_vec, fragment_vec)`.

### Implementation

1. Index time: build all 8 fragments per doc as today.
2. Query time: for each doc in the BM25 pool, pick the
   argmax-fragment by `dot(query, fragment)`. ~10 LOC change to the
   `fvecs` gather step in `fragment_geometry.cpp`.
3. Geometry pipeline runs over these single-fragment-per-doc
   representations. Pool size N is now ≤ pool_size (was pool_size ×
   t=8 ≈ 800).
4. PHSS at N=100 instead of N=800 — 64× cheaper edge sort, may
   unlock larger pools without latency penalty.

### Validate / disprove

- **Validate** if nDCG@10 lifts by ≥0.005 on ≥2/3 corpora vs
  production frontier with cross-fold consistency.
- **Disprove** if quality is within ±0.003 across all corpora and
  folds → multi-fragment information was actually being used by
  the geometry, not just averaged.

### Cost

~1 day:

- ~30 LOC in `fragment_geometry.cpp` for query-time argmax fragment
  selection.
- Bench wiring: 2-3 new recipes.
- Cross-fold bench on 3 (or 4) corpora.

### Risk

- May lose the **redundancy / coverage signal** that richcov
  builder was designed to capture (8 fragments per doc averaged ARE
  more robust to noisy individual fragments).
- Single fragment may overfit to query-stem matches in fiqa-style
  corpora, the same failure mode that killed earlier "concept" rows.
- Very likely outcome: Plan 4 wins on focused-content corpora
  (scifact, possibly trec-covid) and loses on Q&A (fiqa, possibly
  nfcorpus) — same pattern as MaxSim. Difference: Plan 4 also
  unlocks larger pools cheaply, which Plan 3 needs.

### Bottleneck addressed

**Bottleneck 1** (multi-fragment averaging) at the source — different
attack point than MaxSim aggregation. Also indirectly addresses
**Bottleneck 2** by making larger pools affordable (64× edge-sort
reduction at N=100 instead of N=800).

---

## Cross-plan composability + recommended sequencing

The plans are not mutually exclusive:

- **Plans 2 + 3** both attack candidate-set ceilings and could compose.
- **Plans 3 + 4** compose well because Plan 4 makes larger pools cheaper.
- **Plan 1** is the re-validation corpus for all others.
- **Plan 2** is the only purely offline candidate-expansion plan.

Recommended sequencing if all four are pursued:

1. **Plan 4 first** — cheapest, fastest signal.
2. **Plan 2 second** — offline candidate-set attack, independent of Plan 4.
3. **Plan 3 third** if Plans 2/4 unlock larger pools.
4. **Plan 1 last** as the re-validation corpus.

If only one plan is pursued: **Plan 4** remains the highest-leverage single
move.

If only two plans are pursued: **Plans 4 + 2** give one fragment-side lever and
one candidate-side lever.

---

## Discipline rules (carry forward from the methodology lessons)

- lifts ≤0.005 nDCG@10 must be cross-fold validated before being claimed
- per-corpus tuning must validate the dev pick on test
- prefer shipped-but-unused primitives before adding new infrastructure
- land and bench in simeon standalone before any yams integration

---

*2026-04-24. Drafted post-`phss_alpha_sweep_results.md` close.
Builds on `literature_synthesis.md` bottleneck framing. No code
changes required to start; first action under any plan is the
respective bench/fixture engineering described in its own section.*
