# Literature synthesis — bottlenecks identified by the PHSS engineering pass

## Why this doc exists

After the PHSS engineering pass and follow-up fragment-topology work, the
remaining questions collapsed to three mechanism bottlenecks plus the
already-deferred Dictionary/Structural axis. This doc maps those bottlenecks to
the closest literature and the smallest remaining experiments worth trying.

## Bottleneck → relevant literature → candidate experiment

### Bottleneck 1 — Multi-fragment per-doc averaging washes per-fragment signal

**Mechanism (from `phss_1d_triangle_results.md`)**: richcov uses
`top_fragments_per_doc=8`, so each doc contributes up to 8 fragments
to the geometry pool. Per-fragment importance variance averages
across 8 contributions per doc, smoothing any signal. This killed the
Phase A 1D triangle probe (Δ range [−0.0005, +0.0004] across 18
cells; R@10 / R@100 byte-identical).

**Literature (Q5, Q7, Q10)**:

- **ColBERTv2** (Santhanam et al., NAACL 2022, `10.18653/v1/2022.naacl-main.272`).
  Introduces "MaxSim" late-interaction aggregation: per query token,
  take the *max* similarity across all doc tokens and sum these maxima
  across query tokens. The training-free analog: per query, take the
  max-matching fragment per doc and sum across query concepts. This
  preserves the *strongest* per-fragment match instead of averaging.
- **"Simple Projection Variants Improve ColBERT Performance"** (2025)
  — likely a learned-encoder tweak; lower priority for our
  training-free regime.

**Status**: no exact training-free MaxSim precedent surfaced in the search, so
applying it to PMI fragments is genuinely novel.

**Candidate experiment**: replace `geom_pool[doc] += mass[fragment]`
(sum) with `geom_pool[doc] = max(geom_pool[doc], mass[fragment])`
(MaxSim). ~10 LOC change in `fragment_geometry.cpp:858-859`. Test
both pure-max and weighted-MaxSim variants.

**Expected magnitude**: unknown. Easy to falsify.

### Bottleneck 2 — BM25-pool determinism caps R@100

**Mechanism**: geometry rerank only re-orders the BM25 top-K pool.
R@100 has been byte-identical across every PHSS / triangle variant
because no recipe adds documents to the candidate set. Phase C showed
larger pools regress nDCG@10 on 2/3 corpora; nfcorpus pool=500 R@100
+0.0122 was the only positive but at 5 QPS.

**Literature (Q8)**:

- **Bruch & Gai 2023, "An Analysis of Fusion Functions for Hybrid Retrieval"**
  (`10.18653/v1/2023.findings-acl.838`-ish, OpenAlex).
  > "convex combination outperforms RRF in in-domain and out-of-domain
  > settings; convex combination is sample efficient, requiring only a
  > small set of training examples to tune its only parameter to a
  > target domain."

  This validates simeon's existing `α·z(BM25) + (1−α)·z(geom)` blend as the
  right family, but also suggests **α should be tuned per domain**. Simeon
  currently uses a universal `0.8`.

**Candidate experiment**: per-corpus α sweep on dev fold, then validate on
test. Expectation: nfcorpus and fiqa prefer different α than scifact.

**Limitation**: this only changes blending, not candidate-set expansion.

### Bottleneck 3 — Per-corpus optimal recipes (recurring meta-finding)

**Mechanism**: every dimension-tuning experiment (LTD α, RM3 K, WSDM
β, entropy fusion, SDM λ, clarity gate, PHSS pool size) has shown
optimal settings vary per corpus. Universal recipes underperform.

**Literature (Q3 surveys, weak)**:

Generic searches mostly surfaced LLM/RAG surveys rather than training-free
routing work. The only strong adjacent result was Bruch & Gai's per-domain α
tuning. Beyond that, the natural move is a learned router, but T6 already
disproved the cheap-features version (`qpp_post_retrieval_results.md`: routing
discriminator ρ ≈ 0).

**Candidate experiment**: subsumed by Bottleneck 2's per-corpus α
sweep — it's the cheapest way to operationalize the corpus-bound
finding without a learned router.

### Bottleneck 4 — Dictionary/Structural axis (deferred earlier; literature pre-check)

**Status**: AhoCorasick + TextRank ship in simeon as the
"GLiNER-replacement path" (`README.md:49`) but were never wired into
the retrieval pipeline. Earlier deferred per user direction during the
gating-vs-structural choice.

**Literature (Q6)**:

- **"Entity-Oriented Search"** (Krisztian Balog, 2018) — comprehensive
  treatment of entity-centric retrieval, multi-field models, joint
  entity+text scoring. Foundational.
- **"ENT Rank"** (Laura Dietz, 2019) — uses entity-neighbor-text (ENT)
  relations to rank entities; converse direction (entities → text)
  is the standard BM25F entity-aware setup.

**Candidate experiment** (if pursued): wire AhoCorasick entity tags and
TextRank synthetic titles as BM25F-style fields. The implementation sketch was
already drafted and the Phase 1 prototype was already run.

**Status check (verified)**: the Dictionary axis was already
**fully** explored on the BEIR-3 fixture set and disproved:

| Doc | Verdict | Best Δ |
|-----|---------|--------|
| `bm25f_results.md` | E1 plumbing sanity passed (byte-identical at w=0) | n/a |
| `ac_entity_results.md` | **Disproved** | fiqa +0.0026 at w=0.2 (below +0.005 bar) |
| `textrank_title_results.md` | **Disproved** | scifact −0.0135, nfcorpus −0.0050, fiqa −0.0029 (every cell regresses) |

Both docs reach the same conclusion: this needs a structured-document corpus
where entity tags or titles are genuinely distinct from short body text. The
BEIR-3 fixtures are the limiter, not the primitives.

**Disposition**: Bottleneck 4 is closed on this fixture set. Cannot
be the next experiment without a new corpus.

## Synthesis verdict

The literature confirms the bottlenecks, but **none of the surveyed
training-free methods directly address Bottleneck 1 (multi-fragment
averaging)**. MaxSim is the closest adjacent idea, but it arrives via learned
encoders.

## Ranked next-experiment options (post-verification)

Ordered by expected payoff per LOC:

1. **MaxSim aggregation probe**: smallest scope and highest novelty.
2. **Per-corpus α sweep**: lowest novelty, but the cleanest remaining router-side cleanup.
3. ~~Dictionary/Structural axis~~: closed on BEIR-3; only reopens with a structured-doc corpus.

**Recommendation**: MaxSim first. If it fails, α tuning is the last remaining
cleanup before declaring the fixture set training-free-saturated.

## Bibliography (paperbridge-resolved; unverified — no PDF fetch)

- Santhanam et al. 2022, "ColBERTv2: Effective and Efficient
  Retrieval via Lightweight Late Interaction." NAACL.
  DOI: 10.18653/v1/2022.naacl-main.272
- Bruch & Gai 2023, "An Analysis of Fusion Functions for Hybrid
  Retrieval." ACL Findings 2023.
- Nardini, Nguyen et al. 2025, "Effective Inference-Free Retrieval
  for Learned Sparse Representations." (Year/venue per OpenAlex.)
- Balog 2018, "Entity-Oriented Search."
- Dietz 2019, "ENT Rank."

Not verified against full text — paperbridge returned metadata and abstracts
only. Any promoted candidate still needs a source-paper read before
implementation.

---

*2026-04-23. Search results in `/tmp/lit_q{1-10}_*.json`. Method:
paperbridge papers search across openalex + semantic_scholar
(arxiv was rate-limited).*
