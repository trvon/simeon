# Phase XI Results: GloVe Pre-Trained Vectors + Arguana Corpus

Both post-Ceiling-B axes exhausted. GloVe 6B 300d underperforms domain-specific PMI
on trec-covid. Arguana is task-adversarial to cosine fragment geometry.

---

## Experiment A: GloVe 6B 300d on trec-covid

Pre-trained general-English vectors replace the corpus-learned PMI rank-128 embedding.
GloVe 6B (Wikipedia + Gigaword, 6B tokens, 400K vocab, dim=300) loaded via
`load_glove()` with `max_vocab=200K` and L2-normalized on load, then plugged into
`EncoderConfig::pmi_rows` via `PmiEmbeddings::from_external()`.

Implementation: `src/glove_embeddings.cpp` — `load_glove()`. Bench dispatch:
`--fragment-glove <path>` flag in `bench_vs_reference.cpp` →
`run_bm25_fragment_glove_grid()`. No changes to `Encoder`, `fragment_geometry.cpp`,
`persistent_homology.cpp`, or `phss_select_scale`.

Hypothesis: GloVe's richer 300d relational space (trained on 200× more tokens) would
improve inter-fragment separation and MaxSim signal on trec-covid, unlocking Ceiling B.

## Alpha Sweep Results (richcov t8, trec-covid test fold)

Dev fold not run — test fold already exceeds the −0.005 disproof threshold on all configs.

| Config | nDCG@10 (test) | Δ vs BM25 | Δ vs PMI OM α=0.80 |
|--------|----------------|-----------|--------------------|
| BM25 baseline | 0.5649 | — | — |
| PMI OM α=0.80 (Phase III) | 0.5752 | +0.0103 | — |
| GloVe OM α=0.50 | 0.5563 | −0.0086 | **−0.0189** |
| GloVe OM α=0.65 | 0.5631 | −0.0018 | **−0.0121** |
| GloVe OM α=0.80 | 0.5644 | −0.0005 | **−0.0108** |
| GloVe OM α=0.90 | 0.5654 | +0.0005 | **−0.0098** |
| GloVe OM α=0.95 | 0.5659 | +0.0010 | **−0.0093** |
| GloVe PHSS α=0.80 | 0.5597 | −0.0052 | **−0.0155** |

## Cross-Fold Assessment (GloVe)

Not applicable — test fold already falsifies all configs at ≥ −0.009 vs PMI OM baseline,
exceeding the −0.005 per-fold threshold in the negative direction. Dev fold not run.

**Verdict: DISPROVED.**

## Mechanism

trec-covid is a specialized COVID-19 abstract corpus. PMI trained on the same 171K
documents captures domain co-occurrence patterns (COVID–mortality, SARS-CoV-2–spike,
ACE2–receptor) that GloVe 6B cannot: GloVe encodes general-English Wikipedia context,
not biomedical literature.

The comparison directly tests domain specificity vs model scale. Domain specificity wins:
PMI rank-128 in-domain outperforms GloVe dim-300 on 200× more general tokens. The +0.010
gain from fragment geometry on trec-covid is attributable to domain co-occurrence
structure, not compositional depth.

Confirmatory: MiniLM dense retrieval (reference = 0.5008) also underperforms BM25
(0.5649) on trec-covid — the corpus is lexically dominated regardless of model quality.

---

## Experiment B: Arguana Corpus

Claim-to-counterargument retrieval corpus. 8,674 documents, 906 test queries / 500 dev
queries. Hypothesis: argumentative prose has lower content-word density than COVID
abstracts, giving IDF-based approaches room to create separability.

## Cross-Fold Assessment (Arguana, selected configs)

BM25: test=0.3293 / dev=0.3202.

| Config | Test nDCG | Test Δ | Dev nDCG | Dev Δ | Verdict |
|--------|-----------|--------|----------|-------|---------|
| BM25 only | 0.3293 | — | 0.3202 | — | baseline |
| `bm25_fragment_geom_rich_k100_t8` | 0.3324 | +0.003 | 0.3259 | +0.006 | sub-threshold (test) |
| `bm25_fragment_geom_phssapprox_a0.95_richcov` | 0.3276 | −0.002 | 0.3184 | −0.002 | consistent − |
| `bm25_fragment_geom_outermaxsim_a0.95_k100_t8_richcov` | 0.3280 | −0.001 | (n/a) | — | negative |
| `bm25_fragment_geom_outermaxsim_a0.80_k100_t8_richcov` | 0.3181 | −0.011 | (n/a) | — | negative |
| `bm25_fragment_geom_phssapprox_k100_t8_richcov_gap` | 0.3201 | −0.009 | 0.3139 | −0.006 | consistent − |

All outer MaxSim configs degrade on both folds (α=0.80 through α=0.95: −0.001 to −0.011
on test). The sole above-threshold candidate (`bm25_fragment_geom_rich_k100_t8` +0.006 dev)
fails on test (+0.003 < 0.005). No config cross-fold validates.

**Verdict: DISPROVED.**

## Mechanism

Counterargument retrieval rewards topic overlap combined with semantic opposition — the
relevant document addresses the same topic as the query while taking the opposing position.
Fragment cosine similarity maximizes positive cosine alignment; a counterargument shares
vocabulary with the claim but takes the opposing stance.

The density hypothesis was wrong: the failure is task-structural, not vocabulary-structural.
Fragment geometry cannot be sign-flipped to reward dissimilarity without replacing the
cosine kernel entirely.
