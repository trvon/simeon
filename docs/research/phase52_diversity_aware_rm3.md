# Phase LII: Diversity-Aware RM3 (MMR Selection)

## Goal

Phase XLIX's simeon-weighted RM3 improved pseudo-relevance doc weighting but
still used the same BM25 top-K docs. This phase changes the **selection**
mechanism: instead of picking top-K by BM25 score, greedily select docs that
are both relevant AND diverse from each other in simeon embedding space.

The intuition: standard RM3 picks the 10 most BM25-similar docs, which are often
near-duplicates of each other. Expansion terms from these docs are redundant.
MMR selection forces coverage of different aspects of the topic.

## Method

Maximal Marginal Relevance (Carbonell & Goldstein 1998) adapted for PRF:

```text
MMR(d | S) = BM25_score(d) × max(0, simeon_sim(q, d))
           − β × max_{d'∈S} simeon_sim(d, d')
```

where S is the set of already-selected docs, β=0.5 controls the
diversity/relevance tradeoff.

Algorithm:
1. BM25 first-pass to get top-20 candidates
2. Greedily select K=10 docs from candidates using MMR
3. Weight selected docs by BM25 × simeon_sim(q, d) (same as Phase XLIX)
4. Build relevance model, expand, score (same pipeline)

The row is `observed_ordering_rm3_diverse_k10_b0.5`.

Artifact:

```text
/tmp/simeon_diverse_20260505_084103
```

## Test results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| Weighted RM3 | 0.3323 | 0.2034 | 0.2782 | 0.6034 | 0.6059 | 0.4046 |
| **Diversity RM3** | **0.3332** | **0.1993** | **0.2785** | **0.6023** | **0.6300** | **0.4087** |
| Sigmoid 4-way | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | 0.4041 |

### Deltas vs BM25

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| Weighted RM3 | +0.0030 | −0.0019 | +0.0261 | −0.0154 | +0.0410 | +0.0106 |
| **Diversity RM3** | **+0.0039** | **−0.0060** | **+0.0264** | **−0.0165** | **+0.0651** | **+0.0146** |
| Sigmoid 4-way | +0.0004 | +0.0005 | +0.0086 | +0.0016 | +0.0392 | +0.0101 |

### Diversity vs Weighted

| Corpus | Weighted RM3 | Diversity RM3 | Δ |
|---|---:|---:|---:|
| ArguAna | 0.3323 | 0.3332 | **+0.0009** |
| FiQA | 0.2034 | 0.1993 | −0.0041 |
| NFCorpus | 0.2782 | 0.2785 | +0.0003 |
| SciFact | 0.6034 | 0.6023 | −0.0011 |
| TREC-COVID | 0.6059 | 0.6300 | **+0.0241** |
| **Macro** | **0.4046** | **0.4087** | **+0.0040** |

## Interpretation

### 1. TREC-COVID is the biggest winner

+0.0651 over BM25 and +0.0241 over the weighted RM3. This is the largest
single-corpus gain from any experiment in the entire research arc.

Why: TREC-COVID has 171K medical abstracts, many of which are about COVID
treatments. Standard RM3 picks 10 BM25-top documents that are all about the
same narrow subtopic (e.g., remdesivir trials). MMR selection forces the
expansion to cover diverse aspects (trials, mechanisms, outcomes, populations),
producing expansion terms that better match the broad information needs of
TREC-COVID queries.

### 2. ArguAna also improves

+0.0039 over BM25 and +0.0009 over weighted RM3. MMR helps break the
"restatements" problem: BM25 top-K for a claim query contains multiple
restatements of the same claim. MMR selects diverse documents that include
counterarguments alongside restatements.

### 3. FiQA regresses

−0.0060 vs BM25 and −0.0041 vs weighted RM3. FiQA queries are very short
(avg 11 terms). With so few query terms, the MMR diversity penalty may be
spreading the expansion too thin, incorporating terms from tangentially
related financial documents rather than the core question.

### 4. SciFact is flat vs weighted RM3

−0.0011 Δ vs weighted RM3. MMR selection doesn't help or hurt here because
SciFact's problem is that expansion terms from any feedback docs (diverse
or not) are irrelevant to the fact-claim task.

### 5. 0.4087 is the new best raw row

The diversity RM3 reaches 0.4087 macro nDCG@10, +0.0040 better than the
previous best (weighted RM3 at 0.4046) and +0.0046 better than the safest
row (sigmoid 4-way at 0.4041).

## Updated system hierarchy

| Row | Type | Macro | Key strength | Key weakness |
|---|---|---|---|---|
| **Diversity RM3** | 1 gen, MMR | **0.4087** | TREC-COVID +0.0651 | FiQA −0.0060 |
| Weighted RM3 | 1 gen | 0.4046 | NFCorpus +0.0261 | SciFact −0.0154 |
| Sigmoid 4-way | 4 gen, gate | 0.4041 | Zero regressions | Low ceiling |

## Impact on the theorem

The diversity mechanism is the second successful scorer-level improvement after
the simeon weighting. Together they raise the observed ceiling from 0.3941
(BM25) to 0.4087 (diverse RM3), closing 0.0146 of the 0.4148 gap to the oracle
— a 3.5% reduction in the gap.

The remaining gap: 4-way oracle@100 (0.8089) − diverse RM3 (0.4087) = **0.4002**.

The two open problems are:
1. **FiQA regression** (−0.0060): the MMR penalty is too strong for short queries.
2. **SciFact regression** (−0.0165): expansion fundamentally doesn't help fact-claim queries.

## Next direction

Candidate follow-ups:
1. **Dynamic β** — scale the MMR diversity penalty by query length (longer
   queries → more β; shorter queries → less β) to fix FiQA.
2. **Ensemble gate** — use diversity RM3 when BM25 entropy is high, weighted
   RM3 when moderate, and BM25 when low.
3. **SciFact-specific gate using query features** — scq_sum/n_terms/idf_stddev
   to detect fact-claim queries and suppress expansion entirely.
