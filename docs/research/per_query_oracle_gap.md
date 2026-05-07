# Per-query Oracle Gap: First Tail Map

## Purpose

`training_free_optimum.md` showed that the mean gap between the current
training-free floor and the BM25-pool oracle is large. The next question is
where that gap lives: everywhere, or in a hard tail of queries.

This note consumes the per-query dumps emitted by
`SIMEON_PER_QUERY_DUMP=/tmp/perq/<corpus>_<fold>
build-release/benchmarks/simeon_bench_vs_reference_research --dual-only ...`
and summarizes the gap to `oracle_bm25_pool_k100`.

Claude produced the current `/tmp/perq` dump set before stopping; Codex added
`docs/research/analyze_per_query_gap.py` so the analysis is repeatable.

## How to reproduce

```bash
python3 docs/research/analyze_per_query_gap.py /tmp/perq --top-configs 3 --top-queries 5
```

Input files are named:

```text
<corpus>_<fold>.<config>.jsonl
```

The script compares every non-oracle config to the same `(corpus, fold)` oracle,
then reports mean nDCG, mean oracle gap, and a hard-tail count where
`oracle_ndcg - config_ndcg >= 0.5`.

## Headline results

| Corpus | Fold | Best observed config in dump | Best mean | Oracle@100 | Mean gap | Hard tail |
|---|---|---|---:|---:|---:|---:|
| arguana | dev | `kexp_atire_max_a0.80_k500` | 0.3196 | 0.9217 | 0.6021 | 370/498 |
| arguana | test | `kexp_atire_max_a0.80_k500` | 0.3214 | 0.9269 | 0.6055 | 636/903 |
| fiqa | dev | `kexp_atire_max_a0.80_k500` | 0.1942 | 0.4804 | 0.2861 | 55/204 |
| fiqa | test | `routed_layered_geom_a0.80_k100` | 0.2106 | 0.5122 | 0.3016 | 123/444 |
| nfcorpus | dev | `dual_layered_dp100_a0.65` | 0.2679 | 0.4726 | 0.2048 | 10/99 |
| nfcorpus | test | `dual_layered_dp100_a0.80` | 0.2645 | 0.4355 | 0.1710 | 24/224 |
| scifact | dev | `kexp_atire_max_a0.80_k500` | 0.6720 | 0.8492 | 0.1772 | 17/98 |
| scifact | test | `kexp_atire_max_a0.80_k200` | 0.6189 | 0.8755 | 0.2567 | 56/202 |

## Interpretation

The runway is not a smooth, tiny improvement opportunity. It is a tail problem:

1. **Arguana is structurally adversarial.** Most queries have oracle-perfect
   relevant documents inside the pool, yet the best training-free similarity
   recipe scores zero on hundreds of them. This supports the prior diagnosis:
   similarity retrieves same-side restatements instead of counterarguments.
2. **FiQA and Scifact still have many binary failures.** The worst gaps are
   usually `oracle=1.0, config=0.0`. That means the needed document is in
   BM25-top-100, but all current scorers miss it at the top of the reranked
   list.
3. **NFCorpus has a smaller but more actionable tail.** The hard-tail count is
   only 10/99 on dev and 24/224 on test. These are the best targets for the next
   ceiling-closing experiment because the corpus already has a validated
   complementary-pool mechanism.

## Literature hooks from PaperBridge

The current result points away from blind graph diffusion and toward
query-specific selection/calibration:

- Bruch, Gai, and Ingber (2023), *An Analysis of Fusion Functions for Hybrid
  Retrieval*, supports keeping convex fusion as the baseline fusion form rather
  than assuming RRF is parameter-free.
- Meng et al. (2024), *Ranked List Truncation for Large Language Model-based
  Re-Ranking*, is relevant because this tail map is exactly a per-query
  candidate-list truncation/reranking problem, even though simeon remains
  training-free.
- He, Meij, and de Rijke (2011), *Result diversification based on query-specific
  cluster ranking*, is relevant to the next non-learned attempt: the tail may
  need query-specific cluster/facet promotion rather than document-level
  similarity sharpening.
- Csurka, Ah-Pine, and Clinchant (2014), *Unsupervised Visual and Textual
  Information Fusion in Multimedia Retrieval — A Graph-based Point of View*,
  supports the graph-fusion framing, but the in-tree graph rows show that
  diffusion alone was inert or negative on BEIR-3.

## Next experiment

Do **not** spend the next cycle on larger BM25 `K`; the dump confirms the K-exp
rows mostly fail to close the per-query tail.

The next high-leverage experiment is a **tail-oriented pool diagnostic**:

1. For each worst-gap query, dump the top-10 config docs and the oracle-relevant
   pool docs with BM25 rank, BM25 score, geometry score, doc length, and fragment
   count.
2. Compare the positive-in-pool docs against false positives under three feature
   families already in the codebase: lexical exactness, fragment coverage, and
   dense-centroid admission source.
3. Only then add a deterministic scorer branch. Candidate branch: query-specific
   cluster/facet promotion for NFCorpus/FiQA, guarded by an observable tail
   signature, not a universal graph diffusion default.

This turns "close the ceiling" from global recipe search into targeted false
positive/false negative separation on the exact queries that own the runway.
