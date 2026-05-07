# Phase XXX — all-corpus core profile after adapter cleanup

## Scope

Profiled the default benchmark path after moving ArguAna diagnostics behind
`SIMEON_ARGUANA_DIAGNOSTICS=1`. The full default benchmark still includes a large
router grid, so this run captured the stable core rows for every corpus/fold:
reference, `bm25_only`, and BM25 pool oracles at K={50,100,200,500}.

Artifacts: `/tmp/simeon_profile_20260503_175035/core`.

## Core results

| Corpus | Fold | BM25 nDCG@10 | Oracle@100 | Oracle@500 | Runway@100 | K500 lift | BM25 µs/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | dev | 0.3202 | 0.9217 | 0.9699 | +0.6015 | +0.0482 | 770.68 |
| ArguAna | test | 0.3293 | 0.9269 | 0.9701 | +0.5976 | +0.0432 | 780.30 |
| FiQA | dev | 0.1908 | 0.4804 | 0.6207 | +0.2896 | +0.1403 | 330.15 |
| FiQA | test | 0.2053 | 0.5122 | 0.6608 | +0.3069 | +0.1486 | 337.00 |
| NFCorpus | dev | 0.2620 | 0.4726 | 0.5689 | +0.2106 | +0.0963 | 10.49 |
| NFCorpus | test | 0.2521 | 0.4355 | 0.5446 | +0.1834 | +0.1091 | 10.69 |
| SciFact | dev | 0.6623 | 0.8492 | 0.9388 | +0.1869 | +0.0896 | 35.25 |
| SciFact | test | 0.6188 | 0.8755 | 0.9224 | +0.2567 | +0.0469 | 33.87 |
| TREC-COVID | dev | 0.4943 | 0.9484 | 1.0000 | +0.4541 | +0.0516 | 1096.47 |
| TREC-COVID | test | 0.5649 | 0.9613 | 0.9928 | +0.3964 | +0.0315 | 1129.89 |

## ArguAna diagnostic-gated reproduction

Artifacts: `/tmp/simeon_profile_20260503_175035/diagnostics`.

With `SIMEON_ARGUANA_DIAGNOSTICS=1`, the gated diagnostics reproduce the prior
quality pattern while staying out of default output:

| Row | Dev nDCG@10 | Test nDCG@10 | Runtime note |
|---|---:|---:|---|
| text-neighborhood p5 | 0.4480 | 0.4345 | dense diagnostic path; no timing emitted |
| pair discriminator p5 | 0.6267 | 0.6373 | dense diagnostic path; no timing emitted |
| devfit pair ranker p5 | 0.7022 | 0.7025 | supervised diagnostic |
| blended pair ranker p5 | 0.7087 | 0.7097 | supervised diagnostic |
| topic-stem blend p5 | 0.7557 | 0.7694 | structure adapter proxy |
| argument-point diagnostic | 1.0000 | 1.0000 | ~0.14 µs/query |
| pair-id diagnostic | 1.0000 | 1.0000 | ~0.07–0.10 µs/query |

## Findings

1. The cleanup/gating does not change the core oracle story. Every corpus still
   has large BM25→Oracle@100 runway.
2. The largest K-expansion gains remain FiQA and NFCorpus, so candidate-pool
   expansion is still most promising there.
3. ArguAna remains structural: Oracle@50 is already very high, and the gated
   topic/argument diagnostics confirm that the missing signal is relation/field
   structure, not candidate recall.
4. TREC-COVID has near-perfect or perfect K=500 ceiling, so discrimination over
   a larger pool is the obvious next profiling target.

## Next profiling work

- `--core-only` has been added and validated across all five corpora × two folds.
  Latest artifact: `/tmp/simeon_core_only_20260503_182717`; every run emitted the
  expected six rows and stopped at `oracle_bm25_pool_k500`.
- Add timing to the dense ArguAna diagnostic rows (`text_neighborhood`,
  `pair_discriminator`, and ranker blends) so all gated rows are comparable.
- Build a small YAMS metadata fixture to profile `YamsNativeCorpusAdapter`
  directly across path/metadata/English-seed query families.
