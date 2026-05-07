# Phase XXVIII: ArguAna Adapter Runtime

## Goal

After crossing the `.80` quality target, push the adapter path on runtime rather
than only nDCG.

The first implementation of the ArguAna metadata diagnostics used dense
`queries × docs` score arrays. That was fine for ceiling measurement, but it
hid the real runtime shape of a corpus adapter: most adapters should retrieve a
tiny candidate set from a structure index.

## Change

`arguana_pair_id_diagnostic` and `arguana_argument_point_diagnostic` now emit
sparse rankings and record query time:

- pair-id diagnostic: query id -> opposite-side doc id;
- argument-point diagnostic: `(topic, stance, point)` -> candidate doc ids.

Both avoid filling all 8,674 doc scores per query.

## Result

Command:

```bash
build-release/benchmarks/simeon_bench_vs_reference_research \
  --dual-only --queries-from test fixtures/arguana-minilm
```

| Row | nDCG@10 | query_us_per_q | QPS |
|---|---:|---:|---:|
| `arguana_argument_point_diagnostic` | 1.0000 | 0.141 | 7,080,392.9 |
| `arguana_pair_id_diagnostic` | 1.0000 | 0.067 | 14,892,987.5 |

## Interpretation

The adapter direction is not only higher quality; it can be cheaper than
reranking a dense candidate pool when the corpus exposes a compact structural
index.

For product work, this argues for a two-layer design:

1. universal lexical/vector retrieval for candidate recall;
2. optional corpus adapters that produce tiny high-confidence candidate sets
   from structure (path/title/heading/section/issue/citation/debate point).

The adapter should remain honest about scope: schema-specific rows are evidence
for adapter value, not universal retrieval algorithms.
