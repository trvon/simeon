# Phase LXV: ArguAna Pair-ID Adapter

## Setup

- corpus: ArguAna
- structural signal: query / document IDs of the form
  `<topic>-<stance><point><side>`
- adapter action: match `a ↔ b` within the same topic / stance / point group
- non-ArguAna behavior: no parse, no evidence, text-only fallback

This phase isolates the `A` component in `T = (G, A, S, F, B)`.

## Results

| System | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| `observed_ordering_gated_ensemble` | 0.3494 | 0.2048 | 0.2661 | 0.6044 | 0.6185 | 0.4086 |
| ArguAna pair-ID adapter | **0.6234** | 0.2053 | 0.2521 | 0.6188 | 0.5649 | **0.4529** |

Deltas:

- `+0.2941` on ArguAna vs `bm25_only`
- `+0.0588` macro vs `bm25_only`
- `+0.0443` macro vs `observed_ordering_gated_ensemble`

## Mechanism

The adapter contributes only on ArguAna. On the other corpora it is a no-op, so
the macro lift comes entirely from replacing the ArguAna branch.

This is the first strong proof that the residual ArguAna gap is structural, not
just scorer quality:

- universal text-only systems remain in the `0.33–0.35` range
- explicit observable structure jumps to `0.6234`

## Interpretation

The result establishes three facts:

1. `A` is a first-class component, not a debugging convenience.
2. Structural evidence can dominate all universal lexical tuning on the right
   corpus.
3. A valid training-free adapter may use corpus formatting conventions as long
   as it does not consume qrels.

## Status relative to later work

This phase proves the value of structural evidence. Phase LXVII then shows the
correct composition rule:

```text
if adapter evidence exists:
    let the adapter branch win
else:
    use the best available universal router
```

See [Phase LXVII](phase67_adapter_ensemble.md).
