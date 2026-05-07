# Phase LXVII: Adapter-Aware Ensemble

## Setup

Combine:

- ArguAna pair-ID adapter (Phase LXV)
- then-best universal router: `observed_ordering_gated_ensemble`

Decision rule:

```text
if adapter relations are non-empty:
    use adapter branch
else:
    use gated universal router
```

This phase predates the later universal winner
`observed_ordering_entropy_length_router`; the fallback here is historical.

## Results

| System | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| `observed_ordering_gated_ensemble` | 0.3494 | 0.2048 | 0.2661 | 0.6044 | 0.6185 | 0.4086 |
| ArguAna pair-ID adapter | 0.6234 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.4529 |
| **Adapter-aware ensemble** | **0.6234** | **0.2048** | **0.2661** | **0.6044** | **0.6185** | **0.4634** |

## Mechanism

Per-corpus behavior is simple:

| Corpus | Adapter evidence | Branch used |
|---|---|---|
| ArguAna | present | pair-ID relation branch |
| FiQA | absent | universal router fallback |
| NFCorpus | absent | universal router fallback |
| SciFact | absent | universal router fallback |
| TREC-COVID | absent | universal router fallback |

So the gain comes entirely from replacing the ArguAna branch while leaving the
other corpora unchanged.

## Interpretation

This phase is the cleanest empirical proof so far that:

1. `A` and `F` compose cleanly
2. the adapter branch does not need to perturb corpora where it has no evidence
3. structural evidence can dominate universal lexical tuning on the right corpus

Macro effect:

- `+0.0548` vs `observed_ordering_gated_ensemble`
- `+0.0105` vs adapter-only system
- `+0.0693` vs `bm25_only`

## Status relative to later work

This note is historically correct but not the final composition point. The
fallback universal router was later improved from `observed_ordering_gated_ensemble`
to `observed_ordering_entropy_length_router`.

The design lesson remains valid:

```text
if strong adapter evidence exists:
    let the adapter branch win
else:
    use the best available universal router
```

This is the correct adapter composition rule going forward.
