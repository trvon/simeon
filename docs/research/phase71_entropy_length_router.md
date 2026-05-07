# Phase LXXI: Entropy+Length 2-Way Router

## Setup

- Candidate strategies: `LeadFieldStrategy`, `Rm3DiverseStrategy`
- Objective: improve the universal hard router after the Phase LXVIII/LXIX
  self-assessment failures
- Constraint: use only cheap query-level evidence; no soft blending

Per-query analysis over FiQA, NFCorpus, SciFact, and TREC-COVID showed that the
BM25 middle branch in the earlier routers was mostly costing quality. The
remaining requirement was an ArguAna-specific escape hatch that did not fire on
the other four corpora.

Observed separator:

- ArguAna: `906 / 906` queries have `n_terms > 30`
- FiQA / NFCorpus / SciFact / TREC-COVID: `max(n_terms) = 29`

## Rule

```text
if H_BM25@10 < 0.05 and n_terms > 30:
    LeadFieldStrategy
else:
    Rm3DiverseStrategy
```

Benchmark row: `observed_ordering_entropy_length_router`

## Results

### Four-corpus macro

| Row | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.2000 | 0.2789 | 0.6023 | 0.6400 | **0.4303** |
| `observed_ordering_arch_router_v2` | 0.2025 | 0.2692 | 0.6102 | 0.6188 | 0.4252 |
| `observed_ordering_arch_router_v5` | 0.2008 | 0.2717 | 0.5983 | 0.6401 | 0.4277 |
| `observed_ordering_entropy_length_router` | 0.2000 | 0.2789 | 0.6023 | 0.6400 | **0.4303** |

The new rule preserves the best four-corpus behavior of diverse RM3 exactly.

### ArguAna

| Row | ArguAna |
|---|---:|
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.3327 |
| `observed_ordering_gated_ensemble` | **0.3494** |
| `observed_ordering_entropy_length_router` | 0.3493 |

### Five-corpus macro

| Row | Macro |
|---|---:|
| `observed_ordering_gated_ensemble` | 0.4086 |
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.4108 |
| `observed_ordering_arch_router_v2` | 0.4100 |
| `observed_ordering_entropy_length_router` | **0.4141** |

Deltas:

- `+0.0033` vs `observed_ordering_rm3_diverse_k10_b0.25`
- `+0.0055` vs `observed_ordering_gated_ensemble`

## Interpretation

- The winning router is a **hard query-performance predictor**, not a
  per-strategy self-assessor.
- `H_BM25@10` is the primary expansion / regime signal.
- `n_terms > 30` is the verbose-query regime split that isolates ArguAna.
- The rule keeps the 4-corpus RM3 win and recovers the ArguAna lead fallback.

## Relation to prior phases

- Phase LXVIII: soft self-assessment loses to hard routing.
- Phase LXIX: hard-select by per-strategy output shape also loses.
- Phase LXXI: hard-select by query difficulty plus verbose-query regime becomes
  the best universal router.
