# Phase LXXII: QPP Hard-Router Refinement

## Setup

Start from the entropy+length universal router:

```text
if H_BM25@10 < 0.05 and n_terms > 30:
    LeadFieldStrategy
else:
    Rm3DiverseStrategy
```

Add a short-query BM25 rescue gate using post-retrieval QPP signals from the
BM25 top-50:

- `nqc`
- `wig_full`

Winning rule:

```text
if H_BM25@10 < 0.05 and n_terms > 30:
    LeadFieldStrategy
elif H_BM25@10 < 0.8926631 and n_terms < 30 and
     nqc > 1.2376532 and wig_full > 3.0773578:
    Bm25Strategy
else:
    Rm3DiverseStrategy
```

Benchmark row: `observed_ordering_entropy_length_qpp_router`

## Results

### Four-corpus macro

| Row | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4303 |
| `observed_ordering_entropy_length_router` | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4303 |
| `observed_ordering_entropy_length_qpp_router` | **0.2007** | 0.2784 | **0.6162** | **0.6401** | **0.4338** |

### Five-corpus macro

| Row | Macro |
|---|---:|
| `observed_ordering_gated_ensemble` | 0.4086 |
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.4108 |
| `observed_ordering_entropy_length_router` | 0.4141 |
| `observed_ordering_entropy_length_qpp_router` | **0.4169** |

Deltas:

- `+0.0028` vs `observed_ordering_entropy_length_router`
- `+0.0061` vs `observed_ordering_rm3_diverse_k10_b0.25`
- `+0.0083` vs `observed_ordering_gated_ensemble`

## Interpretation

- The gain is real but modest.
- Most of the lift comes from **SciFact recovery**.
- ArguAna is unchanged because the lead branch is unchanged.
- NFCorpus cost is negligible.

So post-retrieval QPP signals can improve the universal router, but the uplift is
still small relative to the remaining structural gap.

## Consequence

This phase validates a stronger claim than Phase LXXI:

```text
query-level hard routing remains the right architecture,
and post-retrieval QPP signals are useful as BM25 rescue gates.
```

It does **not** invalidate the broader conclusion that large remaining gains are
more likely to come from structural adapters than from additional router tuning.
