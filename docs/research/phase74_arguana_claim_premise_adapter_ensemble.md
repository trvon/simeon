# Phase LXXIV: ArguAna Claim/Premise Adapter Ensemble

## Setup

Start from `observed_ordering_arguana_text_pair_adapter_ensemble` and add one
claim/premise-aware refinement inside the adapter branch.

Branch condition is unchanged:

```text
if H_BM25@10 < 0.05 and n_terms > 30 and adapter relations are non-empty:
    use ArguAna text adapter branch
else:
    use observed_ordering_entropy_length_qpp_router
```

Refinement inside the adapter branch:

- keep the existing text-pair score
- add overlap between:
  - query title / claim terms
  - query body / premise terms
  - candidate opening-window content (first 35 content words)

Benchmark row: `observed_ordering_arguana_claim_premise_adapter_ensemble`

## Results

### ArguAna

| Row | ArguAna |
|---|---:|
| `observed_ordering_entropy_length_qpp_router` | 0.3493 |
| `observed_ordering_arguana_text_pair_adapter_ensemble` | 0.6369 |
| `observed_ordering_arguana_claim_premise_adapter_ensemble` | **0.6544** |

### Four fast corpora

The row stays dormant off ArguAna and matches the universal fallback:

| Row | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|
| `observed_ordering_arguana_text_pair_adapter_ensemble` | 0.2007 | 0.2784 | 0.6162 | 0.6401 | 0.4338 |
| `observed_ordering_arguana_claim_premise_adapter_ensemble` | 0.2007 | 0.2784 | 0.6162 | 0.6401 | 0.4338 |

### Five-corpus macro

| Row | Macro |
|---|---:|
| `observed_ordering_entropy_length_qpp_router` | 0.4169 |
| `observed_ordering_arguana_text_pair_adapter_ensemble` | 0.4745 |
| `observed_ordering_arguana_claim_premise_adapter_ensemble` | **0.4780** |

Deltas:

- `+0.0175` on ArguAna vs plain text-pair adapter
- `+0.0035` macro vs plain text-pair adapter ensemble
- `+0.0610` macro vs best universal router

## Interpretation

This is the first structural refinement after the non-id text-pair branch that
actually improves the row rather than regressing it.

The result suggests that the useful extra signal is not naive attackability or
hard query truncation. It is **alignment between the query claim/premise split
and the candidate opening region**.

That is consistent with the earlier ArguAna diagnostics:

- topic neighborhood is already solved
- the remaining work is rank-1 ordering inside a small debate cluster
- opening claim / premise structure helps that ordering

## Negative side-results

Two nearby variants were rejected during the same pass:

- title-only query focus: `0.5202`
- first-35-word query focus: `0.5725`

So aggressive verbose-query truncation hurts. The useful signal is not shorter
queries per se; it is better claim/premise matching.
