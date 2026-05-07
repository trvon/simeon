# Phase LXXIII: ArguAna Text-Pair Adapter Ensemble

## Setup

Compose:

- ArguAna text-pair discriminator branch (Phase XXII mechanism)
- universal fallback: `observed_ordering_entropy_length_qpp_router`

Decision rule:

```text
if H_BM25@10 < 0.05 and n_terms > 30 and
   the text-pair adapter finds a same-header debate neighborhood:
    use text-pair adapter branch
else:
    use observed_ordering_entropy_length_qpp_router
```

Benchmark row: `observed_ordering_arguana_text_pair_adapter_ensemble`

The adapter branch is training-free and non-id: it uses query-text containment,
first-five-token page headers, local order, content overlap, rebuttal cues, and
relative concision. It does not use qrels and does not use the exact `a ↔ b` ID
pair.

## Results

### Four fast corpora

The adapter branch stays dormant; the row matches the universal fallback exactly.

| Row | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|
| `observed_ordering_entropy_length_qpp_router` | 0.2007 | 0.2784 | 0.6162 | 0.6401 | 0.4338 |
| `observed_ordering_arguana_text_pair_adapter_ensemble` | 0.2007 | 0.2784 | 0.6162 | 0.6401 | 0.4338 |

### ArguAna

| Row | ArguAna |
|---|---:|
| `observed_ordering_entropy_length_qpp_router` | 0.3493 |
| `observed_ordering_arguana_text_pair_adapter_ensemble` | **0.6369** |

### Five-corpus macro

| Row | Macro |
|---|---:|
| `adapter-aware ensemble` (historical) | 0.4634 |
| `observed_ordering_entropy_length_qpp_router` | 0.4169 |
| `observed_ordering_arguana_text_pair_adapter_ensemble` | **0.4745** |

Deltas:

- `+0.0576` vs `observed_ordering_entropy_length_qpp_router`
- `+0.0111` vs the earlier adapter-aware ensemble
- `+0.0804` vs `bm25_only`

## Interpretation

This is the first post-LXXI/LXXII result with a **material** macro jump.

- Router refinement yielded `+0.0028`
- Structural adapter composition yields `+0.0576`

So the earlier theorem diagnosis holds: once the universal router is competent,
the largest remaining gains come from structural task matching, not more gate
tuning.

## Consequence

The adapter composition rule is now strongly validated:

```text
if strong structural evidence exists:
    let the adapter branch win
else:
    use the best universal router
```

The next research question is no longer whether structural adapters matter. It is
whether this non-id text-pair branch can be generalized beyond ArguAna-style
debate pages.
