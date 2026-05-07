# Phase LXVIII: Self-Assessed Score-List Router

## Setup

- Strategy pool: BM25, BM25F-lead, RM3, simeon
- Per-strategy quality: `margin + 3 * (1 - entropy) + decay`
- Decision rule: softmax over per-strategy quality, then score-level blend

Two parameterizations were tested:

1. standard RM3, equal shape weights, `T = 0.5`
2. diverse RM3, entropy-heavy weighting, `T = 0.3`

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| self-assess, std RM3 | 0.3402 | 0.2090 | 0.2704 | 0.6163 | 0.5837 | 0.4039 |
| self-assess, diverse RM3 | 0.3391 | 0.2074 | 0.2671 | 0.6077 | 0.5503 | 0.3943 |
| `observed_ordering_gated_ensemble` | 0.3494 | 0.2048 | 0.2661 | 0.6044 | 0.6185 | **0.4086** |

Best self-assessed variant: `0.4039`.

## Failure mode

- The router **blends** all strategies even when one strategy is clearly best.
- Residual weight on weaker strategies acts as noise.
- On corpora where one strategy should dominate completely, the blended score
  cannot match hard selection.

The clearest example is TREC-COVID:

- hard gate: routes fully to diverse RM3 and reaches `0.6185`
- soft self-assessment: retains weight on weaker strategies and drops to
  `0.5837` or `0.5503`

## Interpretation

This experiment rejects the design:

```text
run all strategies -> estimate output health -> softmax blend
```

Per-strategy shape features are not sufficient when the final action is a blend.
Even a good quality estimate is diluted by nonzero weight on suboptimal runs.

## Consequence

- Hard routing is preferable to soft blending for strategy selection.
- The next step is not better blending weights; it is **hard selection**.
- Phase LXIX tests that directly and still fails when the signal is taken from
  per-strategy output shape rather than query-level difficulty.

See [Phase LXIX](phase69_hard_select_router.md),
[Phase LXXI](phase71_entropy_length_router.md), and
[Phase LXXII](phase72_qpp_hard_router.md).
