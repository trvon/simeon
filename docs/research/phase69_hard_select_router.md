# Phase LXIX: Hard-Select Router (Negative)

## Setup

- Same strategy pool as Phase LXVIII
- Same per-strategy quality family: output-shape features derived from margin,
  entropy, and decay
- Decision rule: choose the single strategy with highest estimated quality

This isolates the Phase LXVIII question:

```text
is the failure caused by blending, or by the quality signal itself?
```

## Result

- Macro: `0.3898`
- Worse than Phase LXVIII soft blend: `0.3943`
- Much worse than `observed_ordering_gated_ensemble`: `0.4086`

## Failure mode

Per-strategy entropy is not a quality signal. It is mostly a property of the
strategy's scoring mechanics:

- simeon: high entropy, nearly uniform, almost never selected
- BM25: moderate entropy, often over-selected
- RM3: variable entropy, unstable selection behavior

The selector therefore ranks strategies by score flatness / peaking rather than
by expected retrieval gain.

## Interpretation

This experiment rejects the design:

```text
run all strategies -> choose argmax(output-shape quality)
```

The successful signal is not "which strategy produced the healthiest-looking
distribution?" It is "what query regime are we in?"

BM25 entropy works because it is a proxy for **query difficulty**. Per-strategy
output entropy does not have that semantics.

## Consequence

- Hard routing remains correct.
- Per-strategy output-shape quality remains wrong.
- The next successful step is to return to **query-level routing** and use a
  verbose-query regime split.

That follow-up becomes the entropy+length 2-way router at `0.4141` macro; see
[Phase LXXI](phase71_entropy_length_router.md).
