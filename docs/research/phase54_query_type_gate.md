# Phase LIV: Query-Type Gate + Diverse RM3

## Setup

Two safety mechanisms were tested against
`observed_ordering_rm3_diverse_k10_b0.25`:

1. **avg-IDF gate**

```text
if avg_idf < 2.8 or avg_idf > 3.8:
    use diverse RM3
else:
    use BM25
```

Row: `observed_ordering_diverse_avgidf_gate`

2. **4-way sigmoid fusion with diverse RM3**

Replace the standard RM3 leg inside the existing sigmoid dampening fusion with
diverse RM3.

Row: `observed_4gen_dampen_diverse_rm3`

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| `observed_ordering_rm3_diverse_k10_b0.25` | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | **0.4108** |
| `observed_ordering_diverse_avgidf_gate` | 0.3319 | 0.1975 | 0.2742 | 0.6096 | 0.6313 | 0.4089 |
| `observed_4gen_dampen_diverse_rm3` | 0.3298 | 0.2051 | 0.2637 | 0.6204 | 0.6093 | 0.4057 |
| `observed_4gen_dampen_sigmoid` | 0.3297 | 0.2058 | 0.2607 | 0.6204 | 0.6041 | 0.4041 |

## Interpretation

### avg-IDF gate

- improves SciFact relative to raw diverse RM3
- harms FiQA more than it helps SciFact
- net effect: worse than raw diverse RM3 (`0.4089 < 0.4108`)

The gate is directionally plausible but not transfer-stable.

### Diverse RM3 inside sigmoid dampening

- improves the safer fusion row from `0.4041` to `0.4057`
- preserves the non-regression profile better than raw diverse RM3
- still remains below the raw best scorer

This makes `observed_4gen_dampen_diverse_rm3` the better **safe** row in this
phase, but not the universal winner.

## Consequence

This phase separates two objectives:

1. **max raw quality** -> raw diverse RM3
2. **max safety / smoother portfolio behavior** -> dampened fusion

The later router work supersedes both as the best universal compromise:
`observed_ordering_entropy_length_qpp_router = 0.4169`.

So the durable lesson from this phase is not the specific avg-IDF band. It is
that query-type safety mechanisms are useful, but they work best as **hard
routing rules** rather than as soft suppression inside a broader blend.
