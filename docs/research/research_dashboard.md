# Research Dashboard — Consolidated Numbers (36 Phases)

## Headline rows

| Scope | System / row | Macro nDCG@10 | Δ vs BM25 |
|---|---|---:|---:|
| Best overall | Adapter-aware ensemble | **0.4634** | **+0.0693** |
| Best adapter | ArguAna pair-ID adapter | 0.4529 | +0.0588 |
| Best universal | `observed_ordering_entropy_length_router` | **0.4141** | **+0.0200** |
| Best scorer | `observed_ordering_rm3_diverse_k10_b0.25` | 0.4108 | +0.0167 |
| Best earlier router | `observed_ordering_gated_ensemble` | 0.4086 | +0.0146 |
| Baseline | `bm25_only` | 0.3941 | — |

## Gap summary

```text
BM25 baseline:         0.3941
Best universal:        0.4141    (+0.0200)
Best with adapter:     0.4634    (+0.0693)
BM25 oracle@100:       0.7423
4-way union oracle@100 0.8089

Universal gap:         0.7423 - 0.4141 = 0.3282
Adapter gap:           0.7423 - 0.4634 = 0.2789
```

Interpretation:

- Universal text-only progress remains in the `0.41x` band.
- The largest remaining gains come from structure, not generic scorer tuning.

## Component maxima in `T = (G, A, S, F, B)`

| Component | Best measured contribution | Instantiation |
|---|---:|---|
| `G` | +0.0666 | union oracle lift from adding BM25F + RM3 + simeon |
| `A` | +0.0588 | ArguAna pair-ID adapter |
| `S` | +0.0167 | diversity-aware RM3 |
| `F` | +0.0200 | entropy+length hard router |
| `A + F` | +0.0693 | adapter-aware ensemble |

Observation: `A` alone still exceeds `G + S + F` as an observed macro lift.

## Per-corpus best rows

| Corpus | Best value | Row | Δ vs BM25 |
|---|---:|---|---:|
| ArguAna | **0.6234** | ArguAna adapter (pair-ID) | +0.2941 |
| FiQA | **0.2104** | KS-weighted PLR | +0.0051 |
| NFCorpus | **0.2789** | `observed_ordering_rm3_diverse_k10_b0.25` | +0.0268 |
| SciFact | **0.6263** | LM safe (0.55 / 0.15) | +0.0075 |
| TREC-COVID | **0.6400** | `observed_ordering_rm3_diverse_k10_b0.25` | +0.0751 |

## Major milestones

| Phase | Row / system | Macro | Technical change |
|---|---|---:|---|
| Baseline | BM25 | 0.3941 | lexical baseline |
| XL | shape-risk fusion | 0.4007 | safe 3-generator fusion |
| XLIV | 4-way z-equal | 0.4045 | add simeon generator |
| XLVI | sigmoid dampening | 0.4041 | continuous entropy gate |
| XLIX | weighted RM3 | 0.4046 | simeon-weighted PRF |
| LIII | diverse RM3 β=0.25 | **0.4108** | MMR feedback selection |
| LVI | gated ensemble | 0.4086 | 3-way entropy hard gate |
| LXV | ArguAna adapter | 0.4529 | pair-ID structure |
| LXVII | adapter-aware ensemble | **0.4634** | adapter + routing |
| LXXI | entropy+length router | **0.4141** | verbose-query lead escape hatch |

## Negative controls worth remembering

| Phase | Idea | Outcome |
|---|---|---|
| XLI | lexical overlap / phrase rerank | regress |
| XLII | static structural fields | regress |
| XLVII | multi-signal dampening | no better than entropy alone |
| XLVIII | consensus filtering | FiQA collapse |
| LV | embedding negative filter / MMR | no better than BM25 |
| LVIII | LM convex combination | ties sigmoid mixture |
| LXI | term concentration | regress |
| LXVI | simeon-embedding PRF | regress |
| LXVIII | self-assessed soft blend | loses to hard routing |
| LXIX | hard-select by output shape | loses to query-level routing |

## Claims supported by the numbers

1. Best universal behavior is **query-dependent hard routing**, not soft fusion.
2. BM25 score entropy remains the dominant routing signal; `n_terms > 30` is
   the clean verbose-query regime split.
3. The largest unresolved gap is **structural**. ArguAna remains the clearest
   witness.
4. RM3-diverse is still the strongest universal single strategy.
5. The benchmark has not falsified the `0.41x` universal ceiling; it has only
   nudged the floor upward within that band.
