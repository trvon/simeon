# Phase XXXIII — constructive generator rows

## Goal

Move from oracle-only generator slices to observed, qrel-free ranking rows. The
question is whether the new generator union can improve `BestObserved_TF`, not
just `Oracle_G@K`.

## Added rows

Under `--generator-slices`:

- `observed_gen_bm25f_textrank_w0.5`
- `observed_gen_bm25_rm3_k10_n30_a0.5`
- `observed_union_bm25_bm25f_rm3_rrf_k{50,100,200,500}`

The union row uses RRF over each generator's top-K pool. It is deterministic and
training-free, but intentionally simple.

Artifact:

```text
/tmp/simeon_constructive_generators_20260504_020314
```

## Results, nDCG@10

| Corpus | Fold | BM25 | BM25F | RM3 | Union RRF@100 | Best observed | Union oracle@100 | Remaining gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | dev | 0.3202 | 0.3133 | 0.3105 | 0.3209 | 0.3209 | 0.9598 | +0.6389 |
| ArguAna | test | 0.3293 | 0.3262 | 0.3243 | 0.3311 | 0.3311 | 0.9646 | +0.6335 |
| FiQA | dev | 0.1908 | 0.1641 | 0.1805 | 0.1881 | 0.1908 | 0.5391 | +0.3483 |
| FiQA | test | 0.2053 | 0.1783 | 0.1996 | 0.2046 | 0.2053 | 0.5523 | +0.3470 |
| NFCorpus | dev | 0.2620 | 0.2643 | 0.2784 | 0.2662 | 0.2784 | 0.5714 | +0.2930 |
| NFCorpus | test | 0.2521 | 0.2358 | 0.2726 | 0.2523 | 0.2726 | 0.5286 | +0.2560 |
| SciFact | dev | 0.6623 | 0.6262 | 0.6659 | 0.6656 | 0.6659 | 0.8838 | +0.2179 |
| SciFact | test | 0.6188 | 0.5898 | 0.5998 | 0.6146 | 0.6188 | 0.8939 | +0.2751 |
| TREC-COVID | dev | 0.4943 | 0.4811 | 0.5644 | 0.5521 | 0.5644 | 0.9781 | +0.4137 |
| TREC-COVID | test | 0.5649 | 0.5300 | 0.6064 | 0.5860 | 0.6064 | 0.9847 | +0.3783 |

## Interpretation

The generator variable `G` clearly raises the oracle, but naive RRF over the
union does not close most of the gap. The improvement is corpus-specific:

- RM3 gives real observed gains on NFCorpus and TREC-COVID.
- RRF union is modestly positive on ArguAna and TREC-COVID, but flat or worse on
  FiQA and SciFact.
- The remaining gap is now mostly a **discrimination gap over a better candidate
  union**, not just a BM25 recall gap.

This is useful theorem progress: we can now separate two facts that were
previously conflated:

1. Changing `G` increases the measurable upper bound.
2. Changing `G` alone is insufficient; we need `S`/`F` improvements that exploit
   the larger union pool.

## Next iteration

Try scorer/fusion variables over the same union pool:

- weighted z-score fusion instead of RRF;
- query-adaptive gate: BM25 vs RM3 based on clarity/NQC;
- oracle-assisted diagnostics to learn which unsupervised query features predict
  whether RM3 helps.
