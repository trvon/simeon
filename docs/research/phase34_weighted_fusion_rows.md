# Phase XXXIV — weighted fusion rows over generator union

## Goal

Test `F` (fusion) variables after `G` (generator) variables lifted the oracle.
The same generator set is used:

```text
G_union = BM25 ∪ BM25F(TextRankTopSentence) ∪ RM3
```

Added weighted z-score fusion rows:

- `observed_union_bm25_bm25f_rm3_z_equal`
- `observed_union_bm25_bm25f_rm3_z_bm25_rm3`
- `observed_union_bm25_bm25f_rm3_z_rm3_heavy`

Artifact:

```text
/tmp/simeon_weighted_fusion_20260504_021935
```

## Results, nDCG@10

| Corpus | Fold | BM25 | RM3 | RRF@100 | z equal | z bm25/rm3 | z rm3-heavy | Best observed | Union oracle@100 | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ArguAna | dev | 0.3202 | 0.3105 | 0.3209 | 0.3231 | 0.3208 | 0.3189 | 0.3231 | 0.9598 | +0.6367 |
| ArguAna | test | 0.3293 | 0.3243 | 0.3311 | 0.3334 | 0.3313 | 0.3296 | 0.3334 | 0.9646 | +0.6312 |
| FiQA | dev | 0.1908 | 0.1805 | 0.1881 | 0.1920 | 0.1906 | 0.1900 | 0.1920 | 0.5391 | +0.3471 |
| FiQA | test | 0.2053 | 0.1996 | 0.2046 | 0.2074 | 0.2051 | 0.2063 | 0.2074 | 0.5523 | +0.3449 |
| NFCorpus | dev | 0.2620 | 0.2784 | 0.2662 | 0.2806 | 0.2774 | 0.2773 | 0.2806 | 0.5714 | +0.2908 |
| NFCorpus | test | 0.2521 | 0.2726 | 0.2523 | 0.2603 | 0.2648 | 0.2678 | 0.2726 | 0.5286 | +0.2560 |
| SciFact | dev | 0.6623 | 0.6659 | 0.6656 | 0.6615 | 0.6663 | 0.6593 | 0.6663 | 0.8838 | +0.2175 |
| SciFact | test | 0.6188 | 0.5998 | 0.6146 | 0.6094 | 0.6135 | 0.6100 | 0.6188 | 0.8939 | +0.2751 |
| TREC-COVID | dev | 0.4943 | 0.5644 | 0.5521 | 0.5446 | 0.5465 | 0.5512 | 0.5644 | 0.9781 | +0.4137 |
| TREC-COVID | test | 0.5649 | 0.6064 | 0.5860 | 0.5796 | 0.5891 | 0.5981 | 0.6064 | 0.9847 | +0.3783 |

## Interpretation

Weighted z-score fusion gives small positive movement on ArguAna, FiQA,
NFCorpus-dev, and SciFact-dev, but it still does not exploit most of the union
oracle. On NFCorpus-test and TREC-COVID, the single RM3 generator remains the
best observed row.

This narrows the next theorem iteration: global static fusion weights are not
enough. The missing variable is likely query-adaptive routing/gating:

```text
F(q) = choose_or_weight(BM25, BM25F, RM3) from unsupervised query features
```

Candidate features:

- clarity / simplified clarity
- NQC / WIG
- query length and OOV rate
- RM3 drift indicators
- generator-pool overlap

## Next step

Add a diagnostic that records per-query winner among BM25, RM3, and z-fusion,
then correlate winner with existing unsupervised query features. This keeps the
proof direction clean: learn which feature gates can move `BestObserved_TF`
toward `Oracle_G_union` without qrels in the final system.
