# TextRank synthetic-title field

Experiment: extract `TextRank::top_sentence()` for each document, index it as
an auxiliary field, and score with BM25F weights `w_aux ∈ {0.2, 0.5, 1.0}`
against the body-only BM25 baseline.

Validation bar: lift nDCG@10 by at least `+0.005` on at least one of `{scifact, nfcorpus, fiqa}` with no regression worse than `-0.003` on the other two.

## Results

| Corpus | Baseline | `w=0.2` | `w=0.5` | `w=1.0` | Best Δ |
|--------|---------:|--------:|--------:|--------:|-------:|
| scifact | 0.6188 | 0.6053 | 0.5898 | 0.5546 | -0.0135 |
| nfcorpus | 0.2521 | 0.2471 | 0.2358 | 0.2229 | -0.0050 |
| fiqa | 0.2053 | 0.2024 | 0.1783 | 0.1506 | -0.0029 |

## Verdict

**Disproved.** No weight beats body-only BM25 on any corpus; every setting regresses nDCG@10, and the regressions on scifact / nfcorpus are well beyond the allowed tolerance.

Interpretation: on these body-only BEIR fixtures, the top-ranked sentence is
not acting like a useful title field. Reweighting one sentence mostly throws
away evidence instead of concentrating it. This primitive needs a
structured-document evaluation where a title-like sentence is genuinely
distinct from the body.
