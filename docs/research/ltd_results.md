# Axiomatic LTD correction (Fang-Zhai 2005) — negative result with inversion

Tests Fang-Zhai 2005's axiomatic Length Term-Discrimination (LTD)
correction: replace BM25's linear length normalization
`L(dl) = 1 - b + b · (dl/avg_dl)` with a sublinear power form
`L(dl) = 1 - b + b · (dl/avg_dl)^α`, where α ∈ (0, 1] gates how
aggressively the long-doc penalty is reduced. Implemented as
`Bm25Variant::AtireLTD` with `Bm25Config::ltd_alpha`. Result: on the
three BEIR fixtures the **prediction is inverted** — the plan's
"long-doc corpus benefits" hypothesis fails because FiQA (where the
plan placed its validate gate) is the **shortest-doc corpus** of the
three, not the longest. LTD correction helps recall on the actual
long-doc corpora (scifact, nfcorpus) but the plan's expected nDCG
lift on FiQA materializes as a major regression instead.

## Math

    L_atire(dl) = 1 - b + b · (dl/avg_dl)
    L_LTD(dl)   = 1 - b + b · (dl/avg_dl)^α        (α ∈ (0, 1])

α=1 recovers Atire byte-identically (verified by sanity row below).
α<1 makes the length penalty grow sublinearly: long docs are
penalized less, short docs are penalized more. Fang-Zhai recommend
α≈0.5 as the LTD-axiom-satisfying midpoint; α=0.3 / α=0.7 bracket it.

## Three-corpus measurement

| Corpus   | avg_dl | bm25_atire | LTD α=1.0 ✓ | LTD α=0.7 | LTD α=0.5 | LTD α=0.3 |
|----------|-------:|-----------:|------------:|----------:|----------:|----------:|
| **nDCG@10** |     |            |             |           |           |           |
| scifact  | 214.6  | 0.6188     | 0.6188      | 0.6178    | 0.6151    | 0.6083    |
| nfcorpus | 233.8  | 0.2521     | 0.2521      | 0.2527    | 0.2521    | 0.2539    |
| fiqa     | **132.9** | 0.2053  | 0.2053      | 0.2029    | 0.1882    | 0.1656    |
| **R@100**   |     |            |             |           |           |           |
| scifact  | 214.6  | 0.8736     | 0.8736      | 0.8785    | 0.8785    | 0.8785    |
| nfcorpus | 233.8  | 0.1991     | 0.1991      | 0.1997    | 0.2015    | 0.2020    |
| fiqa     | 132.9  | 0.4672     | 0.4672      | 0.4596    | 0.4449    | 0.4174    |

Corpus avg_dl computed from `fixtures/{c}-minilm/corpus.tsv` (whole
corpus, word-tokenized).

## Plan gate verdicts

Plan validate: **+0.010 nDCG@10 on FiQA**, ±0.003 on
scifact/nfcorpus.
- FiQA observed at recommended α=0.5: **−0.0171** (8% relative loss).
  Validate gate definitively rejected.
- scifact at α=0.5: −0.0037 (just outside ±0.003).
- nfcorpus at α=0.5: 0.0000 (within bound).

Plan disprove: **regression >0.005 on nfcorpus** → "LTD overcorrects
on short-doc corpora."
- nfcorpus actually *improves* (+0.0018 at α=0.3, +0.0006 at α=0.7).
  Disprove gate not met.

Result type: **third category** — neither the plan's validate nor its
disprove gate fires; the experiment is invalidated by the **plan's
wrong premise about which corpus is long**. T5 is functionally
disproved (no useful α setting exists) but for a different reason
than the plan anticipated.

## Mechanism — the inversion

The plan asserted "FiQA avg ~200 words" and used FiQA as the
long-doc corpus where LTD's recall benefit should manifest. The
actual measurement shows FiQA is the **shortest** corpus by avg_dl
(132.9 words/doc), with nfcorpus longest (233.8) and scifact
intermediate (214.6).

LTD's effect, looking at the data side-by-side:

- On the **long-doc corpora** (scifact, nfcorpus) LTD α<1 *helps
  R@100*: scifact +0.0049, nfcorpus +0.0029. The long-doc penalty
  reduction lifts recall by retrieving longer docs that BM25's
  default `b=0.75` was suppressing. nDCG@10 stays roughly flat —
  LTD trades top-10 ranking precision for deeper recall.
- On the **short-doc corpus** (fiqa) LTD α<1 *hurts both*: nDCG@10
  −0.0171 at α=0.5, R@100 −0.0223. With docs already shorter than
  avg_dl, reducing the penalty on a few outlier-long docs amplifies
  their score and pushes them above the genuinely relevant short
  matches. Topic drift wins.

In short: LTD does what it's supposed to do — reduce the long-doc
penalty — but the BEIR-3 fixture set positions FiQA on the **wrong
side** of the corpus length distribution to benefit. The Fang-Zhai
prediction is consistent with the data once the plan's avg_dl claim
is corrected.

This matches the recurring "corpus-bound, not query-bound" finding
from T3 (WSDM) and T4 (RM3 adaptive K): per-corpus optimal length
treatment is real and observable, but no single universal recipe
wins across short-doc + long-doc corpora simultaneously.

## Subfinding — recall-precision split for long-doc corpora

LTD on scifact/nfcorpus is a clean recall-vs-precision trade. R@100
gains land at α=0.7 already (within 1 nDCG@10 point of baseline);
deeper α=0.3 keeps the recall and trades a small nDCG drop on
scifact for an nDCG bump on nfcorpus. This makes LTD α≈0.7 a
plausible drop-in for **recall-targeted** runs on long-doc corpora,
even though it doesn't satisfy the plan's nDCG-targeted gate.

| α    | scifact ΔR@100 | scifact ΔnDCG@10 | nfcorpus ΔR@100 | nfcorpus ΔnDCG@10 |
|------|---------------:|-----------------:|----------------:|------------------:|
| 0.7  | +0.0049        | −0.0010          | +0.0006         | +0.0006           |
| 0.5  | +0.0049        | −0.0037          | +0.0024         |  0.0000           |
| 0.3  | +0.0049        | −0.0105          | +0.0029         | +0.0018           |

scifact's R@100 saturates at α=0.7 (no extra recall from going
deeper); nfcorpus's R@100 grows monotonically through α=0.3. The
saturation point is itself corpus-bound.

## Infrastructure disposition

- `Bm25Variant::AtireLTD` and `Bm25Config::ltd_alpha` ship in
  `simeon/bm25.hpp`. α=1 recovers Atire byte-identically (verified
  on all 3 corpora) so the variant is no-op-safe for callers that
  set α=1 or omit `ltd_alpha`.
- Cost: one `powf` per `(term, doc)` evaluation when α≠1; fast path
  bypasses `powf` at α=1. Net query latency on scifact within ~3% of
  bm25_atire.
- Bench rows (`bm25_atire_ltd_a{1.0, 0.7, 0.5, 0.3}`) stay in the
  three `*_full.jsonl` result files for regression tracking.
- Not invoked by `run_router_cascade()` or any default router
  recipe.

## Next lever

- **Recall-targeted scifact/nfcorpus path**: the +0.005 R@100 lift
  on scifact at α=0.7 is a real (if narrow) improvement and could be
  routed via the existing passE recipe gating on `avg_dl_corpus >
  150`. Cost: one router branch, no new variant. Worth ~30 minutes
  of router code + one bench rerun.
- **The plan's premise assumed a corpus mix simeon doesn't have**.
  Adding a genuinely long-doc corpus (e.g., trec-covid, avg_dl ~290)
  would be the principled way to test the Fang-Zhai prediction at
  the long end. Out of scope for the current 3-fixture set.

## References

- Fang, H. & Zhai, C. (2005). "An Exploration of Axiomatic
  Approaches to Information Retrieval." SIGIR 2005.
- Lv, Y. & Zhai, C. (2011). "When Documents Are Very Long, BM25
  Fails!" SIGIR 2011 (related: BM25L variant already in simeon).
- Per-corpus failure mode mirrors `docs/wsdm_results.md` (T3) and
  `docs/rm3_adaptive_results.md` (T4): no universal recipe across
  short-doc + long-doc fixtures.
