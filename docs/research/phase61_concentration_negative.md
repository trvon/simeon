# Phase LXI: Term-Concentration Structural Boost (Negative)

## Goal

Address the structural gap by rewarding documents where query terms appear in a
concentrated span rather than scattered throughout. The intuition: a document
that mentions all query terms in 50 words is more likely to contain a focused
relevant passage than one that mentions them across 500 words.

## Method

For each document in BM25 top-200, compute the span (last matched position −
first matched position) as a fraction of document length. High concentration
(small span) gets a small additive boost gated by BM25 z-score and entropy.

```text
conc = 1 − span / doc_word_count
boost = λ · conc · 0.3 · sigmoid(z_bm25)
```

Row: `observed_ordering_conc_span`

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| Conc span | 0.3293 | 0.2056 | 0.2524 | 0.6198 | 0.5602 | 0.3935 |
| Diverse RM3 | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4108 |

Term concentration is slightly negative (−0.0006 macro). No corpus benefits
meaningfully.

## Why it fails

BM25's TF normalization already captures term concentration. Documents where
query terms appear with high local frequency get high TF scores. The
concentration signal is redundant — it reinforces what BM25 already does.

More fundamentally: **relevant and irrelevant documents have similar term
concentration patterns.** On TREC-COVID, both a paper about remdesivir trials
(relevant) and a paper that mentions remdesivir in one sentence (irrelevant)
have concentrated query term hits. The signal can't discriminate relevance;
it can only capture the already-captured notion of topical similarity.

The same applies across all corpora: relevance is about the MEANING of the
matched text, not where it appears in the document.

## Impact

Passage-level structural signals (position, concentration, proximity) cannot
close the ordering gap because they don't capture relevance semantics. The gap
requires signals that distinguish "about X" from "mentions X," which requires:

1. **Entity-level evidence** — is remdesivir the SUBJECT of this abstract or
   just a mentioned term?
2. **Claim-fact matching** — does this abstract contain the SPECIFIC fact
   claimed by the query?
3. **Counter-argument detection** — does this argument OPPOSE the query claim?

These are semantic tasks beyond the reach of term-level statistics.
