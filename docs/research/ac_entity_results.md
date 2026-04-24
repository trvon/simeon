# Aho-Corasick self-bootstrapped entity field

Experiment:

1. Use TextRank top-2 sentences per document as the phrase-harvesting source.
2. Build corpus-local bigram/trigram candidates with document frequency in `[10, 0.1·N]`, filtering stopword-heavy phrases.
3. Build one Aho-Corasick dictionary per corpus.
4. Emit matched pattern IDs (`entNNN`) as the auxiliary field for both documents and queries.
5. Score with BM25F weights `w_aux ∈ {0.2, 0.5, 1.0}` against the body-only BM25 baseline.

Validation bar: lift nDCG@10 by at least `+0.005` on at least two corpora or `+0.010` on one corpus, with no regression worse than `-0.003` elsewhere.

## Results

| Corpus | Patterns | Baseline | `w=0.2` | `w=0.5` | `w=1.0` | Best Δ |
|--------|---------:|---------:|--------:|--------:|--------:|-------:|
| scifact | 985 | 0.6188 | 0.6170 | 0.6137 | 0.6124 | -0.0018 |
| nfcorpus | 804 | 0.2521 | 0.2521 | 0.2513 | 0.2513 | +0.0000 |
| fiqa | 24,755 | 0.2053 | 0.2079 | 0.2006 | 0.1780 | +0.0026 |

## Verdict

**Disproved.** The best gain is FiQA `w=0.2` at `+0.0026`, below the promote threshold, and the other corpora are flat-to-negative.

Interpretation: the self-bootstrapped dictionary does produce a tiny lexical
signal on FiQA once queries are transformed into the entity-token space, but it
is not enough to claim a meaningful retrieval lever on BEIR body-only fixtures.
The next honest test of the GLiNER-replacement hypothesis needs either:

- external domain dictionaries (for example UMLS / Wikidata subsets), or
- a structured-document corpus where entity tags are not just another view of the same short body text.
