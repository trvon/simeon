# Language and Corpus Support Contract

## Why this matters

The ArguAna work shows that "getting closer to the ceiling" is not only a
better scorer problem. It is a **corpus-language contract** problem:

- BM25/topical matching supplies candidate recall.
- Corpus structure supplies the small neighborhood where the answer lives.
- Language-specific discourse markers and stopwords help rank the correct
  relation inside that neighborhood.

For English ArguAna, `arguana_text_pair_discriminator_p5` reaches:

| Fold | nDCG@10 |
|---|---:|
| dev | 0.6267 |
| test | 0.6373 |

That branch is still below `.80`, but it is already far above BM25 because it
uses corpus structure plus English relation cues.

## Required support for a new language

A language-specific relation-aware recipe needs four things:

1. **A fixture/corpus with qrels**
   - corpus TSV, query TSV, qrels TSV;
   - dev/test split;
   - a frozen reference embedding file if we want the standard benchmark table.
2. **Tokenizer adequacy**
   - whitespace/lowercase works for English-like languages;
   - CJK, Thai, agglutinative languages, and heavy diacritics need tokenizer or
     normalization changes before cue features are reliable.
3. **Lexical profile**
   - stopwords;
   - contradiction/rebuttal cue terms;
   - optional stemming/diacritic policy.
4. **Corpus-structure adapter**
   - ArguAna uses debate-page headers and source-text containment;
   - other corpora may use titles, sections, paths, citations, doc ids, or
     metadata fields instead.

## Implemented scaffold

`benchmarks/bench_vs_reference.cpp` now has a benchmark-side language profile
hook for the relation discriminator:

```bash
SIMEON_LANGUAGE_PROFILE=en   # default
SIMEON_LANGUAGE_PROFILE=es
SIMEON_LANGUAGE_PROFILE=fr
```

The current profiles provide stopword and cue sets for English, Spanish, and
French. Only English has been measured because the repo currently has only the
English ArguAna fixture.

This is intentionally bench-side first: it lets us validate the corpus/language
contract before moving anything into the public `simeon::` API.

## Next corpus recommendations

To continue toward `.80`, add one of:

1. **A second debate/counterargument corpus in English**
   - validates whether the ArguAna structure branch generalizes beyond a single
     fixture.
2. **A Spanish or French ArguAna-style corpus**
   - validates the new `SIMEON_LANGUAGE_PROFILE` path;
   - requires only stopword/cue tuning if the corpus is whitespace-tokenizable.
3. **A structured-document corpus from yams**
   - lets us test title/path/section structure rather than debate-pair
     structure;
   - likely needs a different adapter but the same ceiling/runway methodology.

## Research rule going forward

Do not call a recipe "universal" until it is validated as:

```text
(corpus structure adapter) × (language profile) × (dev/test qrels)
```

The ceiling is corpus-shaped and language-shaped. The code should make both
axes explicit instead of hiding them inside a single English-only heuristic.
