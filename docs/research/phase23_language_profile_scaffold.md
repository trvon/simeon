# Phase XXIII: Language Profile Scaffold

## Motivation

The user-level question was whether getting closer to the ceiling for specific
languages requires adding corpora and language support. The answer is yes:
relation-aware retrieval depends on both corpus structure and language-specific
lexical/discourse signals.

Phase XXIII therefore adds the first benchmark-side language profile scaffold
before adding more corpora.

## Implementation

`benchmarks/bench_vs_reference.cpp` now supports:

```bash
SIMEON_LANGUAGE_PROFILE=en   # default
SIMEON_LANGUAGE_PROFILE=es
SIMEON_LANGUAGE_PROFILE=fr
```

The profile currently controls:

- stopword removal for content-set Jaccard features;
- contradiction/rebuttal cue counts for the pair discriminator.

Profiles are intentionally local to the benchmark harness until we validate a
second corpus/language.

## Regression check

English ArguAna test still reproduces Phase XXII:

| Row | nDCG@10 |
|---|---:|
| `oracle_bm25_pool_k100` | 0.9269 |
| `arguana_text_neighborhood_p5` | 0.4345 |
| `arguana_text_pair_discriminator_p5` | 0.6373 |
| `arguana_pair_id_diagnostic` | 1.0000 |

## Implication

The next work item is not another English-only tweak first. It is adding a
second fixture so we can test the full contract:

```text
language profile × corpus structure adapter × qrels
```

Recommended next fixture:

1. Spanish/French debate/counterargument corpus if available;
2. otherwise an English second debate corpus;
3. otherwise a structured yams corpus with title/path/section metadata.

Once a second fixture exists, Phase XXIV should measure whether the pair
discriminator transfers or whether each corpus needs a different structure
adapter.
