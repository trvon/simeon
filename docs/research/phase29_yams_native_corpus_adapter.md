# Phase XXIX — YAMS native corpus adapter cleanup

## Goal

Move the ArguAna ceiling lesson into library shape without keeping benchmark-only
helpers on the hot path. The useful abstraction is not an ArguAna ID parser; it
is a deterministic corpus adapter that emits structured evidence into YAMS
fusion.

## Implementation

Parent YAMS now has a first-class `yams::search::CorpusAdapter` interface and a
built-in `YamsNativeCorpusAdapter`.

The native adapter is English-first and seed-based:

1. preserve structured query tokens (`docs/research`, `PBI-043`, file-like or
   path-like fragments);
2. remove common English stopwords from natural-language requests;
3. emit compact content-term and short phrase seeds;
4. query YAMS path metadata with `containsUsesFts=true` for each bounded seed;
5. query exact structured metadata filters (`key=value`) separately;
6. fuse matches as `ComponentResult::Source::CorpusAdapter`.

This turns a request like:

```text
find the research docs about language corpus support
```

into path/content seeds such as:

```text
research, docs, language, corpus, support, language corpus, corpus support
```

which can match repository paths and agent-memory document names even when the
whole sentence would not.

## Profiling hooks

Each adapter result carries debug metadata:

- `corpus_adapter=yams_native`
- `adapter_signal=path_seed|metadata_filter`
- `seed`, `seed_kind`, `seed_count`
- `path_seed_queries`
- `adapter_us`

At the search-engine level the component also appears as `corpus_adapter` in
component timing/debug output when debug stats are enabled.

## Cleanup

The ArguAna diagnostic rows remain available for reproduction, but are no
longer emitted by the default reference benchmark. Set
`SIMEON_ARGUANA_DIAGNOSTICS=1` to reproduce phase20–28 rows. This keeps the
research evidence accessible while preventing fixture-specific diagnostics from
being mistaken for production retrieval recipes.

## Theorem update

The Phase XIX theorem now distinguishes:

- universal training-free recipes, which are corpus-agnostic scoring/fusion
  strategies; and
- corpus adapters, which are still training-free when they expose deterministic
  observed structure without qrel labels.

Adapters raise the constructive floor `Router(C,S)` when the structure is real;
they do not change the BM25-pool oracle unless they also change candidate
generation.
