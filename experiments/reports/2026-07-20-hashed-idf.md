# Bounded hashed-IDF optimization

## Summary

- Baseline: the frozen artifact-free lowercase, word-bounded char+word,
  8,192-bucket FWHT-to-384 compact encoder.
- Selected recipe: the same encoder with a 65,536-bucket all-feature document-
  frequency artifact and Robertson-style smoothed IDF in Q10 fixed point.
- Outcome: frozen holdout nDCG@10 improves by 25.77% on SciFact, 33.06% on
  NFCorpus, and 57.57% on FiQA; recall@100 improves on every fixture.
- Cost: a 128 KiB persistent artifact, 0.13–0.92 seconds one-time build on the
  development corpora, and 12–13% median encoding overhead. Output remains 384
  floats and exact-scoring time is unchanged.
- Training regime: `corpus-adaptive`; documents only, with no qrels, pretrained
  representation, or gradient optimization.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Evidence and mechanism

Paperbridge resolved Salton and Buckley, _Term-weighting approaches in automatic
text retrieval_ (DOI `10.1016/0306-4573(88)90021-0`), Robertson and Zaragoza,
_The Probabilistic Relevance Framework: BM25 and Beyond_ (DOI
`10.1561/1500000019`), and Chisholm and Kolda, _New Term Weighting Formulas for
the Vector Space Method in Information Retrieval_ (DOI `10.2172/5698`). These
works motivate document-frequency-aware term weighting; the compact hashed
artifact and all retrieval claims below are Simeon-specific empirical results.

`HashedIdf::learn()` tokenizes every deployment document with the exact target
encoder identity and counts at most one occurrence per document in a bounded
hash table. It stores
`round(1024 * log((N + 1) / (df + 0.5)))` as a 16-bit weight. Collisions can
only raise observed `df`, conservatively reducing a feature's weight. The
artifact records the n-gram mode/range, normalization, boundary scope, hash
family, seed, feature scope, document count, and table width; an incompatible
encoder rejects it. The deterministic little-endian artifact is serializable
and has its own content fingerprint.

Building is `O(F + H)` time and `O(H)` memory for total emitted corpus features
`F` and table width `H`. At `H = 65,536`, the builder uses two transient
32-bit tables (512 KiB) and emits a 128 KiB 16-bit artifact. Encoding retains
`O(f + S log S + D)` work for per-text features `f`, sketch width `S = 8192`,
and output width `D = 384`, adding one table lookup and fixed-point multiply per
feature. Checked in-place sketch accumulation removed a redundant 64-bit sketch
and full conversion pass, reducing measured overhead from about 30% to 12–13%.
Exact scoring remains `O(Q * N * 384)`.

Lean classifies hashed document frequency as corpus statistics, proves the
composed recipe is allowed in `corpus-adaptive`, and proves it is not
`artifact-free`.

## Development selection

The final capacity manifest is `fnv1a64:4b312a3d3b1e49eb`. Fixture fingerprints
are SciFact `fnv1a64:fe2bdf87a26e671b`, NFCorpus
`fnv1a64:315ceecf6d8146b3`, and FiQA `fnv1a64:7faa27cd473f31d5`.

| Corpus | Raw nDCG@10 | 64K IDF nDCG@10 | Relative delta | Recall@100 before | Recall@100 after |
|---|---:|---:|---:|---:|---:|
| SciFact | 0.445894 | 0.572972 | +28.50% | 0.764998 | 0.842975 |
| NFCorpus | 0.197035 | 0.253910 | +28.87% | 0.175106 | 0.199700 |
| FiQA | 0.117932 | 0.157384 | +33.45% | 0.293665 | 0.383360 |

Capacity and feature-scope neighbors were kept on development data:

| Variant nDCG@10 | SciFact | NFCorpus | FiQA | Artifact |
|---|---:|---:|---:|---:|
| all-feature 16K | 0.555169 | 0.248922 | 0.152085 | 32 KiB |
| all-feature 32K | 0.565541 | 0.252511 | 0.154132 | 64 KiB |
| **all-feature 64K** | **0.572972** | **0.253910** | **0.157384** | **128 KiB** |
| all-feature 256K | 0.575051 | 0.253342 | 0.157961 | 512 KiB |
| all-feature 1M | 0.572534 | 0.254185 | 0.159334 | 2 MiB |

The 64K table is the quality/memory Pareto choice: larger tables trade 4–16x
storage for mixed changes no larger than about 0.0021 nDCG. At 64K,
character-only IDF scores
0.574845/0.252292/0.156719 and word-only scores
0.515960/0.213368/0.134171. Character features carry most of the effect, but
all-feature weighting is the consistent fixed choice across corpora.

After the in-place optimization, median-of-three sequential development runs
show 12.62%, 13.24%, and 12.11% encoding overhead on SciFact, NFCorpus, and
FiQA. Median artifact builds take 0.170, 0.127, and 0.923 seconds. Scoring time
changes by at most 0.13%.

## Frozen holdout

Frozen manifest `fnv1a64:ba50161ed4b26ba4`, linked to the final development
manifest and containing only the selected variant, was read once per test
fixture. Artifact fingerprints are SciFact `fnv1a64:a939e3b00e701f18`,
NFCorpus `fnv1a64:71995d806eef78b2`, and FiQA
`fnv1a64:ec48745f1ef0a9cd`.

| Corpus | Test fingerprint | Raw nDCG@10 | 64K IDF nDCG@10 | Relative delta | Recall@100 before | Recall@100 after |
|---|---|---:|---:|---:|---:|---:|
| SciFact | `fnv1a64:cdc576493e9c566a` | 0.440093 | 0.553510 | +25.77% | 0.735333 | 0.848889 |
| NFCorpus | `fnv1a64:a99992f39ee60a0b` | 0.198537 | 0.264164 | +33.06% | 0.183156 | 0.209313 |
| FiQA | `fnv1a64:a8fd6ec5bfc2f5fe` | 0.089862 | 0.141598 | +57.57% | 0.270579 | 0.404254 |

## Learned-reference gap and next question

This is a standalone compact-embedding comparison, not a claim of learned-model
parity. Against the repository's frozen `all-MiniLM-L6-v2` reference, the new
standalone scores still trail by 0.0916 on SciFact, 0.0525 on NFCorpus, and
0.2271 on FiQA. On FiQA it closes 18.6% of the old compact encoder's gap to that
reference. The existing multi-leg learning-free fusion remains stronger at
0.6990/0.3261/0.2512 and has not yet consumed this IDF leg. The next controlled
experiment is therefore complementarity in candidate generation/fusion, not an
assumption that standalone gains will add linearly.

## Verification and residual risk

- Unit tests cover deterministic learning, rare/common ordering, feature
  scopes, serialization round trips, identity mismatch rejection, malformed
  inputs, exact encoder integration, and the promoted preset.
- The experiment contract strictly separates artifact-free and corpus-adaptive
  variant kinds and records artifact provenance, build time, and resident size.
- Full normal tests, focused sanitizer tests, Python fixture tests, formatting,
  `git diff --check`, and the Lean contract pass.
- Residual risks: document-frequency artifacts must be rebuilt when corpus or
  encoder identity changes; 64K collisions can suppress useful rare features;
  16-bit Q10 weights define embedding identity; three BEIR corpora do not prove
  universal transfer; and the result remains lexical rather than paraphrastic.
