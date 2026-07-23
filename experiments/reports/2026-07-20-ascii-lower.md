# ASCII case-folding optimization

## Summary

- Scope analyzed: exact encoder retrieval and the artifact-free hashed n-gram representation.
- Stack detected: C++20, Meson, and dependency-free Python fixture preparation.
- Test/build commands: Meson focused tests, full Meson suite, and the manifest-driven BEIR runner.
- Highest-impact efficiency hotspot: retaining and sorting every query-document score when the deepest metric cutoff is 100.
- Quality hypothesis: deterministic ASCII case folding removes query/document case mismatches without fitted state.
- Patch status: implemented and selected on development data; holdout remains unread.
- Code state measured: `1f26ef6-dirty`, SIMD tier `neon`.

## Findings

### Exact ranking retention

- Location: `benchmarks/experiment_support.cpp`.
- Previous pattern: materialize all `Q * D` scores and fully sort each query ranking.
- Previous complexity: `O(Q * D)` ranking memory and `O(Q * D log D)` ranking selection work, in addition to exact dot products.
- Change: retain an exact deterministic top-100 min-heap while scoring, then sort only the retained prefix.
- New complexity: `O(Q * K)` ranking memory and `O(Q * D log K + Q * K log K)` selection work for fixed `K = 100`.
- Equivalence: every current metric has cutoff at or below 100; tests and all three real fixtures produced identical metrics before and after the change.
- Risk: a future metric with cutoff above 100 must raise the configured retrieval depth.

For the projected baseline, estimated ranking working memory fell from 34,983,608 to 2,086,432 bytes on SciFact (16.8x), 10,111,008 to 953,472 on NFCorpus (10.6x), and 231,516,608 to 1,364,608 on FiQA (169.7x). Evaluation time fell by approximately 112x, 44x, and 1,288x respectively. Exact dot-product scoring remains `O(Q * D * d)` and is now the dominant cost.

### ASCII case folding

- Location: `EncoderConfig::text_normalization` and `Encoder::Impl::encode_one`.
- Previous pattern: hash original byte case, so otherwise identical query and document terms could occupy unrelated buckets.
- Change: optional locale-independent `A-Z` to `a-z` folding before tokenization; all non-ASCII bytes remain unchanged.
- Complexity: `O(L)` time over input bytes and a thread-local `O(L)` scratch buffer; sketch, projection, vector storage, and retrieval complexity are unchanged.
- Training regime: `artifact-free`; the transform has no corpus statistics, learned weights, or model artifact.
- Compatibility: the default remains `none`, so existing embeddings do not silently change.

## Development results

The wide comparison uses manifest `fnv1a64:e8ee81180f287289`; the compact controlled comparison uses `fnv1a64:0b3dc4e9c483bf2d`. Fixture fingerprints are SciFact `fnv1a64:fe2bdf87a26e671b`, NFCorpus `fnv1a64:315ceecf6d8146b3`, and FiQA `fnv1a64:7faa27cd473f31d5`.

| Configuration | Corpus | Raw nDCG@10 | ASCII-lower nDCG@10 | Absolute delta | Relative delta |
|---|---|---:|---:|---:|---:|
| char+word, 32,768d | SciFact | 0.481130 | 0.502608 | +0.021478 | +4.46% |
| char+word, 32,768d | NFCorpus | 0.223677 | 0.232064 | +0.008387 | +3.75% |
| char+word, 32,768d | FiQA | 0.129883 | 0.142925 | +0.013042 | +10.04% |
| char, 4,096d to 384d Achlioptas | SciFact | 0.375650 | 0.393494 | +0.017844 | +4.75% |
| char, 4,096d to 384d Achlioptas | NFCorpus | 0.149886 | 0.175300 | +0.025413 | +16.96% |
| char, 4,096d to 384d Achlioptas | FiQA | 0.081906 | 0.093749 | +0.011844 | +14.46% |

The compact representation keeps the same 384 output dimensions and identical estimated working memory. Paired encoding and scoring timings were within ordinary run-to-run variation. The compact lowercase variant is frozen in `embedding-compact-normalization-frozen.exp`, linked to the dev selection manifest. It was later consumed once as the frozen holdout baseline for the compact FWHT optimization; that result is reported separately.

## Verification

- Focused regression tests cover opt-in behavior, default case sensitivity, manifest resolution, exact top-K equivalence, deterministic ties, and block-size invariance.
- The real-fixture comparator produced exactly unchanged metrics for the ranking-retention optimization.
- ASCII folding improved the primary metric on all three dev fixtures in both the wide and compact spaces.
- Residual risks: ASCII folding is intentionally not Unicode case folding;
  results are three-domain evidence rather than a universal quality guarantee;
  the later holdout read validates the downstream compact FWHT recipe, not an
  isolated no-fold versus fold comparison.
