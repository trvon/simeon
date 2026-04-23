# PMI soft-match results

Experiment: learn corpus PMI embeddings, expand each query term with its nearest lexical neighbors, then score the resulting weighted query with `Bm25Index::score_weighted_hashes()`. This is the portable soft-matching / transport proxy: no seeded ontology, no supervised model, no external dictionary.

Implementation shipped in `benchmarks/bench_vs_reference.cpp` behind `--softmatch-only`.

## Protocol

- PMI config: rank 128, `min_token_count=5`, `max_vocab_size=20,000`.
- Two neighbor regimes:
  - `k=3`, `min_similarity=0.35`
  - `k=8`, `min_similarity=0.20`
- Blend weights `lambda ∈ {0.2, 0.5, 1.0}` where:
  - `(1 - lambda)` keeps the exact query mass
  - `lambda` redistributes mass over PMI-neighbor terms

## Results

| Corpus | Baseline BM25 | Best soft-match row | Δ | Notes |
|--------|--------------:|--------------------:|--:|-------|
| scifact | 0.6188 | 0.6188 (`k=3/8`, `lambda=0.2/0.5`) | +0.0000 | low-blend rows are ranking-identical at 4 d.p.; `lambda=1.0` collapses to 0.0018 |
| nfcorpus | 0.2521 | 0.2521 (`k=3/8`, `lambda=0.2/0.5`) | +0.0000 | low-blend rows are inert; `lambda=1.0` falls to 0.0319 |
| fiqa | 0.2053 | 0.2053 (`k=3/8`, `lambda=0.2/0.5`) | +0.0000 | low-blend rows are inert; `lambda=1.0` falls to 0.0000 |

## Verdict

**Disproved on the shipped BEIR-3 body-only fixtures.**

Portable PMI-neighbor soft matching does not produce measurable retrieval signal here:

1. Safe blend weights (`lambda=0.2`, `0.5`) are numerically inert at the published precision on all three corpora.
2. Letting the soft-match leg dominate (`lambda=1.0`) destroys ranking quality, so the semantic neighbors are not a viable replacement for exact lexical evidence.
3. Widening the neighborhood (`k=3 -> 8`) and lowering the similarity floor (`0.35 -> 0.20`) does not change the outcome.

Interpretation: corpus-derived unigram-neighbor transport remains too close to lexical reweighting on these short, body-only fixtures. The math space is different from BM25 scoring, but the representation is still too weak: PMI nearest neighbors are either too conservative to change the ranking or too noisy to stand on their own.

That leaves two honest next directions:

- stronger transport structure than unigram-to-unigram neighbor mass (phrase/document graph transport, not just term expansion), or
- a corpus with explicit structure/metadata so non-body fields and normalization can be evaluated fairly.
