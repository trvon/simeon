# Phase LXVI: Simeon-Embedding Pseudo-Relevance Feedback (Negative)

## Goal

Standard RM3 uses BM25 top-K to select pseudo-relevance docs. This phase tests
the opposite: use simeon embedding similarity (dot product) to select the top-10
pseudo-relevance docs. The hypothesis: embedding-similar docs have different
vocabulary than BM25-similar docs, providing complementary expansion terms.

## Method

```text
1. Compute simeon dot product for all docs
2. Top-10 by simeon similarity → pseudo-relevance set
3. Build relevance model from these docs (same as RM3)
4. α-blend (α=0.5) with original query, score with weighted-hash BM25
```

Row: `observed_ordering_simeon_prf`

## Results

| Row | ArguAna | FiQA | NFCorpus | SciFact | TREC-COVID | Macro |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3293 | 0.2053 | 0.2521 | 0.6188 | 0.5649 | 0.3941 |
| Simeon PRF | 0.2994 | 0.2030 | 0.2674 | 0.6071 | 0.5837 | 0.3921 |
| Diverse RM3 (BM25 PRF) | 0.3327 | 0.2000 | 0.2789 | 0.6023 | 0.6400 | 0.4108 |

Simeon PRF scores below BM25 on ArguAna (−0.0299) and SciFact (−0.0117).
The simeon embedding's standalone retrieval is too weak (0.2123) for its
top-K to serve as high-quality pseudo-relevance.

## Why it fails

Simeon encodes topical similarity from n-gram co-occurrence patterns. For
many queries, the top-10 simeon-similar documents are topically related but
not relevant. On ArguAna, the top-10 simeon results are all within the same
debate topic cluster but include many non-counterargument docs whose terms
contaminate the expansion.

The BM25-based PRF works because BM25 at least finds documents that share
query vocabulary — the expansion terms come from documents that are lexically
similar to the query, even if they're not all relevant. Simeon-based PRF
introduces expansion terms from lexically dissimilar documents, which are
more likely to be from the wrong subtopic entirely.

## Library status

The `CorpusAdapter` interface is now available in the library:
- `include/simeon/corpus_adapter.hpp` — interface + TextAdapter + ArguanaAdapter
- `src/corpus_adapter.cpp` — implementations
- Benchmark wired to use `ArguanaAdapter` via the library API

The adapter experiment (Phase LXV, 0.4529 macro) proves the `A` component is
the highest-leverage mechanism. Building corpus-specific adapters for the
remaining BEIR corpora would be the most impactful next step, but requires
corpus-specific structural knowledge.
