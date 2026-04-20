# Reference embedding fixture

A *reference fixture* is a small frozen IR benchmark — corpus, queries, relevance judgments, and pre-computed reference embeddings — used by `bench_vs_reference` to compare simeon against a learned model on a workload that actually has ground truth.

The fixture is generated outside the simeon build (one Python script + a learned model), never committed to the repo (too big), and consumed read-only by the C++ benchmark. This keeps simeon dep-free at build time and the comparison reproducible at evaluation time.

## Layout

```
fixtures/<dataset>/
├── corpus.tsv      # one document per line: <doc_id>\t<text>
├── queries.tsv     # one query per line:    <query_id>\t<text>
├── qrels.tsv       # BEIR-style:            <query_id>\t<doc_id>\t<relevance>
└── reference.bin   # packed reference embeddings for queries + corpus (this order)
```

`text` fields must contain no tab or newline characters; the regen script must replace them with spaces before writing.

## `reference.bin` binary format

All multi-byte integers are little-endian. All floats are IEEE-754 float32, little-endian. Both query and document embedding rows must already be L2-normalized.

| Offset | Bytes      | Field                                                    |
|-------:|-----------:|----------------------------------------------------------|
|      0 |          8 | Magic: ASCII `SIMEONFX` (no NUL terminator).             |
|      8 |          4 | Version (`uint32`). Currently `1`.                       |
|     12 |          4 | `dim` — reference embedding dimension.                   |
|     16 |          4 | `n_queries` — must equal the number of rows in `queries.tsv`. |
|     20 |          4 | `n_docs` — must equal the number of rows in `corpus.tsv`. |
|     24 |          4 | `model_name_len` — length of model name in bytes.        |
|     28 | `mn_len`   | UTF-8 model name (no NUL).                               |
|  28+mn |  4·dim·nq  | Query embeddings, row-major, in `queries.tsv` order.     |
|  …     |  4·dim·nd  | Document embeddings, row-major, in `corpus.tsv` order.   |

The bench loader validates magic, version, dim > 0, and that the file size matches `28 + mn_len + 4·dim·(n_queries + n_docs)`.

## Datasets

Recommended starter datasets (small enough to iterate on, well-known qrels):

| Dataset    | Docs   | Queries | Avg. positives | Source                           |
|------------|-------:|--------:|---------------:|----------------------------------|
| `scifact`  | 5,183  | 300     | ~1.1           | BEIR / arXiv:2104.08663          |
| `nfcorpus` | 3,633  | 323     | ~38            | BEIR                             |
| `arguana`  | 8,674  | 1,406   | 1              | BEIR                             |

These are small enough to run end-to-end in seconds and to ship a fixture under ~100 MB even with a 768-d reference model.

## Reference models

The script supports any `sentence-transformers` model that returns dense, L2-normalizable embeddings. Recommended starting points:

- `sentence-transformers/all-MiniLM-L6-v2` — 384-d, ~22 M params. Common BEIR baseline.
- `BAAI/bge-small-en-v1.5` — 384-d, ~33 M params. Stronger than MiniLM on most BEIR tasks.

Pick one model per fixture build; `model_name` in the binary records which.

## Generating a fixture

Fixture provisioning is out of tree. Any pipeline that downloads the dataset (e.g., via `beir`), encodes queries + corpus with a `sentence-transformers` model, L2-normalizes, and writes the four files in the layout above will produce a valid fixture. Producing an optional held-out dev fold (`queries_dev.tsv`, `qrels_dev.tsv`, `reference_dev.bin`) enables the dev/test workflow described in [build.md](build.md).

Running the comparison bench against the fixture is documented in [build.md](build.md).

## Honest framing

What the comparison should claim:

- **simeon-alone vs reference-alone**: expected to trail on semantic / paraphrase queries; the gap is the cost of being training-free.
- **(BM25 ⊕ simeon) RRF vs reference-alone**: the comparison that matters for production deployments. simeon is a first-stage / fusion tool, not a semantic replacement.

The bench harness emits both rows so the framing is unambiguous in the data, not just in prose.
