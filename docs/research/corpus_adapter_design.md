# Corpus Adapter Design

## Purpose

`A` is the structural component in `T = (G, A, S, F, B)`. It is the highest
observed-leverage component so far:

- ArguAna pair-ID adapter: `0.4529`
- universal baseline: `0.3941`
- gain: `+0.0588`

The library therefore exposes corpus adapters as a first-class interface.

## Shipped interface

```cpp
struct AdapterEvidence {
    std::string aux_field;

    struct DocRelation {
        std::uint32_t target_doc = 0;
        float weight = 1.0f;
    };
    std::vector<DocRelation> relations;

    std::vector<std::string> entities;

    bool empty() const noexcept;
};

class CorpusAdapter {
public:
    virtual ~CorpusAdapter() = default;
    virtual AdapterEvidence process_doc(std::string_view doc_id,
                                        std::string_view doc_text) = 0;
    virtual AdapterEvidence process_query(std::string_view query_id,
                                          std::string_view query_text) = 0;
};
```

## Evidence channels

| Field | Semantics | Downstream use |
|---|---|---|
| `aux_field` | auxiliary lexical text | BM25F / fielded scoring |
| `relations` | explicit query-doc or doc-doc links | relation boost / graph-style rerank |
| `entities` | extracted entity or phrase strings | entity-overlap scoring |

All channels are optional. Empty evidence must degrade to the text-only path.

## Shipped adapters

### `TextAdapter`

- corpus-agnostic default
- `process_doc()` extracts the first 64 space-delimited tokens into `aux_field`
- `process_query()` does the same for queries
- emits no relations and no entities

This is the adapter form of the lead-text bias already used in BM25F rows.

### `ArguanaAdapter`

- corpus-specific relation adapter
- seeded with document IDs via `seed_doc(doc_id, doc_index)`
- parses IDs of the form `<topic>-<stance><point><side>`
- for query side `a`, emits a relation to matching document side `b`, and vice
  versa
- falls back to empty evidence on parse failure

This is a training-free structural adapter because it uses observable ID format,
not relevance labels.

## Invariants

1. **No qrels at inference**
2. **Observable evidence only**
3. **Fail-safe degradation** when IDs or text patterns do not match
4. **Index-time preprocessing** for documents; lightweight query-time lookup

## Execution model

### Index time

1. choose adapter
2. optionally seed corpus-specific state
3. call `process_doc()` once per document
4. feed `aux_field` / `relations` / `entities` into indexing side structures

### Query time

1. call `process_query()` once per query
2. if evidence is non-empty, expose the adapter-aware branch
3. otherwise fall back to the universal router

## Design boundary

`CorpusAdapter` is an evidence interface, not a scoring policy.

- adapter: emits structure
- scorer: consumes structure
- router: decides whether adapter evidence should dominate, complement, or be
  ignored

This separation is what allows one adapter interface to support both
corpus-specific relation adapters and future corpus-agnostic evidence extractors.
