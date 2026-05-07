# Component-Based Retrieval Architecture

## Why restructure

After 36 phases of ad-hoc experimentation, we have proven components but no
clean way to compose them. The benchmark has 60+ ScoreFns that are copy-paste
lambdas, impossible to test independently, and require a PhD in simeon internals
to extend.

A component architecture with interfaces enables:
1. **Independent testing** of each strategy
2. **Dynamic routing** between strategies per query
3. **Extension by users** — plug in a new strategy without modifying the library
4. **Hard query-dependent routing** — pick one strategy per query from cheap,
   observable signals
5. **Gradual learning** — same architecture works for hard-coded rules today and
    learned routing tomorrow

## Why learned systems work better

The core insight: **BM25 works for some queries, RM3 for others, lead-field for
yet others.** The optimal system knows which strategy to use per query. Learned
systems learn this mapping from data. Training-free systems must estimate it
from observable features.

The architecture should be the same for both — the only difference is the
implementation of the router:
- **Training-free:** QPP-style hard gate from entropy, query length, and other
  cheap query / score-shape features
- **Learned:** classifier or configuration ranker trained on query features +
  strategy outcomes

## Component architecture

```
                  ┌──────────────┐
                  │    Query     │
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │ QueryAnalyzer│  ← extracts features (entropy, n_terms, clarity, keyphrases)
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │  AdapterHub  │  ← collects structural evidence (CorpusAdapter)
                  └──────┬───────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────▼──┐   ┌──────▼──┐   ┌──────▼──┐
    │Bm25Strat│   │ Rm3Strat│   │LeadStrat│  ← RetrievalStrategy[]
    └──────┬──┘   └──────┬──┘   └──────┬──┘
           │             │             │
           └─────────────┼─────────────┘
                         │
                  ┌──────▼───────┐
                  │StrategyRouter│  ← hard gate / select
                  └──────┬───────┘
                         │
                  ┌──────▼───────┐
                  │   Ranking    │
                  └──────────────┘
```

## Interfaces

### RetrievalStrategy

```cpp
class RetrievalStrategy {
public:
    virtual ~RetrievalStrategy() = default;

    // Score all documents. out_scores must be pre-sized to doc_count().
    virtual void score(std::string_view query,
                       const AdapterEvidence& evidence,
                       std::span<float> out_scores) const = 0;

    // Self-assessed quality. 0.0 = worst, 1.0 = best.
    // Implementations examine their own score distribution shape.
    virtual float assess_quality(std::span<const float> scores) const;
};
```

### StrategyRouter

```cpp
class StrategyRouter {
public:
    virtual ~StrategyRouter() = default;

    // Route a query to the best strategy.  Scores are written to out_scores.
    virtual void route(std::string_view query,
                       const QueryProfile& profile,
                       const AdapterEvidence& evidence,
                       std::span<RetrievalStrategy*> pool,
                       std::span<float> out_scores) const;
};
```

### QueryAnalyzer

```cpp
struct QueryProfile {
    float bm25_entropy = 0.0f;
    float avg_idf = 0.0f;
    float idf_stddev = 0.0f;
    uint32_t n_terms = 0;
    float scq_sum = 0.0f;
    float simplified_clarity = 0.0f;
    std::vector<std::string> keyphrases;
};

class QueryAnalyzer {
public:
    virtual QueryProfile analyze(std::string_view query,
                                 const Bm25Index& idx) const = 0;
};
```

### AdapterHub

```cpp
class AdapterHub {
public:
    // Register a corpus adapter.  Called once per document at index time.
    void add_doc(std::string_view doc_id, std::string_view doc_text,
                 CorpusAdapter* adapter);

    // Collect structured evidence for a query.
    AdapterEvidence evidence_for(std::string_view query_id,
                                 std::string_view query_text,
                                 CorpusAdapter* adapter) const;
};
```

## Concrete implementations

### Retrieval Strategies (proven by our experiments)

| Strategy | When effective | Phase proven |
|---|---|---|
| `Bm25Strategy` | Fact-claim queries, safe baseline | All phases |
| `Rm3DiverseStrategy` | Default universal strategy; strongest on medical and expansion-friendly queries | LIII (0.4108) |
| `LeadFieldStrategy` | Long, ultra-peaked debate / verbose queries | LXXI (2-way router branch) |
| `KeyphraseStrategy` | Entity-heavy queries (FiQA, drug names) | Proposed |
| `SimeonStrategy` | Topical complement to BM25 | XLIV |

### Routers

| Router | How it routes | Phase proven |
|---|---|---|
| `EntropyLengthQppRouter` | entropy < 0.05 and n_terms > 30 → Lead; otherwise BM25 rescue from NQC+WIG or diverse RM3 | LXXII (0.4169) |
| `EntropyLengthRouter` | entropy < 0.05 and n_terms > 30 → Lead, else diverse RM3 | LXXI (0.4141) |
| `EntropyRouter` | entropy < 0.05 → Lead, > 0.50 → RM3, else BM25 | LVI (0.4086) |
| `SelfAssessRouter` | score all, pick best by quality metric | LXVIII (negative result; loses to hard routing) |
| `AdapterRouter` | if adapter has relations → use adapter | LXVII (0.4634) |

### Adapters

| Adapter | What it extracts | Phase proven |
|---|---|---|
| `TextAdapter` | Lead-64 tokens | Default |
| `ArguanaAdapter` | Debate pair IDs | LXV (0.4529) |
| `ArguanaTextPairAdapter` | Debate-neighborhood text relations; optional claim/premise mode | LXXIV (0.4780 overall ensemble) |
| `KeyphraseAdapter` | RAKE-style from query | Proposed |

## Users extend by

1. **Adding a strategy:** implement `RetrievalStrategy`, pass to router pool
2. **Adding a router:** implement `StrategyRouter`, use different routing logic
3. **Adding an adapter:** implement `CorpusAdapter` for their corpus structure
4. **Learning routing:** replace hard-coded router with a learned classifier
   or configuration ranker trained on query features + strategy outcomes (when
   labels become available)

## Migration path from current code

```
Current:  gen_entropy_length_router lambda → hard-coded dispatch
Target:   StrategyRouter.route(query, profile, evidence, pool[bm25, lead, rm3], out)
```

The benchmark becomes:
```cpp
auto pool = {&bm25_strat, &lead_strat, &rm3_strat};
entropy_length_router.route(query, profile, evidence, pool, out_scores);
```

Instead of 60+ copy-paste lambdas, we have 5 strategies, 4 routers, 3 adapters,
composable and testable independently.
