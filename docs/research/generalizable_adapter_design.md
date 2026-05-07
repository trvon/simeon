# Generalizable Adaptive Corpus Adapter — Corrected Design

## Objective

Design an adapter architecture that improves over the universal text-only floor
without hard-coding corpus IDs into the retrieval logic.

Current empirical anchors:

- best universal row: `observed_ordering_entropy_length_qpp_router = 0.4169`
- best adapter-only system: ArguAna pair-ID adapter `= 0.4529`
- best adapter + routing system: `observed_ordering_arguana_text_pair_adapter_ensemble = 0.4745`

## Rejected design

The original hypothesis for a generalizable adapter was:

```text
extract multiple evidence streams -> run many strategies -> self-assess each
strategy from its output shape -> blend or hard-select by output quality
```

This is not the validated path.

- Phase LXVIII: soft self-assessment loses to hard routing
- Phase LXIX: hard-select by per-strategy output shape also loses

The failure is architectural: per-strategy output entropy is not a stable proxy
for retrieval quality across heterogeneous scoring mechanisms.

## Validated design rule

The validated adapter architecture is:

```text
adapter evidence extraction
    -> explicit adapter branch if strong structural evidence exists
    -> otherwise universal hard router from cheap query-level signals
```

In current terms:

1. **Adapter layer** emits observable structural evidence
2. **Branching policy** prefers explicit adapter relations when available
3. **Fallback** is `observed_ordering_entropy_length_qpp_router`, not
   per-strategy self-assessment

## Evidence types

The adapter side should only emit evidence that is:

1. observable from document IDs, text, or formatting
2. usable without qrels
3. fail-safe when absent

Shipped evidence channels already support this:

| Evidence | Intended use | Current example |
|---|---|---|
| `aux_field` | field-aware lexical scoring | lead-64 text |
| `relations` | explicit query-doc or doc-doc links | ArguAna pair IDs |
| `entities` | entity / phrase overlap signals | future medical / financial adapters |

## Minimal generalizable architecture

### Layer 1: structural evidence

Extract only cheap, observable signals:

- lead text
- ID-derived relations
- entity / keyphrase candidates
- discourse cues

### Layer 2: strategy pool

Keep the strategy set small and justified:

- lexical baseline
- structural / lead field strategy
- expansion strategy
- optional entity or relation scorer when evidence exists

### Layer 3: routing

Use query-level routing, not output self-assessment:

```text
if adapter relations are present:
    use adapter-aware branch
else:
    use universal hard router
```

The current universal hard router is the entropy+length 2-way selector.

## What "generalizable" means here

"Generalizable" does not mean one scoring rule for every corpus. It means one
interface and one branching discipline:

- emit only qrel-free structural evidence
- degrade cleanly when no evidence is available
- reuse the same universal fallback when the adapter is silent

Under this definition, the ArguAna pair-ID adapter is the first proof of concept
for the interface, and the ArguAna text-pair adapter is the first proof that a
non-id structural branch can still produce a large macro jump. Neither is yet a
fully general adapter family.

## Open problem

The unsolved problem is not routing. The unsolved problem is discovering
non-leaking structural evidence for non-ArguAna corpora that is as strong as the
pair-ID relation signal.

Priority targets:

1. argument / stance relations
2. claim / premise structure
3. medical or financial entity structure
4. passage / lead cues for verbose queries

The universal router is already good enough to serve as the default branch. The
adapter research problem is now primarily an **evidence-discovery** problem.
