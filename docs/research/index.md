# Research notes

This directory contains the benchmark, routing, and ablation notes behind
the published simeon numbers.

## What to cite

- **Software use:** cite the repository via [`CITATION.cff`](../CITATION.cff).
- **Benchmark claims:** cite [benchmarks.md](benchmarks.md).
- **Router behavior and transfer limits:** cite [router_design.md](router_design.md).
- **PMI / co-occurrence projection analysis:** cite [pmi_projection.md](pmi_projection.md).
- **Prior-work list used by this repo:** see [works_cited.md](../works_cited.md).

## Scope of the shipped research artifacts

The in-tree documents preserve:

- the headline benchmark tables used in the public README
- the routing rules, tuning workflow, and honest failure modes
- negative-result documentation for PMI and cross-corpus transfer
- the prior-work list most directly used in the implementation and docs

The repository intentionally does **not** ship large raw benchmark dumps,
temporary telemetry, or local helper scripts used during development.

## Framing guidance

When describing simeon publicly:

- treat it as a **training-free retrieval backend**, not a learned semantic replacement
- separate **headline claims** from **corpus-specific ablations**
- preserve negative results alongside positive ones
- avoid treating simeon as blanket validation of any single upstream paper

NUMEN is one important antecedent, but not the only one. The library also
draws on late interaction, product quantization, sparse random projection,
nested representations, and query-difficulty routing. The notes in this folder
also document where the measured behavior does **not** support a simplistic
"this validates NUMEN" reading.
