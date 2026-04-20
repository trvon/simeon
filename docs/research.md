# Research notes

This repository ships a compact engineering surface plus a small set of
research-facing notes explaining how the published numbers were obtained,
where the method works, and where it does not.

## What to cite

- **Software use:** cite the repository via [`CITATION.cff`](../CITATION.cff).
- **Benchmark claims:** cite [benchmarks.md](benchmarks.md).
- **Router behavior and transfer limits:** cite [router_design.md](router_design.md).
- **PMI / co-occurrence projection analysis:** cite [pmi_projection.md](pmi_projection.md).
- **Prior-work list used by this repo:** see [works_cited.md](works_cited.md).

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

- treat it as a **training-free retrieval backend**, not a semantic replacement
  for a learned encoder
- separate **headline production claims** from **corpus-specific ablations**
- preserve negative results alongside positive ones
- avoid treating simeon as a blanket validation of any single upstream paper

NUMEN is one important antecedent, but it is not the only one. The library also
draws on late-interaction retrieval, product quantization, sparse random
projection, nested representations, and query-difficulty routing literature.
The shipped benchmark and router notes document places where simeon's measured
behavior does **not** support a simplistic "this validates NUMEN" reading.

That framing keeps the library useful to downstream users without overstating
what the benchmark tables show.
