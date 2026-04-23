# Works cited

This is the compact prior-work list for the ideas used directly in simeon's
implementation, benchmarks, and design notes. It is intentionally selective:
these are the papers and books a downstream reader is most likely to need when
understanding the library's design or citing its evaluation context.

## Core retrieval context

- **Sharma (2026), _NUMEN: Non-neural embeddings that match transformer embeddings on retrieval_.**
  Prior art for the training-free retrieval framing that simeon reimplements and
  stress-tests. The benchmark and router docs in this repo should be cited for
  simeon's own measured behavior rather than treating this repository as an
  independent validation of NUMEN's headline claims.
- **Khattab and Zaharia (2020), _ColBERT_.** DOI: [10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)
  Reference point for late-interaction retrieval and the broader multi-vector
  retrieval framing used when positioning simeon's dense/lexical hybrid path.

## Compression and representation

- **Jégou, Douze, and Schmid (2011), _Product Quantization for Nearest Neighbor Search_.** DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
  Basis for the PQ/ADC compression path.
- **Li, Hastie, and Church (2006), _Very sparse random projections_.** DOI: [10.1145/1150402.1150436](https://doi.org/10.1145/1150402.1150436)
  Reference for the very-sparse projection option.
- **Bhatt et al. (2022), _Matryoshka Representation Learning_.** DOI: [10.52202/068431-2192](https://doi.org/10.52202/068431-2192)
  Motivation for the nested-prefix representation interface; simeon uses a
  training-free analog rather than the learned method itself.

## Routing and query-difficulty estimation

- **Carmel and Yom-Tov (2010), _Estimating the Query Difficulty for Information Retrieval_.** DOI: [10.1007/978-3-031-02272-2](https://doi.org/10.1007/978-3-031-02272-2)
  Background for the pre-retrieval query-difficulty family used by the router.
- **Cronen-Townsend, Zhou, and Croft (2002), _Predicting query performance_.** DOI: [10.1145/564376.564429](https://doi.org/10.1145/564376.564429)
  Reference for predictor families reused in the router notes.

## How to cite this repo responsibly

- Cite **the software** with [`CITATION.cff`](../CITATION.cff).
- Cite **the benchmark tables and negative findings** with
  [docs/research/benchmarks.md](research/benchmarks.md), [docs/research/router_design.md](research/router_design.md), and
  [docs/research/pmi_projection.md](research/pmi_projection.md).
- Cite the **upstream papers above** when discussing the underlying prior work.
