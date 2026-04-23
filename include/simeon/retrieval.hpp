#pragma once

// Stable retrieval-facing umbrella header.
//
// This groups the library's default first-stage retrieval surface:
//   - BM25 and its shipped variants
//   - fusion helpers used by the default cascades
//   - the query router that selects among the proven sparse/cascade recipes
//   - fragment-geometry reranking used by yams's simeon backend
//
// More corpus-sensitive or research-only extensions remain in their dedicated
// headers (`prf.hpp`, `concept_mining.hpp`) so callers opt into them
// deliberately.

#include "simeon/bm25.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/fusion.hpp"
#include "simeon/query_router.hpp"
