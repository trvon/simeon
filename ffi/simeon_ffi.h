// C ABI for the vendored simeon encoder (training-free SIMD text embeddings),
// consumed from Dart via dart:ffi. The encoder configuration is pinned in
// simeon_ffi.cpp — same input on any platform yields the same vector
// ("simeon-v1-384").
#pragma once

#include <stdint.h>

// `used` + default visibility: when built as a static library linked into the
// iOS app binary nothing references these symbols, so without the attribute
// the linker dead-strips them and dart:ffi lookups fail.
#if defined(__GNUC__) || defined(__clang__)
#define SIMEON_FFI_EXPORT __attribute__((used, visibility("default")))
#else
#define SIMEON_FFI_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Embedding dimension of the pinned encoder (384).
SIMEON_FFI_EXPORT int32_t simeon_ffi_dim(void);

// Encode UTF-8 `text` (byte length `len`) into `out`, which must hold
// simeon_ffi_dim() floats. Returns 0 on success, nonzero on failure.
SIMEON_FFI_EXPORT int32_t simeon_ffi_encode(const char* text, int32_t len, float* out);

// Re-rank `n_docs` candidate documents against `query` using simeon's
// fragment-geometry pipeline (BM25 + semantic-fragment geometry blend; the
// upstream perf-optimized query path). Writes one score per doc to
// `scores_out` (higher = more relevant). Returns 0 on success.
SIMEON_FFI_EXPORT int32_t simeon_ffi_rerank(const char* query, int32_t query_len,
                                            const char* const* docs, const int32_t* doc_lens,
                                            int32_t n_docs, float* scores_out);

#ifdef __cplusplus
}
#endif
