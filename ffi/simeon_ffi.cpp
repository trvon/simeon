#include "simeon_ffi.h"

#include <simeon/bm25.hpp>
#include <simeon/fragment_geometry.hpp>
#include <simeon/simeon.hpp>

#include <algorithm>
#include <string_view>
#include <vector>

namespace {

constexpr uint32_t kOutputDim = 384;

// "simeon-v1-384": pinned configuration. Changing ANY field here changes the
// vector space — bump the version tag and re-embed the index if you do.
const simeon::Encoder& encoder() {
    static const simeon::Encoder enc = [] {
        simeon::EncoderConfig cfg;
        cfg.sketch_dim = 4096;
        cfg.output_dim = kOutputDim;
        cfg.projection = simeon::ProjectionMode::AchlioptasSparse;
        // ngram/hash/seed/l2_normalize stay at simeon defaults (deterministic).
        return simeon::Encoder(cfg);
    }();
    return enc;
}

} // namespace

extern "C" {

SIMEON_FFI_EXPORT int32_t simeon_ffi_dim(void) {
    return static_cast<int32_t>(kOutputDim);
}

SIMEON_FFI_EXPORT int32_t simeon_ffi_encode(const char* text, int32_t len, float* out) {
    if (text == nullptr || out == nullptr || len < 0) {
        return 1;
    }
    try {
        encoder().encode(std::string_view(text, static_cast<size_t>(len)), out);
        return 0;
    } catch (...) {
        return 2;
    }
}

SIMEON_FFI_EXPORT int32_t simeon_ffi_rerank(const char* query, int32_t query_len,
                                            const char* const* docs, const int32_t* doc_lens,
                                            int32_t n_docs, float* scores_out) {
    if (query == nullptr || docs == nullptr || doc_lens == nullptr || scores_out == nullptr ||
        query_len < 0 || n_docs <= 0) {
        return 1;
    }
    try {
        const auto& enc = encoder();

        std::vector<std::string_view> views;
        views.reserve(static_cast<size_t>(n_docs));
        simeon::Bm25Index idx;
        for (int32_t i = 0; i < n_docs; ++i) {
            if (docs[i] == nullptr || doc_lens[i] < 0)
                return 1;
            views.emplace_back(docs[i], static_cast<size_t>(doc_lens[i]));
            idx.add_doc(views.back());
        }
        idx.finalize();

        std::vector<std::vector<simeon::SemanticFragment>> doc_frags;
        doc_frags.reserve(views.size());
        for (const auto& view : views) {
            doc_frags.push_back(
                simeon::build_doc_semantic_fragments_rich(enc, view, idx,
                                                          /*top_sentence_fragments=*/4,
                                                          /*fragment_signature_terms=*/8));
        }

        // Defaults include the upstream perf-optimized rerank query path.
        const simeon::FragmentGeometryConfig cfg{};
        const auto scores = simeon::score_fragment_geometry(
            std::string_view(query, static_cast<size_t>(query_len)), idx, enc, doc_frags, cfg);
        if (scores.size() != static_cast<size_t>(n_docs))
            return 3;
        std::copy(scores.begin(), scores.end(), scores_out);
        return 0;
    } catch (...) {
        return 2;
    }
}

} // extern "C"
