// libFuzzer harness for simeon::Bm25Index build, finalize and score.
//
// Input layout:
//   byte 0  variant (index into kVariants, modulo)
//   byte 1  flags: bit0 word bigrams, bit1 aux field, bit2 memory budget,
//           bits 3-7 budget size ((1 + n) * 32 KiB)
//   rest    newline-separated lines. A line starting with '?' is a query;
//           any other line is a doc, split at the first '\t' into body and
//           aux text when the aux flag is set.
//
// Checks, aborting on the first failure:
//   - memory_bytes() never shrinks while adding docs
//   - a refused doc leaves doc_count and memory_bytes() unchanged, and
//     refusal is sticky
//   - validate() is clean during the build, at the end, and after finalize()
//   - finalize() does not grow the index
//   - every score, df/idf and relevance-model weight is finite
//
// Run with allocation limits so runaway reservations fail fast:
//
//   CXX=/opt/homebrew/opt/llvm/bin/clang++ meson setup buildfuzz -Denable_fuzz=true \
//     -Db_sanitize=address,undefined -Db_lundef=false --buildtype=debug
//   meson compile -C buildfuzz
//   ./buildfuzz/tests/fuzz/fuzz_bm25 -max_len=8192 -malloc_limit_mb=64 \
//     -rss_limit_mb=1024 -max_total_time=300 tests/fuzz/corpus/bm25

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/bm25.hpp"

namespace {

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::Bm25Variant;

constexpr Bm25Variant kVariants[] = {
    Bm25Variant::Atire,    Bm25Variant::BM25Plus,
    Bm25Variant::BM25L,    Bm25Variant::DLH13,
    Bm25Variant::PL2,      Bm25Variant::DPH,
    Bm25Variant::Dcm,      Bm25Variant::SubwordAwareBackoff,
    Bm25Variant::AtireLTD, Bm25Variant::Layered,
    Bm25Variant::LayeredW,
};
constexpr std::size_t kMaxDocs = 512;
constexpr std::size_t kMaxQueries = 16;

[[noreturn]] void fail(const char* what, const std::string& detail = {}) {
    std::fprintf(stderr, "fuzz_bm25 invariant violated: %s %s\n", what, detail.c_str());
    std::abort();
}

void require(bool ok, const char* what) {
    if (!ok)
        fail(what);
}

void require_valid(const Bm25Index& idx, const char* when) {
    if (auto err = idx.validate(); !err.empty())
        fail(when, err);
}

void require_finite(std::span<const float> xs, const char* what) {
    for (float x : xs)
        if (!std::isfinite(x))
            fail(what);
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 2)
        return 0;
    const Bm25Variant variant = kVariants[data[0] % std::size(kVariants)];
    const std::uint8_t flags = data[1];
    const bool aux = (flags & 0x02u) != 0;

    Bm25Config cfg;
    cfg.variant = variant;
    cfg.build_word_bigrams =
        (flags & 0x01u) != 0 || variant == Bm25Variant::Layered || variant == Bm25Variant::LayeredW;
    cfg.subword_gamma = static_cast<float>(data[0] >> 4);
    if (flags & 0x04u)
        cfg.max_memory_bytes = (1u + (flags >> 3)) * (std::size_t{32} << 10);

    std::vector<std::string_view> docs;
    std::vector<std::string_view> aux_texts;
    std::vector<std::string_view> queries;
    const std::string_view rest(reinterpret_cast<const char*>(data + 2), size - 2);
    std::size_t pos = 0;
    while (pos <= rest.size() && docs.size() < kMaxDocs) {
        std::size_t nl = rest.find('\n', pos);
        if (nl == std::string_view::npos)
            nl = rest.size();
        std::string_view line = rest.substr(pos, nl - pos);
        pos = nl + 1;
        if (!line.empty() && line.front() == '?') {
            if (queries.size() < kMaxQueries)
                queries.push_back(line.substr(1));
            continue;
        }
        std::string_view body = line;
        std::string_view aux_text;
        if (aux) {
            if (auto tab = line.find('\t'); tab != std::string_view::npos) {
                body = line.substr(0, tab);
                aux_text = line.substr(tab + 1);
            }
        }
        docs.push_back(body);
        aux_texts.push_back(aux_text);
    }

    Bm25Index idx(cfg);
    idx.reserve_docs(docs.size());
    require(cfg.max_memory_bytes == 0 || idx.memory_bytes() <= cfg.max_memory_bytes / 2,
            "reserve_docs exceeded half the budget");

    bool refused = false;
    std::size_t prev_bytes = idx.memory_bytes();
    for (std::size_t i = 0; i < docs.size(); ++i) {
        const std::uint32_t before_docs = idx.doc_count();
        const bool added = idx.try_add_doc(docs[i], aux_texts[i]);
        const std::size_t now = idx.memory_bytes();
        if (added) {
            require(!refused, "doc accepted after a refusal");
            require(cfg.max_memory_bytes == 0 || prev_bytes < cfg.max_memory_bytes,
                    "doc accepted at or over budget");
            require(idx.doc_count() == before_docs + 1, "doc_count did not advance");
            require(now >= prev_bytes, "memory_bytes shrank while adding");
        } else {
            require(cfg.max_memory_bytes > 0, "refusal without a budget");
            require(idx.doc_count() == before_docs, "refused doc changed doc_count");
            require(now == prev_bytes, "refused doc changed memory_bytes");
            refused = true;
        }
        prev_bytes = now;
        if (i % 16 == 15)
            require_valid(idx, "during build:");
    }
    require_valid(idx, "after build:");

    idx.finalize();
    require(idx.memory_bytes() <= prev_bytes, "finalize grew the index");
    require_valid(idx, "after finalize:");

    const std::size_t n = idx.doc_count();
    std::vector<float> scores(n);
    for (auto q : queries) {
        idx.score(q, scores);
        require_finite(scores, "score");
        if (aux) {
            idx.score_bm25f(q, scores, 1.0f, 0.5f);
            require_finite(scores, "score_bm25f");
        }
        if (cfg.build_word_bigrams) {
            idx.score_sdm(q, scores);
            require_finite(scores, "score_sdm");
        }
        require(idx.df(q) <= n, "df > doc_count");
        require(std::isfinite(idx.idf(q)) && idx.idf(q) >= 0.0f, "idf not finite or negative");
    }

    if (n > 0) {
        const std::uint32_t top[] = {0, static_cast<std::uint32_t>(n / 2),
                                     static_cast<std::uint32_t>(n - 1)};
        const float weights[] = {1.0f, 0.5f, 0.25f};
        std::vector<std::pair<std::uint64_t, float>> terms;
        idx.build_relevance_model(top, weights, terms);
        for (const auto& [h, w] : terms) {
            (void)h;
            if (!std::isfinite(w))
                fail("relevance model weight not finite");
        }
        if (!terms.empty()) {
            idx.score_weighted_hashes(terms, scores);
            require_finite(scores, "score_weighted_hashes");
        }
        require_valid(idx, "after relevance model:");
    }
    return 0;
}
