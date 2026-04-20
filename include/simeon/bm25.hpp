#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simeon/simeon.hpp"

namespace simeon {

// BM25 scoring variant. Atire is the conventional Robertson-style BM25 with
// IDF = log((N-df+0.5)/(df+0.5)+1); the others are well-studied refinements.
enum class Bm25Variant : std::uint8_t {
    // Robertson BM25 (Atire-style IDF). Default; matches pre-variant behavior.
    Atire,
    // Lv & Zhai 2011 (CIKM, "Lower-Bounding Term Frequency Normalization").
    // Adds a δ floor to per-term contribution to fix long-doc lower-bound
    // violation: idf*(tf*(k1+1)/(tf + k1*(1-b+b*dl/avg)) + δ).
    BM25Plus,
    // Lv & Zhai 2011 (SIGIR, "When Documents Are Very Long, BM25 Fails!").
    // Replaces normalized-tf with c' = tf / (1 - b + b*dl/avg), then
    // (k1+1)*(c'+δ)/(k1+c'+δ). Different functional form than BM25+.
    BM25L,
    // Amati DFR DLH13 (Terrier reference). Parameter-free divergence-from-
    // randomness scorer; uses corpus-wide term frequency, no k1/b/δ.
    DLH13,
    // Subword-aware BM25+ backoff (this codebase, training-free): for each
    // query term t, blend exact BM25+ with a score computed from the
    // character n-grams of t against a parallel n-gram inverted index of
    // the corpus. Per-term mix is α = df(t)/(df(t)+γ) when df(t)>0, 0 when
    // df(t)=0; γ=0 recovers the strict OOV-fallback variant. Roughly
    // doubles index size and finalize time.
    SubwordAwareBackoff,
};

struct Bm25Config {
    float k1 = 1.2f;
    float b = 0.75f;
    HashFamily hash = HashFamily::SplitMix64;
    std::uint64_t hash_seed = 0xB252B252B252B252ULL;

    Bm25Variant variant = Bm25Variant::Atire;
    // δ floor for BM25Plus / BM25L. Lv & Zhai recommend 1.0 as default.
    float delta = 1.0f;
    // Smoothing factor for SubwordAwareBackoff: per-term mix between exact
    // and n-gram score is α = df / (df + subword_gamma). Set to 0 for the
    // strict "n-grams only when df=0" backoff.
    float subword_gamma = 0.0f;
    // Char n-gram range used by SubwordAwareBackoff for the secondary index.
    // Ignored by other variants.
    std::uint32_t ngram_min = 3;
    std::uint32_t ngram_max = 5;
};

// Streaming BM25 index over word tokens. Tokenization reuses the simeon
// tokenizer with `emit_word=true, emit_char=false` so term boundaries match
// other simeon components when the same text is processed by both.
//
// Add docs in id order via `add_doc`, then call `finalize()` once before
// scoring. `score()` writes one BM25 score per doc into out_scores.
class Bm25Index {
public:
    explicit Bm25Index(Bm25Config cfg = {}) noexcept;

    void add_doc(std::string_view text);
    void finalize();
    void score(std::string_view query, std::span<float> out_scores) const;

    std::uint32_t doc_count() const noexcept { return static_cast<std::uint32_t>(doc_lengths_.size()); }
    const Bm25Config& config() const noexcept { return cfg_; }

    // Per-term lookup for pre-retrieval predictors (QueryRouter). Tokenizes
    // `term` exactly like score() does (word-only) and returns the document
    // frequency / IDF; both are 0 when the term is OOV. Cheap: one tokenize
    // + one hash + one map lookup. Requires finalize() to have been called.
    std::uint32_t df(std::string_view term) const noexcept;
    float idf(std::string_view term) const noexcept;

private:
    // After finalize(), each posting list carries the term's IDF inline, so
    // score() can iterate one map lookup per query term instead of two
    // (postings + idf). total_tf is the corpus-wide term frequency used by
    // DLH13. The (doc_id, tf) layout is unchanged.
    struct TermPostings {
        float idf = 0.0f;
        std::uint64_t total_tf = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> docs;
    };

    Bm25Config cfg_;
    std::vector<std::uint32_t> doc_lengths_;
    std::unordered_map<std::uint64_t, TermPostings> postings_;
    float avg_dl_ = 0.0f;
    bool finalized_ = false;

    // SubwordAwareBackoff secondary index: char n-gram postings over the
    // same docs. Only populated when cfg_.variant == SubwordAwareBackoff.
    std::vector<std::uint32_t> ngram_doc_lengths_;
    std::unordered_map<std::uint64_t, TermPostings> ngram_postings_;
    float ngram_avg_dl_ = 0.0f;
};

}  // namespace simeon
