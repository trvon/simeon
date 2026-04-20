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
    // Amati & van Rijsbergen 2002 DFR PL2 (Poisson with Laplace after-effect
    // and L2 length normalization, Terrier reference). Score uses
    // tfn = tf * log2(1 + c * avg_dl / dl), then the Stirling-Poisson form
    // (1/(tfn+1)) * (tfn*log2(tfn/λ) + (λ-tfn)*log2(e) + 0.5*log2(2π·tfn))
    // with λ = ttf / N. Free parameter c defaults to 1.0 per Terrier.
    PL2,
    // Amati 2007 DFR DPH (hypergeometric, Terrier reference). Parameter-free:
    // norm = (1 - tf/dl)² / (tf + 1) gates the Stirling log-argument
    // tf * avg_dl / dl * N / ttf. No free parameter.
    DPH,
    // Dirichlet-smoothed LM / DCM (Madsen-Kauchak-Elkan 2005, ICML, "Modeling
    // Word Burstiness"; related to Zhai-Lafferty 2001 Dirichlet smoothing).
    // Per-term contribution log(1 + tf * total_tokens / (α_sum * ttf)), where
    // α_sum defaults to avg_dl. Captures burstiness via the Polya-urn-style
    // log-tf growth without needing per-term α_t tuning.
    Dcm,
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
    // Dirichlet concentration for Dcm variant. 0 means "derive from corpus"
    // (uses avg_dl at finalize() time, matching Zhai-Lafferty μ=avg_dl).
    // Ignored by other variants.
    float dcm_alpha_sum = 0.0f;
    // Free parameter c for PL2. Terrier default is 1.0. Ignored by other
    // variants.
    float pl2_c = 1.0f;
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

    std::uint32_t doc_count() const noexcept {
        return static_cast<std::uint32_t>(doc_lengths_.size());
    }
    const Bm25Config& config() const noexcept { return cfg_; }

    // Per-term lookup for pre-retrieval predictors (QueryRouter). Tokenizes
    // `term` exactly like score() does (word-only) and returns the document
    // frequency / IDF; both are 0 when the term is OOV. Cheap: one tokenize
    // + one hash + one map lookup. Requires finalize() to have been called.
    std::uint32_t df(std::string_view term) const noexcept;
    float idf(std::string_view term) const noexcept;
    // Corpus-wide term frequency (Σ_d tf(term, d)). 0 for OOV terms. Used by
    // Step 1k Pass E predictors (SCQ, simplified clarity) which need
    // collection statistics beyond df/idf.
    std::uint64_t total_tf(std::string_view term) const noexcept;
    // Total token count across all docs = Σ_d doc_length(d). Cheap; cached
    // at finalize(). Used by simplified clarity to normalize collection LM.
    std::uint64_t total_tokens() const noexcept { return total_tokens_; }

    // Hash `term` using the same scheme score() / df() / idf() use. Returns
    // the raw uint64 hash (not the IDF). Cheap: one tokenize of the single
    // term plus one hash call; exposes no corpus state. Used by PRF to build
    // weighted-term queries without re-tokenizing through the full pipeline.
    std::uint64_t hash_term(std::string_view term) const noexcept;

    // Per-doc length in word tokens (the same dl used in BM25 scoring).
    // Returns 0 for out-of-range doc_id.
    std::uint32_t doc_length(std::uint32_t doc_id) const noexcept {
        return doc_id < doc_lengths_.size() ? doc_lengths_[doc_id] : 0u;
    }

    // Build an RM1-style relevance model from the top-K feedback docs.
    // For each term w present in any feedback doc, writes
    //   weight(w) = Σ_{d ∈ top_k_docs} (tf(w, d) / dl(d)) * doc_weights[d_idx]
    // into `out_terms` as (term_hash, weight). Caller normalizes / trims.
    // top_k_docs and doc_weights must have the same size; doc_weights are
    // typically normalized first-pass BM25 scores.
    void build_relevance_model(
        std::span<const std::uint32_t> top_k_docs, std::span<const float> doc_weights,
        std::vector<std::pair<std::uint64_t, float>>& out_terms) const;

    // Score each doc by Σ_w weight(w) * per-term-contribution(w, d), using
    // the same Bm25Variant-dispatched scoring inner loop score() uses.
    // `weighted_terms` is (term_hash, weight) — hashes must be produced via
    // this index's hash_term() or df()/idf() will mis-match. Unlike score(),
    // this path never invokes the SubwordAwareBackoff n-gram fallback (no
    // per-term strings available); for SAB indexes the expansion is scored
    // on the exact word-posting path only.
    void score_weighted_hashes(
        std::span<const std::pair<std::uint64_t, float>> weighted_terms,
        std::span<float> out_scores) const;

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
    // Cached at finalize(): Σ doc_lengths_ (used by Dcm, computed once).
    std::uint64_t total_tokens_ = 0;
    // Cached at finalize() when variant == Dcm: α_sum override or avg_dl.
    float alpha_sum_ = 0.0f;
    bool finalized_ = false;

    // SubwordAwareBackoff secondary index: char n-gram postings over the
    // same docs. Only populated when cfg_.variant == SubwordAwareBackoff.
    std::vector<std::uint32_t> ngram_doc_lengths_;
    std::unordered_map<std::uint64_t, TermPostings> ngram_postings_;
    float ngram_avg_dl_ = 0.0f;
};

} // namespace simeon
