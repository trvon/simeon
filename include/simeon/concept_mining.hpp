#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/simeon.hpp"

namespace simeon {

// Latent Concept Model (Bendersky 2008, revisited 2024) — training-free
// concept mining from corpus statistics. At finalize() time, we discover
// word-bigram "concepts" whose pointwise mutual information (PMI) exceeds a
// floor and whose total corpus frequency clears a minimum. At query time,
// matched concepts in the query contribute a PMI-weighted BM25 term score
// that is linearly blended with the base BM25 variant's score.
//
// This is simeon's answer to FiQA's paraphrase gap (−0.147 vs MiniLM): a
// training-free, corpus-statistic mechanism that rewards documents matching
// high-coherence multi-word terms from the query (e.g. "time value of
// money"). The concept index is deterministic and reproducible — given the
// same corpus and config, mine_concepts() emits byte-identical output.
//
// Initial ship is bigrams-only. Trigrams are deferred to a follow-on step.

struct ConceptConfig {
    // Concepts with corpus-wide frequency below min_ttf are discarded as
    // noise. 5 matches the Bendersky 2008 "appears in >= 5 docs" heuristic
    // tightened for small corpora.
    std::uint32_t min_ttf = 5;
    // PMI floor in natural-log units. PMI(a, b) >= 2 means the bigram
    // (a, b) is at least e^2 ≈ 7.4× more frequent than chance.
    float pmi_floor = 2.0f;
    // Cap on concept count. When exceeded, lowest-PMI concepts are evicted.
    // 200k is ~4 MB at 20 bytes/entry and covers all three fixtures.
    std::uint32_t max_concepts = 200000;
    // BM25 saturation parameters applied to concept postings. Defaults
    // match Bm25Config; concepts behave like word tokens for scoring.
    float k1 = 1.2f;
    float b = 0.75f;
    // Weight applied to the PMI-weighted concept contribution. Final
    // fused score = base_bm25 + concept_weight * concept_score.
    // 0.0 disables concept scoring (equivalent to mine_concepts=false).
    float concept_weight = 0.5f;
};

struct ConceptEntry {
    float pmi = 0.0f; // PMI in natural log units
    float idf = 0.0f; // log((N - df + 0.5) / (df + 0.5) + 1)
    std::uint64_t total_tf = 0;
    // Posting list (doc_id, per-doc tf) sorted by doc_id.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> docs;
};

// Concept index: concept_hash -> ConceptEntry. Exposed read-only after
// mine_concepts() returns. Query-time scoring is via score().
class ConceptIndex {
public:
    ConceptIndex() = default;

    std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(concepts_.size()); }
    std::uint32_t doc_count() const noexcept { return doc_count_; }
    float avg_dl() const noexcept { return avg_bigram_dl_; }

    // Read-only concept lookup. Returns nullptr when hash is unknown.
    const ConceptEntry* find(std::uint64_t concept_hash) const noexcept;

    // Score a query against the concept index. For each matched concept
    // (bigram found in the query), adds PMI-weighted BM25 contribution
    // to the corresponding doc's entry in out_scores. Caller is
    // responsible for zero-initializing out_scores (mine_concepts does
    // not own a baseline). out_scores.size() must equal doc_count().
    void score(std::string_view query, std::span<float> out_scores) const;

    // Public bigram hash so tests and callers can verify determinism.
    // Uses the same splitmix64_mix canonicalization as Bm25Index.
    static std::uint64_t hash_bigram(std::uint64_t a, std::uint64_t b) noexcept;

private:
    friend ConceptIndex mine_concepts(const Bm25Index& idx, std::span<const std::string_view> docs,
                                      const ConceptConfig& cfg);

    std::unordered_map<std::uint64_t, ConceptEntry> concepts_;
    std::uint32_t doc_count_ = 0;
    // Average per-doc bigram count (used as avg_dl in BM25 for concept
    // postings). Stored separately from Bm25Index::avg_dl because a doc's
    // bigram count is roughly (word_count - 1), not word_count.
    float avg_bigram_dl_ = 0.0f;
    std::vector<std::uint32_t> bigram_doc_lengths_;
    float k1_ = 1.2f;
    float b_ = 0.75f;
    // Mirrors Bm25Index's hash scheme so query-time tokenization produces
    // compatible hashes.
    HashFamily hash_family_ = HashFamily::SplitMix64;
    std::uint64_t hash_seed_ = 0xB252B252B252B252ULL;
};

// One-pass corpus mining. `idx` must have been finalize()'d already; its
// unigram postings provide total_tf for PMI denominators, and its hash
// scheme is mirrored into the resulting ConceptIndex so that query-time
// bigram hashes match.
//
// `docs` must be the same corpus texts added to idx in the same order
// (doc_id == index into docs span). Re-tokenizes each doc once with the
// same word-only tokenizer Bm25Index uses.
//
// Deterministic: same inputs produce byte-identical output.
ConceptIndex mine_concepts(const Bm25Index& idx, std::span<const std::string_view> docs,
                           const ConceptConfig& cfg);

// Score helper: computes base BM25 + concept_weight * concept_score in a
// single call. Zero-initializes out_scores. cfg.concept_weight controls
// the blend; 0.0 recovers pure base BM25. Provided for bench/test
// convenience — callers that need custom blending can call
// Bm25Index::score() and ConceptIndex::score() separately.
void score_bm25_with_concepts(const Bm25Index& idx, const ConceptIndex& concepts,
                              std::string_view query, float concept_weight,
                              std::span<float> out_scores);

} // namespace simeon
