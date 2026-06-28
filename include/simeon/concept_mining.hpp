#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/simeon.hpp"

namespace simeon {

struct ConceptConfig {
    std::uint32_t min_ttf = 5;
    float pmi_floor = 2.0f;
    std::uint32_t max_concepts = 200000;
    float k1 = 1.2f;
    float b = 0.75f;
    float concept_weight = 0.5f;
};

struct ConceptEntry {
    float pmi = 0.0f;
    float idf = 0.0f;
    std::uint64_t total_tf = 0;
    std::uint64_t a_hash = 0;
    std::uint64_t b_hash = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> docs;
};

class ConceptIndex {
public:
    ConceptIndex() = default;

    std::uint32_t size() const noexcept { return static_cast<std::uint32_t>(concepts_.size()); }
    std::uint32_t doc_count() const noexcept { return doc_count_; }
    float avg_dl() const noexcept { return avg_bigram_dl_; }

    const ConceptEntry* find(std::uint64_t concept_hash) const noexcept;

    void score(std::string_view query, std::span<float> out_scores) const;

    // Sparse blend into an existing score vector: out[did] += weight * pmi *
    // BM25(concept) for every doc containing a matched query-bigram concept.
    // Avoids the caller allocating/zeroing a full-corpus buffer and running a
    // dense add; touches only docs with a matched concept. score() == weight 1.
    void blend_into(std::string_view query, std::span<float> out_scores, float weight) const;

    static std::uint64_t hash_bigram(std::uint64_t a, std::uint64_t b) noexcept;

    // Calls fn(concept_hash, pmi) for each concept bigram found in doc_text.
    void collect_doc_concepts(std::string_view doc_text,
                              const std::function<void(std::uint64_t, float)>& fn) const;

private:
    friend ConceptIndex mine_concepts(const Bm25Index& idx, std::span<const std::string_view> docs,
                                      const ConceptConfig& cfg);

    std::unordered_map<std::uint64_t, ConceptEntry> concepts_;
    std::uint32_t doc_count_ = 0;
    float avg_bigram_dl_ = 0.0f;
    std::vector<std::uint32_t> bigram_doc_lengths_;
    float k1_ = 1.2f;
    float b_ = 0.75f;
    HashFamily hash_family_ = HashFamily::SplitMix64;
    std::uint64_t hash_seed_ = 0xB252B252B252B252ULL;
};

ConceptIndex mine_concepts(const Bm25Index& idx, std::span<const std::string_view> docs,
                           const ConceptConfig& cfg);

void score_bm25_with_concepts(const Bm25Index& idx, const ConceptIndex& concepts,
                              std::string_view query, float concept_weight,
                              std::span<float> out_scores);

} // namespace simeon
