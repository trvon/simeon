#pragma once

#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace simeon {

// Structural evidence produced by a CorpusAdapter for a single document or
// query.  Index-time evidence feeds into the retrieval index; query-time
// evidence feeds into per-query boost / re-rank signals.
struct AdapterEvidence {
    // Optional auxiliary field text.  Scored alongside the body text via
    // BM25F at both index and query time.  Empty → no field contribution.
    std::string aux_field;

    // Document-to-document relations.  Each entry signals that this document
    // (or query) is semantically related to `target_doc`.
    struct DocRelation {
        std::uint32_t target_doc = 0;
        float weight = 1.0f;
    };
    std::vector<DocRelation> relations;

    // Extracted entity strings (e.g. named entities, topic labels, key
    // phrases).  Used for entity-overlap scoring.
    std::vector<std::string> entities;

    bool empty() const noexcept {
        return aux_field.empty() && relations.empty() && entities.empty();
    }
};

// Abstract interface for corpus-specific structural awareness.
// Implementations are training-free: they may inspect document IDs, text
// patterns, and formatting conventions, but never relevance labels.
class CorpusAdapter {
public:
    virtual ~CorpusAdapter() = default;

    // Process a document at index time.  Called once per document.
    virtual AdapterEvidence process_doc(std::string_view doc_id, std::string_view doc_text) = 0;

    // Process a query at query time.  Called once per query.
    virtual AdapterEvidence process_query(std::string_view query_id,
                                          std::string_view query_text) = 0;
};

// ---------------------------------------------------------------------------
// TextAdapter — universal corpus-agnostic default.
//
// Extracts the first 64 space-delimited tokens of each document as the
// aux_field (lead-text bias).  Does not produce relations or entities.
// Safe on any corpus.
// ---------------------------------------------------------------------------
class TextAdapter final : public CorpusAdapter {
public:
    AdapterEvidence process_doc(std::string_view doc_id, std::string_view doc_text) override;

    AdapterEvidence process_query(std::string_view query_id, std::string_view query_text) override;
};

// Extract at most `max_tokens` leading tokens from `text`.
// Token boundaries are defined by std::isspace().
std::string extract_lead_tokens(std::string_view text, std::uint32_t max_tokens);

// ---------------------------------------------------------------------------
// ArguAnaAdapter — debate pair-ID parser.
// Research-only: gated behind SIMEON_ENABLE_RESEARCH.
// ---------------------------------------------------------------------------
#ifdef SIMEON_ENABLE_RESEARCH
class ArguanaAdapter final : public CorpusAdapter {
public:
    // Register a document that will later be matched by queries.
    // Must be called once per document at index time.
    void seed_doc(std::string_view doc_id, std::uint32_t doc_index);

    // Number of successfully parsed documents.
    std::size_t seeded_docs() const noexcept { return seeded_; }

    AdapterEvidence process_doc(std::string_view doc_id, std::string_view doc_text) override;

    AdapterEvidence process_query(std::string_view query_id, std::string_view query_text) override;

private:
    // Key: topic|stance|point — stable key for same-topic/stance/point docs.
    // Value: map from side ('a' or 'b') to doc_index.
    std::unordered_map<std::string, std::unordered_map<char, std::uint32_t>> topics_;
    std::size_t seeded_ = 0;
    bool seeded_all_ = false; // set after seed phase

    // Parse an ArguAna ID. Returns valid-on-success struct.
    struct Id {
        std::string topic;
        char stance[4] = {}; // "pro" or "con" + null
        std::string point;   // digits, e.g. "03"
        char side = 0;       // 'a' or 'b'
        bool valid = false;
    };
    static Id parse(std::string_view id);
};
#endif // SIMEON_ENABLE_RESEARCH

// ---------------------------------------------------------------------------
// ArguanaTextPairAdapter — non-id text-only ArguAna relation adapter.
// Research-only: gated behind SIMEON_ENABLE_RESEARCH.
// ---------------------------------------------------------------------------
#ifdef SIMEON_ENABLE_RESEARCH
class ArguanaTextPairAdapter final : public CorpusAdapter {
public:
    explicit ArguanaTextPairAdapter(std::uint32_t prefix_terms = 5,
                                    bool claim_premise_mode = false) noexcept
        : prefix_terms_(prefix_terms), claim_premise_mode_(claim_premise_mode) {}

    void seed_doc(std::string_view doc_id, std::string_view doc_text, std::uint32_t doc_index);

    std::size_t seeded_docs() const noexcept { return docs_.size(); }

    AdapterEvidence process_doc(std::string_view doc_id, std::string_view doc_text) override;

    AdapterEvidence process_query(std::string_view query_id, std::string_view query_text) override;

private:
    struct SeededDoc {
        std::uint32_t index = 0;
        std::string normalized;
        std::vector<std::string> tokens;
        std::unordered_set<std::string> content;
        std::unordered_set<std::string> first35_content;
    };

    std::uint32_t prefix_terms_ = 5;
    bool claim_premise_mode_ = false;
    std::vector<SeededDoc> docs_;

    static std::string normalize_ws_lower(std::string_view text);
    static std::vector<std::string> word_tokens(std::string_view text);
    static std::unordered_set<std::string> content_set(std::string_view text);
    static std::unordered_set<std::string> content_set_first_words(std::string_view text,
                                                                   std::uint32_t max_words);
    static float jaccard_set(const std::unordered_set<std::string>& a,
                             const std::unordered_set<std::string>& b);
    static std::uint32_t cue_count(const std::unordered_set<std::string>& toks);
    bool same_prefix(const SeededDoc& a, const SeededDoc& b) const noexcept;
};
#endif // SIMEON_ENABLE_RESEARCH

// ---------------------------------------------------------------------------
// ScientificAdapter — corpus-agnostic scientific/technical entity extraction.
//
// Training-free: uses regex patterns and suffix-matching to identify
// biomedical and technical entities in prose text. Works on any scientific
// corpus (SciFact, TREC-COVID, NFCorpus) without per-corpus tuning.
//
// Entity types extracted:
//   - Capitalized multi-word technical terms (genes, processes, methods)
//   - ALL_CAPS acronyms and gene/protein symbols
//   - Biomedical-suffix words (-ase, -itis, -oma, -in, -gen, -cyte, etc.)
//   - Measurement patterns (number + unit)
//
// At query time, entities are extracted for use by KeyphraseStrategy
// and other entity-aware retrieval lanes.  No ML dependencies.
// ---------------------------------------------------------------------------
class ScientificAdapter final : public CorpusAdapter {
public:
    AdapterEvidence process_doc(std::string_view doc_id, std::string_view doc_text) override;

    AdapterEvidence process_query(std::string_view query_id, std::string_view query_text) override;

    static bool is_biomedical_suffix(std::string_view word);
    static std::vector<std::string> extract_entities(std::string_view text,
                                                     std::size_t max_entities = 32);
};

} // namespace simeon
