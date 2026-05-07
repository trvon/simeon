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
//
// Parses BEIR ArguAna document/query IDs of the form:
//     <topic>-<stance><point><side>
//   e.g.  "guncontrol-pro03a" → topic=guncontrol  stance=pro  point=03  side=a
//
// For each query, adds a DocRelation pointing to the document with the
// same topic/stance/point but opposite side (a↔b).  Documents that are
// not in the expected format produce empty evidence.
//
// This adapter must be pre-seeded with all document IDs before any query
// processing (via seed_doc).  It is only useful for the ArguAna corpus;
// on other corpora all IDs fail parsing and the adapter degrades to no-op.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// ArguanaTextPairAdapter — non-id text-only ArguAna relation adapter.
//
// Seeds on document text, then at query time:
//   1. locates the self/source document by normalized query containment;
//   2. clusters candidates by a shared first-N-token header;
//   3. emits weighted DocRelation scores for the cluster using overlap,
//      rebuttal-cue, and locality features.
//
// Unlike ArguanaAdapter, this does not use the exact a↔b id pair. It is still
// corpus-structured, but the signal comes from observable text only.
// ---------------------------------------------------------------------------
class ArguanaTextPairAdapter final : public CorpusAdapter {
public:
    explicit ArguanaTextPairAdapter(std::uint32_t prefix_terms = 5) noexcept
        : prefix_terms_(prefix_terms) {}

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
    };

    std::uint32_t prefix_terms_ = 5;
    std::vector<SeededDoc> docs_;

    static std::string normalize_ws_lower(std::string_view text);
    static std::vector<std::string> word_tokens(std::string_view text);
    static std::unordered_set<std::string> content_set(std::string_view text);
    static float jaccard_set(const std::unordered_set<std::string>& a,
                             const std::unordered_set<std::string>& b);
    static std::uint32_t cue_count(const std::unordered_set<std::string>& toks);
    bool same_prefix(const SeededDoc& a, const SeededDoc& b) const noexcept;
};

} // namespace simeon
