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
