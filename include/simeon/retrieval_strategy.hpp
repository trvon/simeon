#pragma once
// simeon — component-based retrieval architecture
//
// RetrievalStrategy  : scores documents for a query (one strategy)
// StrategyRouter     : selects the best strategy per query (routing)
// QueryProfile       : extracted query features for routing decisions

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simeon {

class Bm25Index;
struct AdapterEvidence;
class CorpusAdapter;

// ---------------------------------------------------------------------------
// QueryProfile — features extracted from query + index for routing
// ---------------------------------------------------------------------------
struct QueryProfile {
    float bm25_entropy = 0.0f;
    float avg_idf = 0.0f;
    float idf_stddev = 0.0f;
    std::uint32_t n_terms = 0;
    float scq_sum = 0.0f;
    std::vector<std::string> keyphrases;
};

// ---------------------------------------------------------------------------
// RetrievalStrategy — scores all documents for a query
// ---------------------------------------------------------------------------
class RetrievalStrategy {
public:
    virtual ~RetrievalStrategy() = default;
    virtual void score(std::string_view query, const AdapterEvidence& evidence,
                       std::span<float> out_scores) const = 0;
    virtual void score_indexed(std::string_view query, std::uint32_t qi,
                               const AdapterEvidence& evidence, std::span<float> out_scores) const {
        score(query, evidence, out_scores); // fallback: ignore qi
    }
    virtual float assess_quality(std::span<const float> scores) const;
};

// ---------------------------------------------------------------------------
// StrategyRouter — selects the best strategy per query
// ---------------------------------------------------------------------------
class StrategyRouter {
public:
    virtual ~StrategyRouter() = default;
    virtual void route(std::string_view query, const QueryProfile& profile,
                       const AdapterEvidence& evidence, std::span<RetrievalStrategy* const> pool,
                       std::span<float> out_scores) const = 0;
};

// ---------------------------------------------------------------------------
// EntropyRouter — proven hard gate (Phases XLVI/LVI/LXVII)
//   entropy < 0.05  → pool[0] (Lead/BM25F)
//   entropy > 0.50  → pool[2] (RM3)
//   else            → pool[1] (BM25)
// ---------------------------------------------------------------------------
class EntropyRouter final : public StrategyRouter {
public:
    void route(std::string_view query, const QueryProfile& profile, const AdapterEvidence& evidence,
               std::span<RetrievalStrategy* const> pool,
               std::span<float> out_scores) const override;
};

// ---------------------------------------------------------------------------
// SelfAssessRouter — score all strategies, pick best by assess_quality
// ---------------------------------------------------------------------------
class SelfAssessRouter final : public StrategyRouter {
public:
    void route(std::string_view query, const QueryProfile& profile, const AdapterEvidence& evidence,
               std::span<RetrievalStrategy* const> pool,
               std::span<float> out_scores) const override;
};

// ---------------------------------------------------------------------------
// Bm25Strategy — wraps a Bm25Index
// ---------------------------------------------------------------------------
class Bm25Strategy final : public RetrievalStrategy {
public:
    explicit Bm25Strategy(const Bm25Index& idx, float relation_boost = 0.0f)
        : idx_(&idx), relation_boost_(relation_boost) {}
    void score(std::string_view query, const AdapterEvidence& evidence,
               std::span<float> out_scores) const override;

private:
    const Bm25Index* idx_;
    float relation_boost_ = 0.0f;
};

// ---------------------------------------------------------------------------
// LeadFieldStrategy — BM25F with a pre-computed lead-text aux field.
// lead_texts[i] is the aux text for document i.
// ---------------------------------------------------------------------------
class LeadFieldStrategy final : public RetrievalStrategy {
public:
    LeadFieldStrategy(const Bm25Index& idx, std::span<const std::string> lead_texts,
                      float body_weight = 0.85f, float aux_weight = 0.15f);
    void score(std::string_view query, const AdapterEvidence& evidence,
               std::span<float> out_scores) const override;

private:
    const Bm25Index* idx_;
    std::vector<std::string> lead_texts_;
    float body_w_, aux_w_;
};

// ---------------------------------------------------------------------------
// Rm3DiverseStrategy — MMR-diverse PRF via simeon embeddings (β=0.25).
// query_embs is row-major: query_embs[qi * sdim + d].
// Use score_query() with a query INDEX, not score() with query text.
// ---------------------------------------------------------------------------
class Rm3DiverseStrategy final : public RetrievalStrategy {
public:
    Rm3DiverseStrategy(const Bm25Index& idx, std::span<const float> doc_embs,
                       std::span<const float> query_embs, std::uint32_t sdim,
                       float mmr_beta = 0.25f, std::uint32_t cand_k = 20, std::uint32_t sel_k = 10,
                       std::uint32_t n_terms = 30);
    void score(std::string_view query, const AdapterEvidence& evidence,
               std::span<float> out_scores) const override;
    void score_indexed(std::string_view query, std::uint32_t qi, const AdapterEvidence& evidence,
                       std::span<float> out_scores) const override;

private:
    const Bm25Index* idx_;
    const float* doc_embs_;
    const float* query_embs_;
    std::uint32_t sdim_;
    float mmr_beta_;
    std::uint32_t cand_k_, sel_k_, n_terms_;
};

// Utility: vector dot product
inline float emb_dot(const float* a, const float* b, std::size_t n) {
    return std::inner_product(a, a + n, b, 0.0f);
}

// Utility: RAKE-style keyphrase extraction from a query.
// Returns contiguous non-stopword sequences.
std::vector<std::string> extract_keyphrases(std::string_view text);

// ---------------------------------------------------------------------------
// KeyphraseStrategy — boosts documents where query keyphrases appear
// close together.  High concentration → higher boost.  Gated by BM25
// entropy: only activates when BM25 is uncertain (entropy > 0.30).
// ---------------------------------------------------------------------------
class KeyphraseStrategy final : public RetrievalStrategy {
public:
    explicit KeyphraseStrategy(const Bm25Index& idx, float boost_scale = 0.3f,
                               float entropy_thresh = 0.30f);
    void score(std::string_view query, const AdapterEvidence& evidence,
               std::span<float> out_scores) const override;
    float assess_quality(std::span<const float> scores) const override;

private:
    const Bm25Index* idx_;
    float boost_scale_;
    float entropy_thresh_;
};

} // namespace simeon
