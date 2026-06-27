#include "simeon/retrieval_strategy.hpp"
#include "simeon/bm25.hpp"
#include "simeon/corpus_adapter.hpp"
#include "simeon/fusion.hpp"
#include "simeon/prf.hpp"
#include "simeon/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace simeon {

// Local z-score normalisation (not in library API, used internally by strategies)
static void zscore_inplace(std::vector<float>& v) {
    if (v.empty())
        return;
    double mean = 0.0;
    for (float x : v)
        mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v)
        var += (static_cast<double>(x) - mean) * (static_cast<double>(x) - mean);
    float sd = static_cast<float>(std::sqrt(var / static_cast<double>(v.size())) + 1e-12);
    for (float& x : v)
        x = static_cast<float>((static_cast<double>(x) - mean) / sd);
}

// =========================================================================
// assess_quality — default: 3×(1−entropy10) + margin2
// =========================================================================
float RetrievalStrategy::assess_quality(std::span<const float> scores) const {
    auto tk = simeon::top_k(scores,
                            std::min<std::uint32_t>(10, static_cast<std::uint32_t>(scores.size())));
    if (tk.size() < 2)
        return 0.0f;

    // Softmax entropy
    double sum_exp = 0.0;
    const std::uint32_t kk = std::min<std::uint32_t>(10, static_cast<std::uint32_t>(tk.size()));
    for (std::uint32_t i = 0; i < kk; ++i)
        sum_exp += std::exp(static_cast<double>(tk[i].second));
    double entropy = 0.0;
    for (std::uint32_t i = 0; i < kk; ++i) {
        double p = std::exp(static_cast<double>(tk[i].second)) / sum_exp;
        if (p > 1e-12)
            entropy -= p * std::log(p);
    }
    float norm_e = static_cast<float>(entropy / std::log(static_cast<double>(kk > 1 ? kk : 2)));

    float denom = std::max(1e-6f, std::fabs(tk[0].second));
    float margin = (tk[0].second - tk[1].second) / denom;

    return 3.0f * (1.0f - norm_e) + margin;
}

// =========================================================================
// EntropyRouter
// =========================================================================
void EntropyRouter::route(std::string_view query, const QueryProfile& profile,
                          const AdapterEvidence& evidence, std::span<RetrievalStrategy* const> pool,
                          std::span<float> out_scores) const {
    if (pool.size() < 3) {
        if (!pool.empty())
            pool[0]->score(query, evidence, out_scores);
        return;
    }
    if (profile.bm25_entropy < 0.05f)
        pool[0]->score(query, evidence, out_scores);
    else if (profile.bm25_entropy > 0.50f)
        pool[2]->score(query, evidence, out_scores);
    else
        pool[1]->score(query, evidence, out_scores);
}

// =========================================================================
// =========================================================================
// =========================================================================
// Bm25Strategy
// =========================================================================
void Bm25Strategy::score(std::string_view query, const AdapterEvidence& evidence,
                         std::span<float> out_scores) const {
    if (evidence.aux_field.empty()) {
        idx_->score(query, out_scores);
    } else {
        idx_->score_bm25f(query, evidence.aux_field, out_scores, 0.85f, 0.15f);
    }
    if (relation_boost_ > 0.0f && !evidence.relations.empty()) {
        for (const auto& rel : evidence.relations) {
            if (rel.target_doc < out_scores.size())
                out_scores[rel.target_doc] += relation_boost_ * rel.weight;
        }
    }
}

// =========================================================================
// LeadFieldStrategy
// =========================================================================
LeadFieldStrategy::LeadFieldStrategy(const Bm25Index& idx, std::span<const std::string> lead_texts,
                                     float body_weight, float aux_weight)
    : idx_(&idx), lead_texts_(lead_texts.begin(), lead_texts.end()), body_w_(body_weight),
      aux_w_(aux_weight) {}

void LeadFieldStrategy::score(std::string_view query, const AdapterEvidence& /*evidence*/,
                              std::span<float> out_scores) const {
    if (lead_texts_.empty() || aux_w_ <= 0.0f) {
        idx_->score(query, out_scores);
        return;
    }
    // Build a temporary aux query: the query text itself (same for lead field).
    // For lead-64, the aux field is pre-computed in lead_texts_.
    // We score BM25F with the query text as both body and aux query.
    idx_->score_bm25f(query, query, out_scores, body_w_, aux_w_);
    // Note: this uses the Bm25Index's internal aux_postings_ which are
    // populated via add_doc(text, aux_text).  The lead_texts_ must have been
    // passed as aux_text at index time.
}

// =========================================================================
// =========================================================================

// =========================================================================
// Keyphrase extraction — RAKE-style
// =========================================================================

namespace {

const std::unordered_set<std::string_view> kStopwords = {
    "a",       "an",     "the",    "and",     "or",   "but",   "in",   "on",   "at",    "to",
    "for",     "of",     "with",   "by",      "from", "is",    "are",  "was",  "were",  "be",
    "been",    "being",  "have",   "has",     "had",  "do",    "does", "did",  "will",  "would",
    "could",   "should", "may",    "might",   "can",  "shall", "this", "that", "these", "those",
    "it",      "its",    "they",   "them",    "we",   "us",    "he",   "she",  "his",   "her",
    "their",   "our",    "my",     "your",    "not",  "no",    "nor",  "so",   "as",    "if",
    "than",    "then",   "also",   "very",    "just", "about", "into", "over", "after", "before",
    "between", "under",  "during", "without",
};

bool is_stopword(std::string_view w) {
    return kStopwords.find(w) != kStopwords.end();
}

} // namespace

std::vector<std::string> extract_keyphrases(std::string_view text) {
    std::vector<std::string> phrases;
    std::vector<std::string_view> words;
    std::string lower;
    lower.reserve(text.size());
    std::string_view remaining = text;

    // Tokenize into lowercase words
    while (!remaining.empty()) {
        // Skip non-alpha
        while (!remaining.empty() && !std::isalpha(static_cast<unsigned char>(remaining.front())))
            remaining.remove_prefix(1);
        if (remaining.empty())
            break;
        const char* start = remaining.data();
        while (!remaining.empty() && std::isalpha(static_cast<unsigned char>(remaining.front())))
            remaining.remove_prefix(1);
        std::string_view word(start, remaining.data() - start);
        if (word.size() < 2)
            continue;
        lower.clear();
        for (char c : word)
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        if (!is_stopword(lower)) {
            phrases.push_back(lower);
        } else if (!phrases.empty()) {
            // Stopword ends a contiguous keyphrase run — but we emit multi-word phrases too.
            // For simplicity, emit each non-stopword as a 1-word keyphrase.
            // Multi-word phrases would require accumulating them.
        }
    }
    // Deduplicate
    std::sort(phrases.begin(), phrases.end());
    phrases.erase(std::unique(phrases.begin(), phrases.end()), phrases.end());
    return phrases;
}

// =========================================================================
// KeyphraseStrategy
// =========================================================================

KeyphraseStrategy::KeyphraseStrategy(const Bm25Index& idx, float boost_scale, float entropy_thresh)
    : idx_(&idx), boost_scale_(boost_scale), entropy_thresh_(entropy_thresh) {}

void KeyphraseStrategy::score(std::string_view query, const AdapterEvidence& /*evidence*/,
                              std::span<float> out_scores) const {
    const std::uint32_t nd = static_cast<std::uint32_t>(idx_->doc_count());
    std::vector<float> bm25_scores(nd, 0.0f);
    idx_->score(query, bm25_scores);

    // Gat by BM25 entropy
    auto tk = simeon::top_k(bm25_scores, 10);
    double ent = 0.0;
    if (tk.size() >= 2) {
        double sum_exp = 0.0;
        for (auto& [did, s] : tk)
            sum_exp += std::exp(static_cast<double>(s));
        for (auto& [did, s] : tk) {
            double p = std::exp(static_cast<double>(s)) / sum_exp;
            if (p > 1e-12)
                ent -= p * std::log(p);
        }
        ent /= std::log(static_cast<double>(tk.size()));
    }
    float entropy = static_cast<float>(ent);
    if (entropy < entropy_thresh_) {
        std::copy(bm25_scores.begin(), bm25_scores.end(), out_scores.begin());
        return;
    }

    auto kps = extract_keyphrases(query);
    if (kps.empty()) {
        std::copy(bm25_scores.begin(), bm25_scores.end(), out_scores.begin());
        return;
    }

    // Use posting-list lookups to find docs containing keyphrase words
    std::vector<std::uint64_t> kp_hashes;
    for (auto& kp : kps)
        kp_hashes.push_back(idx_->hash_term(kp));

    // Count how many keyphrase words appear in each doc
    std::vector<std::uint32_t> kp_match_count(nd, 0u);
    for (auto h : kp_hashes) {
        // We need to access postings per term. Use score_weighted_hashes trick:
        // score with weight 1 for this term to get per-doc presence.
        std::vector<float> term_scores(nd, 0.0f);
        std::pair<std::uint64_t, float> single_term{h, 1.0f};
        std::span<const std::pair<std::uint64_t, float>> st_span(&single_term, 1);
        idx_->score_weighted_hashes(st_span, term_scores);
        for (std::uint32_t di = 0; di < nd; ++di)
            if (term_scores[di] > 0.0f)
                ++kp_match_count[di];
    }

    // Z-score BM25 and apply boost
    zscore_inplace(bm25_scores);
    float n_kp = static_cast<float>(kps.size());
    float lam = 1.0f / (1.0f + std::exp(-(entropy - 0.48f) * 8.0f));
    for (std::uint32_t di = 0; di < nd; ++di) {
        float frac = static_cast<float>(kp_match_count[di]) / n_kp;
        bm25_scores[di] += lam * boost_scale_ * frac;
    }
    std::copy(bm25_scores.begin(), bm25_scores.end(), out_scores.begin());
}

float KeyphraseStrategy::assess_quality(std::span<const float> scores) const {
    return RetrievalStrategy::assess_quality(scores);
}

} // namespace simeon
