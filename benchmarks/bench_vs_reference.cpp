// Reference embedding comparison bench.
//
// Loads a frozen IR fixture (corpus + queries + qrels + pre-computed reference
// embeddings) and evaluates simeon configurations against the reference model
// on the same workload. Emits one JSONL record per configuration.
//
// Fixture format documented in docs/reference_fixture.md.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simeon/aho_corasick.hpp"
#include "simeon/bm25.hpp"
#include "simeon/concept_mining.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/fusion.hpp"
#include "simeon/matryoshka.hpp"
#include "simeon/minhash.hpp"
#include "simeon/persistent_homology.hpp"
#include "simeon/pmi.hpp"
#include "simeon/pq.hpp"
#include "simeon/prf.hpp"
#include "simeon/query_router.hpp"
#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"
#include "simeon/text_rank.hpp"
#include "simeon/tokenizer.hpp"

namespace {

struct Fixture {
    std::vector<std::string> query_ids;
    std::vector<std::string> query_texts;
    std::vector<std::string> doc_ids;
    std::vector<std::string> doc_texts;

    struct Qrel {
        std::uint32_t q;
        std::uint32_t d;
        std::uint32_t rel;
    };
    std::vector<Qrel> qrels;

    std::uint32_t ref_dim = 0;
    std::string ref_model;
    std::vector<float> ref_query_embs;
    std::vector<float> ref_doc_embs;
};

// Optional per-query routing telemetry. When non-null, router-grid and
// oracle-router rows append one JSONL line per (config, query) describing the
// chosen recipe, query features, and per-query metrics. Lets post-hoc analysis
// run without re-executing the bench.
FILE* g_router_per_query_fp = nullptr;

#ifdef SIMEON_RESEARCH_BENCH
enum class AuxFieldMode : std::uint8_t {
    None,
    TextRankTitle,
    AhoCorasickEntity,
};

struct SoftMatchConfig {
    std::uint32_t target_rank = 128;
    std::uint32_t min_token_count = 5;
    std::uint32_t max_vocab_size = 20'000;
    std::uint32_t nearest_k = 3;
    float lambda = 0.2f;
    float min_similarity = 0.35f;
};

struct SoftMatchTerm {
    std::string token;
    std::uint64_t hash = 0;
    const float* row = nullptr;
    float norm = 0.0f;
};

enum class TransportLegMode : std::uint8_t {
    ConceptsOnly,
    OrderedOnly,
    OrderedUnordered,
    OrderedUnorderedConcepts,
};

struct TransportConfig {
    std::uint32_t pool_size = 100;
    float alpha = 0.8f; // BM25 weight in the z-scored pool blend
    bool normalize_by_query_phrase_count = true;
    float max_positive_fraction = 2.0f; // >1 => disabled
    TransportLegMode mode = TransportLegMode::OrderedUnordered;
};

struct PooledConceptHit {
    const simeon::ConceptEntry* entry = nullptr;
    std::uint32_t tf = 0;
    float dl = 1.0f;
};

struct QueryPhraseNode {
    std::uint64_t a_hash = 0;
    std::uint64_t b_hash = 0;
    float query_weight = 0.0f;
};

struct GraphConfig {
    std::uint32_t pool_size = 100;
    float alpha = 0.8f;
    float damping = 0.85f;
    bool include_docdoc = false;
    std::uint32_t unordered_window = 8;
};

using WeightedHashTerm = simeon::WeightedHashTerm;
using SparseSignature = simeon::SparseSignature;

struct ClusterFragment {
    SparseSignature signature;
};

struct ClusterConfig {
    std::uint32_t pool_size = 100;
    float alpha = 0.8f;
    std::uint32_t top_fragments_per_doc = 3;
    std::uint32_t fragment_signature_terms = 8;
    std::uint32_t cluster_signature_terms = 24;
    float min_cluster_overlap = 0.35f;
    float min_query_cover = 0.35f;
};

using SemanticFragment = simeon::SemanticFragment;

struct FragmentGraphConfig {
    std::uint32_t pool_size = 100;
    float alpha = 0.8f;
    float damping = 0.85f;
    std::uint32_t top_fragments_per_doc = 3;
    float min_query_sim = 0.20f;
    float min_fragment_sim = 0.35f;
    float lexical_bridge_weight = 0.0f;
    float min_bridge_overlap = 0.35f;
    std::uint32_t fragment_signature_terms = 8;
    std::uint32_t max_iters = 20;
};

using GeometryGraphConfig = simeon::FragmentGeometryConfig;

struct StringTokenSink final : simeon::NGramEmitter {
    std::vector<std::string>* out = nullptr;
    void on_token(std::string_view tok, float) override { out->emplace_back(tok); }
};

constexpr simeon::TokenizerConfig word_only_cfg() noexcept {
    return simeon::TokenizerConfig{0, 0, false, true};
}

void tokenize_words(std::string_view text, std::vector<std::string>& out) {
    out.clear();
    StringTokenSink sink{};
    sink.out = &out;
    simeon::tokenize(text, word_only_cfg(), sink);
}

bool is_stopword(std::string_view token);

std::uint32_t query_bigram_count(std::string_view text) {
    std::vector<std::string> toks;
    tokenize_words(text, toks);
    return toks.size() >= 2 ? static_cast<std::uint32_t>(toks.size() - 1) : 0u;
}

std::uint32_t query_concept_count(const simeon::Bm25Index& idx,
                                  const simeon::ConceptIndex& concepts, std::string_view query) {
    std::vector<std::string> toks;
    tokenize_words(query, toks);
    if (toks.size() < 2)
        return 0u;
    std::uint32_t count = 0u;
    for (std::size_t i = 1; i < toks.size(); ++i) {
        const auto a = idx.hash_term(toks[i - 1]);
        const auto b = idx.hash_term(toks[i]);
        if (concepts.find(simeon::ConceptIndex::hash_bigram(a, b)))
            ++count;
    }
    return count;
}

void build_query_content_hashes(const simeon::Bm25Index& idx, std::string_view query,
                                std::unordered_set<std::uint64_t>& out) {
    out.clear();
    std::vector<std::string> toks;
    tokenize_words(query, toks);
    for (const auto& tok : toks) {
        if (is_stopword(tok) || tok.size() < 3)
            continue;
        out.insert(idx.hash_term(tok));
    }
}

float bounded_pmi(float pmi) {
    return pmi > 0.0f ? pmi / (1.0f + pmi) : 0.0f;
}

std::vector<QueryPhraseNode> build_query_phrase_nodes(const simeon::Bm25Index& idx,
                                                      std::string_view query) {
    std::vector<std::string> toks;
    tokenize_words(query, toks);
    std::vector<QueryPhraseNode> out;
    if (toks.size() < 2)
        return out;
    out.reserve(toks.size() - 1);
    std::unordered_set<std::uint64_t> seen;
    for (std::size_t i = 1; i < toks.size(); ++i) {
        const auto a = idx.hash_term(toks[i - 1]);
        const auto b = idx.hash_term(toks[i]);
        const auto key = simeon::ConceptIndex::hash_bigram(a, b);
        if (seen.contains(key))
            continue;
        seen.insert(key);
        const float q_weight = 0.5f * (idx.idf(toks[i - 1]) + idx.idf(toks[i]));
        out.push_back(QueryPhraseNode{a, b, q_weight > 0.0f ? q_weight : 1.0f});
    }
    return out;
}

void build_doc_hash_tokens(const simeon::Bm25Index& idx, std::string_view text,
                           std::vector<std::uint64_t>& out) {
    std::vector<std::string> toks;
    tokenize_words(text, toks);
    out.clear();
    out.reserve(toks.size());
    for (const auto& tok : toks)
        out.push_back(idx.hash_term(tok));
}

std::vector<float> compute_doc_phrase_weights(std::span<const std::uint64_t> doc_tokens,
                                              std::span<const QueryPhraseNode> phrases,
                                              std::uint32_t unordered_window) {
    std::vector<float> out(phrases.size(), 0.0f);
    if (doc_tokens.size() < 2 || phrases.empty())
        return out;
    const float dl = static_cast<float>(doc_tokens.size() - 1);
    for (std::size_t pi = 0; pi < phrases.size(); ++pi) {
        const auto& p = phrases[pi];
        std::uint32_t ordered_tf = 0;
        std::uint32_t unordered_hits = 0;
        for (std::size_t i = 1; i < doc_tokens.size(); ++i) {
            ordered_tf += (doc_tokens[i - 1] == p.a_hash && doc_tokens[i] == p.b_hash) ? 1u : 0u;
        }
        for (std::size_t i = 0; i < doc_tokens.size(); ++i) {
            const auto j_hi = std::min<std::size_t>(doc_tokens.size(), i + unordered_window);
            bool hit = false;
            for (std::size_t j = i + 1; j < j_hi; ++j) {
                const auto x = doc_tokens[i];
                const auto y = doc_tokens[j];
                if ((x == p.a_hash && y == p.b_hash) || (x == p.b_hash && y == p.a_hash)) {
                    hit = true;
                    break;
                }
            }
            unordered_hits += hit ? 1u : 0u;
        }
        const float ordered = static_cast<float>(ordered_tf) / dl;
        const float unordered = static_cast<float>(unordered_hits) / dl;
        out[pi] = p.query_weight * (ordered + 0.5f * unordered);
    }
    return out;
}

void trim_signature(std::vector<WeightedHashTerm>& terms, std::uint32_t max_terms) {
    if (terms.size() <= max_terms)
        return;
    std::sort(terms.begin(), terms.end(), [](const auto& a, const auto& b) {
        if (a.weight != b.weight)
            return a.weight > b.weight;
        return a.hash < b.hash;
    });
    terms.resize(max_terms);
    std::sort(terms.begin(), terms.end(),
              [](const auto& a, const auto& b) { return a.hash < b.hash; });
}

SparseSignature build_sparse_signature(const simeon::Bm25Index& idx, std::string_view text,
                                       std::uint32_t max_terms) {
    std::vector<std::string> toks;
    tokenize_words(text, toks);
    std::unordered_map<std::uint64_t, float> weights;
    weights.reserve(toks.size());
    for (const auto& tok : toks) {
        if (is_stopword(tok) || tok.size() < 3)
            continue;
        const auto h = idx.hash_term(tok);
        const float w = std::max(0.25f, idx.idf(tok));
        auto it = weights.find(h);
        if (it == weights.end()) {
            weights.emplace(h, w);
        } else if (w > it->second) {
            it->second = w;
        }
    }
    SparseSignature sig;
    sig.terms.reserve(weights.size());
    for (const auto& [h, w] : weights)
        sig.terms.push_back(WeightedHashTerm{h, w});
    trim_signature(sig.terms, max_terms);
    for (const auto& term : sig.terms)
        sig.weight_sum += term.weight;
    return sig;
}

float weighted_intersection(const SparseSignature& a, const SparseSignature& b) {
    float out = 0.0f;
    std::size_t ia = 0, ib = 0;
    while (ia < a.terms.size() && ib < b.terms.size()) {
        if (a.terms[ia].hash == b.terms[ib].hash) {
            out += std::min(a.terms[ia].weight, b.terms[ib].weight);
            ++ia;
            ++ib;
        } else if (a.terms[ia].hash < b.terms[ib].hash) {
            ++ia;
        } else {
            ++ib;
        }
    }
    return out;
}

float weighted_overlap_coeff(const SparseSignature& a, const SparseSignature& b) {
    if (a.weight_sum <= 0.0f || b.weight_sum <= 0.0f)
        return 0.0f;
    const float inter = weighted_intersection(a, b);
    return inter / std::max(1e-6f, std::min(a.weight_sum, b.weight_sum));
}

float weighted_containment(const SparseSignature& subset, const SparseSignature& superset) {
    if (subset.weight_sum <= 0.0f)
        return 0.0f;
    const float inter = weighted_intersection(subset, superset);
    return inter / std::max(1e-6f, subset.weight_sum);
}

void merge_signature_max(SparseSignature& dst, const SparseSignature& src,
                         std::uint32_t max_terms) {
    std::vector<WeightedHashTerm> merged;
    merged.reserve(dst.terms.size() + src.terms.size());
    std::size_t i = 0, j = 0;
    while (i < dst.terms.size() || j < src.terms.size()) {
        if (j == src.terms.size() ||
            (i < dst.terms.size() && dst.terms[i].hash < src.terms[j].hash)) {
            merged.push_back(dst.terms[i++]);
        } else if (i == dst.terms.size() || src.terms[j].hash < dst.terms[i].hash) {
            merged.push_back(src.terms[j++]);
        } else {
            merged.push_back(WeightedHashTerm{dst.terms[i].hash,
                                              std::max(dst.terms[i].weight, src.terms[j].weight)});
            ++i;
            ++j;
        }
    }
    trim_signature(merged, max_terms);
    dst.terms.swap(merged);
    dst.weight_sum = 0.0f;
    for (const auto& term : dst.terms)
        dst.weight_sum += term.weight;
}

std::vector<ClusterFragment> build_doc_fragments(const simeon::Bm25Index& idx, std::string_view doc,
                                                 std::uint32_t top_fragments_per_doc,
                                                 std::uint32_t fragment_signature_terms) {
    simeon::TextRank ranker;
    auto ranked = ranker.rank(doc, top_fragments_per_doc);
    std::vector<ClusterFragment> out;
    out.reserve(ranked.empty() ? 1 : ranked.size());
    for (const auto& sent : ranked) {
        auto sig = build_sparse_signature(idx, sent.text, fragment_signature_terms);
        if (!sig.terms.empty())
            out.push_back(ClusterFragment{std::move(sig)});
    }
    if (out.empty()) {
        auto sig = build_sparse_signature(idx, doc, fragment_signature_terms);
        if (!sig.terms.empty())
            out.push_back(ClusterFragment{std::move(sig)});
    }
    return out;
}

float vector_l2_norm(const float* v, std::uint32_t dim) {
    double sq = 0.0;
    for (std::uint32_t i = 0; i < dim; ++i)
        sq += static_cast<double>(v[i]) * v[i];
    return static_cast<float>(std::sqrt(sq));
}

struct ClusterAssignment {
    std::uint32_t pool_index = 0;
    float doc_seed = 0.0f;
    float query_overlap = 0.0f;
    const SparseSignature* fragment = nullptr;
};

struct ClusterCoverNode {
    SparseSignature signature;
    std::vector<ClusterAssignment> members;
    float seed_mass = 0.0f;
    float query_cover = 0.0f;
};

void zscore_inplace(std::vector<float>& v) {
    if (v.empty())
        return;
    double mean = 0.0;
    for (float x : v)
        mean += x;
    mean /= static_cast<double>(v.size());
    double var = 0.0;
    for (float x : v)
        var += (static_cast<double>(x) - mean) * (static_cast<double>(x) - mean);
    const float sd = static_cast<float>(std::sqrt(var / static_cast<double>(v.size())) + 1e-12);
    for (float& x : v)
        x = static_cast<float>((static_cast<double>(x) - mean) / sd);
}

bool is_stopword(std::string_view token) {
    static const std::unordered_set<std::string_view> kStopwords = {
        "a",    "an",    "and",  "are", "as",  "at",   "be",   "by", "for",
        "from", "in",    "into", "is",  "it",  "of",   "on",   "or", "that",
        "the",  "their", "this", "to",  "was", "were", "with",
    };
    return kStopwords.contains(token);
}

bool keep_phrase_window(std::span<const std::string> tokens) {
    std::size_t stopwords = 0;
    std::size_t informative = 0;
    for (const auto& tok : tokens) {
        stopwords += is_stopword(tok) ? 1u : 0u;
        informative += tok.size() >= 3 ? 1u : 0u;
    }
    return informative > 0 && stopwords * 2 < tokens.size();
}

std::vector<std::string> build_textrank_titles(const Fixture& fx) {
    simeon::TextRank ranker;
    std::vector<std::string> out;
    out.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts) {
        const auto top = ranker.top_sentence(doc);
        out.emplace_back(top);
    }
    return out;
}

std::vector<std::string> build_ac_dictionary(const Fixture& fx) {
    simeon::TextRank ranker;
    std::unordered_map<std::string, std::uint32_t> phrase_df;
    std::vector<std::string> tokens;
    std::unordered_set<std::string> doc_phrases;
    const std::uint32_t max_df =
        std::max<std::uint32_t>(10u, static_cast<std::uint32_t>(fx.doc_texts.size() / 10));

    for (const auto& doc : fx.doc_texts) {
        doc_phrases.clear();
        const auto ranked = ranker.rank(doc, 2);
        for (const auto& sent : ranked) {
            tokenize_words(sent.text, tokens);
            for (std::size_t n = 2; n <= 3; ++n) {
                if (tokens.size() < n)
                    continue;
                for (std::size_t i = 0; i + n <= tokens.size(); ++i) {
                    const auto window = std::span<const std::string>(tokens).subspan(i, n);
                    if (!keep_phrase_window(window))
                        continue;
                    std::string phrase = tokens[i];
                    for (std::size_t j = 1; j < n; ++j) {
                        phrase.push_back(' ');
                        phrase += tokens[i + j];
                    }
                    doc_phrases.insert(std::move(phrase));
                }
            }
        }
        for (const auto& phrase : doc_phrases) {
            ++phrase_df[phrase];
        }
    }

    std::vector<std::pair<std::string, std::uint32_t>> kept;
    kept.reserve(phrase_df.size());
    for (auto& [phrase, df] : phrase_df) {
        if (df >= 10u && df <= max_df)
            kept.emplace_back(std::move(phrase), df);
    }
    std::sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
    if (kept.size() > 100'000)
        kept.resize(100'000);

    std::vector<std::string> patterns;
    patterns.reserve(kept.size());
    for (auto& [phrase, _df] : kept)
        patterns.push_back(std::move(phrase));
    return patterns;
}

std::vector<std::string> build_ac_token_field(std::span<const std::string> texts,
                                              const std::vector<std::string>& patterns) {
    std::vector<std::string> out(texts.size());
    if (patterns.empty())
        return out;

    std::vector<std::string_view> pattern_views;
    pattern_views.reserve(patterns.size());
    for (const auto& p : patterns)
        pattern_views.emplace_back(p);
    std::vector<std::uint16_t> type_ids(patterns.size(), 0u);

    simeon::AhoCorasick ac;
    if (const auto err = ac.build(pattern_views, type_ids)) {
        throw std::runtime_error("AhoCorasick build failed: " + err->message);
    }

    std::vector<std::uint32_t> seen(patterns.size(), 0u);
    std::uint32_t epoch = 1u;
    for (std::size_t di = 0; di < texts.size(); ++di) {
        if (epoch == 0u) {
            std::fill(seen.begin(), seen.end(), 0u);
            epoch = 1u;
        }
        auto& field = out[di];
        for (const auto& m : ac.match(texts[di])) {
            if (seen[m.pattern_id] == epoch)
                continue;
            seen[m.pattern_id] = epoch;
            if (!field.empty())
                field.push_back(' ');
            field += "ent";
            field += std::to_string(m.pattern_id);
        }
        ++epoch;
    }
    return out;
}

std::vector<SoftMatchTerm> build_softmatch_vocab(const Fixture& fx,
                                                 const simeon::PmiEmbeddings& pmi,
                                                 const simeon::Bm25Index& idx,
                                                 std::uint32_t min_token_count,
                                                 std::uint32_t max_vocab_size) {
    std::unordered_map<std::string, std::uint32_t> counts;
    std::vector<std::string> tokens;
    for (const auto& doc : fx.doc_texts) {
        tokenize_words(doc, tokens);
        for (const auto& tok : tokens)
            ++counts[tok];
    }

    std::vector<std::pair<std::string, std::uint32_t>> ranked;
    ranked.reserve(counts.size());
    for (auto& [tok, count] : counts) {
        if (count >= min_token_count && pmi.row(tok) != nullptr)
            ranked.emplace_back(std::move(tok), count);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });
    if (ranked.size() > max_vocab_size)
        ranked.resize(max_vocab_size);

    std::vector<SoftMatchTerm> out;
    out.reserve(ranked.size());
    for (auto& [tok, _count] : ranked) {
        const float* row = pmi.row(tok);
        if (!row)
            continue;
        double sq = 0.0;
        for (std::uint32_t d = 0; d < pmi.dim(); ++d)
            sq += static_cast<double>(row[d]) * row[d];
        const float norm = static_cast<float>(std::sqrt(sq));
        if (norm <= 0.0f)
            continue;
        out.push_back(SoftMatchTerm{std::move(tok), idx.hash_term(tok), row, norm});
    }
    return out;
}

std::vector<std::pair<std::uint64_t, float>>
build_softmatch_query(const simeon::Bm25Index& idx, const simeon::PmiEmbeddings& pmi,
                      std::span<const SoftMatchTerm> vocab, std::string_view query,
                      const SoftMatchConfig& cfg) {
    std::vector<std::string> q_tokens;
    tokenize_words(query, q_tokens);
    std::unordered_map<std::uint64_t, float> base;
    std::unordered_map<std::uint64_t, float> expansion;
    if (q_tokens.empty())
        return {};

    const float q_total = static_cast<float>(q_tokens.size());
    for (const auto& tok : q_tokens)
        base[idx.hash_term(tok)] += 1.0f / q_total;

    for (const auto& tok : q_tokens) {
        const float* qrow = pmi.row(tok);
        if (!qrow)
            continue;
        double sq = 0.0;
        for (std::uint32_t d = 0; d < pmi.dim(); ++d)
            sq += static_cast<double>(qrow[d]) * qrow[d];
        const float qnorm = static_cast<float>(std::sqrt(sq));
        if (qnorm <= 0.0f)
            continue;

        std::vector<std::pair<float, std::uint64_t>> best;
        best.reserve(cfg.nearest_k);
        for (const auto& cand : vocab) {
            if (cand.token == tok)
                continue;
            double dotp = 0.0;
            for (std::uint32_t d = 0; d < pmi.dim(); ++d)
                dotp += static_cast<double>(qrow[d]) * cand.row[d];
            const float sim = static_cast<float>(dotp / (static_cast<double>(qnorm) * cand.norm));
            if (sim < cfg.min_similarity)
                continue;
            if (best.size() < cfg.nearest_k) {
                best.emplace_back(sim, cand.hash);
                continue;
            }
            auto worst =
                std::min_element(best.begin(), best.end(),
                                 [](const auto& a, const auto& b) { return a.first < b.first; });
            if (sim > worst->first)
                *worst = {sim, cand.hash};
        }
        for (const auto& [sim, h] : best)
            expansion[h] += sim;
    }

    std::vector<std::pair<std::uint64_t, float>> weighted;
    weighted.reserve(base.size() + expansion.size());
    if (expansion.empty() || cfg.lambda <= 0.0f) {
        for (const auto& [h, w] : base)
            weighted.emplace_back(h, w);
        std::sort(weighted.begin(), weighted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        return weighted;
    }

    double exp_sum = 0.0;
    for (const auto& [_, w] : expansion)
        exp_sum += w;
    std::unordered_map<std::uint64_t, float> combined;
    combined.reserve(base.size() + expansion.size());
    const float one_minus_lambda = 1.0f - cfg.lambda;
    for (const auto& [h, w] : base)
        combined[h] += one_minus_lambda * w;
    if (exp_sum > 0.0) {
        for (const auto& [h, w] : expansion)
            combined[h] += cfg.lambda * static_cast<float>(w / exp_sum);
    }
    for (const auto& [h, w] : combined)
        weighted.emplace_back(h, w);
    std::sort(weighted.begin(), weighted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return weighted;
}

#endif // SIMEON_RESEARCH_BENCH

std::vector<std::pair<std::string, std::string>> read_tsv2(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::pair<std::string, std::string>> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("malformed TSV (missing tab) in " + path);
        }
        out.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    return out;
}

std::vector<std::array<std::string, 3>> read_tsv3(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::array<std::string, 3>> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const auto t1 = line.find('\t');
        const auto t2 = line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) {
            throw std::runtime_error("malformed qrels (need 3 cols) in " + path);
        }
        out.push_back({line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1), line.substr(t2 + 1)});
    }
    return out;
}

template <typename T> T read_le(std::istream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in)
        throw std::runtime_error("short read");
    return v;
}

void load_reference_bin(const std::string& path, Fixture& fx) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "SIMEONFX", 8) != 0) {
        throw std::runtime_error("bad magic in " + path);
    }
    const std::uint32_t version = read_le<std::uint32_t>(in);
    if (version != 1u) {
        throw std::runtime_error("unsupported reference.bin version");
    }
    fx.ref_dim = read_le<std::uint32_t>(in);
    const std::uint32_t nq = read_le<std::uint32_t>(in);
    const std::uint32_t nd = read_le<std::uint32_t>(in);
    const std::uint32_t mn_len = read_le<std::uint32_t>(in);
    fx.ref_model.resize(mn_len);
    in.read(fx.ref_model.data(), mn_len);

    if (nq != fx.query_ids.size() || nd != fx.doc_ids.size()) {
        throw std::runtime_error("reference.bin row counts disagree with TSVs");
    }
    if (fx.ref_dim == 0) {
        throw std::runtime_error("reference.bin dim must be > 0");
    }
    fx.ref_query_embs.resize(static_cast<std::size_t>(nq) * fx.ref_dim);
    fx.ref_doc_embs.resize(static_cast<std::size_t>(nd) * fx.ref_dim);
    in.read(reinterpret_cast<char*>(fx.ref_query_embs.data()),
            fx.ref_query_embs.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(fx.ref_doc_embs.data()),
            fx.ref_doc_embs.size() * sizeof(float));
    if (!in)
        throw std::runtime_error("short read on reference.bin payload");
}

Fixture load_fixture(const std::string& dir, const std::string& split = "test") {
    namespace fs = std::filesystem;
    Fixture fx;

    const std::string suffix = (split == "dev") ? "_dev" : "";
    const std::string queries_name = "queries" + suffix + ".tsv";
    const std::string qrels_name = "qrels" + suffix + ".tsv";
    const std::string ref_name = "reference" + suffix + ".bin";

    auto corpus = read_tsv2((fs::path(dir) / "corpus.tsv").string());
    fx.doc_ids.reserve(corpus.size());
    fx.doc_texts.reserve(corpus.size());
    for (auto& [id, text] : corpus) {
        fx.doc_ids.push_back(std::move(id));
        fx.doc_texts.push_back(std::move(text));
    }

    auto queries = read_tsv2((fs::path(dir) / queries_name).string());
    fx.query_ids.reserve(queries.size());
    fx.query_texts.reserve(queries.size());
    for (auto& [id, text] : queries) {
        fx.query_ids.push_back(std::move(id));
        fx.query_texts.push_back(std::move(text));
    }

    std::unordered_map<std::string, std::uint32_t> qmap, dmap;
    qmap.reserve(fx.query_ids.size());
    dmap.reserve(fx.doc_ids.size());
    for (std::uint32_t i = 0; i < fx.query_ids.size(); ++i)
        qmap[fx.query_ids[i]] = i;
    for (std::uint32_t i = 0; i < fx.doc_ids.size(); ++i)
        dmap[fx.doc_ids[i]] = i;

    auto qrels = read_tsv3((fs::path(dir) / qrels_name).string());
    fx.qrels.reserve(qrels.size());
    for (const auto& [qid, did, relstr] : qrels) {
        const auto qit = qmap.find(qid);
        const auto dit = dmap.find(did);
        if (qit == qmap.end() || dit == dmap.end())
            continue;
        const int rel = std::atoi(relstr.c_str());
        if (rel <= 0)
            continue;
        fx.qrels.push_back({qit->second, dit->second, static_cast<std::uint32_t>(rel)});
    }

    load_reference_bin((fs::path(dir) / ref_name).string(), fx);
    return fx;
}

// Routes through the SIMD-dispatched dot kernel so the bench's per-query
// scoring measures the same code path retrieval users hit at runtime.
inline float dot(const float* a, const float* b, std::size_t n) {
    return simeon::simd::dot(a, b, static_cast<std::uint32_t>(n));
}

struct Metrics {
    double ndcg_at_10;
    double recall_at_10;
    double recall_at_100;
    double mrr_at_10;
    std::size_t evaluated_queries; // queries with ≥1 relevant doc
};

// Per-query: scored[i] is (score, doc_idx). Higher score = better rank.
Metrics score_rankings(const std::vector<std::vector<std::pair<float, std::uint32_t>>>& rankings,
                       const Fixture& fx) {
    // Bucket qrels by query.
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    double ndcg10_sum = 0.0, r10_sum = 0.0, r100_sum = 0.0, mrr10_sum = 0.0;
    std::size_t n_eval = 0;

    for (std::uint32_t qi = 0; qi < rankings.size(); ++qi) {
        auto it = rel.find(qi);
        if (it == rel.end() || it->second.empty())
            continue;
        ++n_eval;
        const auto& qrel = it->second;

        // Sort by score descending; tie-break by doc index for determinism.
        auto sorted = rankings[qi];
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        // DCG@10
        double dcg = 0.0;
        std::size_t hits10 = 0, hits100 = 0;
        double first_rel_rank = 0.0;
        for (std::size_t r = 0; r < sorted.size(); ++r) {
            const auto rit = qrel.find(sorted[r].second);
            const std::uint32_t g = rit != qrel.end() ? rit->second : 0;
            if (r < 10 && g > 0) {
                dcg += static_cast<double>(g) / std::log2(static_cast<double>(r) + 2.0);
                ++hits10;
                if (first_rel_rank == 0.0)
                    first_rel_rank = static_cast<double>(r) + 1.0;
            }
            if (r < 100 && g > 0)
                ++hits100;
        }

        // IDCG@10
        std::vector<std::uint32_t> rels;
        rels.reserve(qrel.size());
        for (const auto& [_, g] : qrel)
            rels.push_back(g);
        std::sort(rels.begin(), rels.end(), std::greater<>());
        double idcg = 0.0;
        for (std::size_t r = 0; r < rels.size() && r < 10; ++r) {
            idcg += static_cast<double>(rels[r]) / std::log2(static_cast<double>(r) + 2.0);
        }
        ndcg10_sum += idcg > 0.0 ? dcg / idcg : 0.0;

        const double denom10 = static_cast<double>(std::min<std::size_t>(10, qrel.size()));
        const double denom100 = static_cast<double>(std::min<std::size_t>(100, qrel.size()));
        r10_sum += static_cast<double>(hits10) / denom10;
        r100_sum += static_cast<double>(hits100) / denom100;
        mrr10_sum += first_rel_rank > 0.0 ? 1.0 / first_rel_rank : 0.0;
    }

    const double n = static_cast<double>(n_eval == 0 ? 1 : n_eval);
    return {ndcg10_sum / n, r10_sum / n, r100_sum / n, mrr10_sum / n, n_eval};
}

// Optional per-config timings. Zero means "not measured" — emit() omits the field.
struct Timing {
    double doc_encode_us = 0.0;  // total simeon doc-encode time (sum over corpus)
    double index_build_us = 0.0; // BM25/PQ build/finalize/train time
    double query_us = 0.0; // total per-query work (encode + score + rerank), summed across queries
};

void emit(const char* config_name, const Fixture& fx, const Metrics& m,
          std::uint32_t code_bytes_per_doc, const Timing& t = {}) {
    const double nd = static_cast<double>(fx.doc_ids.size());
    const double nq = static_cast<double>(fx.query_ids.size());
    const double encode_us_per_doc = t.doc_encode_us > 0 ? t.doc_encode_us / nd : 0.0;
    const double query_us_per_q = t.query_us > 0 ? t.query_us / nq : 0.0;
    const double encode_dps = encode_us_per_doc > 0 ? 1.0e6 / encode_us_per_doc : 0.0;
    const double query_qps = query_us_per_q > 0 ? 1.0e6 / query_us_per_q : 0.0;

    std::printf("{\"bench\":\"vs_reference\",\"config\":\"%s\",\"model\":\"%s\","
                "\"queries\":%zu,\"docs\":%zu,\"evaluated_queries\":%zu,"
                "\"code_bytes_per_doc\":%u,"
                "\"ndcg_at_10\":%.4f,\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                "\"mrr_at_10\":%.4f,"
                "\"encode_us_per_doc\":%.3f,\"query_us_per_q\":%.3f,"
                "\"encode_throughput_dps\":%.1f,\"query_throughput_qps\":%.1f,"
                "\"index_build_us\":%.1f,"
                "\"simd_tier\":\"%s\"}\n",
                config_name, fx.ref_model.c_str(), fx.query_ids.size(), fx.doc_ids.size(),
                m.evaluated_queries, code_bytes_per_doc, m.ndcg_at_10, m.recall_at_10,
                m.recall_at_100, m.mrr_at_10, encode_us_per_doc, query_us_per_q, encode_dps,
                query_qps, t.index_build_us, simeon::simd_tier_name(simeon::active_simd_tier()));
    std::fflush(stdout);
}

using Clock = std::chrono::steady_clock;
inline double elapsed_us(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

void run_simeon(const char* name, simeon::EncoderConfig cfg, const Fixture& fx) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    std::vector<float> dembs(static_cast<std::size_t>(fx.doc_ids.size()) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < fx.doc_ids.size(); ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        enc.encode(fx.query_texts[qi], qembs.data() + qi * dim);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(dot(qembs.data() + qi * dim, dembs.data() + di * dim, dim),
                                      di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
}

void run_simeon_pq(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                   std::uint32_t pq_m) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    simeon::PQConfig pcfg{.dim = dim, .m = pq_m, .k = 256, .seed = 0xBEEF1234ULL};
    simeon::ProductQuantizer pq(pcfg);
    t0 = Clock::now();
    pq.train(dembs.data(), nd, 15);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(nd) * pq_m, 0);
    pq.encode_batch(dembs.data(), nd, codes.data());
    t.index_build_us = elapsed_us(t0);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::PQQuery pq_q(pq, qembs.data() + qi * dim);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(pq_q.inner_product(codes.data() + di * pq_m), di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), pq_m, t);
}

void run_reference(const Fixture& fx) {
    Timing t;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(dot(fx.ref_query_embs.data() + qi * fx.ref_dim,
                                          fx.ref_doc_embs.data() + di * fx.ref_dim, fx.ref_dim),
                                      di);
        }
    }
    t.query_us = elapsed_us(t0);
    // Note: doc_encode_us is intentionally 0 here — reference-model encoding is
    // measured out of process so the benchmark can compare against a realistic
    // CPU throughput number without adding Python/ML dependencies to simeon.
    emit("reference", fx, score_rankings(rankings, fx),
         static_cast<std::uint32_t>(fx.ref_dim * sizeof(float)), t);
}

// Build a BM25 index over fx.doc_texts. Returns the index and the build time;
// caller reuses both across multiple bench rows so we only pay indexing once.
struct Bm25WithTiming {
    simeon::Bm25Index idx;
    double build_us = 0.0;
};

Bm25WithTiming build_bm25(const Fixture& fx, simeon::Bm25Config cfg = {}) {
    Bm25WithTiming out{simeon::Bm25Index(cfg), 0.0};
    auto t0 = Clock::now();
    for (const auto& d : fx.doc_texts)
        out.idx.add_doc(d);
    out.idx.finalize();
    out.build_us = elapsed_us(t0);
    return out;
}

Bm25WithTiming build_bm25f(const Fixture& fx, std::span<const std::string> aux_texts,
                           simeon::Bm25Config cfg = {}, double extra_build_us = 0.0) {
    if (aux_texts.size() != fx.doc_texts.size()) {
        throw std::runtime_error("build_bm25f aux_texts size mismatch");
    }
    Bm25WithTiming out{simeon::Bm25Index(cfg), 0.0};
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < fx.doc_texts.size(); ++i)
        out.idx.add_doc(fx.doc_texts[i], aux_texts[i]);
    out.idx.finalize();
    out.build_us = extra_build_us + elapsed_us(t0);
    return out;
}

void run_bm25(const char* name, const Fixture& fx, const simeon::Bm25Index& idx, double build_us) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], s);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    // BM25 has no per-doc fixed footprint (variable inverted-index size); emit 0.
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

#ifdef SIMEON_RESEARCH_BENCH
void run_bm25_softmatch(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                        double build_us, const simeon::PmiEmbeddings& pmi,
                        std::span<const SoftMatchTerm> vocab, const SoftMatchConfig& cfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto weighted = build_softmatch_query(idx, pmi, vocab, fx.query_texts[qi], cfg);
        idx.score_weighted_hashes(weighted, s);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(s[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_softmatch_grid(const Fixture& fx, const simeon::Bm25Index& idx, double build_us) {
    simeon::PmiConfig pcfg;
    pcfg.target_rank = 128;
    pcfg.min_token_count = 5;
    pcfg.max_vocab_size = 20'000;
    std::vector<std::string_view> seed_views;
    seed_views.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        seed_views.emplace_back(d);

    auto t0 = Clock::now();
    auto pmi = simeon::PmiEmbeddings::learn(std::span<const std::string_view>(seed_views), pcfg);
    auto vocab = build_softmatch_vocab(fx, pmi, idx, pcfg.min_token_count, pcfg.max_vocab_size);
    const double total_build_us = build_us + elapsed_us(t0);

    for (const auto [k, min_sim] :
         {std::pair<std::uint32_t, float>{3, 0.35f}, std::pair<std::uint32_t, float>{8, 0.20f}}) {
        for (float lambda : {0.2f, 0.5f, 1.0f}) {
            SoftMatchConfig cfg;
            cfg.target_rank = pcfg.target_rank;
            cfg.min_token_count = pcfg.min_token_count;
            cfg.max_vocab_size = pcfg.max_vocab_size;
            cfg.nearest_k = k;
            cfg.lambda = lambda;
            cfg.min_similarity = min_sim;
            char name[128];
            std::snprintf(name, sizeof(name), "bm25_pmi_softmatch_k%u_s%.2f_l%.1f",
                          static_cast<unsigned>(k), static_cast<double>(min_sim),
                          static_cast<double>(lambda));
            run_bm25_softmatch(name, fx, idx, total_build_us, pmi, vocab, cfg);
        }
    }
}

void run_bm25f(const char* name, const Fixture& fx, const simeon::Bm25Index& idx, double build_us,
               float weight_body, float weight_aux, std::span<const std::string> aux_queries = {}) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const std::string_view aux_query = aux_queries.empty()
                                               ? std::string_view(fx.query_texts[qi])
                                               : std::string_view(aux_queries[qi]);
        idx.score_bm25f(fx.query_texts[qi], aux_query, s, weight_body, weight_aux);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25f_textrank(const Fixture& fx, std::span<const float> aux_weights) {
    auto t0 = Clock::now();
    auto aux_texts = build_textrank_titles(fx);
    const double aux_build_us = elapsed_us(t0);
    auto idx = build_bm25f(fx, aux_texts, simeon::Bm25Config{}, aux_build_us);
    run_bm25f("bm25_textrank_title_w0.0", fx, idx.idx, idx.build_us, 1.0f, 0.0f);
    for (float w : aux_weights) {
        char name[64];
        std::snprintf(name, sizeof(name), "bm25_textrank_title_w%.1f", static_cast<double>(w));
        run_bm25f(name, fx, idx.idx, idx.build_us, 1.0f, w);
    }
}

void run_bm25f_ac_entity(const Fixture& fx, std::span<const float> aux_weights) {
    auto t0 = Clock::now();
    auto patterns = build_ac_dictionary(fx);
    auto aux_texts = build_ac_token_field(std::span<const std::string>(fx.doc_texts), patterns);
    auto aux_queries = build_ac_token_field(std::span<const std::string>(fx.query_texts), patterns);
    const double aux_build_us = elapsed_us(t0);
    auto idx = build_bm25f(fx, aux_texts, simeon::Bm25Config{}, aux_build_us);
    std::fprintf(stderr, "[bm25f-ac] patterns=%zu\n", patterns.size());
    run_bm25f("bm25_ac_entity_w0.0", fx, idx.idx, idx.build_us, 1.0f, 0.0f, aux_queries);
    for (float w : aux_weights) {
        char name[64];
        std::snprintf(name, sizeof(name), "bm25_ac_entity_w%.1f", static_cast<double>(w));
        run_bm25f(name, fx, idx.idx, idx.build_us, 1.0f, w, aux_queries);
    }
}
#endif // SIMEON_RESEARCH_BENCH

// BM25 with RM3 pseudo-relevance feedback expansion. Same BM25 variant the
// passed-in idx was built with (Atire / BM25+ / SAB etc.); PRF only changes
// the query, not the scorer. See include/simeon/prf.hpp for algorithm.
void run_bm25_prf(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                  double build_us, const simeon::PrfConfig& pc) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::score_with_prf(idx, fx.query_texts[qi], s, pc);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

// T4 — clarity-adaptive RM3 (Bendersky-Metzler-Croft 2011). Per-query the
// expansion-term count K is interpolated from simplified_clarity via
// simeon::n_terms_for_clarity. Compute clarity inline via QueryRouter::features
// (cheap: O(n_query_terms) df lookups, no first-pass scoring).
void run_bm25_prf_adaptive(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                           double build_us, simeon::PrfConfig base_pc, std::uint32_t n_min = 5,
                           std::uint32_t n_max = 50, float clarity_lo = 0.5f,
                           float clarity_hi = 5.0f) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    simeon::QueryRouter router(idx, simeon::RouterConfig{});
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto features = router.features(fx.query_texts[qi]);
        base_pc.n_terms = simeon::n_terms_for_clarity(features.simplified_clarity, n_min, n_max,
                                                      clarity_lo, clarity_hi);
        simeon::score_with_prf(idx, fx.query_texts[qi], s, base_pc);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

// SDM (Metzler & Croft 2005): three-leg BM25 blend (unigram + ordered bigram
// + unordered bigram). idx must have been built with
// Bm25Config::build_word_bigrams = true. cfg.lambda_* default to Metzler's
// published weights (0.85, 0.10, 0.05).
void run_bm25_sdm(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                  double build_us, const simeon::SdmConfig& scfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score_sdm(fx.query_texts[qi], s, scfg);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

// T3 — Weighted SDM (Bendersky-Croft 2010). Per-bigram λ scales with the
// bigram's IDF (training-free reduction of full WSDM; see
// docs/wsdm_results.md when populated).
void run_bm25_wsdm(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                   double build_us, const simeon::WeightedSdmConfig& wcfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score_wsdm(fx.query_texts[qi], s, wcfg);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

// Step 1n — Latent Concept Model (Bendersky 2008, revisited 2024).
// Mines high-PMI word-bigram concepts from the corpus at fixture-setup time;
// at query time fuses base BM25 (via idx's variant) with PMI-weighted concept
// BM25 inside the same corpus. Training-free; concept_weight is fixed
// corpus-insensitive (0.5 default, can be overridden via bench row name).
void run_bm25_concepts(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                       double base_build_us, float concept_weight,
                       const simeon::ConceptConfig& ccfg) {
    Timing t;
    // Concept mining cost is treated as additive to the BM25 index build
    // (one-time, offline) — fold it into index_build_us so the reported
    // build time matches the configuration the row actually evaluates.
    std::vector<std::string_view> docs_view;
    docs_view.reserve(fx.doc_texts.size());
    for (const auto& s : fx.doc_texts)
        docs_view.emplace_back(s);
    auto tb = Clock::now();
    auto concepts = simeon::mine_concepts(idx, std::span<const std::string_view>(docs_view), ccfg);
    const double mine_us = elapsed_us(tb);
    t.index_build_us = base_build_us + mine_us;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::score_bm25_with_concepts(idx, concepts, fx.query_texts[qi], concept_weight,
                                         std::span<float>{s});
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

#ifdef SIMEON_RESEARCH_BENCH
void run_bm25_transport(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                        double build_us, const simeon::ConceptIndex* concepts,
                        const TransportConfig& cfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<float> phrase_scores(nd, 0.0f);
    std::vector<float> concept_scores(nd, 0.0f);

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, cfg.pool_size);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(neg_inf, di);

        std::fill(phrase_scores.begin(), phrase_scores.end(), 0.0f);
        std::fill(concept_scores.begin(), concept_scores.end(), 0.0f);

        if (cfg.mode == TransportLegMode::OrderedOnly ||
            cfg.mode == TransportLegMode::OrderedUnordered ||
            cfg.mode == TransportLegMode::OrderedUnorderedConcepts) {
            simeon::SdmConfig scfg;
            scfg.lambda_unigram = 0.0f;
            scfg.lambda_ordered = 1.0f;
            scfg.lambda_unordered = cfg.mode == TransportLegMode::OrderedOnly ? 0.0f : 0.5f;
            idx.score_sdm(fx.query_texts[qi], phrase_scores, scfg);
            if (cfg.normalize_by_query_phrase_count) {
                const auto qbg = query_bigram_count(fx.query_texts[qi]);
                if (qbg > 0) {
                    const float inv = 1.0f / static_cast<float>(qbg);
                    for (float& x : phrase_scores)
                        x *= inv;
                }
            }
        }

        if (concepts && (cfg.mode == TransportLegMode::ConceptsOnly ||
                         cfg.mode == TransportLegMode::OrderedUnorderedConcepts)) {
            concepts->score(fx.query_texts[qi], concept_scores);
            if (cfg.normalize_by_query_phrase_count) {
                const auto qc = query_concept_count(idx, *concepts, fx.query_texts[qi]);
                if (qc > 0) {
                    const float inv = 1.0f / static_cast<float>(qc);
                    for (float& x : concept_scores)
                        x *= inv;
                }
            }
        }

        std::vector<float> bm_pool(pool.size(), 0.0f);
        std::vector<float> tr_pool(pool.size(), 0.0f);
        std::size_t positive_transport = 0;
        for (std::size_t i = 0; i < pool.size(); ++i) {
            bm_pool[i] = pool[i].second;
            const auto did = pool[i].first;
            tr_pool[i] = phrase_scores[did] + concept_scores[did];
            positive_transport += tr_pool[i] > 0.0f ? 1u : 0u;
        }
        const float positive_fraction =
            pool.empty() ? 0.0f
                         : static_cast<float>(positive_transport) / static_cast<float>(pool.size());
        if (positive_fraction == 0.0f || positive_fraction > cfg.max_positive_fraction) {
            std::fill(tr_pool.begin(), tr_pool.end(), 0.0f);
        }
        zscore_inplace(bm_pool);
        zscore_inplace(tr_pool);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            rankings[qi][pool[i].first].first =
                cfg.alpha * bm_pool[i] + (1.0f - cfg.alpha) * tr_pool[i];
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_docanchored_concepts(const char* name, const Fixture& fx,
                                   const simeon::Bm25Index& idx, double build_us,
                                   const simeon::ConceptIndex& concepts, std::uint32_t pool_size,
                                   float alpha) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::string> toks;
    std::unordered_set<std::uint64_t> query_content_hashes;
    std::unordered_map<const simeon::ConceptEntry*, float> concept_mass;
    std::unordered_map<const simeon::ConceptEntry*, std::uint32_t> local_tf;

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, pool_size);
        build_query_content_hashes(idx, fx.query_texts[qi], query_content_hashes);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(neg_inf, di);

        std::vector<std::vector<PooledConceptHit>> pooled_hits(pool.size());
        concept_mass.clear();
        float bm25_sum = 0.0f;
        for (const auto& [_, sc] : pool)
            bm25_sum += std::max(sc, 0.0f);

        if (query_content_hashes.size() >= 2 && bm25_sum > 0.0f) {
            for (std::size_t i = 0; i < pool.size(); ++i) {
                const auto did = pool[i].first;
                tokenize_words(fx.doc_texts[did], toks);
                if (toks.size() < 2)
                    continue;
                local_tf.clear();
                for (std::size_t j = 1; j < toks.size(); ++j) {
                    const auto a = idx.hash_term(toks[j - 1]);
                    const auto b = idx.hash_term(toks[j]);
                    const auto* e = concepts.find(simeon::ConceptIndex::hash_bigram(a, b));
                    if (!e)
                        continue;
                    if (!query_content_hashes.contains(e->a_hash) ||
                        !query_content_hashes.contains(e->b_hash))
                        continue;
                    ++local_tf[e];
                }
                if (local_tf.empty())
                    continue;
                const float dl = static_cast<float>(toks.size() - 1);
                const float doc_weight = std::max(pool[i].second, 0.0f) / bm25_sum;
                pooled_hits[i].reserve(local_tf.size());
                for (const auto& [entry, tf] : local_tf) {
                    pooled_hits[i].push_back(PooledConceptHit{entry, tf, dl});
                    concept_mass[entry] += doc_weight * (static_cast<float>(tf) / dl) *
                                           bounded_pmi(entry->pmi) * entry->idf;
                }
            }
        }

        std::vector<float> bm_pool(pool.size(), 0.0f);
        std::vector<float> tr_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            bm_pool[i] = pool[i].second;
            float score = 0.0f;
            for (const auto& hit : pooled_hits[i]) {
                const auto it = concept_mass.find(hit.entry);
                if (it == concept_mass.end())
                    continue;
                score += it->second * (static_cast<float>(hit.tf) / hit.dl);
            }
            tr_pool[i] = score;
        }
        zscore_inplace(bm_pool);
        zscore_inplace(tr_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            rankings[qi][pool[i].first].first = alpha * bm_pool[i] + (1.0f - alpha) * tr_pool[i];
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_filtered_query_concepts(const char* name, const Fixture& fx,
                                      const simeon::Bm25Index& idx, double build_us,
                                      const simeon::ConceptIndex& concepts, std::uint32_t pool_size,
                                      float alpha) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::string> toks;
    std::unordered_set<std::uint64_t> query_content_hashes;
    std::unordered_map<const simeon::ConceptEntry*, std::uint32_t> query_concepts;
    std::unordered_map<const simeon::ConceptEntry*, std::uint32_t> local_tf;

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, pool_size);
        build_query_content_hashes(idx, fx.query_texts[qi], query_content_hashes);
        query_concepts.clear();

        tokenize_words(fx.query_texts[qi], toks);
        for (std::size_t i = 1; i < toks.size(); ++i) {
            const auto a = idx.hash_term(toks[i - 1]);
            const auto b = idx.hash_term(toks[i]);
            const auto* e = concepts.find(simeon::ConceptIndex::hash_bigram(a, b));
            if (!e)
                continue;
            if (!query_content_hashes.contains(e->a_hash) ||
                !query_content_hashes.contains(e->b_hash))
                continue;
            ++query_concepts[e];
        }

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(neg_inf, di);

        std::vector<float> bm_pool(pool.size(), 0.0f);
        std::vector<float> tr_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            bm_pool[i] = pool[i].second;
            if (query_concepts.empty())
                continue;
            tokenize_words(fx.doc_texts[pool[i].first], toks);
            if (toks.size() < 2)
                continue;
            local_tf.clear();
            for (std::size_t j = 1; j < toks.size(); ++j) {
                const auto a = idx.hash_term(toks[j - 1]);
                const auto b = idx.hash_term(toks[j]);
                const auto* e = concepts.find(simeon::ConceptIndex::hash_bigram(a, b));
                if (!e)
                    continue;
                const auto qit = query_concepts.find(e);
                if (qit == query_concepts.end())
                    continue;
                local_tf[e] += qit->second;
            }
            if (local_tf.empty())
                continue;
            const float dl = static_cast<float>(toks.size() - 1);
            float score = 0.0f;
            for (const auto& [entry, tf] : local_tf)
                score += bounded_pmi(entry->pmi) * entry->idf * (static_cast<float>(tf) / dl);
            tr_pool[i] = score;
        }
        zscore_inplace(bm_pool);
        zscore_inplace(tr_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            rankings[qi][pool[i].first].first = alpha * bm_pool[i] + (1.0f - alpha) * tr_pool[i];
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_graph_ppr(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                        double build_us, const GraphConfig& cfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::uint64_t> doc_hashes;

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, cfg.pool_size);
        const auto phrases = build_query_phrase_nodes(idx, fx.query_texts[qi]);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(neg_inf, di);
        if (pool.empty() || phrases.empty()) {
            for (const auto& [did, sc] : pool)
                rankings[qi][did].first = sc;
            continue;
        }

        const std::size_t query_node = 0;
        const std::size_t phrase_offset = 1;
        const std::size_t doc_offset = phrase_offset + phrases.size();
        const std::size_t n_nodes = doc_offset + pool.size();
        std::vector<std::vector<std::pair<std::size_t, float>>> adj(n_nodes);
        std::vector<std::vector<float>> doc_phrase(pool.size());

        for (std::size_t pi = 0; pi < phrases.size(); ++pi) {
            const auto pnode = phrase_offset + pi;
            const float w = phrases[pi].query_weight;
            adj[query_node].push_back({pnode, w});
            adj[pnode].push_back({query_node, w});
        }

        for (std::size_t di = 0; di < pool.size(); ++di) {
            build_doc_hash_tokens(idx, fx.doc_texts[pool[di].first], doc_hashes);
            doc_phrase[di] = compute_doc_phrase_weights(doc_hashes, phrases, cfg.unordered_window);
            for (std::size_t pi = 0; pi < phrases.size(); ++pi) {
                const float w = doc_phrase[di][pi];
                if (w <= 0.0f)
                    continue;
                const auto pnode = phrase_offset + pi;
                const auto dnode = doc_offset + di;
                adj[pnode].push_back({dnode, w});
                adj[dnode].push_back({pnode, w});
            }
        }

        if (cfg.include_docdoc) {
            for (std::size_t i = 0; i < pool.size(); ++i) {
                double norm_i = 0.0;
                for (float x : doc_phrase[i])
                    norm_i += static_cast<double>(x) * x;
                if (norm_i <= 0.0)
                    continue;
                for (std::size_t j = i + 1; j < pool.size(); ++j) {
                    double norm_j = 0.0;
                    double dotp = 0.0;
                    for (std::size_t pi = 0; pi < phrases.size(); ++pi) {
                        dotp += static_cast<double>(doc_phrase[i][pi]) * doc_phrase[j][pi];
                        norm_j += static_cast<double>(doc_phrase[j][pi]) * doc_phrase[j][pi];
                    }
                    if (norm_j <= 0.0 || dotp <= 0.0)
                        continue;
                    const float sim =
                        static_cast<float>(dotp / (std::sqrt(norm_i) * std::sqrt(norm_j)));
                    if (sim <= 0.0f)
                        continue;
                    const auto di_node = doc_offset + i;
                    const auto dj_node = doc_offset + j;
                    adj[di_node].push_back({dj_node, sim});
                    adj[dj_node].push_back({di_node, sim});
                }
            }
        }

        std::vector<float> row_sum(n_nodes, 0.0f);
        for (std::size_t i = 0; i < n_nodes; ++i) {
            for (const auto& [j, w] : adj[i])
                row_sum[i] += w;
        }

        std::vector<float> s(n_nodes, 0.0f), ns(n_nodes, 0.0f);
        s[query_node] = 1.0f;
        const float d = cfg.damping;
        for (std::uint32_t iter = 0; iter < 30; ++iter) {
            std::fill(ns.begin(), ns.end(), 0.0f);
            ns[query_node] = 1.0f - d;
            for (std::size_t j = 0; j < n_nodes; ++j) {
                if (row_sum[j] <= 0.0f || s[j] == 0.0f)
                    continue;
                const float scale = d * s[j] / row_sum[j];
                for (const auto& [i, w] : adj[j])
                    ns[i] += scale * w;
            }
            float delta = 0.0f;
            for (std::size_t i = 0; i < n_nodes; ++i)
                delta += std::fabs(ns[i] - s[i]);
            s.swap(ns);
            if (delta < 1e-4f)
                break;
        }

        std::vector<float> bm_pool(pool.size(), 0.0f), gr_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            bm_pool[i] = pool[i].second;
            gr_pool[i] = s[doc_offset + i];
        }
        zscore_inplace(bm_pool);
        zscore_inplace(gr_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            rankings[qi][pool[i].first].first =
                cfg.alpha * bm_pool[i] + (1.0f - cfg.alpha) * gr_pool[i];
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_cluster_cover(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                            double build_us,
                            std::span<const std::vector<ClusterFragment>> doc_frags,
                            const ClusterConfig& cfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, cfg.pool_size);
        auto query_sig =
            build_sparse_signature(idx, fx.query_texts[qi], cfg.fragment_signature_terms);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(neg_inf, di);
        if (pool.empty() || query_sig.terms.empty()) {
            for (const auto& [did, sc] : pool)
                rankings[qi][did].first = sc;
            continue;
        }

        std::vector<float> seed_pool(pool.size(), 0.0f);
        const float seed_denom = pool.empty() ? 1.0f : static_cast<float>(pool.size());
        for (std::size_t i = 0; i < pool.size(); ++i)
            seed_pool[i] = 1.0f / std::sqrt(1.0f + static_cast<float>(i));

        std::vector<ClusterAssignment> fragments;
        for (std::size_t pi = 0; pi < pool.size(); ++pi) {
            const auto did = pool[pi].first;
            for (const auto& frag : doc_frags[did]) {
                const float q_overlap = weighted_containment(query_sig, frag.signature);
                if (q_overlap <= 0.0f)
                    continue;
                fragments.push_back(ClusterAssignment{
                    .pool_index = static_cast<std::uint32_t>(pi),
                    .doc_seed = seed_pool[pi] / seed_denom,
                    .query_overlap = q_overlap,
                    .fragment = &frag.signature,
                });
            }
        }
        if (fragments.empty()) {
            for (const auto& [did, sc] : pool)
                rankings[qi][did].first = sc;
            continue;
        }

        std::sort(fragments.begin(), fragments.end(), [](const auto& a, const auto& b) {
            if (a.query_overlap != b.query_overlap)
                return a.query_overlap > b.query_overlap;
            return a.doc_seed > b.doc_seed;
        });

        std::vector<ClusterCoverNode> clusters;
        clusters.reserve(fragments.size());
        for (const auto& frag : fragments) {
            std::size_t best_idx = clusters.size();
            float best_overlap = 0.0f;
            for (std::size_t ci = 0; ci < clusters.size(); ++ci) {
                const float ov = weighted_overlap_coeff(*frag.fragment, clusters[ci].signature);
                if (ov > best_overlap) {
                    best_overlap = ov;
                    best_idx = ci;
                }
            }
            if (best_idx == clusters.size() || best_overlap < cfg.min_cluster_overlap) {
                ClusterCoverNode node;
                node.signature = *frag.fragment;
                node.members.push_back(frag);
                node.seed_mass = frag.doc_seed * frag.query_overlap;
                clusters.push_back(std::move(node));
            } else {
                merge_signature_max(clusters[best_idx].signature, *frag.fragment,
                                    cfg.cluster_signature_terms);
                clusters[best_idx].members.push_back(frag);
                clusters[best_idx].seed_mass += frag.doc_seed * frag.query_overlap;
            }
        }

        for (auto& cluster : clusters)
            cluster.query_cover = weighted_containment(query_sig, cluster.signature);

        std::vector<float> bm_pool(pool.size(), 0.0f), cl_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i)
            bm_pool[i] = pool[i].second;

        for (const auto& cluster : clusters) {
            if (cluster.query_cover < cfg.min_query_cover)
                continue;
            const float activation = cluster.query_cover * cluster.seed_mass /
                                     std::sqrt(static_cast<float>(cluster.members.size()));
            for (const auto& member : cluster.members) {
                const float membership = weighted_containment(*member.fragment, cluster.signature);
                if (membership <= 0.0f)
                    continue;
                const float mass = activation * membership * (0.5f + 0.5f * member.query_overlap);
                cl_pool[member.pool_index] += mass;
            }
        }

        zscore_inplace(bm_pool);
        zscore_inplace(cl_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            rankings[qi][pool[i].first].first =
                cfg.alpha * bm_pool[i] + (1.0f - cfg.alpha) * cl_pool[i];
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_fragment_graph(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                             double build_us, const simeon::Encoder& enc,
                             std::span<const std::vector<SemanticFragment>> doc_frags,
                             const FragmentGraphConfig& cfg) {
    struct FragRef {
        std::uint32_t pool_index = 0;
        const float* vec = nullptr;
        const SparseSignature* sig = nullptr;
    };

    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const std::uint32_t dim = enc.output_dim();
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<float> qvec(dim, 0.0f);

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, cfg.pool_size);
        enc.encode(fx.query_texts[qi], qvec.data());
        auto query_sig =
            build_sparse_signature(idx, fx.query_texts[qi], cfg.fragment_signature_terms);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(neg_inf, di);
        if (pool.empty() || vector_l2_norm(qvec.data(), dim) <= 0.0f || query_sig.terms.empty()) {
            for (const auto& [did, sc] : pool)
                rankings[qi][did].first = sc;
            continue;
        }

        std::vector<FragRef> frags;
        frags.reserve(pool.size() * cfg.top_fragments_per_doc);
        for (std::size_t pi = 0; pi < pool.size(); ++pi) {
            const auto did = pool[pi].first;
            const auto limit =
                std::min<std::size_t>(cfg.top_fragments_per_doc, doc_frags[did].size());
            for (std::size_t fi = 0; fi < limit; ++fi) {
                const auto& frag = doc_frags[did][fi];
                frags.push_back(FragRef{.pool_index = static_cast<std::uint32_t>(pi),
                                        .vec = frag.vec.data(),
                                        .sig = &frag.signature});
            }
        }
        if (frags.empty()) {
            for (const auto& [did, sc] : pool)
                rankings[qi][did].first = sc;
            continue;
        }

        std::vector<float> seed_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i)
            seed_pool[i] = 1.0f / std::sqrt(1.0f + static_cast<float>(i));

        std::vector<float> teleport(frags.size(), 0.0f);
        float teleport_sum = 0.0f;
        for (std::size_t fi = 0; fi < frags.size(); ++fi) {
            const float qsim = dot(qvec.data(), frags[fi].vec, dim);
            const float qlex = weighted_containment(query_sig, *frags[fi].sig);
            const float sem_mass = qsim > cfg.min_query_sim ? (qsim - cfg.min_query_sim) : 0.0f;
            const float lex_mass = qlex > cfg.min_bridge_overlap
                                       ? cfg.lexical_bridge_weight * (qlex - cfg.min_bridge_overlap)
                                       : 0.0f;
            const float total_mass = sem_mass + lex_mass;
            if (total_mass <= 0.0f)
                continue;
            const float mass = total_mass * seed_pool[frags[fi].pool_index];
            teleport[fi] = mass;
            teleport_sum += mass;
        }
        if (teleport_sum <= 0.0f) {
            for (const auto& [did, sc] : pool)
                rankings[qi][did].first = sc;
            continue;
        }
        for (float& x : teleport)
            x /= teleport_sum;

        std::vector<std::vector<std::pair<std::uint32_t, float>>> adj(frags.size());
        std::vector<float> row_sum(frags.size(), 0.0f);
        for (std::size_t i = 0; i < frags.size(); ++i) {
            for (std::size_t j = i + 1; j < frags.size(); ++j) {
                const float sem_sim = dot(frags[i].vec, frags[j].vec, dim);
                const float bridge = weighted_overlap_coeff(*frags[i].sig, *frags[j].sig);
                const float edge =
                    (sem_sim > cfg.min_fragment_sim ? (sem_sim - cfg.min_fragment_sim) : 0.0f) +
                    (bridge > cfg.min_bridge_overlap
                         ? cfg.lexical_bridge_weight * (bridge - cfg.min_bridge_overlap)
                         : 0.0f);
                if (edge <= 0.0f)
                    continue;
                adj[i].push_back({static_cast<std::uint32_t>(j), edge});
                adj[j].push_back({static_cast<std::uint32_t>(i), edge});
                row_sum[i] += edge;
                row_sum[j] += edge;
            }
        }

        std::vector<float> s = teleport, ns(frags.size(), 0.0f);
        const float d = cfg.damping;
        for (std::uint32_t iter = 0; iter < cfg.max_iters; ++iter) {
            std::fill(ns.begin(), ns.end(), 0.0f);
            for (std::size_t i = 0; i < frags.size(); ++i)
                ns[i] += (1.0f - d) * teleport[i];
            for (std::size_t j = 0; j < frags.size(); ++j) {
                if (s[j] <= 0.0f)
                    continue;
                if (row_sum[j] <= 0.0f) {
                    ns[j] += d * s[j];
                    continue;
                }
                const float scale = d * s[j] / row_sum[j];
                for (const auto& [i, w] : adj[j])
                    ns[i] += scale * w;
            }
            float delta = 0.0f;
            for (std::size_t i = 0; i < frags.size(); ++i)
                delta += std::fabs(ns[i] - s[i]);
            s.swap(ns);
            if (delta < 1e-4f)
                break;
        }

        std::vector<float> bm_pool(pool.size(), 0.0f), fg_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i)
            bm_pool[i] = pool[i].second;
        for (std::size_t fi = 0; fi < frags.size(); ++fi)
            fg_pool[frags[fi].pool_index] += s[fi];

        zscore_inplace(bm_pool);
        zscore_inplace(fg_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            rankings[qi][pool[i].first].first =
                cfg.alpha * bm_pool[i] + (1.0f - cfg.alpha) * fg_pool[i];
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_fragment_geometry(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                                double build_us, const simeon::Encoder& enc,
                                std::span<const std::vector<SemanticFragment>> doc_frags,
                                const GeometryGraphConfig& cfg) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].reserve(nd);
        const auto scores =
            simeon::score_fragment_geometry(fx.query_texts[qi], idx, enc, doc_frags, cfg);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(scores[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_bm25_transport_grid(const Fixture& fx) {
    simeon::Bm25Config cfg;
    cfg.build_word_bigrams = true;
    auto idx = build_bm25(fx, cfg);

    std::vector<std::string_view> docs_view;
    docs_view.reserve(fx.doc_texts.size());
    for (const auto& s : fx.doc_texts)
        docs_view.emplace_back(s);
    auto tb = Clock::now();
    const simeon::ConceptConfig ccfg{};
    auto concepts =
        simeon::mine_concepts(idx.idx, std::span<const std::string_view>(docs_view), ccfg);
    const double concept_mine_us = elapsed_us(tb);

    run_bm25_transport("bm25_transport_concepts_k100_a0.8", fx, idx.idx,
                       idx.build_us + concept_mine_us, &concepts,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .mode = TransportLegMode::ConceptsOnly});
    run_bm25_transport("bm25_transport_ordered_k100_a0.8", fx, idx.idx, idx.build_us, nullptr,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .mode = TransportLegMode::OrderedOnly});
    run_bm25_transport("bm25_transport_phrase_k100_a0.8", fx, idx.idx, idx.build_us, nullptr,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .mode = TransportLegMode::OrderedUnordered});
    run_bm25_transport("bm25_transport_phrase_concepts_k100_a0.8", fx, idx.idx,
                       idx.build_us + concept_mine_us, &concepts,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .mode = TransportLegMode::OrderedUnorderedConcepts});
    run_bm25_transport("bm25_transport_phrase_k500_a0.8", fx, idx.idx, idx.build_us, nullptr,
                       TransportConfig{.pool_size = 500,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .mode = TransportLegMode::OrderedUnordered});
    run_bm25_transport("bm25_transport_phrase_concepts_k500_a0.8", fx, idx.idx,
                       idx.build_us + concept_mine_us, &concepts,
                       TransportConfig{.pool_size = 500,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .mode = TransportLegMode::OrderedUnorderedConcepts});
    run_bm25_transport("bm25_transport_phrase_concepts_k100_a0.5", fx, idx.idx,
                       idx.build_us + concept_mine_us, &concepts,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.5f,
                                       .normalize_by_query_phrase_count = true,
                                       .max_positive_fraction = 2.0f,
                                       .mode = TransportLegMode::OrderedUnorderedConcepts});
    run_bm25_transport("bm25_transport_phrase_k100_a0.8_gate0.2", fx, idx.idx, idx.build_us,
                       nullptr,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .max_positive_fraction = 0.2f,
                                       .mode = TransportLegMode::OrderedUnordered});
    run_bm25_transport("bm25_transport_phrase_k100_a0.8_gate0.5", fx, idx.idx, idx.build_us,
                       nullptr,
                       TransportConfig{.pool_size = 100,
                                       .alpha = 0.8f,
                                       .normalize_by_query_phrase_count = true,
                                       .max_positive_fraction = 0.5f,
                                       .mode = TransportLegMode::OrderedUnordered});
    run_bm25_docanchored_concepts("bm25_transport_docconcept_k100_a0.8", fx, idx.idx,
                                  idx.build_us + concept_mine_us, concepts, 100, 0.8f);
    run_bm25_docanchored_concepts("bm25_transport_docconcept_k500_a0.8", fx, idx.idx,
                                  idx.build_us + concept_mine_us, concepts, 500, 0.8f);
    run_bm25_filtered_query_concepts("bm25_transport_filtered_concepts_k100_a0.8", fx, idx.idx,
                                     idx.build_us + concept_mine_us, concepts, 100, 0.8f);
}

void run_bm25_graph_grid(const Fixture& fx) {
    simeon::Bm25Config cfg;
    cfg.build_word_bigrams = true;
    auto idx = build_bm25(fx, cfg);
    run_bm25_graph_ppr("bm25_graph_phrase_ppr_k100_d0.85_a0.8", fx, idx.idx, idx.build_us,
                       GraphConfig{.pool_size = 100,
                                   .alpha = 0.8f,
                                   .damping = 0.85f,
                                   .include_docdoc = false,
                                   .unordered_window = 8});
    run_bm25_graph_ppr("bm25_graph_phrase_docdoc_ppr_k100_d0.85_a0.8", fx, idx.idx, idx.build_us,
                       GraphConfig{.pool_size = 100,
                                   .alpha = 0.8f,
                                   .damping = 0.85f,
                                   .include_docdoc = true,
                                   .unordered_window = 8});
    run_bm25_graph_ppr("bm25_graph_phrase_ppr_k300_d0.85_a0.8", fx, idx.idx, idx.build_us,
                       GraphConfig{.pool_size = 300,
                                   .alpha = 0.8f,
                                   .damping = 0.85f,
                                   .include_docdoc = false,
                                   .unordered_window = 8});
    run_bm25_graph_ppr("bm25_graph_phrase_docdoc_ppr_k100_d0.70_a0.8", fx, idx.idx, idx.build_us,
                       GraphConfig{.pool_size = 100,
                                   .alpha = 0.8f,
                                   .damping = 0.70f,
                                   .include_docdoc = true,
                                   .unordered_window = 8});
}

void run_bm25_cluster_grid(const Fixture& fx) {
    simeon::Bm25Config cfg;
    cfg.build_word_bigrams = true;
    auto idx = build_bm25(fx, cfg);

    auto tb = Clock::now();
    std::vector<std::vector<ClusterFragment>> doc_frags;
    doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        doc_frags.push_back(build_doc_fragments(idx.idx, doc, 3, 8));
    const double frag_build_us = elapsed_us(tb);

    run_bm25_cluster_cover("bm25_cluster_cover_k100_o0.35_q0.35_a0.8", fx, idx.idx,
                           idx.build_us + frag_build_us, doc_frags,
                           ClusterConfig{.pool_size = 100,
                                         .alpha = 0.8f,
                                         .top_fragments_per_doc = 3,
                                         .fragment_signature_terms = 8,
                                         .cluster_signature_terms = 24,
                                         .min_cluster_overlap = 0.35f,
                                         .min_query_cover = 0.35f});
    run_bm25_cluster_cover("bm25_cluster_cover_k100_o0.50_q0.35_a0.8", fx, idx.idx,
                           idx.build_us + frag_build_us, doc_frags,
                           ClusterConfig{.pool_size = 100,
                                         .alpha = 0.8f,
                                         .top_fragments_per_doc = 3,
                                         .fragment_signature_terms = 8,
                                         .cluster_signature_terms = 24,
                                         .min_cluster_overlap = 0.50f,
                                         .min_query_cover = 0.35f});
    run_bm25_cluster_cover("bm25_cluster_cover_k100_o0.35_q0.20_a0.8", fx, idx.idx,
                           idx.build_us + frag_build_us, doc_frags,
                           ClusterConfig{.pool_size = 100,
                                         .alpha = 0.8f,
                                         .top_fragments_per_doc = 3,
                                         .fragment_signature_terms = 8,
                                         .cluster_signature_terms = 24,
                                         .min_cluster_overlap = 0.35f,
                                         .min_query_cover = 0.20f});
    run_bm25_cluster_cover("bm25_cluster_cover_k300_o0.35_q0.35_a0.8", fx, idx.idx,
                           idx.build_us + frag_build_us, doc_frags,
                           ClusterConfig{.pool_size = 300,
                                         .alpha = 0.8f,
                                         .top_fragments_per_doc = 3,
                                         .fragment_signature_terms = 8,
                                         .cluster_signature_terms = 24,
                                         .min_cluster_overlap = 0.35f,
                                         .min_query_cover = 0.35f});
}

void run_bm25_fragment_graph_grid(const Fixture& fx) {
    simeon::Bm25Config bcfg;
    bcfg.build_word_bigrams = true;
    auto idx = build_bm25(fx, bcfg);

    std::vector<std::string_view> seed_views;
    seed_views.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        seed_views.emplace_back(d);

    auto tb = Clock::now();
    simeon::PmiConfig pcfg;
    pcfg.target_rank = 128;
    pcfg.min_token_count = 5;
    pcfg.max_vocab_size = 20'000;
    auto pmi = simeon::PmiEmbeddings::learn(std::span<const std::string_view>(seed_views), pcfg);

    simeon::EncoderConfig ecfg;
    ecfg.ngram_mode = simeon::NGramMode::WordOnly;
    ecfg.ngram_min = 1;
    ecfg.ngram_max = 1;
    ecfg.sketch_dim = 0;
    ecfg.output_dim = pmi.dim();
    ecfg.projection = simeon::ProjectionMode::None;
    ecfg.l2_normalize = true;
    ecfg.pmi_rows = &pmi;
    simeon::Encoder enc(ecfg);

    std::vector<std::vector<SemanticFragment>> doc_frags;
    doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        doc_frags.push_back(simeon::build_doc_semantic_fragments(enc, doc, idx.idx, 6, 8));
    const double total_build_us = idx.build_us + elapsed_us(tb);
    auto rich_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_doc_frags;
    rich_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_doc_frags.push_back(
            simeon::build_doc_semantic_fragments_rich(enc, doc, idx.idx, 6, 8));
    const double rich_total_build_us = total_build_us + elapsed_us(rich_tb);
    auto cover_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_cov_doc_frags;
    rich_cov_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_cov_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_covered(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f));
    const double rich_cov_total_build_us = total_build_us + elapsed_us(cover_tb);
    auto mmr_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_mmr_doc_frags;
    rich_mmr_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_mmr_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_mmr(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f, 0.35f, 0.30f, 0.15f));
    const double rich_mmr_total_build_us = total_build_us + elapsed_us(mmr_tb);
    auto mmr_novel_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_mmr_novel_doc_frags;
    rich_mmr_novel_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_mmr_novel_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_mmr(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f, 0.50f, 0.24f, 0.12f));
    const double rich_mmr_novel_total_build_us = total_build_us + elapsed_us(mmr_novel_tb);
    auto budget_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_budget_doc_frags;
    rich_budget_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_budget_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_budgeted(
            enc, doc, idx.idx, 6, 8, 0.60f, 4, 0.80f, 0.15f, 1));
    const double rich_budget_total_build_us = total_build_us + elapsed_us(budget_tb);

    run_bm25_fragment_graph("bm25_fragment_graph_k100_d0.85_q0.20_f0.35_a0.8", fx, idx.idx,
                            total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 100,
                                                .alpha = 0.8f,
                                                .damping = 0.85f,
                                                .top_fragments_per_doc = 3,
                                                .min_query_sim = 0.20f,
                                                .min_fragment_sim = 0.35f,
                                                .lexical_bridge_weight = 0.0f,
                                                .min_bridge_overlap = 0.35f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});
    run_bm25_fragment_graph("bm25_fragment_graph_k100_d0.70_q0.20_f0.35_a0.8", fx, idx.idx,
                            total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 100,
                                                .alpha = 0.8f,
                                                .damping = 0.70f,
                                                .top_fragments_per_doc = 3,
                                                .min_query_sim = 0.20f,
                                                .min_fragment_sim = 0.35f,
                                                .lexical_bridge_weight = 0.0f,
                                                .min_bridge_overlap = 0.35f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});
    run_bm25_fragment_graph("bm25_fragment_graph_k100_d0.85_q0.10_f0.20_a0.8", fx, idx.idx,
                            total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 100,
                                                .alpha = 0.8f,
                                                .damping = 0.85f,
                                                .top_fragments_per_doc = 3,
                                                .min_query_sim = 0.10f,
                                                .min_fragment_sim = 0.20f,
                                                .lexical_bridge_weight = 0.0f,
                                                .min_bridge_overlap = 0.20f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});
    run_bm25_fragment_graph("bm25_fragment_graph_k300_d0.85_q0.20_f0.35_a0.8", fx, idx.idx,
                            total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 300,
                                                .alpha = 0.8f,
                                                .damping = 0.85f,
                                                .top_fragments_per_doc = 3,
                                                .min_query_sim = 0.20f,
                                                .min_fragment_sim = 0.35f,
                                                .lexical_bridge_weight = 0.0f,
                                                .min_bridge_overlap = 0.35f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});
    run_bm25_fragment_graph("bm25_fragment_hybrid_k100_t6_d0.70_q0.20_f0.35_b0.20_a0.8", fx,
                            idx.idx, total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 100,
                                                .alpha = 0.8f,
                                                .damping = 0.70f,
                                                .top_fragments_per_doc = 6,
                                                .min_query_sim = 0.20f,
                                                .min_fragment_sim = 0.35f,
                                                .lexical_bridge_weight = 0.20f,
                                                .min_bridge_overlap = 0.20f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});
    run_bm25_fragment_graph("bm25_fragment_hybrid_k100_t6_d0.85_q0.10_f0.20_b0.35_a0.8", fx,
                            idx.idx, total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 100,
                                                .alpha = 0.8f,
                                                .damping = 0.85f,
                                                .top_fragments_per_doc = 6,
                                                .min_query_sim = 0.10f,
                                                .min_fragment_sim = 0.20f,
                                                .lexical_bridge_weight = 0.35f,
                                                .min_bridge_overlap = 0.10f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});
    run_bm25_fragment_graph("bm25_fragment_hybrid_k300_t6_d0.70_q0.20_f0.35_b0.20_a0.8", fx,
                            idx.idx, total_build_us, enc, doc_frags,
                            FragmentGraphConfig{.pool_size = 300,
                                                .alpha = 0.8f,
                                                .damping = 0.70f,
                                                .top_fragments_per_doc = 6,
                                                .min_query_sim = 0.20f,
                                                .min_fragment_sim = 0.35f,
                                                .lexical_bridge_weight = 0.20f,
                                                .min_bridge_overlap = 0.20f,
                                                .fragment_signature_terms = 8,
                                                .max_iters = 20});

    run_bm25_fragment_geometry("bm25_fragment_geom_k100_t4_s8_k8_p2_a0.8", fx, idx.idx,
                               total_build_us, enc, doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 4,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_k100_t4_s4_k16_p2_a0.8", fx, idx.idx,
                               total_build_us, enc, doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 4,
                                                   .attention_scale = 4.0f,
                                                   .knn = 16,
                                                   .steps = 2,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_k100_t6_s8_k8_p3_a0.8", fx, idx.idx,
                               total_build_us, enc, doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 6,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 3,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_adapt_k100_t4_a0.70_0.98_s4_10_k4_16_p1_3", fx,
                               idx.idx, total_build_us, enc, doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 4,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .adaptive = true,
                                                   .adaptive_idf_lo = 2.0f,
                                                   .adaptive_idf_hi = 5.0f,
                                                   .adaptive_decay_lo = 0.25f,
                                                   .adaptive_decay_hi = 0.95f,
                                                   .adaptive_alpha_lo = 0.70f,
                                                   .adaptive_alpha_hi = 0.98f,
                                                   .adaptive_scale_lo = 4.0f,
                                                   .adaptive_scale_hi = 10.0f,
                                                   .adaptive_knn_lo = 4,
                                                   .adaptive_knn_hi = 16,
                                                   .adaptive_steps_lo = 1,
                                                   .adaptive_steps_hi = 3,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_adapt_k100_t6_a0.65_0.95_s3_8_k8_20_p2_4", fx,
                               idx.idx, total_build_us, enc, doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 6,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 3,
                                                   .adaptive = true,
                                                   .adaptive_idf_lo = 1.5f,
                                                   .adaptive_idf_hi = 4.5f,
                                                   .adaptive_decay_lo = 0.20f,
                                                   .adaptive_decay_hi = 0.90f,
                                                   .adaptive_alpha_lo = 0.65f,
                                                   .adaptive_alpha_hi = 0.95f,
                                                   .adaptive_scale_lo = 3.0f,
                                                   .adaptive_scale_hi = 8.0f,
                                                   .adaptive_knn_lo = 8,
                                                   .adaptive_knn_hi = 20,
                                                   .adaptive_steps_lo = 2,
                                                   .adaptive_steps_hi = 4,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_rich_k100_t8_s8_k8_p2_a0.8", fx, idx.idx,
                               rich_total_build_us, enc, rich_doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 8,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_gsig_k100_t4_a0.65_0.98_s3_10_k4_16_p1_3", fx,
                               idx.idx, total_build_us, enc, doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 4,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .geometry_signal_adaptive = true,
                                                   .geometry_alpha_lo = 0.65f,
                                                   .geometry_alpha_hi = 0.98f,
                                                   .geometry_scale_lo = 3.0f,
                                                   .geometry_scale_hi = 10.0f,
                                                   .geometry_knn_lo = 4,
                                                   .geometry_knn_hi = 16,
                                                   .geometry_steps_lo = 1,
                                                   .geometry_steps_hi = 3,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_rich_gsig_k100_t8_a0.65_0.98_s3_10_k4_16_p1_3",
                               fx, idx.idx, rich_total_build_us, enc, rich_doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 8,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .geometry_signal_adaptive = true,
                                                   .geometry_alpha_lo = 0.65f,
                                                   .geometry_alpha_hi = 0.98f,
                                                   .geometry_scale_lo = 3.0f,
                                                   .geometry_scale_hi = 10.0f,
                                                   .geometry_knn_lo = 4,
                                                   .geometry_knn_hi = 16,
                                                   .geometry_steps_lo = 1,
                                                   .geometry_steps_hi = 3,
                                                   .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_richcov_k100_t8_o0.60_0.80_s8_k8_p2_a0.8", fx,
                               idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 8,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .use_phss = false});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_richcov_gsig_k100_t8_o0.60_0.80_a0.65_0.98_s3_10_k4_16_p1_3", fx,
        idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .geometry_signal_adaptive = true,
                            .geometry_alpha_lo = 0.65f,
                            .geometry_alpha_hi = 0.98f,
                            .geometry_scale_lo = 3.0f,
                            .geometry_scale_hi = 10.0f,
                            .geometry_knn_lo = 4,
                            .geometry_knn_hi = 16,
                            .geometry_steps_lo = 1,
                            .geometry_steps_hi = 3,
                            .use_phss = false});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_richmmr_k100_t8_l0.35_m0.30_0.15_o0.60_0.80_s8_k8_p2_a0.8", fx, idx.idx,
        rich_mmr_total_build_us, enc, rich_mmr_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = false});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_richmmr_k100_t8_l0.50_m0.24_0.12_o0.60_0.80_s8_k8_p2_a0.8", fx, idx.idx,
        rich_mmr_novel_total_build_us, enc, rich_mmr_novel_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = false});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_richbud_k100_t6_s4_a1_n0.15_o0.60_0.80_s8_k8_p2_a0.8", fx, idx.idx,
        rich_budget_total_build_us, enc, rich_budget_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 6,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = false});
    run_bm25_fragment_geometry("bm25_fragment_geom_richbud_gsig_k100_t6_s4_a1_n0.15_o0.60_0.80_a0."
                               "65_0.98_s3_10_k4_16_p1_3",
                               fx, idx.idx, rich_budget_total_build_us, enc, rich_budget_doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 6,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .geometry_signal_adaptive = true,
                                                   .geometry_alpha_lo = 0.65f,
                                                   .geometry_alpha_hi = 0.98f,
                                                   .geometry_scale_lo = 3.0f,
                                                   .geometry_scale_hi = 10.0f,
                                                   .geometry_knn_lo = 4,
                                                   .geometry_knn_hi = 16,
                                                   .geometry_steps_lo = 1,
                                                   .geometry_steps_hi = 3,
                                                   .use_phss = false});

    // Asymmetric two-stage: richcov sentences + MMR anchors
    auto asym_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_asym_doc_frags;
    rich_asym_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_asym_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_asymmetric(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f, 0.35f, 0.15f));
    const double rich_asym_total_build_us = total_build_us + elapsed_us(asym_tb);
    run_bm25_fragment_geometry("bm25_fragment_geom_richasym_k100_t6_s8_k8_p2_a0.8", fx, idx.idx,
                               rich_asym_total_build_us, enc, rich_asym_doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 6,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .use_phss = false});

    // Asymmetric variant 2: stricter anchor novelty (lambda=0.50)
    auto asym2_tb = Clock::now();
    std::vector<std::vector<SemanticFragment>> rich_asym2_doc_frags;
    rich_asym2_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_asym2_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_asymmetric(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f, 0.50f, 0.15f));
    const double rich_asym2_total_build_us = total_build_us + elapsed_us(asym2_tb);
    run_bm25_fragment_geometry("bm25_fragment_geom_richasym2_k100_t6_s8_k8_p2_a0.8", fx, idx.idx,
                               rich_asym2_total_build_us, enc, rich_asym2_doc_frags,
                               GeometryGraphConfig{.pool_size = 100,
                                                   .alpha = 0.8f,
                                                   .top_fragments_per_doc = 6,
                                                   .attention_scale = 8.0f,
                                                   .knn = 8,
                                                   .steps = 2,
                                                   .use_phss = false});

    // Persistent Homology Scale Selection (PHSS) variants
    // Replace fixed knn/top-k with data-driven threshold from 0D persistence.
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phss_k100_t4_gap", fx, idx.idx, total_build_us, enc, doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 4,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGap}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phssapprox_k100_t4_gap", fx, idx.idx, total_build_us, enc, doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 4,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phss_k100_t4_persist", fx, idx.idx, total_build_us, enc, doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 4,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::MaxPersistence}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phss_k100_t4_elbow", fx, idx.idx, total_build_us, enc, doc_frags,
        GeometryGraphConfig{
            .pool_size = 100,
            .alpha = 0.8f,
            .top_fragments_per_doc = 4,
            .attention_scale = 8.0f,
            .knn = 8,
            .steps = 2,
            .use_phss = true,
            .phss_config = simeon::PhssConfig{.criterion = simeon::PhssConfig::Criterion::Elbow}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phss_k100_t8_richcov_gap", fx, idx.idx, rich_cov_total_build_us, enc,
        rich_cov_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGap}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phssapprox_k100_t8_richcov_gap", fx, idx.idx, rich_cov_total_build_us,
        enc, rich_cov_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phssadapt_k100_t4_gap_c0.55", fx, idx.idx, total_build_us, enc,
        doc_frags,
        GeometryGraphConfig{
            .pool_size = 100,
            .alpha = 0.8f,
            .top_fragments_per_doc = 4,
            .attention_scale = 8.0f,
            .knn = 8,
            .steps = 2,
            .use_phss = true,
            .phss_config =
                simeon::PhssConfig{.criterion = simeon::PhssConfig::Criterion::LargestGap},
            .phss_adaptive = true,
            .phss_confidence_threshold = 0.55f});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phssadapt_k100_t8_richcov_gap_c0.55", fx, idx.idx,
        rich_cov_total_build_us, enc, rich_cov_doc_frags,
        GeometryGraphConfig{
            .pool_size = 100,
            .alpha = 0.8f,
            .top_fragments_per_doc = 8,
            .attention_scale = 8.0f,
            .knn = 8,
            .steps = 2,
            .use_phss = true,
            .phss_config =
                simeon::PhssConfig{.criterion = simeon::PhssConfig::Criterion::LargestGap},
            .phss_adaptive = true,
            .phss_confidence_threshold = 0.55f});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_richmmr_k100_t8_l0.35_phss_gap", fx, idx.idx, rich_mmr_total_build_us,
        enc, rich_mmr_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGap}});
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_richmmr_k100_t8_l0.50_phss_gap", fx, idx.idx,
        rich_mmr_novel_total_build_us, enc, rich_mmr_novel_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.8f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGap}});
}
#endif // SIMEON_RESEARCH_BENCH

void run_simeon_bm25_rrf(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                         const simeon::Bm25Index& idx) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);
    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::pair<std::uint32_t, float>> r_simeon(nd), r_bm25(nd);
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (std::uint32_t di = 0; di < nd; ++di) {
            r_simeon[di] = {di, dot(qembs.data() + qi * dim, dembs.data() + di * dim, dim)};
        }
        idx.score(fx.query_texts[qi], bm25_scores);
        for (std::uint32_t di = 0; di < nd; ++di)
            r_bm25[di] = {di, bm25_scores[di]};

        std::array<simeon::Ranking, 2> ins = {simeon::Ranking(r_simeon), simeon::Ranking(r_bm25)};
        auto fused = simeon::rrf_fuse(ins, 60.0f);

        rankings[qi].reserve(fused.size());
        for (const auto& [did, sc] : fused) {
            rankings[qi].emplace_back(sc, did);
        }
    }
    t.query_us = elapsed_us(t0);
    // Footprint is the simeon vectors (BM25 index sits in a different storage tier
    // and is variable-size; flagged in docs).
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
}

// Cascade: BM25 top-K pool → re-rank within the pool by one of three modes.
// Documents outside the pool are assigned -inf so they never appear in top-N
// during metric scoring. The cascade is the only ensemble pattern that can
// improve over BM25-alone on lexical-heavy corpora; see plan Step 1.
enum class RerankMode {
    SimeonCosine, // pure simeon cosine within pool
    LinearAlpha,  // alpha * bm25_z + (1-alpha) * simeon_z, both z-scored within pool
    PoolRrf,      // RRF restricted to the pool (avoids the global-RRF dilution)
    EntropyAlpha, // per-query α from entropy_alpha(); Step 1m novel contribution
};

void run_bm25_pool_simeon_rerank(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                                 const simeon::Bm25Index& idx, std::uint32_t pool_size,
                                 RerankMode mode, float alpha = 0.5f) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);
    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, pool_size);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(neg_inf, di);
        }

        if (mode == RerankMode::SimeonCosine) {
            for (const auto& [did, _] : pool) {
                const float s = dot(qembs.data() + qi * dim, dembs.data() + did * dim, dim);
                rankings[qi][did].first = s;
            }
        } else if (mode == RerankMode::LinearAlpha) {
            // Z-score both signals within the pool so alpha is a meaningful weight.
            std::vector<float> bm(pool.size()), si(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) {
                bm[i] = pool[i].second;
                si[i] = dot(qembs.data() + qi * dim, dembs.data() + pool[i].first * dim, dim);
            }
            auto zscore = [](std::vector<float>& v) {
                if (v.empty())
                    return;
                double mean = 0.0;
                for (float x : v)
                    mean += x;
                mean /= v.size();
                double var = 0.0;
                for (float x : v)
                    var += (x - mean) * (x - mean);
                const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
                for (float& x : v)
                    x = static_cast<float>((x - mean) / sd);
            };
            zscore(bm);
            zscore(si);
            for (std::size_t i = 0; i < pool.size(); ++i) {
                rankings[qi][pool[i].first].first = alpha * bm[i] + (1.0f - alpha) * si[i];
            }
        } else if (mode == RerankMode::PoolRrf) {
            std::vector<std::pair<std::uint32_t, float>> r_bm, r_si;
            r_bm.reserve(pool.size());
            r_si.reserve(pool.size());
            for (const auto& [did, score] : pool)
                r_bm.emplace_back(did, score);
            for (const auto& [did, _] : pool) {
                const float s = dot(qembs.data() + qi * dim, dembs.data() + did * dim, dim);
                r_si.emplace_back(did, s);
            }
            std::array<simeon::Ranking, 2> ins = {simeon::Ranking(r_bm), simeon::Ranking(r_si)};
            auto fused = simeon::rrf_fuse(ins, 60.0f);
            for (const auto& [did, sc] : fused) {
                rankings[qi][did].first = sc;
            }
        } else { // EntropyAlpha
            // Per-query α from Shannon entropy of each leg's top-K softmax.
            // The raw pool scores are the BM25 leg's top-K; for the simeon leg
            // we compute cosine scores over the same pool, then reuse the
            // top-K slice for the entropy estimate. Pool_jaccard is 1 here
            // because both legs share the same pool — so the collapse branch
            // is opt-in via the config threshold set below.
            std::vector<float> bm_pool(pool.size()), si_pool(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) {
                bm_pool[i] = pool[i].second;
                si_pool[i] = dot(qembs.data() + qi * dim, dembs.data() + pool[i].first * dim, dim);
            }
            std::vector<std::pair<std::uint32_t, float>> si_ranked(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) {
                si_ranked[i] = {pool[i].first, si_pool[i]};
            }
            std::sort(si_ranked.begin(), si_ranked.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            const std::size_t tk = std::min<std::size_t>(pool.size(), 50);
            std::vector<float> bm_top(tk), si_top(tk);
            for (std::size_t i = 0; i < tk; ++i) {
                bm_top[i] = pool[i].second;
                si_top[i] = si_ranked[i].second;
            }
            // pool_jaccard = 1.0 here (shared pool); set threshold > 1 so the
            // collapse branch never fires for this bench — we want to measure
            // the pure entropy-weighted regime.
            simeon::EntropyAlphaConfig cfg_ea;
            cfg_ea.top_k = 50;
            cfg_ea.agreement_threshold = 2.0f;
            const float a_q = simeon::entropy_alpha(bm_top, si_top, /*pool_jaccard=*/1.0f, cfg_ea);
            auto zscore_local = [](std::vector<float>& v) {
                if (v.empty())
                    return;
                double mean = 0.0;
                for (float x : v)
                    mean += x;
                mean /= v.size();
                double var = 0.0;
                for (float x : v)
                    var += (x - mean) * (x - mean);
                const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
                for (float& x : v)
                    x = static_cast<float>((x - mean) / sd);
            };
            zscore_local(bm_pool);
            zscore_local(si_pool);
            for (std::size_t i = 0; i < pool.size(); ++i) {
                rankings[qi][pool[i].first].first = a_q * bm_pool[i] + (1.0f - a_q) * si_pool[i];
            }
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
}

// Cascade: rank top-K by PQ codes (cheap), then re-rank those candidates by
// full simeon vectors. Index footprint is just the PQ codes; the full vectors
// are assumed to live in a slower storage tier (or be re-encoded on demand
// from the doc text).
void run_pq_first_then_full(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                            std::uint32_t pq_m, std::uint32_t candidate_k = 100) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);
    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    simeon::PQConfig pcfg{.dim = dim, .m = pq_m, .k = 256, .seed = 0xBEEF1234ULL};
    simeon::ProductQuantizer pq(pcfg);
    t0 = Clock::now();
    pq.train(dembs.data(), nd, 15);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(nd) * pq_m, 0);
    pq.encode_batch(dembs.data(), nd, codes.data());
    t.index_build_us = elapsed_us(t0);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> pq_scores(nd, 0.0f);
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::PQQuery pq_q(pq, qembs.data() + qi * dim);
        for (std::uint32_t di = 0; di < nd; ++di) {
            pq_scores[di] = pq_q.inner_product(codes.data() + di * pq_m);
        }
        auto cands = simeon::top_k(pq_scores, candidate_k);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(-std::numeric_limits<float>::infinity(), di);
        }
        for (const auto& [did, _] : cands) {
            const float s = dot(qembs.data() + qi * dim, dembs.data() + did * dim, dim);
            rankings[qi][did].first = s;
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), pq_m, t);
}

// MinHash retrieval: encode all docs once, score every doc per query by
// jaccard_estimate. Deterministic given (cfg, seed). Code-bytes-per-doc is
// k * sizeof(uint32) — directly comparable to simeon dense rows.
void run_minhash(const char* name, simeon::MinHashConfig cfg, const Fixture& fx) {
    Timing t;
    simeon::MinHashEncoder enc(cfg);
    const std::uint32_t k = enc.k();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::uint32_t> dsigs(static_cast<std::size_t>(nd) * k, 0);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dsigs.data() + i * k);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<std::uint32_t> qsig(k, 0);
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        enc.encode(fx.query_texts[qi], qsig.data());
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(
                simeon::jaccard_estimate(qsig.data(), dsigs.data() + di * k, k), di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx),
         static_cast<std::uint32_t>(k * sizeof(std::uint32_t)), t);
}

// Three-way RRF: BM25 ⊕ simeon ⊕ MinHash. The MinHash leg covers
// near-duplicate / boilerplate cases that cosine-space simeon misses.
void run_simeon_bm25_minhash_rrf(const char* name, simeon::EncoderConfig sc,
                                 simeon::MinHashConfig mc, const Fixture& fx,
                                 const simeon::Bm25Index& bm25_idx) {
    Timing t;
    simeon::Encoder senc(sc);
    simeon::MinHashEncoder menc(mc);
    const std::uint32_t dim = senc.output_dim();
    const std::uint32_t k = menc.k();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    std::vector<std::uint32_t> dsigs(static_cast<std::size_t>(nd) * k, 0);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        senc.encode(fx.doc_texts[i], dembs.data() + i * dim);
        menc.encode(fx.doc_texts[i], dsigs.data() + i * k);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    std::vector<std::uint32_t> qsigs(static_cast<std::size_t>(fx.query_ids.size()) * k, 0);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        senc.encode(fx.query_texts[i], qembs.data() + i * dim);
        menc.encode(fx.query_texts[i], qsigs.data() + i * k);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::pair<std::uint32_t, float>> r_si(nd), r_bm(nd), r_mh(nd);
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (std::uint32_t di = 0; di < nd; ++di) {
            r_si[di] = {di, dot(qembs.data() + qi * dim, dembs.data() + di * dim, dim)};
        }
        bm25_idx.score(fx.query_texts[qi], bm25_scores);
        for (std::uint32_t di = 0; di < nd; ++di)
            r_bm[di] = {di, bm25_scores[di]};
        for (std::uint32_t di = 0; di < nd; ++di) {
            r_mh[di] = {di,
                        simeon::jaccard_estimate(qsigs.data() + qi * k, dsigs.data() + di * k, k)};
        }
        std::array<simeon::Ranking, 3> ins = {simeon::Ranking(r_si), simeon::Ranking(r_bm),
                                              simeon::Ranking(r_mh)};
        auto fused = simeon::rrf_fuse(ins, 60.0f);
        rankings[qi].reserve(fused.size());
        for (const auto& [did, sc_] : fused) {
            rankings[qi].emplace_back(sc_, did);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx),
         static_cast<std::uint32_t>(dim * sizeof(float) + k * sizeof(std::uint32_t)), t);
}

// Router-driven cascade. Pre-builds Atire + SAB-smooth indexes and the
// simeon encoder; per query, the QueryRouter picks one of three recipes
// based on cheap pre-retrieval predictors (see docs/router_design.md). The
// emitted row reflects the *aggregated* metric across whichever recipes the
// router selected. Index-build cost is the sum of the two BM25 indexes plus
// simeon doc-encoding; query cost is the actual per-query dispatched work.
struct RouterCounts {
    std::uint32_t atire = 0;
    std::uint32_t sab = 0;
    std::uint32_t cascade = 0;
};

void run_router_cascade(const char* name, simeon::EncoderConfig sc, const Fixture& fx,
                        const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                        double bm25_build_us, std::uint32_t pool_size = 500,
                        float cascade_alpha = 0.75f, simeon::RouterConfig rc = {},
                        const simeon::PrfConfig* prf = nullptr) {
    Timing t;
    t.index_build_us = bm25_build_us;

    simeon::Encoder enc(sc);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    // Router uses the Atire index for IDF lookups (any finalized index works;
    // both share the corpus so df values are identical).
    simeon::QueryRouter router(atire_idx, rc);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    std::vector<float> q_emb(dim, 0.0f);
    RouterCounts counts;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto recipe = router.choose(fx.query_texts[qi]);
        rankings[qi].reserve(nd);
        if (recipe == simeon::Recipe::Bm25Atire) {
            ++counts.atire;
            if (prf)
                simeon::score_with_prf(atire_idx, fx.query_texts[qi], scores, *prf);
            else
                atire_idx.score(fx.query_texts[qi], scores);
            for (std::uint32_t di = 0; di < nd; ++di) {
                rankings[qi].emplace_back(scores[di], di);
            }
        } else if (recipe == simeon::Recipe::Bm25SabSmooth) {
            ++counts.sab;
            if (prf)
                simeon::score_with_prf(sab_idx, fx.query_texts[qi], scores, *prf);
            else
                sab_idx.score(fx.query_texts[qi], scores);
            for (std::uint32_t di = 0; di < nd; ++di) {
                rankings[qi].emplace_back(scores[di], di);
            }
        } else { // CascadeLinearAlpha — SAB pool, simeon rerank, z-score combine
            ++counts.cascade;
            sab_idx.score(fx.query_texts[qi], scores);
            auto pool = simeon::top_k(scores, pool_size);
            for (std::uint32_t di = 0; di < nd; ++di) {
                rankings[qi].emplace_back(neg_inf, di);
            }
            enc.encode(fx.query_texts[qi], q_emb.data());
            std::vector<float> bm(pool.size()), si(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) {
                bm[i] = pool[i].second;
                si[i] = dot(q_emb.data(), dembs.data() + pool[i].first * dim, dim);
            }
            auto zscore = [](std::vector<float>& v) {
                if (v.empty())
                    return;
                double mean = 0.0;
                for (float x : v)
                    mean += x;
                mean /= v.size();
                double var = 0.0;
                for (float x : v)
                    var += (x - mean) * (x - mean);
                const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
                for (float& x : v)
                    x = static_cast<float>((x - mean) / sd);
            };
            zscore(bm);
            zscore(si);
            for (std::size_t i = 0; i < pool.size(); ++i) {
                rankings[qi][pool[i].first].first =
                    cascade_alpha * bm[i] + (1.0f - cascade_alpha) * si[i];
            }
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
    std::fprintf(stderr, "[router] %s: atire=%u sab=%u cascade=%u (of %u queries)\n", name,
                 counts.atire, counts.sab, counts.cascade,
                 static_cast<std::uint32_t>(fx.query_ids.size()));
}

// Per-query metric computation for the per-query telemetry log and the
// oracle router. Mirrors the aggregate logic in score_rankings(): sort by
// score (tie-break by doc index), compute nDCG@10 / R@10 / R@100 / MRR@10
// using the saturating-recall convention.
struct PerQueryMetric {
    double ndcg_at_10 = 0.0;
    double recall_at_10 = 0.0;
    double recall_at_100 = 0.0;
    double mrr_at_10 = 0.0;
    bool has_relevant = false; // false → caller should skip from aggregates
};

PerQueryMetric compute_per_query(const std::vector<std::pair<float, std::uint32_t>>& ranking,
                                 const std::unordered_map<std::uint32_t, std::uint32_t>& qrel) {
    PerQueryMetric out;
    if (qrel.empty())
        return out;
    out.has_relevant = true;

    auto sorted = ranking;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first)
            return a.first > b.first;
        return a.second < b.second;
    });

    double dcg = 0.0;
    std::size_t hits10 = 0, hits100 = 0;
    double first_rel_rank = 0.0;
    for (std::size_t r = 0; r < sorted.size(); ++r) {
        const auto rit = qrel.find(sorted[r].second);
        const std::uint32_t g = rit != qrel.end() ? rit->second : 0;
        if (r < 10 && g > 0) {
            dcg += static_cast<double>(g) / std::log2(static_cast<double>(r) + 2.0);
            ++hits10;
            if (first_rel_rank == 0.0)
                first_rel_rank = static_cast<double>(r) + 1.0;
        }
        if (r < 100 && g > 0)
            ++hits100;
    }
    std::vector<std::uint32_t> rels;
    rels.reserve(qrel.size());
    for (const auto& [_, g] : qrel)
        rels.push_back(g);
    std::sort(rels.begin(), rels.end(), std::greater<>());
    double idcg = 0.0;
    for (std::size_t r = 0; r < rels.size() && r < 10; ++r) {
        idcg += static_cast<double>(rels[r]) / std::log2(static_cast<double>(r) + 2.0);
    }
    out.ndcg_at_10 = idcg > 0.0 ? dcg / idcg : 0.0;
    const double denom10 = static_cast<double>(std::min<std::size_t>(10, qrel.size()));
    const double denom100 = static_cast<double>(std::min<std::size_t>(100, qrel.size()));
    out.recall_at_10 = static_cast<double>(hits10) / denom10;
    out.recall_at_100 = static_cast<double>(hits100) / denom100;
    out.mrr_at_10 = first_rel_rank > 0.0 ? 1.0 / first_rel_rank : 0.0;
    return out;
}

// Per-spec config for run_router_grid. Each spec emits one labeled JSONL row.
struct RouterSweepSpec {
    std::string tag; // suffix appended to the row name
    simeon::RouterConfig rc;
    std::uint32_t pool_size = 500;
    float cascade_alpha = 0.75f;
    // Step 1g.1: when true, populate post-retrieval-lite predictors via
    // QueryRouter::features_with_pool() so Atire pool/decay AND-gates can
    // see real signal. Adds ~one extra BM25 score() per query for the SAB
    // pool used in pool_overlap_jaccard. Off by default to keep Pass A/B/C
    // behavior byte-identical.
    bool use_post_retrieval = false;
    std::uint32_t post_retrieval_k = 50;
};

// Score one query under a Recipe. Fills `out_ranking` with size-nd
// (score, doc_idx) pairs. SimeonCascade also requires the encoded corpus +
// pre-encoded query embedding.
void score_query_for_recipe(simeon::Recipe recipe, const Fixture& fx, std::uint32_t qi,
                            const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                            const std::vector<float>* dembs, std::uint32_t dim,
                            const float* q_emb_or_null, std::uint32_t pool_size,
                            float cascade_alpha, std::vector<float>& scratch_scores,
                            std::vector<std::pair<float, std::uint32_t>>& out_ranking) {
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();
    out_ranking.clear();
    out_ranking.reserve(nd);

    if (recipe == simeon::Recipe::Bm25Atire) {
        atire_idx.score(fx.query_texts[qi], scratch_scores);
        for (std::uint32_t di = 0; di < nd; ++di) {
            out_ranking.emplace_back(scratch_scores[di], di);
        }
    } else if (recipe == simeon::Recipe::Bm25SabSmooth) {
        sab_idx.score(fx.query_texts[qi], scratch_scores);
        for (std::uint32_t di = 0; di < nd; ++di) {
            out_ranking.emplace_back(scratch_scores[di], di);
        }
    } else { // CascadeLinearAlpha — SAB pool, simeon rerank, z-scored combine
        sab_idx.score(fx.query_texts[qi], scratch_scores);
        auto pool = simeon::top_k(scratch_scores, pool_size);
        for (std::uint32_t di = 0; di < nd; ++di) {
            out_ranking.emplace_back(neg_inf, di);
        }
        std::vector<float> bm(pool.size()), si(pool.size());
        for (std::size_t i = 0; i < pool.size(); ++i) {
            bm[i] = pool[i].second;
            si[i] = dot(q_emb_or_null, dembs->data() + pool[i].first * dim, dim);
        }
        auto zscore = [](std::vector<float>& v) {
            if (v.empty())
                return;
            double mean = 0.0;
            for (float x : v)
                mean += x;
            mean /= v.size();
            double var = 0.0;
            for (float x : v)
                var += (x - mean) * (x - mean);
            const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
            for (float& x : v)
                x = static_cast<float>((x - mean) / sd);
        };
        zscore(bm);
        zscore(si);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            out_ranking[pool[i].first].first =
                cascade_alpha * bm[i] + (1.0f - cascade_alpha) * si[i];
        }
    }
}

// Sweep helper: encode the corpus once, then run each RouterSweepSpec as its
// own bench row. Cheaper than calling run_router_cascade() N times because the
// expensive simeon doc-encode happens exactly once. Per-query telemetry is
// written to g_router_per_query_fp when set.
void run_router_grid(const std::string& prefix, simeon::EncoderConfig sc, const Fixture& fx,
                     const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                     double bm25_build_us, std::span<const RouterSweepSpec> specs) {
    simeon::Encoder enc(sc);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    const double doc_encode_us = elapsed_us(t0);

    // Bucket qrels by query for telemetry. (Aggregate scoring uses the same
    // bucketed view inside score_rankings.)
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel_bucket;
    for (const auto& q : fx.qrels)
        rel_bucket[q.q][q.d] = q.rel;
    static const std::unordered_map<std::uint32_t, std::uint32_t> empty_qrel;

    for (const auto& spec : specs) {
        const std::string name = prefix + "_" + spec.tag;

        Timing t;
        t.doc_encode_us = doc_encode_us;
        t.index_build_us = bm25_build_us;

        simeon::QueryRouter router(atire_idx, spec.rc);
        std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
        std::vector<float> scratch(nd, 0.0f);
        std::vector<float> q_emb(dim, 0.0f);
        std::array<std::uint32_t, 3> route_counts{0, 0, 0};

        // Pool span for Step 1g.1 post-retrieval-lite predictors. Convention
        // matches docs/router_design.md: pools[0]=Atire, pools[1]=SAB so that
        // pool_overlap_jaccard measures the routing-relevant disagreement.
        const std::array<const simeon::Bm25Index*, 2> pool_span{&atire_idx, &sab_idx};

        auto t1 = Clock::now();
        for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
            const auto features =
                spec.use_post_retrieval
                    ? router.features_with_pool(fx.query_texts[qi],
                                                std::span<const simeon::Bm25Index* const>(
                                                    pool_span.data(), pool_span.size()),
                                                spec.post_retrieval_k)
                    : router.features(fx.query_texts[qi]);
            const auto recipe = router.choose(features);
            ++route_counts[static_cast<std::size_t>(recipe)];
            const float* q_ptr = nullptr;
            if (recipe == simeon::Recipe::CascadeLinearAlpha) {
                enc.encode(fx.query_texts[qi], q_emb.data());
                q_ptr = q_emb.data();
            }
            score_query_for_recipe(recipe, fx, qi, atire_idx, sab_idx, &dembs, dim, q_ptr,
                                   spec.pool_size, spec.cascade_alpha, scratch, rankings[qi]);

            if (g_router_per_query_fp) {
                const auto rit = rel_bucket.find(qi);
                const auto& qrel = rit != rel_bucket.end() ? rit->second : empty_qrel;
                const auto pq = compute_per_query(rankings[qi], qrel);
                std::fprintf(g_router_per_query_fp,
                             "{\"config\":\"%s\",\"query_id\":\"%s\",\"recipe\":\"%s\","
                             "\"oov_rate\":%.4f,\"avg_idf\":%.4f,\"max_idf\":%.4f,"
                             "\"min_idf\":%.4f,\"idf_stddev\":%.4f,"
                             "\"n_terms\":%u,\"avg_term_chars\":%.2f,"
                             "\"score_decay_rate\":%.4f,\"score_normalized_var\":%.4f,"
                             "\"top_k_score_entropy\":%.4f,\"pool_overlap_jaccard\":%.4f,"
                             "\"nqc\":%.4f,\"wig_full\":%.4f,"
                             "\"has_relevant\":%s,\"ndcg_at_10\":%.4f,"
                             "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                             "\"mrr_at_10\":%.4f}\n",
                             name.c_str(), fx.query_ids[qi].c_str(), simeon::recipe_name(recipe),
                             features.oov_rate, features.avg_idf, features.max_idf,
                             features.min_idf, features.idf_stddev, features.n_terms,
                             features.avg_term_chars, features.score_decay_rate,
                             features.score_normalized_var, features.top_k_score_entropy,
                             features.pool_overlap_jaccard, features.nqc, features.wig_full,
                             pq.has_relevant ? "true" : "false", pq.ndcg_at_10, pq.recall_at_10,
                             pq.recall_at_100, pq.mrr_at_10);
            }
        }
        t.query_us = elapsed_us(t1);
        emit(name.c_str(), fx, score_rankings(rankings, fx),
             static_cast<std::uint32_t>(dim * sizeof(float)), t);
        std::fprintf(stderr, "[router-grid] %s: atire=%u sab=%u cascade=%u (of %u queries)\n",
                     name.c_str(),
                     route_counts[static_cast<std::size_t>(simeon::Recipe::Bm25Atire)],
                     route_counts[static_cast<std::size_t>(simeon::Recipe::Bm25SabSmooth)],
                     route_counts[static_cast<std::size_t>(simeon::Recipe::CascadeLinearAlpha)],
                     static_cast<std::uint32_t>(fx.query_ids.size()));
    }
    if (g_router_per_query_fp)
        std::fflush(g_router_per_query_fp);
}

// Oracle router: per query, score under all three recipes and use the one with
// the best per-query nDCG@10 (tie-break: best R@100, then SAB > Atire > Cascade
// for stability). The aggregate is the upper bound on what any pre-retrieval
// router that picks among these three recipes can achieve at the given
// (pool_size, cascade_alpha) setting. Per-query telemetry logs the oracle
// choice for downstream regret analysis.
void run_oracle_router(const char* name, simeon::EncoderConfig sc, const Fixture& fx,
                       const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                       double bm25_build_us, std::uint32_t pool_size = 500,
                       float cascade_alpha = 0.75f) {
    Timing t;
    t.index_build_us = bm25_build_us;

    simeon::Encoder enc(sc);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    simeon::QueryRouter router(atire_idx, simeon::RouterConfig{}); // for features only

    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel_bucket;
    for (const auto& q : fx.qrels)
        rel_bucket[q.q][q.d] = q.rel;
    static const std::unordered_map<std::uint32_t, std::uint32_t> empty_qrel;

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scratch(nd, 0.0f);
    std::vector<float> q_emb(dim, 0.0f);
    std::array<std::uint32_t, 3> oracle_counts{0, 0, 0};

    // Pool span: oracle telemetry includes Step 1g.1 post-retrieval predictors
    // so per-query rows have the full feature vector (the oracle itself does
    // not consult them — its choice is the per-query argmax over recipes).
    const std::array<const simeon::Bm25Index*, 2> pool_span{&atire_idx, &sab_idx};

    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto features = router.features_with_pool(
            fx.query_texts[qi],
            std::span<const simeon::Bm25Index* const>(pool_span.data(), pool_span.size()), 50);
        // Cascade always needs a query embedding; precompute once.
        enc.encode(fx.query_texts[qi], q_emb.data());

        std::vector<std::pair<float, std::uint32_t>> r_atire, r_sab, r_cascade;
        score_query_for_recipe(simeon::Recipe::Bm25Atire, fx, qi, atire_idx, sab_idx, &dembs, dim,
                               q_emb.data(), pool_size, cascade_alpha, scratch, r_atire);
        score_query_for_recipe(simeon::Recipe::Bm25SabSmooth, fx, qi, atire_idx, sab_idx, &dembs,
                               dim, q_emb.data(), pool_size, cascade_alpha, scratch, r_sab);
        score_query_for_recipe(simeon::Recipe::CascadeLinearAlpha, fx, qi, atire_idx, sab_idx,
                               &dembs, dim, q_emb.data(), pool_size, cascade_alpha, scratch,
                               r_cascade);

        const auto rit = rel_bucket.find(qi);
        const auto& qrel = rit != rel_bucket.end() ? rit->second : empty_qrel;
        const auto m_atire = compute_per_query(r_atire, qrel);
        const auto m_sab = compute_per_query(r_sab, qrel);
        const auto m_cascade = compute_per_query(r_cascade, qrel);

        // Pick best by nDCG@10; tie-break by R@100, then by stable preference
        // SAB > Atire > Cascade (matches the default-fallback recipe order).
        struct Candidate {
            simeon::Recipe r;
            const PerQueryMetric* m;
            std::vector<std::pair<float, std::uint32_t>>* ranking;
            int tie_pref; // higher = preferred on metric ties
        };
        std::array<Candidate, 3> cands = {{
            {simeon::Recipe::Bm25SabSmooth, &m_sab, &r_sab, 2},
            {simeon::Recipe::Bm25Atire, &m_atire, &r_atire, 1},
            {simeon::Recipe::CascadeLinearAlpha, &m_cascade, &r_cascade, 0},
        }};
        const Candidate* best = &cands[0];
        for (const auto& c : cands) {
            if (c.m->ndcg_at_10 > best->m->ndcg_at_10 ||
                (c.m->ndcg_at_10 == best->m->ndcg_at_10 &&
                 c.m->recall_at_100 > best->m->recall_at_100) ||
                (c.m->ndcg_at_10 == best->m->ndcg_at_10 &&
                 c.m->recall_at_100 == best->m->recall_at_100 && c.tie_pref > best->tie_pref)) {
                best = &c;
            }
        }
        rankings[qi] = std::move(*best->ranking);
        ++oracle_counts[static_cast<std::size_t>(best->r)];

        if (g_router_per_query_fp) {
            std::fprintf(g_router_per_query_fp,
                         "{\"config\":\"%s\",\"query_id\":\"%s\",\"recipe\":\"%s\","
                         "\"oov_rate\":%.4f,\"avg_idf\":%.4f,\"max_idf\":%.4f,"
                         "\"min_idf\":%.4f,\"idf_stddev\":%.4f,"
                         "\"n_terms\":%u,\"avg_term_chars\":%.2f,"
                         "\"score_decay_rate\":%.4f,\"score_normalized_var\":%.4f,"
                         "\"top_k_score_entropy\":%.4f,\"pool_overlap_jaccard\":%.4f,"
                         "\"nqc\":%.4f,\"wig_full\":%.4f,"
                         "\"scq_sum\":%.4f,\"simplified_clarity\":%.4f,"
                         "\"has_relevant\":%s,\"ndcg_at_10\":%.4f,"
                         "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                         "\"mrr_at_10\":%.4f,"
                         "\"ndcg_atire\":%.4f,\"ndcg_sab\":%.4f,\"ndcg_cascade\":%.4f}\n",
                         name, fx.query_ids[qi].c_str(), simeon::recipe_name(best->r),
                         features.oov_rate, features.avg_idf, features.max_idf, features.min_idf,
                         features.idf_stddev, features.n_terms, features.avg_term_chars,
                         features.score_decay_rate, features.score_normalized_var,
                         features.top_k_score_entropy, features.pool_overlap_jaccard, features.nqc,
                         features.wig_full, features.scq_sum, features.simplified_clarity,
                         best->m->has_relevant ? "true" : "false", best->m->ndcg_at_10,
                         best->m->recall_at_10, best->m->recall_at_100, best->m->mrr_at_10,
                         m_atire.ndcg_at_10, m_sab.ndcg_at_10, m_cascade.ndcg_at_10);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
    std::fprintf(stderr, "[router-oracle] %s: atire=%u sab=%u cascade=%u (of %u queries)\n", name,
                 oracle_counts[static_cast<std::size_t>(simeon::Recipe::Bm25Atire)],
                 oracle_counts[static_cast<std::size_t>(simeon::Recipe::Bm25SabSmooth)],
                 oracle_counts[static_cast<std::size_t>(simeon::Recipe::CascadeLinearAlpha)],
                 static_cast<std::uint32_t>(fx.query_ids.size()));
    if (g_router_per_query_fp)
        std::fflush(g_router_per_query_fp);
}

} // namespace

int main(int argc, char** argv) {
    const char* fixture_dir = nullptr;
    std::string queries_from = "test"; // "test" | "dev"
    const char* router_per_query_path = nullptr;
#ifdef SIMEON_RESEARCH_BENCH
    AuxFieldMode aux_mode = AuxFieldMode::None;
    bool softmatch_only = false;
    bool transport_only = false;
    bool graph_only = false;
    bool cluster_only = false;
    bool fragment_only = false;
#endif

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--queries-from" && i + 1 < argc) {
            queries_from = argv[++i];
            if (queries_from != "test" && queries_from != "dev") {
                std::fprintf(stderr, "--queries-from must be test|dev\n");
                return 2;
            }
#ifdef SIMEON_RESEARCH_BENCH
        } else if (a == "--aux-from" && i + 1 < argc) {
            const std::string v = argv[++i];
            if (v == "none") {
                aux_mode = AuxFieldMode::None;
            } else if (v == "textrank") {
                aux_mode = AuxFieldMode::TextRankTitle;
            } else if (v == "ac") {
                aux_mode = AuxFieldMode::AhoCorasickEntity;
            } else {
                std::fprintf(stderr, "--aux-from must be none|textrank|ac\n");
                return 2;
            }
        } else if (a == "--softmatch-only") {
            softmatch_only = true;
        } else if (a == "--transport-only") {
            transport_only = true;
        } else if (a == "--graph-only") {
            graph_only = true;
        } else if (a == "--cluster-only") {
            cluster_only = true;
        } else if (a == "--fragment-only") {
            fragment_only = true;
#endif
        } else if (a == "--router-per-query" && i + 1 < argc) {
            router_per_query_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
#ifdef SIMEON_RESEARCH_BENCH
            std::fprintf(
                stderr,
                "usage: %s [flags] <fixture_dir>\n"
                "  --queries-from {test,dev}    pick split (default test)\n"
                "  --aux-from {none,textrank,ac} add BM25F structural rows (default none)\n"
                "  --softmatch-only             run PMI soft-match rows only\n"
                "  --transport-only             run phrase/document transport rows only\n"
                "  --graph-only                 run graph transport rows only\n"
                "  --cluster-only               run fragment/cluster topology rows only\n"
                "  --fragment-only              run PMI fragment graph rows only\n"
                "  --router-per-query <path>    write per-query router telemetry JSONL\n"
                "  fixture_dir expects corpus.tsv, queries[_dev].tsv,\n"
                "  qrels[_dev].tsv, reference[_dev].bin\n"
                "  see docs/reference_fixture.md\n",
                argv[0]);
#else
            std::fprintf(stderr,
                         "usage: %s [flags] <fixture_dir>\n"
                         "  --queries-from {test,dev}    pick split (default test)\n"
                         "  --router-per-query <path>    write per-query router telemetry JSONL\n"
                         "  fixture_dir expects corpus.tsv, queries[_dev].tsv,\n"
                         "  qrels[_dev].tsv, reference[_dev].bin\n"
                         "  see docs/reference_fixture.md\n"
                         "\n"
                         "For archived probe-style experiment grids, use\n"
                         "  build/benchmarks/simeon_bench_vs_reference_research\n",
                         argv[0]);
#endif
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            return 2;
        } else if (!fixture_dir) {
            fixture_dir = argv[i];
        } else {
            std::fprintf(stderr, "extra positional argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (!fixture_dir) {
        std::fprintf(stderr, "missing fixture_dir (try --help)\n");
        return 2;
    }
    if (router_per_query_path) {
        g_router_per_query_fp = std::fopen(router_per_query_path, "w");
        if (!g_router_per_query_fp) {
            std::fprintf(stderr, "cannot open --router-per-query path %s\n", router_per_query_path);
            return 1;
        }
    }
    Fixture fx;
    try {
        fx = load_fixture(fixture_dir, queries_from);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fixture load failed: %s\n", e.what());
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 1;
    }
    std::fprintf(stderr, "[bench] fixture=%s split=%s nq=%zu nd=%zu\n", fixture_dir,
                 queries_from.c_str(), fx.query_ids.size(), fx.doc_ids.size());

    using PM = simeon::ProjectionMode;
    using NM = simeon::NGramMode;

    auto base = [](std::uint32_t out, PM p, NM nm = NM::CharAndWord, std::uint32_t lo = 3,
                   std::uint32_t hi = 5) {
        simeon::EncoderConfig c;
        c.ngram_mode = nm;
        c.ngram_min = lo;
        c.ngram_max = hi;
        c.sketch_dim = 4096;
        c.output_dim = out;
        c.projection = p;
        c.l2_normalize = true;
        return c;
    };

    run_reference(fx);

    auto bm25 = build_bm25(fx);
    run_bm25("bm25_only", fx, bm25.idx, bm25.build_us);
#ifdef SIMEON_RESEARCH_BENCH
    static constexpr std::array<float, 3> kBm25fAuxWeights{0.2f, 0.5f, 1.0f};
    if (aux_mode == AuxFieldMode::TextRankTitle) {
        run_bm25f_textrank(fx, kBm25fAuxWeights);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (aux_mode == AuxFieldMode::AhoCorasickEntity) {
        run_bm25f_ac_entity(fx, kBm25fAuxWeights);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (softmatch_only) {
        run_bm25_softmatch_grid(fx, bm25.idx, bm25.build_us);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (transport_only) {
        run_bm25_transport_grid(fx);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (graph_only) {
        run_bm25_graph_grid(fx);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (cluster_only) {
        run_bm25_cluster_grid(fx);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (fragment_only) {
        run_bm25_fragment_graph_grid(fx);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    }
#endif

    run_simeon("achlioptas_4096_384", base(384, PM::AchlioptasSparse), fx);
    run_simeon("achlioptas_4096_768", base(768, PM::AchlioptasSparse), fx);
    run_simeon("very_sparse_4096_384", base(384, PM::VerySparse), fx);
    run_simeon("gaussian_4096_384", base(384, PM::DenseGaussian), fx);

    auto matryoshka = base(384, PM::AchlioptasSparse);
    matryoshka.matryoshka = true;
    run_simeon("achlioptas_matryoshka_4096_384", matryoshka, fx);

    run_simeon_pq("achlioptas_4096_384_pq8", base(384, PM::AchlioptasSparse), fx, 8);
    run_simeon_pq("achlioptas_4096_384_pq16", base(384, PM::AchlioptasSparse), fx, 16);
    run_simeon_pq("achlioptas_4096_384_pq32", base(384, PM::AchlioptasSparse), fx, 32);
    run_simeon_bm25_rrf("bm25_rrf_simeon_4096_384", base(384, PM::AchlioptasSparse), fx, bm25.idx);
    run_pq_first_then_full("pq16_first_stage_then_full_k100", base(384, PM::AchlioptasSparse), fx,
                           16, 100);
    run_pq_first_then_full("pq16_first_stage_then_full_k500", base(384, PM::AchlioptasSparse), fx,
                           16, 500);

    // Cascade: BM25 top-K pool → simeon rerank within pool. Three rerank modes
    // and two simeon widths to bound headroom (see plan Step 1).
    auto cfg384 = base(384, PM::AchlioptasSparse);
    auto cfg768 = base(768, PM::AchlioptasSparse);
    using RM = RerankMode;

    run_bm25_pool_simeon_rerank("bm25_pool100_simeon_cos_4096_384", cfg384, fx, bm25.idx, 100,
                                RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_pool500_simeon_cos_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_pool1000_simeon_cos_4096_384", cfg384, fx, bm25.idx, 1000,
                                RM::SimeonCosine);

    run_bm25_pool_simeon_rerank("bm25_pool500_linear_alpha050_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::LinearAlpha, 0.5f);
    run_bm25_pool_simeon_rerank("bm25_pool500_linear_alpha075_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::LinearAlpha, 0.75f);
    run_bm25_pool_simeon_rerank("bm25_pool500_pool_rrf_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::PoolRrf);

    // Higher-quality simeon variant — bounds cascade headroom on this corpus.
    run_bm25_pool_simeon_rerank("bm25_pool500_simeon_cos_4096_768", cfg768, fx, bm25.idx, 500,
                                RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_pool500_linear_alpha075_4096_768", cfg768, fx, bm25.idx, 500,
                                RM::LinearAlpha, 0.75f);
    // Step 1m — entropy-weighted runtime α. Replaces the fixed α=0.75 with a
    // per-query value derived from softmax-entropy of each leg's top-50.
    // Strict generalization of the linear-α row at matched (cfg, pool).
    run_bm25_pool_simeon_rerank("bm25_pool500_entropy_alpha_4096_768", cfg768, fx, bm25.idx, 500,
                                RM::EntropyAlpha);

    // Step 1c — BM25 formulation ablation. Standalone variant rows + cascade
    // headline using the strongest pool source (SubwordAwareBackoff).
    auto bm25_variant = [&](const char* name, simeon::Bm25Variant v, float delta = 1.0f,
                            float subword_gamma = 0.0f) {
        simeon::Bm25Config cfg;
        cfg.variant = v;
        cfg.delta = delta;
        cfg.subword_gamma = subword_gamma;
        auto idx = build_bm25(fx, cfg);
        run_bm25(name, fx, idx.idx, idx.build_us);
        return idx;
    };
    bm25_variant("bm25_atire", simeon::Bm25Variant::Atire);
    bm25_variant("bm25_plus", simeon::Bm25Variant::BM25Plus);
    bm25_variant("bm25_l", simeon::Bm25Variant::BM25L);
    bm25_variant("bm25_dlh13", simeon::Bm25Variant::DLH13);
    bm25_variant("bm25_pl2", simeon::Bm25Variant::PL2);
    bm25_variant("bm25_dph", simeon::Bm25Variant::DPH);
    bm25_variant("bm25_dcm", simeon::Bm25Variant::Dcm);
    // T5 — Atire BM25 with Fang-Zhai 2005 axiomatic LTD correction.
    // Sweep α ∈ {0.3, 0.5, 0.7, 1.0}; α=1 is the byte-identity sanity row
    // against bm25_atire, α=0.5 is Fang-Zhai's recommended midpoint, α=0.3
    // / α=0.7 bracket it. See docs/ltd_results.md for the expected
    // long-doc-corpus (FiQA) lift hypothesis.
    auto ltd_variant = [&](const char* name, float alpha) {
        simeon::Bm25Config cfg;
        cfg.variant = simeon::Bm25Variant::AtireLTD;
        cfg.ltd_alpha = alpha;
        auto idx = build_bm25(fx, cfg);
        run_bm25(name, fx, idx.idx, idx.build_us);
    };
    ltd_variant("bm25_atire_ltd_a1.0", 1.0f);
    ltd_variant("bm25_atire_ltd_a0.7", 0.7f);
    ltd_variant("bm25_atire_ltd_a0.5", 0.5f);
    ltd_variant("bm25_atire_ltd_a0.3", 0.3f);
    auto sab_strict =
        bm25_variant("bm25_sab_strict", simeon::Bm25Variant::SubwordAwareBackoff, 1.0f, 0.0f);
    auto sab_smooth = bm25_variant("bm25_sab_smooth_gamma5",
                                   simeon::Bm25Variant::SubwordAwareBackoff, 1.0f, 5.0f);
    // Headline cascade: SAB pool (strongest morphological recall) → simeon
    // cosine rerank inside the pool. This is the row that should beat
    // bm25_only on lexical+morphological corpora like scifact.
    run_bm25_pool_simeon_rerank("bm25_sab_pool500_simeon_cos_4096_768", cfg768, fx, sab_smooth.idx,
                                500, RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_sab_strict_pool500_simeon_cos_4096_768", cfg768, fx,
                                sab_strict.idx, 500, RM::SimeonCosine);
    // Step 1m — entropy-α on SAB-smooth pool (the strongest morphological
    // source). Companion to bm25_pool500_entropy_alpha_4096_768.
    run_bm25_pool_simeon_rerank("bm25_sab_pool500_entropy_alpha_4096_768", cfg768, fx,
                                sab_smooth.idx, 500, RM::EntropyAlpha);

    // Step 1k A1 — RM3 pseudo-relevance feedback (Lavrenko & Croft 2001).
    // Canonical TREC settings (K=10, N=20, α=0.5) applied on top of the two
    // strongest BM25 pool sources. Attacks the FiQA-style short+paraphrase
    // query failure mode; expected no-op to modest-lift on scifact's short
    // abstracts.
    simeon::PrfConfig prf_canonical;
    prf_canonical.k = 10;
    prf_canonical.n_terms = 20;
    prf_canonical.alpha = 0.5f;
    run_bm25_prf("bm25_atire_rm3_k10_a0.5", fx, bm25.idx, bm25.build_us, prf_canonical);
    run_bm25_prf("bm25_sab_smooth_rm3_k10_a0.5", fx, sab_smooth.idx, sab_smooth.build_us,
                 prf_canonical);

    // T4 — fixed-K sweep + clarity-adaptive K (Bendersky-Metzler-Croft 2011).
    // SAB-only (RM3 lift is dominantly a SAB-side phenomenon per sdm_results
    // mechanism notes). The k10_a0.5 row above is N=20 by default and serves
    // as the middle of the sweep. Fixed-N rows isolate the sweet spot;
    // adaptive row tests whether per-query clarity beats any single fixed N.
    auto prf_n = [&](std::uint32_t n) {
        simeon::PrfConfig p = prf_canonical;
        p.n_terms = n;
        return p;
    };
    run_bm25_prf("bm25_sab_smooth_rm3_n10_a0.5", fx, sab_smooth.idx, sab_smooth.build_us,
                 prf_n(10));
    run_bm25_prf("bm25_sab_smooth_rm3_n30_a0.5", fx, sab_smooth.idx, sab_smooth.build_us,
                 prf_n(30));
    run_bm25_prf("bm25_sab_smooth_rm3_n50_a0.5", fx, sab_smooth.idx, sab_smooth.build_us,
                 prf_n(50));
    run_bm25_prf_adaptive("bm25_sab_smooth_rm3_adaptive_a0.5", fx, sab_smooth.idx,
                          sab_smooth.build_us, prf_canonical);

    // Step 1l — SDM (Sequential Dependence Model, Metzler & Croft 2005).
    // Builds a separate Atire and SAB-smooth index with build_word_bigrams=true
    // so the base bm25/sab_smooth indexes above stay byte-identical. Three
    // rows: Metzler defaults on Atire, same on SAB-smooth, and a weight-
    // sensitivity row (0.90/0.05/0.05) to check whether fixed weights are
    // corpus-agnostic on the current fixture.
    simeon::Bm25Config cfg_atire_sdm;
    cfg_atire_sdm.variant = simeon::Bm25Variant::Atire;
    cfg_atire_sdm.build_word_bigrams = true;
    auto bm25_sdm_atire = build_bm25(fx, cfg_atire_sdm);
    simeon::Bm25Config cfg_sab_sdm;
    cfg_sab_sdm.variant = simeon::Bm25Variant::SubwordAwareBackoff;
    cfg_sab_sdm.delta = 1.0f;
    cfg_sab_sdm.subword_gamma = 5.0f;
    cfg_sab_sdm.build_word_bigrams = true;
    auto bm25_sdm_sab = build_bm25(fx, cfg_sab_sdm);

    simeon::SdmConfig sdm_default; // 0.85 / 0.10 / 0.05
    simeon::SdmConfig sdm_unigram_heavy;
    sdm_unigram_heavy.lambda_unigram = 0.90f;
    sdm_unigram_heavy.lambda_ordered = 0.05f;
    sdm_unigram_heavy.lambda_unordered = 0.05f;

    run_bm25_sdm("bm25_atire_sdm_l0.85_0.10_0.05", fx, bm25_sdm_atire.idx, bm25_sdm_atire.build_us,
                 sdm_default);
    run_bm25_sdm("bm25_sab_smooth_sdm_l0.85_0.10_0.05", fx, bm25_sdm_sab.idx, bm25_sdm_sab.build_us,
                 sdm_default);
    run_bm25_sdm("bm25_atire_sdm_l0.90_0.05_0.05", fx, bm25_sdm_atire.idx, bm25_sdm_atire.build_us,
                 sdm_unigram_heavy);

    // T3 — Weighted SDM sweep over β at canonical λ. β=0 sanity-checks
    // byte-identity with sdm_default; β∈{0.5, 1.0, 1.5} brackets the
    // Bendersky-Croft 2010 IDF-reweighting recipe. Run on both Atire and
    // SAB-smooth so comparison is direct against the existing SDM rows.
    auto wsdm = [&](float beta) {
        simeon::WeightedSdmConfig w;
        w.beta = beta;
        return w;
    };
    run_bm25_wsdm("bm25_atire_wsdm_b0.0", fx, bm25_sdm_atire.idx, bm25_sdm_atire.build_us,
                  wsdm(0.0f));
    run_bm25_wsdm("bm25_atire_wsdm_b0.5", fx, bm25_sdm_atire.idx, bm25_sdm_atire.build_us,
                  wsdm(0.5f));
    run_bm25_wsdm("bm25_atire_wsdm_b1.0", fx, bm25_sdm_atire.idx, bm25_sdm_atire.build_us,
                  wsdm(1.0f));
    run_bm25_wsdm("bm25_atire_wsdm_b1.5", fx, bm25_sdm_atire.idx, bm25_sdm_atire.build_us,
                  wsdm(1.5f));
    run_bm25_wsdm("bm25_sab_smooth_wsdm_b1.0", fx, bm25_sdm_sab.idx, bm25_sdm_sab.build_us,
                  wsdm(1.0f));

    // Step 1n — Latent Concept Model (Bendersky 2008). Corpus-PMI word-bigram
    // concept mining + PMI-weighted concept BM25 fused with the base variant.
    // Two base variants (Atire, SAB-smooth) × concept_weight 0.5; low-weight
    // variant (0.25) at Atire to bound the blend sensitivity. FiQA promote
    // threshold is +0.010 nDCG@10 over the base variant; see
    // docs/concept_mining.md.
    simeon::ConceptConfig ccfg_default; // min_ttf=5, pmi_floor=2.0, max=200k
    run_bm25_concepts("bm25_atire_concepts_l0.50", fx, bm25.idx, bm25.build_us, 0.5f, ccfg_default);
    run_bm25_concepts("bm25_atire_concepts_l0.25", fx, bm25.idx, bm25.build_us, 0.25f,
                      ccfg_default);
    run_bm25_concepts("bm25_sab_smooth_concepts_l0.50", fx, sab_smooth.idx, sab_smooth.build_us,
                      0.5f, ccfg_default);

    // Step 1g.2 — training-free PMI / co-occurrence embeddings. Rows tagged
    // `_incorpus` use the evaluation corpus itself as the seed corpus; that is
    // a known leakage configuration and serves as a sanity ceiling only — the
    // headline number must come from a held-out fold or external seed corpus
    // (see docs/pmi_projection.md). In-corpus rows land first because they are
    // reproducible from the shipped fixture alone.
    std::vector<std::string_view> pmi_seed_view;
    pmi_seed_view.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        pmi_seed_view.emplace_back(d);
    auto make_pmi = [&](std::uint32_t rank) {
        simeon::PmiConfig pc;
        pc.target_rank = rank;
        pc.min_token_count = 5;
        pc.max_vocab_size = 50'000;
        pc.svd_iters = 4;
        pc.svd_oversample = 10;
        return simeon::PmiEmbeddings::learn(std::span<const std::string_view>(pmi_seed_view), pc);
    };
    auto pmi256 = make_pmi(256);
    auto pmi512 = make_pmi(512);

    auto pmi_cfg = [&](const simeon::PmiEmbeddings& e) {
        simeon::EncoderConfig c;
        c.ngram_mode = NM::WordOnly;
        c.ngram_min = 1;
        c.ngram_max = 1;
        c.sketch_dim = 0;
        c.output_dim = e.dim();
        c.projection = PM::None;
        c.l2_normalize = true;
        c.pmi_rows = &e;
        return c;
    };
    auto pmi256_cfg = pmi_cfg(pmi256);
    auto pmi512_cfg = pmi_cfg(pmi512);

    run_simeon("simeon_pmi256_incorpus", pmi256_cfg, fx);
    run_simeon("simeon_pmi512_incorpus", pmi512_cfg, fx);
    run_simeon_bm25_rrf("simeon_pmi256_rrf_bm25_incorpus", pmi256_cfg, fx, bm25.idx);
    run_bm25_pool_simeon_rerank("bm25_sab_pool500_simeon_pmi256_cos_rerank_incorpus", pmi256_cfg,
                                fx, sab_smooth.idx, 500, RM::SimeonCosine);

    // Step 4 — wider sketch sweep. Establishes max-quality reference points.
    auto wide = [&](std::uint32_t sketch, std::uint32_t out, PM p) {
        auto c = base(out, p);
        c.sketch_dim = sketch;
        return c;
    };
    run_simeon("achlioptas_8192_512", wide(8192, 512, PM::AchlioptasSparse), fx);
    run_simeon("achlioptas_8192_1024", wide(8192, 1024, PM::AchlioptasSparse), fx);
    run_simeon("achlioptas_16384_1024", wide(16384, 1024, PM::AchlioptasSparse), fx);

    // Step 5 — Mixed Tabulation hash + parameterized Sparse-JL.
    auto mixed_tab_cfg = base(384, PM::AchlioptasSparse);
    mixed_tab_cfg.hash = simeon::HashFamily::MixedTabulation;
    run_simeon("achlioptas_4096_384_mixed_tab", mixed_tab_cfg, fx);

    auto sparse_jl_cfg = base(384, PM::SparseJL);
    sparse_jl_cfg.sparse_jl_eps = 0.10f;
    run_simeon("sparse_jl_4096_384_eps0.10", sparse_jl_cfg, fx);
    sparse_jl_cfg.sparse_jl_eps = 0.05f;
    run_simeon("sparse_jl_4096_384_eps0.05", sparse_jl_cfg, fx);

    // Step 6 — data-aware matryoshka weights. Uses the corpus itself as the
    // seed for the variance estimate (no held-out fold available in this
    // fixture); documented in docs/benchmarks.md so the row's evaluation
    // context is explicit.
    {
        auto probe_cfg = base(384, PM::AchlioptasSparse);
        std::vector<std::string_view> seed_views;
        seed_views.reserve(fx.doc_texts.size());
        for (const auto& d : fx.doc_texts)
            seed_views.emplace_back(d);
        auto weights = simeon::compute_matryoshka_weights(probe_cfg, seed_views);
        auto data_aware_cfg = probe_cfg;
        data_aware_cfg.matryoshka = true;
        data_aware_cfg.matryoshka_weights = std::move(weights);
        run_simeon("achlioptas_4096_384_matryoshka_data_aware", data_aware_cfg, fx);
    }

    // Step 7 — FWHT (subsampled randomized Hadamard transform).
    run_simeon("fwht_4096_384", base(384, PM::Fwht), fx);
    run_simeon("fwht_8192_1024", wide(8192, 1024, PM::Fwht), fx);

    // Step 2 — Densified MinHash retrieval + three-way fusion.
    simeon::MinHashConfig mh256;
    mh256.k = 256;
    simeon::MinHashConfig mh512;
    mh512.k = 512;
    run_minhash("minhash_256", mh256, fx);
    run_minhash("minhash_512", mh512, fx);
    run_simeon_bm25_minhash_rrf("simeon_4096_384_rrf_bm25_rrf_minhash_256", cfg384, mh256, fx,
                                bm25.idx);

    // Router-driven cascade. Picks per-query among Bm25Atire / Bm25SabSmooth /
    // CascadeLinearAlpha using cheap pre-retrieval predictors. The Atire
    // index from `bm25` and SAB-smooth from `sab_smooth` are reused — no
    // additional index-build cost. Two router presets:
    //   - default thresholds (oov>0, idf>6, nterms>=4 + idf<=5)
    //   - aggressive cascade (lower nterms threshold, higher max_idf)
    {
        run_router_cascade("router_default_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                           bm25.build_us + sab_smooth.build_us);

        simeon::RouterConfig raggro;
        raggro.cascade_min_terms = 2;
        raggro.cascade_max_idf = 8.0f;
        run_router_cascade("router_cascade_aggressive_4096_768", cfg768, fx, bm25.idx,
                           sab_smooth.idx, bm25.build_us + sab_smooth.build_us, 500, 0.75f, raggro);

        simeon::RouterConfig rsab_only;
        rsab_only.cascade_min_terms = 9999;  // disable cascade route
        rsab_only.high_idf_threshold = 9999; // disable Atire route
        run_router_cascade("router_sab_only_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                           bm25.build_us + sab_smooth.build_us, 500, 0.75f, rsab_only);

        // Step 1g.2 — default-threshold router with the PMI256 encoder used
        // inside the cascade route (replaces the Achlioptas reranker).
        run_router_cascade("router_default_with_pmi256_cascade_incorpus", pmi256_cfg, fx, bm25.idx,
                           sab_smooth.idx, bm25.build_us + sab_smooth.build_us);

        // Step 1k A1 — router integration with RM3 expansion on the Atire and
        // SAB routes (cascade route's first-pass left un-expanded to keep its
        // pool membership comparable to router_default_4096_768).
        simeon::PrfConfig prf_router;
        prf_router.k = 10;
        prf_router.n_terms = 20;
        prf_router.alpha = 0.5f;
        run_router_cascade("router_default_with_rm3_k10_a0.5", cfg768, fx, bm25.idx, sab_smooth.idx,
                           bm25.build_us + sab_smooth.build_us, 500, 0.75f, simeon::RouterConfig{},
                           &prf_router);
    }

    // Step 1e — router ablation expansion. Two sweeps share an encoded corpus
    // and the existing Atire/SAB indices via run_router_grid:
    //   pass A: vary one threshold knob at a time around the default config
    //           (small Cartesian: 36 specs).
    //   pass B: fix thresholds to default, sweep pool_size × cascade_alpha
    //           (9 specs).
    // Plus an oracle row that picks the per-query best recipe (upper bound).
    {
        std::vector<RouterSweepSpec> specs;
        const float oovs[] = {0.0f, 0.25f};
        const float idfs[] = {3.0f, 5.0f, 9999.0f};
        const std::uint32_t nts[] = {2u, 4u, 6u};
        const float maxidfs[] = {5.0f, 7.0f};
        char tagbuf[128];
        for (float oov : oovs)
            for (float idf : idfs)
                for (std::uint32_t nt : nts)
                    for (float mi : maxidfs) {
                        simeon::RouterConfig rc;
                        rc.oov_threshold = oov;
                        rc.high_idf_threshold = idf;
                        rc.cascade_min_terms = nt;
                        rc.cascade_max_idf = mi;
                        std::snprintf(tagbuf, sizeof(tagbuf), "passA_oov%.2f_idf%.0f_nt%u_mi%.0f",
                                      oov, idf, nt, mi);
                        specs.push_back({tagbuf, rc, 500u, 0.75f});
                    }
        const std::uint32_t pools[] = {250u, 500u, 1000u};
        const float alphas[] = {0.5f, 0.75f, 0.85f};
        for (std::uint32_t p : pools)
            for (float a : alphas) {
                simeon::RouterConfig rc; // defaults
                std::snprintf(tagbuf, sizeof(tagbuf), "passB_pool%u_a%.2f", p, a);
                specs.push_back({tagbuf, rc, p, a});
            }
        // Pass C — Step 1f predictor enrichment. Hold the Pass A winners
        // (oov=0, idf=3, cascade nt=4 / max_idf=5, pool=500, alpha=0.75) and
        // sweep the new atire-route AND-gates: atire_min_terms (n_terms floor)
        // and atire_min_idf_floor (min_idf floor). The (0, 0) cell duplicates
        // the Pass A winner row and acts as a regression check.
        const std::uint32_t atire_nts[] = {0u, 6u, 10u, 14u};
        const float atire_mi_floors[] = {0.0f, 1.5f, 3.0f};
        for (std::uint32_t ant : atire_nts)
            for (float amif : atire_mi_floors) {
                simeon::RouterConfig rc;
                rc.oov_threshold = 0.0f;
                rc.high_idf_threshold = 3.0f;
                rc.cascade_min_terms = 4u;
                rc.cascade_max_idf = 5.0f;
                rc.atire_min_terms = ant;
                rc.atire_min_idf_floor = amif;
                std::snprintf(tagbuf, sizeof(tagbuf), "passC_ant%u_amif%.1f", ant, amif);
                specs.push_back({tagbuf, rc, 500u, 0.75f, false, 50u});
            }
        // Pass D — Step 1g.1 post-retrieval-lite predictors. Hold the Pass A
        // winners and Step 1f Pass C defaults; sweep the new Atire AND-gates
        // atire_max_pool_jaccard (route only when Atire-vs-SAB top-K pools
        // disagree enough) and atire_min_score_decay (route only when the
        // BM25 top is sharply peaked). use_post_retrieval=true triggers
        // QueryRouter::features_with_pool() in run_router_grid.
        const float atire_jacs[] = {0.5f, 0.7f, 0.9f, 1.0f};
        const float atire_decays[] = {0.0f, 0.3f, 0.6f};
        for (float jac : atire_jacs)
            for (float dec : atire_decays) {
                simeon::RouterConfig rc;
                rc.oov_threshold = 0.0f;
                rc.high_idf_threshold = 3.0f;
                rc.cascade_min_terms = 4u;
                rc.cascade_max_idf = 5.0f;
                rc.atire_max_pool_jaccard = jac;
                rc.atire_min_score_decay = dec;
                std::snprintf(tagbuf, sizeof(tagbuf), "passD_jac%.1f_dec%.1f", jac, dec);
                specs.push_back({tagbuf, rc, 500u, 0.75f, true, 50u});
            }
        // Pass E — Step 1k B2. Sum-SCQ (Zhao 2008) and simplified clarity
        // (Cronen-Townsend 2002) AND-gates on the Atire route. Both are
        // pre-retrieval (no pool needed), so use_post_retrieval=false.
        const float atire_scq_floors[] = {0.0f, 5.0f, 10.0f, 20.0f};
        const float atire_clarity_ceils[] = {std::numeric_limits<float>::infinity(), 3.0f, 5.0f,
                                             8.0f};
        for (float scq : atire_scq_floors)
            for (float clar : atire_clarity_ceils) {
                simeon::RouterConfig rc;
                rc.oov_threshold = 0.0f;
                rc.high_idf_threshold = 3.0f;
                rc.cascade_min_terms = 4u;
                rc.cascade_max_idf = 5.0f;
                rc.atire_min_scq = scq;
                rc.atire_max_clarity = clar;
                const double clar_d = std::isinf(clar) ? 99.0 : clar;
                std::snprintf(tagbuf, sizeof(tagbuf), "passE_scq%.0f_clar%.1f", scq, clar_d);
                specs.push_back({tagbuf, rc, 500u, 0.75f, false, 50u});
            }

        run_router_grid("router_grid_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                        bm25.build_us + sab_smooth.build_us,
                        std::span<const RouterSweepSpec>(specs));

        run_oracle_router("router_oracle_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                          bm25.build_us + sab_smooth.build_us, 500u, 0.75f);
    }

    if (g_router_per_query_fp)
        std::fclose(g_router_per_query_fp);
    return 0;
}
