// Reference embedding comparison bench.
//
// Loads a frozen IR fixture (corpus + queries + qrels + pre-computed reference
// embeddings) and evaluates simeon configurations against the reference model
// on the same workload. Emits one JSONL record per configuration.
//
// Fixture format documented in docs/reference_fixture.md.

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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
#include "simeon/corpus_adapter.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/fusion.hpp"
#include "simeon/glove_embeddings.hpp"
#include "simeon/matryoshka.hpp"
#include "simeon/minhash.hpp"
#include "simeon/persistent_homology.hpp"
#include "simeon/pmi.hpp"
#include "simeon/pq.hpp"
#include "simeon/prf.hpp"
#include "simeon/query_router.hpp"
#include "simeon/retrieval_strategy.hpp"
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

bool env_truthy(const char* name) {
    const char* raw = std::getenv(name);
    return raw && *raw && std::string_view(raw) != "0";
}

bool arguana_diagnostics_enabled() {
    return env_truthy("SIMEON_ARGUANA_DIAGNOSTICS");
}

// -----------------------------------------------------------------------------
// Research-only ArguAna corpus-structure diagnostics. These helpers intentionally
// live in the benchmark binary rather than the library: they encode BEIR ArguAna
// fixture structure used to locate the candidate-pool ceiling, not a general
// retrieval algorithm. Keep default benchmark output production-facing; enable
// these rows only with SIMEON_ARGUANA_DIAGNOSTICS=1.
// -----------------------------------------------------------------------------

struct BenchStringTokenSink final : simeon::NGramEmitter {
    std::vector<std::string>* out = nullptr;
    void on_token(std::string_view tok, float) override { out->emplace_back(tok); }
};

std::vector<std::string> bench_word_tokens(std::string_view text) {
    std::vector<std::string> out;
    BenchStringTokenSink sink{};
    sink.out = &out;
    simeon::tokenize(text, simeon::TokenizerConfig{0, 0, false, true}, sink);
    return out;
}

enum class BenchLanguageProfile : std::uint8_t { English, Spanish, French };

BenchLanguageProfile bench_language_profile_from_env() {
    const char* raw = std::getenv("SIMEON_LANGUAGE_PROFILE");
    if (!raw || !*raw)
        return BenchLanguageProfile::English;
    std::string v;
    for (const unsigned char ch : std::string_view(raw)) {
        if (!std::isspace(ch))
            v.push_back(static_cast<char>(std::tolower(ch)));
    }
    if (v == "es" || v == "spa" || v == "spanish")
        return BenchLanguageProfile::Spanish;
    if (v == "fr" || v == "fra" || v == "fre" || v == "french")
        return BenchLanguageProfile::French;
    return BenchLanguageProfile::English;
}

const std::unordered_set<std::string_view>& bench_stopwords(BenchLanguageProfile lang) {
    static const std::unordered_set<std::string_view> kEnglish = {
        "a",     "an",     "and",   "are", "as",  "at",   "be",   "by",   "for",   "from", "in",
        "into",  "is",     "it",    "of",  "on",  "or",   "that", "the",  "their", "this", "to",
        "was",   "were",   "with",  "we",  "our", "you",  "your", "have", "has",   "had",  "will",
        "would", "should", "could", "can", "do",  "does", "did",  "they", "them",
    };
    static const std::unordered_set<std::string_view> kSpanish = {
        "un",    "una",      "unos",    "unas",    "y",     "o",      "de",   "del",
        "la",    "las",      "el",      "los",     "en",    "por",    "para", "con",
        "sin",   "que",      "se",      "es",      "son",   "ser",    "fue",  "fueron",
        "como",  "al",       "lo",      "su",      "sus",   "este",   "esta", "estos",
        "estas", "nosotros", "nuestro", "nuestra", "puede", "pueden",
    };
    static const std::unordered_set<std::string_view> kFrench = {
        "un",  "une",  "des",   "et",    "ou",   "de",    "du",   "la",      "le",
        "les", "en",   "par",   "pour",  "avec", "sans",  "que",  "qui",     "se",
        "est", "sont", "etre",  "comme", "au",   "aux",   "son",  "sa",      "ses",
        "ce",  "cet",  "cette", "ces",   "nous", "notre", "peut", "peuvent",
    };
    switch (lang) {
        case BenchLanguageProfile::Spanish:
            return kSpanish;
        case BenchLanguageProfile::French:
            return kFrench;
        case BenchLanguageProfile::English:
        default:
            return kEnglish;
    }
}

bool bench_stopword(std::string_view token, BenchLanguageProfile lang) {
    return bench_stopwords(lang).contains(token);
}

std::unordered_set<std::string> bench_content_set(std::string_view text,
                                                  BenchLanguageProfile lang) {
    std::unordered_set<std::string> out;
    for (auto& tok : bench_word_tokens(text)) {
        if (tok.size() > 2 && !bench_stopword(tok, lang))
            out.insert(std::move(tok));
    }
    return out;
}

std::unordered_set<std::string> bench_content_set_first_words(std::string_view text,
                                                              BenchLanguageProfile lang,
                                                              std::uint32_t max_words) {
    std::unordered_set<std::string> out;
    auto words = bench_word_tokens(text);
    const std::uint32_t n =
        std::min<std::uint32_t>(max_words, static_cast<std::uint32_t>(words.size()));
    for (std::uint32_t i = 0; i < n; ++i) {
        auto& tok = words[i];
        if (tok.size() > 2 && !bench_stopword(tok, lang))
            out.insert(std::move(tok));
    }
    return out;
}

float jaccard_set(const std::unordered_set<std::string>& a,
                  const std::unordered_set<std::string>& b) {
    if (a.empty() && b.empty())
        return 0.0f;
    const auto* small = &a;
    const auto* large = &b;
    if (small->size() > large->size())
        std::swap(small, large);
    std::uint32_t inter = 0;
    for (const auto& x : *small) {
        if (large->contains(x))
            ++inter;
    }
    const std::uint32_t uni = static_cast<std::uint32_t>(a.size() + b.size() - inter);
    return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

std::string arguana_topic_stem(std::string_view id) {
    const std::size_t dash = id.rfind('-');
    if (dash == std::string_view::npos)
        return std::string(id);
    const std::string_view tail = id.substr(dash + 1);
    if (tail.size() >= 5 && (tail.substr(0, 3) == "pro" || tail.substr(0, 3) == "con") &&
        (tail.back() == 'a' || tail.back() == 'b')) {
        bool digits = true;
        for (std::size_t i = 3; i + 1 < tail.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(tail[i]))) {
                digits = false;
                break;
            }
        }
        if (digits)
            return std::string(id.substr(0, dash));
    }
    return std::string(id);
}

struct ArguanaPointKey {
    std::string topic;
    std::string_view stance;
    std::string_view point;
    bool valid = false;
};

ArguanaPointKey arguana_point_key(std::string_view id) {
    const std::size_t dash = id.rfind('-');
    if (dash == std::string_view::npos)
        return {};
    const std::string_view tail = id.substr(dash + 1);
    if (tail.size() < 5 || (tail.substr(0, 3) != "pro" && tail.substr(0, 3) != "con") ||
        (tail.back() != 'a' && tail.back() != 'b'))
        return {};
    for (std::size_t i = 3; i + 1 < tail.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(tail[i])))
            return {};
    }
    return ArguanaPointKey{std::string(id.substr(0, dash)), tail.substr(0, 3),
                           tail.substr(3, tail.size() - 4), true};
}

const std::unordered_set<std::string_view>& bench_cues(BenchLanguageProfile lang) {
    static const std::unordered_set<std::string_view> kEnglish = {
        "not",      "no",        "never",      "however", "but",          "although", "despite",
        "instead",  "rather",    "whereas",    "while",   "counterpoint", "fail",     "fails",
        "failed",   "unlikely",  "cannot",     "less",    "insufficient", "wrong",    "problem",
        "problems", "expensive", "harm",       "harmful", "risk",         "risks",    "impossible",
        "oppose",   "opposing",  "opposition",
    };
    static const std::unordered_set<std::string_view> kSpanish = {
        "no",        "nunca",      "jamas",     "sin embargo",  "pero",       "aunque",
        "a pesar",   "en cambio",  "mientras",  "contrapunto",  "fracasa",    "fallo",
        "fallan",    "improbable", "menos",     "insuficiente", "incorrecto", "problema",
        "problemas", "caro",       "danino",    "riesgo",       "riesgos",    "imposible",
        "opone",     "opuesto",    "oposicion",
    };
    static const std::unordered_set<std::string_view> kFrench = {
        "non",         "ne",         "jamais",      "cependant",  "mais",  "bien",       "malgre",
        "plutot",      "tandis",     "contrepoint", "echoue",     "echec", "improbable", "moins",
        "insuffisant", "faux",       "probleme",    "problemes",  "cher",  "nocif",      "risque",
        "risques",     "impossible", "oppose",      "opposition",
    };
    switch (lang) {
        case BenchLanguageProfile::Spanish:
            return kSpanish;
        case BenchLanguageProfile::French:
            return kFrench;
        case BenchLanguageProfile::English:
        default:
            return kEnglish;
    }
}

std::uint32_t cue_count(const std::unordered_set<std::string>& toks, BenchLanguageProfile lang) {
    const auto& cues = bench_cues(lang);
    std::uint32_t n = 0;
    for (const auto& tok : toks) {
        if (cues.contains(tok))
            ++n;
    }
    return n;
}

std::string normalize_ws_lower(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = true;
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!in_space) {
                out.push_back(' ');
                in_space = true;
            }
        } else {
            out.push_back(static_cast<char>(std::tolower(ch)));
            in_space = false;
        }
    }
    if (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

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

std::vector<std::string> build_lead_token_field(const Fixture& fx, std::uint32_t max_tokens) {
    std::vector<std::string> out;
    out.reserve(fx.doc_texts.size());
    std::vector<std::string> toks;
    for (const auto& doc : fx.doc_texts) {
        tokenize_words(doc, toks);
        std::string lead;
        const std::uint32_t n =
            std::min<std::uint32_t>(max_tokens, static_cast<std::uint32_t>(toks.size()));
        for (std::uint32_t i = 0; i < n; ++i) {
            if (!lead.empty())
                lead.push_back(' ');
            lead += toks[i];
        }
        out.push_back(std::move(lead));
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
    double precision_at_10;
    double recall_at_10;
    double recall_at_100;
    double mrr_at_10;
    std::size_t evaluated_queries; // queries with ≥1 relevant doc
};

// Per-query: scored[i] is (score, doc_idx). Higher score = better rank.
// `per_query_ndcg` (if non-null) is filled with one nDCG@10 value per query
// in fx order; queries without qrels get NaN (matches the n_eval skip).
Metrics
score_rankings_full(const std::vector<std::vector<std::pair<float, std::uint32_t>>>& rankings,
                    const Fixture& fx, std::vector<double>* per_query_ndcg = nullptr) {
    // Bucket qrels by query.
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    double ndcg10_sum = 0.0, p10_sum = 0.0, r10_sum = 0.0, r100_sum = 0.0, mrr10_sum = 0.0;
    std::size_t n_eval = 0;
    if (per_query_ndcg)
        per_query_ndcg->assign(rankings.size(), std::numeric_limits<double>::quiet_NaN());

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
        const double q_ndcg = idcg > 0.0 ? dcg / idcg : 0.0;
        ndcg10_sum += q_ndcg;
        if (per_query_ndcg)
            (*per_query_ndcg)[qi] = q_ndcg;
        p10_sum += static_cast<double>(hits10) / 10.0;

        const double denom10 = static_cast<double>(std::min<std::size_t>(10, qrel.size()));
        const double denom100 = static_cast<double>(std::min<std::size_t>(100, qrel.size()));
        r10_sum += static_cast<double>(hits10) / denom10;
        r100_sum += static_cast<double>(hits100) / denom100;
        mrr10_sum += first_rel_rank > 0.0 ? 1.0 / first_rel_rank : 0.0;
    }

    const double n = static_cast<double>(n_eval == 0 ? 1 : n_eval);
    return {ndcg10_sum / n, p10_sum / n, r10_sum / n, r100_sum / n, mrr10_sum / n, n_eval};
}

Metrics score_rankings(const std::vector<std::vector<std::pair<float, std::uint32_t>>>& rankings,
                       const Fixture& fx) {
    return score_rankings_full(rankings, fx, nullptr);
}

// If env var SIMEON_PER_QUERY_DUMP is set, write per-query nDCG to
// ${SIMEON_PER_QUERY_DUMP}.${config_name}.jsonl with one line per query:
// {"qi":N,"qid":"...","n_qrels":N,"ndcg":0.xxxx}
void dump_per_query_ndcg_if_env(const char* config_name, const Fixture& fx,
                                const std::vector<double>& per_query_ndcg) {
    const char* base = std::getenv("SIMEON_PER_QUERY_DUMP");
    if (!base || !*base)
        return;
    std::string path = std::string(base) + "." + config_name + ".jsonl";
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp)
        return;
    // Per-query qrel count for downstream filtering / interpretation.
    std::unordered_map<std::uint32_t, std::uint32_t> n_qrels_per_q;
    for (const auto& q : fx.qrels)
        ++n_qrels_per_q[q.q];
    for (std::uint32_t qi = 0; qi < per_query_ndcg.size(); ++qi) {
        const double v = per_query_ndcg[qi];
        if (!std::isfinite(v))
            continue;
        const std::uint32_t nq = n_qrels_per_q.count(qi) ? n_qrels_per_q[qi] : 0u;
        std::fprintf(fp, "{\"qi\":%u,\"qid\":\"%s\",\"n_qrels\":%u,\"ndcg\":%.6f}\n", qi,
                     qi < fx.query_ids.size() ? fx.query_ids[qi].c_str() : "", nq, v);
    }
    std::fclose(fp);
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

    std::printf(
        "{\"bench\":\"vs_reference\",\"config\":\"%s\",\"model\":\"%s\","
        "\"queries\":%zu,\"docs\":%zu,\"evaluated_queries\":%zu,"
        "\"code_bytes_per_doc\":%u,"
        "\"ndcg_at_10\":%.4f,\"precision_at_10\":%.4f,\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
        "\"mrr_at_10\":%.4f,"
        "\"encode_us_per_doc\":%.3f,\"query_us_per_q\":%.3f,"
        "\"encode_throughput_dps\":%.1f,\"query_throughput_qps\":%.1f,"
        "\"index_build_us\":%.1f,"
        "\"simd_tier\":\"%s\"}\n",
        config_name, fx.ref_model.c_str(), fx.query_ids.size(), fx.doc_ids.size(),
        m.evaluated_queries, code_bytes_per_doc, m.ndcg_at_10, m.precision_at_10, m.recall_at_10,
        m.recall_at_100, m.mrr_at_10, encode_us_per_doc, query_us_per_q, encode_dps, query_qps,
        t.index_build_us, simeon::simd_tier_name(simeon::active_simd_tier()));
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

// Plan 3 — RRF (reciprocal rank fusion) over multiple BM25 variant rankings.
// Each variant produces a per-query top-K via BM25. Rankings are RRF-fused
// to produce a pool diversified over length-normalization / weighting
// differences. Attacks Bottleneck 2 (BM25-pool R@100 ceiling) via
// first-pass candidate diversification instead of fragment-geometry rerank.
void run_bm25_variants_rrf(const char* name, const Fixture& fx,
                           std::span<const simeon::Bm25Index* const> variants, double build_us) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<std::vector<std::pair<std::uint32_t, float>>> per_variant_rank(variants.size());
    for (auto& r : per_variant_rank)
        r.resize(nd);
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::vector<simeon::Ranking> ins;
        ins.reserve(variants.size());
        for (std::size_t v = 0; v < variants.size(); ++v) {
            std::fill(s.begin(), s.end(), 0.0f);
            variants[v]->score(fx.query_texts[qi], s);
            for (std::uint32_t di = 0; di < nd; ++di)
                per_variant_rank[v][di] = {di, s[di]};
            ins.emplace_back(per_variant_rank[v]);
        }
        auto fused = simeon::rrf_fuse(ins, 60.0f);
        rankings[qi].reserve(fused.size());
        for (const auto& [did, sc] : fused)
            rankings[qi].emplace_back(sc, did);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
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

// Oracle upper bound for any reranker over the BM25 top-K candidate pool.
// For each query: take the BM25 top-K, score each pool member by its qrel
// value (graded relevance preserved), score everything else as -inf. The
// resulting nDCG@10 is the supremum of nDCG@10(reranker(BM25_top_K)) over
// all permutations, i.e. the achievable ceiling of any candidate-set-bound
// rerank strategy. Quantifies "Ceiling A" (the candidate-pool determinism
// bound) for the first time as a per-corpus per-fold number.
void run_oracle_pool(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                     double build_us, std::uint32_t pool_size = 100) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());

    // Bucket qrels by query index for O(1) lookup during scoring.
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, pool_size);
        rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi][di].second = di;
        // Score pool members by their qrel (0 if not in qrels). Outside-pool
        // docs keep -inf so they sort last.
        const auto rit = rel.find(qi);
        for (const auto& [did, _bm] : pool) {
            float g = 0.0f;
            if (rit != rel.end()) {
                const auto qit = rit->second.find(did);
                if (qit != rit->second.end())
                    g = static_cast<float>(qit->second);
            }
            rankings[qi][did].first = g;
        }
    }
    t.query_us = elapsed_us(t0);
    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env(name, fx, per_q);
    emit(name, fx, m, 0, t);
}

using ScoreFn = std::function<void(std::uint32_t, std::vector<float>&)>;

void run_generator_slice_oracle(const char* name, const Fixture& fx, double build_us,
                                std::uint32_t pool_size, ScoreFn score_fn) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());

    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(scores.begin(), scores.end(), 0.0f);
        score_fn(qi, scores);
        auto pool = simeon::top_k(scores, pool_size);
        rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi][di].second = di;
        const auto rit = rel.find(qi);
        for (const auto& [did, _score] : pool) {
            float g = 0.0f;
            if (rit != rel.end()) {
                const auto qit = rit->second.find(did);
                if (qit != rit->second.end())
                    g = static_cast<float>(qit->second);
            }
            rankings[qi][did].first = g;
        }
    }
    t.query_us = elapsed_us(t0);
    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env(name, fx, per_q);
    emit(name, fx, m, 0, t);
}

void run_union_generator_slice_oracle(const char* name, const Fixture& fx, double build_us,
                                      std::uint32_t pool_size,
                                      std::span<const ScoreFn> generators) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());

    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    std::vector<std::uint8_t> in_pool(nd, 0u);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(in_pool.begin(), in_pool.end(), 0u);
        for (const auto& gen : generators) {
            std::fill(scores.begin(), scores.end(), 0.0f);
            gen(qi, scores);
            for (const auto& [did, _score] : simeon::top_k(scores, pool_size))
                in_pool[did] = 1u;
        }
        rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi][di].second = di;
        const auto rit = rel.find(qi);
        for (std::uint32_t did = 0; did < nd; ++did) {
            if (!in_pool[did])
                continue;
            float g = 0.0f;
            if (rit != rel.end()) {
                const auto qit = rit->second.find(did);
                if (qit != rit->second.end())
                    g = static_cast<float>(qit->second);
            }
            rankings[qi][did].first = g;
        }
    }
    t.query_us = elapsed_us(t0);
    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env(name, fx, per_q);
    emit(name, fx, m, 0, t);
}

void run_generator_observed(const char* name, const Fixture& fx, double build_us,
                            ScoreFn generator) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(scores.begin(), scores.end(), 0.0f);
        generator(qi, scores);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(scores[di], di);
    }
    t.query_us = elapsed_us(t0);
    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env(name, fx, per_q);
    emit(name, fx, m, 0, t);
}

void run_union_generator_weighted_z(const char* name, const Fixture& fx, double build_us,
                                    std::span<const ScoreFn> generators,
                                    std::span<const float> weights) {
    if (generators.size() != weights.size())
        throw std::runtime_error("weighted_z generator/weight size mismatch");
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    std::vector<float> fused(nd, 0.0f);

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(fused.begin(), fused.end(), 0.0f);
        for (std::size_t gi = 0; gi < generators.size(); ++gi) {
            std::fill(scores.begin(), scores.end(), 0.0f);
            generators[gi](qi, scores);
            zscore_inplace(scores);
            for (std::uint32_t di = 0; di < nd; ++di)
                fused[di] += weights[gi] * scores[di];
        }
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(fused[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

float topk_jaccard_from_scores(const std::vector<float>& a, const std::vector<float>& b,
                               std::uint32_t k) {
    auto ta = simeon::top_k(a, std::min<std::uint32_t>(k, static_cast<std::uint32_t>(a.size())));
    auto tb = simeon::top_k(b, std::min<std::uint32_t>(k, static_cast<std::uint32_t>(b.size())));
    if (ta.empty() && tb.empty())
        return 1.0f;
    std::unordered_set<std::uint32_t> sa;
    sa.reserve(ta.size() * 2);
    for (const auto& [did, score] : ta)
        sa.insert(did);
    std::uint32_t inter = 0;
    for (const auto& [did, score] : tb) {
        (void)score;
        if (sa.find(did) != sa.end())
            ++inter;
    }
    const std::uint32_t uni = static_cast<std::uint32_t>(ta.size() + tb.size()) - inter;
    return uni == 0 ? 0.0f : static_cast<float>(inter) / static_cast<float>(uni);
}

constexpr float topk_score_decay_from_scores(const std::vector<float>& s, std::uint32_t k) {
    auto t = simeon::top_k(s, std::min<std::uint32_t>(k, static_cast<std::uint32_t>(s.size())));
    if (t.size() < 2)
        return 0.0f;
    const float top = t.front().second;
    const float kth = t.back().second;
    const float denom = std::max(1e-6f, std::fabs(top));
    return (top - kth) / denom;
}

float pearson_over_union_pool(const std::vector<float>& a, const std::vector<float>& b,
                              std::uint32_t k) {
    auto ta = simeon::top_k(a, std::min<std::uint32_t>(k, static_cast<std::uint32_t>(a.size())));
    auto tb = simeon::top_k(b, std::min<std::uint32_t>(k, static_cast<std::uint32_t>(b.size())));
    std::unordered_set<std::uint32_t> pool;
    pool.reserve((ta.size() + tb.size()) * 2);
    for (const auto& [did, score] : ta)
        pool.insert(did);
    for (const auto& [did, score] : tb)
        pool.insert(did);
    if (pool.size() < 2)
        return 0.0f;
    double ma = 0.0, mb = 0.0;
    for (std::uint32_t did : pool) {
        ma += a[did];
        mb += b[did];
    }
    ma /= static_cast<double>(pool.size());
    mb /= static_cast<double>(pool.size());
    double num = 0.0, va = 0.0, vb = 0.0;
    for (std::uint32_t did : pool) {
        const double da = static_cast<double>(a[did]) - ma;
        const double db = static_cast<double>(b[did]) - mb;
        num += da * db;
        va += da * da;
        vb += db * db;
    }
    const double den = std::sqrt(va * vb);
    return den <= 1e-12 ? 0.0f : static_cast<float>(num / den);
}

float topk_softmax_entropy_from_scores(const std::vector<float>& s, std::uint32_t k) {
    auto t = simeon::top_k(s, std::min<std::uint32_t>(k, static_cast<std::uint32_t>(s.size())));
    if (t.size() < 2)
        return 0.0f;
    const float mx = t.front().second;
    double sum = 0.0;
    std::vector<double> weights;
    weights.reserve(t.size());
    for (const auto& [did, score] : t) {
        (void)did;
        const double w = std::exp(static_cast<double>(score - mx));
        weights.push_back(w);
        sum += w;
    }
    if (sum <= 0.0)
        return 0.0f;
    double h = 0.0;
    for (double w : weights) {
        const double p = w / sum;
        if (p > 0.0)
            h -= p * std::log(p);
    }
    return static_cast<float>(h / std::log(static_cast<double>(weights.size())));
}

constexpr float top2_margin_from_scores(const std::vector<float>& s) {
    auto t = simeon::top_k(s, std::min<std::uint32_t>(2, static_cast<std::uint32_t>(s.size())));
    if (t.size() < 2)
        return 0.0f;
    const float denom = std::max(1e-6f, std::fabs(t[0].second));
    return (t[0].second - t[1].second) / denom;
}

constexpr float top1_same_from_scores(const std::vector<float>& a, const std::vector<float>& b) {
    auto ta = simeon::top_k(a, std::min<std::uint32_t>(1, static_cast<std::uint32_t>(a.size())));
    auto tb = simeon::top_k(b, std::min<std::uint32_t>(1, static_cast<std::uint32_t>(b.size())));
    if (ta.empty() || tb.empty())
        return 0.0f;
    return ta.front().first == tb.front().first ? 1.0f : 0.0f;
}

void zscore_pool_values(std::vector<float>& values,
                        std::span<const std::pair<std::uint32_t, float>> pool) {
    if (pool.empty())
        return;
    double mean = 0.0;
    for (const auto& [did, score] : pool)
        mean += values[did];
    mean /= static_cast<double>(pool.size());
    double var = 0.0;
    for (const auto& [did, score] : pool) {
        (void)score;
        const double d = static_cast<double>(values[did]) - mean;
        var += d * d;
    }
    const double sd = std::sqrt(var / static_cast<double>(pool.size()));
    if (sd <= 1e-9) {
        for (const auto& [did, score] : pool)
            values[did] = 0.0f;
        return;
    }
    for (const auto& [did, score] : pool)
        values[did] = static_cast<float>((static_cast<double>(values[did]) - mean) / sd);
}

void run_pool_lexical_evidence_rerank(const char* name, const Fixture& fx,
                                      const simeon::Bm25Index& idx, double build_us,
                                      float overlap_weight, float phrase_weight,
                                      std::uint32_t pool_k) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<SparseSignature> doc_sigs;
    doc_sigs.reserve(nd);
    std::vector<std::vector<std::uint64_t>> doc_hashes(nd);
    auto prep0 = Clock::now();
    for (std::uint32_t di = 0; di < nd; ++di) {
        doc_sigs.push_back(build_sparse_signature(idx, fx.doc_texts[di], 64));
        build_doc_hash_tokens(idx, fx.doc_texts[di], doc_hashes[di]);
    }
    t.index_build_us += elapsed_us(prep0);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25(nd, 0.0f), bm25_z(nd, 0.0f), overlap(nd, 0.0f), phrase(nd, 0.0f),
        fused(nd, -std::numeric_limits<float>::infinity());

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(bm25.begin(), bm25.end(), 0.0f);
        idx.score(fx.query_texts[qi], bm25);
        auto pool = simeon::top_k(bm25, std::min<std::uint32_t>(pool_k, nd));
        bm25_z = bm25;
        zscore_inplace(bm25_z);
        std::fill(overlap.begin(), overlap.end(), 0.0f);
        std::fill(phrase.begin(), phrase.end(), 0.0f);
        const auto qsig = build_sparse_signature(idx, fx.query_texts[qi], 48);
        const auto phrases = build_query_phrase_nodes(idx, fx.query_texts[qi]);
        for (const auto& [did, score] : pool) {
            (void)score;
            overlap[did] = weighted_containment(qsig, doc_sigs[did]);
            if (!phrases.empty()) {
                auto pw = compute_doc_phrase_weights(doc_hashes[did], phrases, 8);
                phrase[did] = std::accumulate(pw.begin(), pw.end(), 0.0f);
            }
        }
        zscore_pool_values(overlap, std::span<const std::pair<std::uint32_t, float>>(pool));
        zscore_pool_values(phrase, std::span<const std::pair<std::uint32_t, float>>(pool));
        std::fill(fused.begin(), fused.end(), -std::numeric_limits<float>::infinity());
        for (const auto& [did, score] : pool) {
            (void)score;
            fused[did] = bm25_z[did] + overlap_weight * overlap[did] + phrase_weight * phrase[did];
        }
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(fused[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

// =========================================================================
// Consolidated helpers shared across research ScoreFns (Phases LII–LXIX).
// A single copy of each algorithm, parameterised, testable, no duplication.
// =========================================================================

// Greedy MMR selection: from `candidates` (top-K BM25), select `sel_k` docs
// maximising BM25 × simeon_sim_q − β × max_sim_to_already_selected.
// Returns (selected_doc_ids, selected_bm25_scores).
struct MmrResult {
    std::vector<std::uint32_t> ids;
    std::vector<float> bm25_scores;
};
MmrResult mmr_select_docs(const std::vector<std::pair<std::uint32_t, float>>& candidates,
                          const std::vector<float>& cand_sim, // simeon sim per candidate
                          const std::vector<float>& simeon_dembs, std::uint32_t sdim,
                          std::uint32_t sel_k, float beta) {
    const auto n = candidates.size();
    std::vector<bool> used(n, false);
    std::vector<std::uint32_t> sel_dids, sel_ids;
    std::vector<float> sel_bm25;
    for (std::uint32_t sel = 0; sel < sel_k && sel < n; ++sel) {
        float best_mmr = -1e9f;
        std::size_t best_i = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i])
                continue;
            float reward = candidates[i].second * std::max(0.0f, cand_sim[i]);
            float penalty = 0.0f;
            for (std::size_t j = 0; j < sel_dids.size(); ++j) {
                float sim = dot(simeon_dembs.data() + candidates[i].first * sdim,
                                simeon_dembs.data() + sel_dids[j] * sdim, sdim);
                if (sim > penalty)
                    penalty = sim;
            }
            float mmr = reward - beta * penalty;
            if (mmr > best_mmr) {
                best_mmr = mmr;
                best_i = i;
            }
        }
        used[best_i] = true;
        sel_dids.push_back(candidates[best_i].first);
        sel_ids.push_back(candidates[best_i].first);
        sel_bm25.push_back(candidates[best_i].second);
    }
    return {std::move(sel_ids), std::move(sel_bm25)};
}

// Tokenize query into words, hash each via idx.hash_term().
// Returns the term hashes in arbitrary order.
std::vector<std::uint64_t> hash_query_words(std::string_view query, const simeon::Bm25Index& idx) {
    struct Sink : simeon::NGramEmitter {
        std::vector<std::uint64_t>* hashes = nullptr;
        const simeon::Bm25Index* idx_ptr = nullptr;
        void on_token(std::string_view tok, float) override {
            if (hashes && idx_ptr)
                hashes->push_back(idx_ptr->hash_term(tok));
        }
    };
    std::vector<std::uint64_t> out;
    simeon::TokenizerConfig tcfg;
    tcfg.emit_word = true;
    tcfg.emit_char = false;
    Sink sink;
    sink.hashes = &out;
    sink.idx_ptr = &idx;
    simeon::tokenize(query, tcfg, sink);
    return out;
}

// Blend original query hashes with RM expansion terms (α=0.5), score via
// weighted-hash path.  out_scores must be pre-zeroed and sized to doc_count().
void score_rm_blended(const simeon::Bm25Index& idx, std::span<const std::uint64_t> q_hashes,
                      std::span<const std::pair<std::uint64_t, float>> rm_terms,
                      std::span<float> out_scores) {
    constexpr float alpha = 0.5f;
    std::unordered_map<std::uint64_t, float> combined;
    combined.reserve(q_hashes.size() + rm_terms.size());
    if (!q_hashes.empty()) [[likely]] {
        const float q_total = static_cast<float>(q_hashes.size());
        for (std::uint64_t h : q_hashes)
            combined[h] += (1.0f - alpha) * (1.0f / q_total);
    }
    for (const auto& [h, w] : rm_terms)
        combined[h] += alpha * w;

    std::vector<std::pair<std::uint64_t, float>> weighted;
    weighted.reserve(combined.size());
    for (const auto& [h, w] : combined)
        weighted.emplace_back(h, w);
    std::sort(weighted.begin(), weighted.end());
    idx.score_weighted_hashes(weighted, out_scores);
}

// Compute simeon-weighted + MMR-diverse doc weights from BM25 top-candidates.
// Returns (selected_doc_ids, doc_weights).  Empty if no feedback docs found.
struct DocWeightResult {
    std::vector<std::uint32_t> ids;
    std::vector<float> weights;
};
DocWeightResult compute_diverse_feedback_weights(const simeon::Bm25Index& idx,
                                                 const std::vector<float>& first_pass_scores,
                                                 const std::vector<float>& simeon_dembs,
                                                 const std::vector<float>& simeon_qembs,
                                                 std::uint32_t qi, std::uint32_t sdim,
                                                 std::uint32_t candidate_k, std::uint32_t select_k,
                                                 float mmr_beta) {
    auto tk = simeon::top_k(first_pass_scores, candidate_k);
    if (tk.empty())
        return {};
    const float* qemb = simeon_qembs.data() + qi * sdim;
    std::vector<float> cand_sim(tk.size(), 0.0f);
    for (std::size_t i = 0; i < tk.size(); ++i)
        cand_sim[i] = dot(qemb, simeon_dembs.data() + tk[i].first * sdim, sdim);

    auto mmr = mmr_select_docs(tk, cand_sim, simeon_dembs, sdim, select_k, mmr_beta);
    if (mmr.ids.empty())
        return {};

    float bm25_sum = 0.0f;
    for (float s : mmr.bm25_scores)
        bm25_sum += s;
    std::vector<float> doc_weights(mmr.ids.size(), 0.0f);
    float w_sum = 0.0f;
    for (std::size_t i = 0; i < mmr.ids.size(); ++i) {
        float sim_q = dot(qemb, simeon_dembs.data() + mmr.ids[i] * sdim, sdim);
        doc_weights[i] = (mmr.bm25_scores[i] / bm25_sum) * std::max(0.0f, sim_q);
        w_sum += doc_weights[i];
    }
    if (w_sum > 0.0f)
        for (float& w : doc_weights)
            w /= w_sum;
    else
        return {};

    return {std::move(mmr.ids), std::move(doc_weights)};
}

// -------------------------------------------------------------------------
// End consolidated helpers
// -------------------------------------------------------------------------

void run_shape_risk_z_fusion(const char* name, const Fixture& fx, double build_us, ScoreFn gen_bm25,
                             ScoreFn gen_bm25f, ScoreFn gen_rm3, bool hard_gate) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> raw_bm25(nd, 0.0f), raw_bm25f(nd, 0.0f), raw_rm3(nd, 0.0f);
    std::vector<float> z_bm25(nd, 0.0f), z_bm25f(nd, 0.0f), z_rm3(nd, 0.0f), z_equal(nd, 0.0f),
        fused(nd, 0.0f);

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(raw_bm25.begin(), raw_bm25.end(), 0.0f);
        std::fill(raw_bm25f.begin(), raw_bm25f.end(), 0.0f);
        std::fill(raw_rm3.begin(), raw_rm3.end(), 0.0f);
        gen_bm25(qi, raw_bm25);
        gen_bm25f(qi, raw_bm25f);
        gen_rm3(qi, raw_rm3);

        z_bm25 = raw_bm25;
        z_bm25f = raw_bm25f;
        z_rm3 = raw_rm3;
        zscore_inplace(z_bm25);
        zscore_inplace(z_bm25f);
        zscore_inplace(z_rm3);
        for (std::uint32_t di = 0; di < nd; ++di)
            z_equal[di] = z_bm25[di] + z_bm25f[di] + z_rm3[di];

        const float bm25_margin2 = top2_margin_from_scores(raw_bm25);
        const float bm25_rm3_jaccard100 = topk_jaccard_from_scores(raw_bm25, raw_rm3, 100);
        float lambda = 0.0f;
        if (hard_gate) {
            lambda = (bm25_margin2 >= 0.01939f && bm25_rm3_jaccard100 <= 0.7544f) ? 1.0f : 0.0f;
        } else {
            const float confidence = std::clamp(bm25_margin2 / 0.01939f, 0.0f, 1.0f);
            const float drift = std::clamp((0.7544f - bm25_rm3_jaccard100) / 0.2544f, 0.0f, 1.0f);
            lambda = confidence * drift;
        }
        for (std::uint32_t di = 0; di < nd; ++di)
            fused[di] = (1.0f - lambda) * z_bm25[di] + lambda * z_equal[di];

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(fused[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void dump_structural_winner_features_if_env(const Fixture& fx, const simeon::Bm25Index& feature_idx,
                                            ScoreFn gen_bm25, ScoreFn gen_lead64,
                                            ScoreFn gen_shape_risk) {
    const char* path = std::getenv("SIMEON_STRUCT_WINNER_DUMP");
    if (!path || !*path)
        return;
    std::FILE* fp = std::fopen(path, "w");
    if (!fp)
        return;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    simeon::QueryRouter router(feature_idx);
    std::array<std::vector<float>, 3> scores{
        std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f)};
    constexpr std::array<const char*, 3> names{"bm25", "lead64", "shape_risk"};

    auto q_ndcg = [&](std::uint32_t qi, const std::vector<float>& s) -> double {
        std::vector<std::pair<float, std::uint32_t>> ranking;
        ranking.reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            ranking.emplace_back(s[di], di);
        std::vector<std::vector<std::pair<float, std::uint32_t>>> one_rankings(1);
        one_rankings[0] = std::move(ranking);
        Fixture one_fx;
        one_fx.query_ids.push_back(fx.query_ids[qi]);
        one_fx.query_texts.push_back(fx.query_texts[qi]);
        one_fx.doc_ids = fx.doc_ids;
        one_fx.doc_texts = fx.doc_texts;
        one_fx.ref_dim = fx.ref_dim;
        one_fx.ref_model = fx.ref_model;
        const auto rit = rel.find(qi);
        if (rit != rel.end()) {
            for (const auto& [d, r] : rit->second)
                one_fx.qrels.push_back(Fixture::Qrel{0, d, r});
        }
        return score_rankings(one_rankings, one_fx).ndcg_at_10;
    };

    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (auto& s : scores)
            std::fill(s.begin(), s.end(), 0.0f);
        gen_bm25(qi, scores[0]);
        gen_lead64(qi, scores[1]);
        gen_shape_risk(qi, scores[2]);
        std::array<double, 3> ndcgs{};
        std::uint32_t best = 0;
        for (std::uint32_t i = 0; i < ndcgs.size(); ++i) {
            ndcgs[i] = q_ndcg(qi, scores[i]);
            if (ndcgs[i] > ndcgs[best])
                best = i;
        }
        const float bm25_lead_jaccard10 = topk_jaccard_from_scores(scores[0], scores[1], 10);
        const float bm25_lead_jaccard50 = topk_jaccard_from_scores(scores[0], scores[1], 50);
        const float bm25_lead_jaccard100 = topk_jaccard_from_scores(scores[0], scores[1], 100);
        const float bm25_lead_corr100 = pearson_over_union_pool(scores[0], scores[1], 100);
        const float bm25_lead_top1_same = top1_same_from_scores(scores[0], scores[1]);
        const float bm25_margin2 = top2_margin_from_scores(scores[0]);
        const float lead_margin2 = top2_margin_from_scores(scores[1]);
        const float lead_minus_bm25_margin2 = lead_margin2 - bm25_margin2;
        const float bm25_entropy10 = topk_softmax_entropy_from_scores(scores[0], 10);
        const float lead_entropy10 = topk_softmax_entropy_from_scores(scores[1], 10);
        const float lead_minus_bm25_entropy10 = lead_entropy10 - bm25_entropy10;
        const auto f = router.features(fx.query_texts[qi]);
        std::fprintf(fp,
                     "{\"qi\":%u,\"qid\":\"%s\",\"winner\":\"%s\","
                     "\"ndcg_bm25\":%.6f,\"ndcg_lead64\":%.6f,\"ndcg_shape_risk\":%.6f,"
                     "\"bm25_lead_jaccard10\":%.6f,\"bm25_lead_jaccard50\":%.6f,"
                     "\"bm25_lead_jaccard100\":%.6f,\"bm25_lead_corr100\":%.6f,"
                     "\"bm25_lead_top1_same\":%.6f,\"bm25_margin2\":%.6f,"
                     "\"lead_margin2\":%.6f,\"lead_minus_bm25_margin2\":%.6f,"
                     "\"bm25_entropy10\":%.6f,\"lead_entropy10\":%.6f,"
                     "\"lead_minus_bm25_entropy10\":%.6f,\"oov_rate\":%.6f,"
                     "\"avg_idf\":%.6f,\"max_idf\":%.6f,\"min_idf\":%.6f,"
                     "\"idf_stddev\":%.6f,\"n_terms\":%u,\"avg_term_chars\":%.6f,"
                     "\"scq_sum\":%.6f,\"simplified_clarity\":%.6f}\n",
                     qi, fx.query_ids[qi].c_str(), names[best], ndcgs[0], ndcgs[1], ndcgs[2],
                     bm25_lead_jaccard10, bm25_lead_jaccard50, bm25_lead_jaccard100,
                     bm25_lead_corr100, bm25_lead_top1_same, bm25_margin2, lead_margin2,
                     lead_minus_bm25_margin2, bm25_entropy10, lead_entropy10,
                     lead_minus_bm25_entropy10, f.oov_rate, f.avg_idf, f.max_idf, f.min_idf,
                     f.idf_stddev, f.n_terms, f.avg_term_chars, f.scq_sum, f.simplified_clarity);
    }
    std::fclose(fp);
}

void dump_generator_winner_features_if_env(const Fixture& fx, const simeon::Bm25Index& feature_idx,
                                           ScoreFn gen_bm25, ScoreFn gen_bm25f, ScoreFn gen_rm3,
                                           ScoreFn gen_z_equal) {
    const char* path = std::getenv("SIMEON_GENERATOR_WINNER_DUMP");
    if (!path || !*path)
        return;
    std::FILE* fp = std::fopen(path, "w");
    if (!fp)
        return;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    simeon::QueryRouter router(feature_idx);
    const std::array<const simeon::Bm25Index*, 1> pool_span{&feature_idx};
    std::array<std::vector<float>, 4> scores{
        std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f),
        std::vector<float>(nd, 0.0f)};
    constexpr std::array<const char*, 4> names{"bm25", "bm25f", "rm3", "z_equal"};

    auto q_ndcg = [&](std::uint32_t qi, const std::vector<float>& s) -> double {
        std::vector<std::pair<float, std::uint32_t>> ranking;
        ranking.reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            ranking.emplace_back(s[di], di);
        std::vector<std::vector<std::pair<float, std::uint32_t>>> one_rankings(1);
        one_rankings[0] = std::move(ranking);
        Fixture one_fx;
        one_fx.query_ids.push_back(fx.query_ids[qi]);
        one_fx.query_texts.push_back(fx.query_texts[qi]);
        one_fx.doc_ids = fx.doc_ids;
        one_fx.doc_texts = fx.doc_texts;
        one_fx.ref_dim = fx.ref_dim;
        one_fx.ref_model = fx.ref_model;
        const auto rit = rel.find(qi);
        if (rit != rel.end()) {
            for (const auto& [d, r] : rit->second)
                one_fx.qrels.push_back(Fixture::Qrel{0, d, r});
        }
        return score_rankings(one_rankings, one_fx).ndcg_at_10;
    };

    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (auto& s : scores)
            std::fill(s.begin(), s.end(), 0.0f);
        gen_bm25(qi, scores[0]);
        gen_bm25f(qi, scores[1]);
        gen_rm3(qi, scores[2]);
        gen_z_equal(qi, scores[3]);
        std::array<double, 4> ndcgs{};
        std::uint32_t best = 0;
        for (std::uint32_t i = 0; i < ndcgs.size(); ++i) {
            ndcgs[i] = q_ndcg(qi, scores[i]);
            if (ndcgs[i] > ndcgs[best])
                best = i;
        }
        const float bm25_rm3_jaccard10 = topk_jaccard_from_scores(scores[0], scores[2], 10);
        const float bm25_rm3_jaccard50 = topk_jaccard_from_scores(scores[0], scores[2], 50);
        const float bm25_rm3_jaccard100 = topk_jaccard_from_scores(scores[0], scores[2], 100);
        const float bm25_bm25f_jaccard10 = topk_jaccard_from_scores(scores[0], scores[1], 10);
        const float bm25_bm25f_jaccard50 = topk_jaccard_from_scores(scores[0], scores[1], 50);
        const float bm25_rm3_corr100 = pearson_over_union_pool(scores[0], scores[2], 100);
        const float bm25_rm3_top1_same = top1_same_from_scores(scores[0], scores[2]);
        const float bm25_decay10 = topk_score_decay_from_scores(scores[0], 10);
        const float rm3_decay10 = topk_score_decay_from_scores(scores[2], 10);
        const float rm3_minus_bm25_decay10 = rm3_decay10 - bm25_decay10;
        const float bm25_entropy10 = topk_softmax_entropy_from_scores(scores[0], 10);
        const float rm3_entropy10 = topk_softmax_entropy_from_scores(scores[2], 10);
        const float z_equal_entropy10 = topk_softmax_entropy_from_scores(scores[3], 10);
        const float rm3_minus_bm25_entropy10 = rm3_entropy10 - bm25_entropy10;
        const float bm25_margin2 = top2_margin_from_scores(scores[0]);
        const float rm3_margin2 = top2_margin_from_scores(scores[2]);
        const float z_equal_margin2 = top2_margin_from_scores(scores[3]);
        const float rm3_minus_bm25_margin2 = rm3_margin2 - bm25_margin2;
        const auto f = router.features_with_pool(
            fx.query_texts[qi],
            std::span<const simeon::Bm25Index* const>(pool_span.data(), pool_span.size()), 50);
        std::fprintf(fp,
                     "{\"qi\":%u,\"qid\":\"%s\",\"winner\":\"%s\","
                     "\"ndcg_bm25\":%.6f,\"ndcg_bm25f\":%.6f,"
                     "\"ndcg_rm3\":%.6f,\"ndcg_z_equal\":%.6f,"
                     "\"bm25_rm3_jaccard10\":%.6f,\"bm25_rm3_jaccard50\":%.6f,"
                     "\"bm25_rm3_jaccard100\":%.6f,\"bm25_bm25f_jaccard10\":%.6f,"
                     "\"bm25_bm25f_jaccard50\":%.6f,\"bm25_rm3_corr100\":%.6f,"
                     "\"bm25_rm3_top1_same\":%.6f,\"bm25_decay10\":%.6f,"
                     "\"rm3_decay10\":%.6f,\"rm3_minus_bm25_decay10\":%.6f,"
                     "\"bm25_entropy10\":%.6f,\"rm3_entropy10\":%.6f,"
                     "\"z_equal_entropy10\":%.6f,\"rm3_minus_bm25_entropy10\":%.6f,"
                     "\"bm25_margin2\":%.6f,\"rm3_margin2\":%.6f,"
                     "\"z_equal_margin2\":%.6f,\"rm3_minus_bm25_margin2\":%.6f,"
                     "\"oov_rate\":%.6f,\"avg_idf\":%.6f,\"max_idf\":%.6f,"
                     "\"min_idf\":%.6f,\"idf_stddev\":%.6f,\"n_terms\":%u,"
                     "\"avg_term_chars\":%.6f,\"score_decay_rate\":%.6f,"
                     "\"score_normalized_var\":%.6f,\"top_k_score_entropy\":%.6f,"
                     "\"nqc\":%.6f,\"wig_full\":%.6f,\"scq_sum\":%.6f,"
                     "\"simplified_clarity\":%.6f}\n",
                     qi, fx.query_ids[qi].c_str(), names[best], ndcgs[0], ndcgs[1], ndcgs[2],
                     ndcgs[3], bm25_rm3_jaccard10, bm25_rm3_jaccard50, bm25_rm3_jaccard100,
                     bm25_bm25f_jaccard10, bm25_bm25f_jaccard50, bm25_rm3_corr100,
                     bm25_rm3_top1_same, bm25_decay10, rm3_decay10, rm3_minus_bm25_decay10,
                     bm25_entropy10, rm3_entropy10, z_equal_entropy10, rm3_minus_bm25_entropy10,
                     bm25_margin2, rm3_margin2, z_equal_margin2, rm3_minus_bm25_margin2, f.oov_rate,
                     f.avg_idf, f.max_idf, f.min_idf, f.idf_stddev, f.n_terms, f.avg_term_chars,
                     f.score_decay_rate, f.score_normalized_var, f.top_k_score_entropy, f.nqc,
                     f.wig_full, f.scq_sum, f.simplified_clarity);
    }
    std::fclose(fp);
}

// Phase XLV: 4-generator winner dump including simeon embedding features.
void dump_4gen_winner_features_if_env(const Fixture& fx, const simeon::Bm25Index& feature_idx,
                                      ScoreFn gen_bm25, ScoreFn gen_bm25f, ScoreFn gen_rm3,
                                      ScoreFn gen_simeon, ScoreFn gen_4way_z) {
    const char* path = std::getenv("SIMEON_4GEN_WINNER_DUMP");
    if (!path || !*path)
        return;
    std::FILE* fp = std::fopen(path, "w");
    if (!fp)
        return;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    simeon::QueryRouter router(feature_idx);
    std::array<std::vector<float>, 5> scores{
        std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f),
        std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f)};
    constexpr std::array<const char*, 5> names{"bm25", "bm25f", "rm3", "simeon", "4way_z"};

    auto q_ndcg = [&](std::uint32_t qi, const std::vector<float>& s) -> double {
        std::vector<std::pair<float, std::uint32_t>> ranking;
        ranking.reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            ranking.emplace_back(s[di], di);
        std::vector<std::vector<std::pair<float, std::uint32_t>>> one_rankings(1);
        one_rankings[0] = std::move(ranking);
        Fixture one_fx;
        one_fx.query_ids.push_back(fx.query_ids[qi]);
        one_fx.query_texts.push_back(fx.query_texts[qi]);
        one_fx.doc_ids = fx.doc_ids;
        one_fx.doc_texts = fx.doc_texts;
        one_fx.ref_dim = fx.ref_dim;
        one_fx.ref_model = fx.ref_model;
        const auto rit = rel.find(qi);
        if (rit != rel.end()) {
            for (const auto& [d, r] : rit->second)
                one_fx.qrels.push_back(Fixture::Qrel{0, d, r});
        }
        return score_rankings(one_rankings, one_fx).ndcg_at_10;
    };

    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (auto& s : scores)
            std::fill(s.begin(), s.end(), 0.0f);
        gen_bm25(qi, scores[0]);
        gen_bm25f(qi, scores[1]);
        gen_rm3(qi, scores[2]);
        gen_simeon(qi, scores[3]);
        gen_4way_z(qi, scores[4]);
        std::array<double, 5> ndcgs{};
        std::uint32_t best = 0;
        for (std::uint32_t i = 0; i < ndcgs.size(); ++i) {
            ndcgs[i] = q_ndcg(qi, scores[i]);
            if (ndcgs[i] > ndcgs[best])
                best = i;
        }
        const float bm25_rm3_jaccard50 = topk_jaccard_from_scores(scores[0], scores[2], 50);
        const float bm25_rm3_corr100 = pearson_over_union_pool(scores[0], scores[2], 100);
        const float bm25_simeon_jaccard50 = topk_jaccard_from_scores(scores[0], scores[3], 50);
        const float bm25_simeon_corr100 = pearson_over_union_pool(scores[0], scores[3], 100);
        const float bm25_simeon_top1_same = top1_same_from_scores(scores[0], scores[3]);
        const float bm25_decay10 = topk_score_decay_from_scores(scores[0], 10);
        const float rm3_decay10 = topk_score_decay_from_scores(scores[2], 10);
        const float simeon_decay10 = topk_score_decay_from_scores(scores[3], 10);
        const float bm25_entropy10 = topk_softmax_entropy_from_scores(scores[0], 10);
        const float rm3_entropy10 = topk_softmax_entropy_from_scores(scores[2], 10);
        const float simeon_entropy10 = topk_softmax_entropy_from_scores(scores[3], 10);
        const float bm25_margin2 = top2_margin_from_scores(scores[0]);
        const float rm3_margin2 = top2_margin_from_scores(scores[2]);
        const float simeon_margin2 = top2_margin_from_scores(scores[3]);
        const float simeon_minus_bm25_margin2 = simeon_margin2 - bm25_margin2;
        const auto f = router.features(fx.query_texts[qi]);
        std::fprintf(fp,
                     "{\"qi\":%u,\"qid\":\"%s\",\"winner\":\"%s\","
                     "\"ndcg_bm25\":%.6f,\"ndcg_bm25f\":%.6f,"
                     "\"ndcg_rm3\":%.6f,\"ndcg_simeon\":%.6f,\"ndcg_4way_z\":%.6f,"
                     "\"bm25_rm3_jaccard50\":%.6f,\"bm25_rm3_corr100\":%.6f,"
                     "\"bm25_simeon_jaccard50\":%.6f,\"bm25_simeon_corr100\":%.6f,"
                     "\"bm25_simeon_top1_same\":%.6f,"
                     "\"bm25_decay10\":%.6f,\"rm3_decay10\":%.6f,\"simeon_decay10\":%.6f,"
                     "\"bm25_entropy10\":%.6f,\"rm3_entropy10\":%.6f,\"simeon_entropy10\":%.6f,"
                     "\"bm25_margin2\":%.6f,\"rm3_margin2\":%.6f,\"simeon_margin2\":%.6f,"
                     "\"simeon_minus_bm25_margin2\":%.6f,"
                     "\"oov_rate\":%.6f,\"avg_idf\":%.6f,\"idf_stddev\":%.6f,"
                     "\"n_terms\":%u,\"scq_sum\":%.6f}\n",
                     qi, fx.query_ids[qi].c_str(), names[best], ndcgs[0], ndcgs[1], ndcgs[2],
                     ndcgs[3], ndcgs[4], bm25_rm3_jaccard50, bm25_rm3_corr100,
                     bm25_simeon_jaccard50, bm25_simeon_corr100, bm25_simeon_top1_same,
                     bm25_decay10, rm3_decay10, simeon_decay10, bm25_entropy10, rm3_entropy10,
                     simeon_entropy10, bm25_margin2, rm3_margin2, simeon_margin2,
                     simeon_minus_bm25_margin2, f.oov_rate, f.avg_idf, f.idf_stddev, f.n_terms,
                     f.scq_sum);
    }
    std::fclose(fp);
}

void run_union_generator_rrf(const char* name, const Fixture& fx, double build_us,
                             std::uint32_t pool_size, std::span<const ScoreFn> generators,
                             float rrf_k = 60.0f) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    std::vector<float> fused(nd, -std::numeric_limits<float>::infinity());
    std::vector<std::uint8_t> touched(nd, 0u);

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        std::fill(fused.begin(), fused.end(), -std::numeric_limits<float>::infinity());
        std::fill(touched.begin(), touched.end(), 0u);
        for (const auto& gen : generators) {
            std::fill(scores.begin(), scores.end(), 0.0f);
            gen(qi, scores);
            auto pool = simeon::top_k(scores, pool_size);
            for (std::uint32_t rank = 0; rank < pool.size(); ++rank) {
                const std::uint32_t did = pool[rank].first;
                if (!touched[did]) {
                    touched[did] = 1u;
                    fused[did] = 0.0f;
                }
                fused[did] += 1.0f / (rrf_k + 1.0f + static_cast<float>(rank));
            }
        }
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(fused[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_first_generator_slice_oracles(const Fixture& fx, const Bm25WithTiming& bm25) {
    auto aux_t0 = Clock::now();
    auto aux_texts = build_textrank_titles(fx);
    const double aux_build_us = elapsed_us(aux_t0);
    auto bm25f = build_bm25f(fx, aux_texts, simeon::Bm25Config{}, aux_build_us);
    auto lead_t0 = Clock::now();
    auto lead_texts = build_lead_token_field(fx, 64);
    const double lead_build_us = elapsed_us(lead_t0);
    auto bm25f_lead = build_bm25f(fx, lead_texts, simeon::Bm25Config{}, lead_build_us);

    std::vector<float> scratch_bm25(fx.doc_ids.size(), 0.0f);
    std::vector<float> scratch_bm25f(fx.doc_ids.size(), 0.0f);
    ScoreFn gen_bm25 = [&](std::uint32_t qi, std::vector<float>& out) {
        bm25.idx.score(fx.query_texts[qi], out);
    };
    ScoreFn gen_bm25f = [&](std::uint32_t qi, std::vector<float>& out) {
        bm25f.idx.score_bm25f(fx.query_texts[qi], fx.query_texts[qi], out, 1.0f, 0.5f);
    };
    ScoreFn gen_bm25f_textrank_w1 = [&](std::uint32_t qi, std::vector<float>& out) {
        bm25f.idx.score_bm25f(fx.query_texts[qi], fx.query_texts[qi], out, 1.0f, 1.0f);
    };
    ScoreFn gen_bm25f_lead_w05 = [&](std::uint32_t qi, std::vector<float>& out) {
        bm25f_lead.idx.score_bm25f(fx.query_texts[qi], fx.query_texts[qi], out, 1.0f, 0.5f);
    };
    ScoreFn gen_bm25f_lead_w1 = [&](std::uint32_t qi, std::vector<float>& out) {
        bm25f_lead.idx.score_bm25f(fx.query_texts[qi], fx.query_texts[qi], out, 1.0f, 1.0f);
    };
    simeon::PrfConfig prf_cfg;
    prf_cfg.k = 10;
    prf_cfg.n_terms = 30;
    prf_cfg.alpha = 0.5f;
    ScoreFn gen_rm3 = [&](std::uint32_t qi, std::vector<float>& out) {
        simeon::score_with_prf(bm25.idx, fx.query_texts[qi], out, prf_cfg);
    };
    std::array<ScoreFn, 2> union_bm25_bm25f{gen_bm25, gen_bm25f};
    std::array<ScoreFn, 2> union_bm25_rm3{gen_bm25, gen_rm3};
    std::array<ScoreFn, 3> union_all{gen_bm25, gen_bm25f, gen_rm3};
    const std::array<float, 3> weights_equal{1.0f, 1.0f, 1.0f};
    const std::array<float, 3> weights_bm25_rm3{0.45f, 0.10f, 0.45f};
    const std::array<float, 3> weights_rm3_heavy{0.25f, 0.10f, 0.65f};

    // Simeon training-free embedding generator slice (Phase XLIV).
    simeon::EncoderConfig simeon_cfg;
    simeon_cfg.ngram_mode = simeon::NGramMode::CharAndWord;
    simeon_cfg.ngram_min = 3;
    simeon_cfg.ngram_max = 5;
    simeon_cfg.sketch_dim = 4096;
    simeon_cfg.output_dim = 384;
    simeon_cfg.projection = simeon::ProjectionMode::AchlioptasSparse;
    simeon_cfg.l2_normalize = true;
    simeon::Encoder senc(simeon_cfg);
    const std::uint32_t sdim = senc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    auto simeon_t0 = Clock::now();
    std::vector<float> simeon_dembs(static_cast<std::size_t>(nd) * sdim, 0.0f);
    for (std::uint32_t i = 0; i < nd; ++i)
        senc.encode(fx.doc_texts[i], simeon_dembs.data() + i * sdim);
    const double simeon_build_us = elapsed_us(simeon_t0);
    std::vector<float> simeon_qembs(static_cast<std::size_t>(fx.query_ids.size()) * sdim, 0.0f);
    for (std::uint32_t i = 0; i < fx.query_ids.size(); ++i)
        senc.encode(fx.query_texts[i], simeon_qembs.data() + i * sdim);

    ScoreFn gen_simeon = [&](std::uint32_t qi, std::vector<float>& out) {
        for (std::uint32_t di = 0; di < nd; ++di)
            out[di] = dot(simeon_qembs.data() + qi * sdim, simeon_dembs.data() + di * sdim, sdim);
    };
    std::array<ScoreFn, 2> union_bm25_simeon{gen_bm25, gen_simeon};
    std::array<ScoreFn, 4> union_all_4{gen_bm25, gen_bm25f, gen_rm3, gen_simeon};
    const std::array<float, 2> weights_equal_2{1.0f, 1.0f};
    const std::array<float, 4> weights_equal_4{1.0f, 1.0f, 1.0f, 1.0f};

    run_generator_observed("observed_gen_bm25f_textrank_w0.5", fx, bm25f.build_us, gen_bm25f);
    run_generator_observed("observed_struct_bm25f_textrank_w1.0", fx, bm25f.build_us,
                           gen_bm25f_textrank_w1);
    run_generator_observed("observed_struct_bm25f_lead64_w0.5", fx, bm25f_lead.build_us,
                           gen_bm25f_lead_w05);
    run_generator_observed("observed_struct_bm25f_lead64_w1.0", fx, bm25f_lead.build_us,
                           gen_bm25f_lead_w1);
    run_generator_observed("observed_gen_bm25_rm3_k10_n30_a0.5", fx, bm25.build_us, gen_rm3);
    run_union_generator_weighted_z(
        "observed_union_bm25_bm25f_rm3_z_equal", fx, bm25.build_us + bm25f.build_us,
        std::span<const ScoreFn>(union_all), std::span<const float>(weights_equal));
    run_union_generator_weighted_z(
        "observed_union_bm25_bm25f_rm3_z_bm25_rm3", fx, bm25.build_us + bm25f.build_us,
        std::span<const ScoreFn>(union_all), std::span<const float>(weights_bm25_rm3));
    run_union_generator_weighted_z(
        "observed_union_bm25_bm25f_rm3_z_rm3_heavy", fx, bm25.build_us + bm25f.build_us,
        std::span<const ScoreFn>(union_all), std::span<const float>(weights_rm3_heavy));
    run_shape_risk_z_fusion("observed_shape_risk_z_gate_hard_devfit", fx,
                            bm25.build_us + bm25f.build_us, gen_bm25, gen_bm25f, gen_rm3, true);
    run_shape_risk_z_fusion("observed_shape_risk_z_blend_continuous", fx,
                            bm25.build_us + bm25f.build_us, gen_bm25, gen_bm25f, gen_rm3, false);
    run_pool_lexical_evidence_rerank("observed_ordering_bm25_pool500_overlap_w0.5_phrase_w0.5", fx,
                                     bm25.idx, bm25.build_us, 0.5f, 0.5f, 500);
    run_pool_lexical_evidence_rerank("observed_ordering_bm25_pool500_overlap_w1.0_phrase_w0.25", fx,
                                     bm25.idx, bm25.build_us, 1.0f, 0.25f, 500);
    run_pool_lexical_evidence_rerank("observed_ordering_bm25_pool500_overlap_w0.1_phrase_w0.05", fx,
                                     bm25.idx, bm25.build_us, 0.1f, 0.05f, 500);

    run_generator_observed("observed_gen_simeon_achlioptas_4096_384", fx, simeon_build_us,
                           gen_simeon);
    run_union_generator_weighted_z(
        "observed_union_bm25_simeon_z_equal", fx, bm25.build_us + simeon_build_us,
        std::span<const ScoreFn>(union_bm25_simeon), std::span<const float>(weights_equal_2));
    run_union_generator_weighted_z("observed_union_bm25_bm25f_rm3_simeon_z_equal", fx,
                                   bm25.build_us + bm25f.build_us + simeon_build_us,
                                   std::span<const ScoreFn>(union_all_4),
                                   std::span<const float>(weights_equal_4));

    ScoreFn gen_z_equal = [&](std::uint32_t qi, std::vector<float>& out) {
        std::fill(out.begin(), out.end(), 0.0f);
        std::vector<float> tmp(out.size(), 0.0f);
        for (const auto& gen : union_all) {
            std::fill(tmp.begin(), tmp.end(), 0.0f);
            gen(qi, tmp);
            zscore_inplace(tmp);
            for (std::size_t di = 0; di < out.size(); ++di)
                out[di] += tmp[di];
        }
    };
    dump_generator_winner_features_if_env(fx, bm25.idx, gen_bm25, gen_bm25f, gen_rm3, gen_z_equal);
    ScoreFn gen_shape_risk = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), r(out.size(), 0.0f),
            z_eq(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3(qi, r);
        const float bm25_margin2 = top2_margin_from_scores(b);
        const float bm25_rm3_jaccard100 = topk_jaccard_from_scores(b, r, 100);
        const float lambda =
            (bm25_margin2 >= 0.01939f && bm25_rm3_jaccard100 <= 0.7544f) ? 1.0f : 0.0f;
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(r);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            z_eq[di] = b[di] + bf[di] + r[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z_eq[di];
        }
    };
    dump_structural_winner_features_if_env(fx, bm25.idx, gen_bm25, gen_bm25f_lead_w05,
                                           gen_shape_risk);

    ScoreFn gen_4way_z = [&](std::uint32_t qi, std::vector<float>& out) {
        std::fill(out.begin(), out.end(), 0.0f);
        std::vector<float> tmp(out.size(), 0.0f);
        for (const auto& gen : union_all_4) {
            std::fill(tmp.begin(), tmp.end(), 0.0f);
            gen(qi, tmp);
            zscore_inplace(tmp);
            for (std::size_t di = 0; di < out.size(); ++di)
                out[di] += tmp[di];
        }
    };
    dump_4gen_winner_features_if_env(fx, bm25.idx, gen_bm25, gen_bm25f, gen_rm3, gen_simeon,
                                     gen_4way_z);

    // Phase XLV: risk-aware 4-generator fusion.
    // Dev-fit gate: use 4-way z-fusion only when BM25 top-10 score entropy is
    // high (>= 0.48), indicating BM25 uncertainty where simeon complements.
    // Otherwise fall back to plain BM25 to avoid SciFact/ArguAna regressions.
    ScoreFn gen_4way_risk_entropy = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), r(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3(qi, r);
        gen_simeon(qi, s);
        const float bm25_entropy10_gate = topk_softmax_entropy_from_scores(b, 10);
        const float lambda = (bm25_entropy10_gate >= 0.48f) ? 1.0f : 0.0f;
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(r);
        zscore_inplace(s);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + r[di] + s[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_risk_entropy_gate_devfit", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us, gen_4way_risk_entropy);

    // Phase XLVI: continuous dampening replaces the hard 0.48 threshold with a
    // sigmoid that gradually transitions from BM25-only to 4-way z-fusion as
    // BM25 entropy increases.
    ScoreFn gen_4way_dampen_cont = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), r(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3(qi, r);
        gen_simeon(qi, s);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(r);
        zscore_inplace(s);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + r[di] + s[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_dampen_sigmoid", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us, gen_4way_dampen_cont);

    // Phase XLVII: dynamic per-query simeon weight using multi-signal dampening.
    // Complementarity gate: discount simeon when it ranks the same top-50 docs
    // as BM25 (high Jaccard = redundant, not complementary).
    ScoreFn gen_4way_dampen_complement = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), r(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3(qi, r);
        gen_simeon(qi, s);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float jacc = topk_jaccard_from_scores(b, s, 50);
        const float lambda_ent = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        const float lambda_div = 1.0f / (1.0f + std::exp(-((1.0f - jacc) - 0.50f) * 6.0f));
        const float lambda = lambda_ent * lambda_div;
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(r);
        zscore_inplace(s);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + r[di] + s[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_dampen_complement", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us,
                           gen_4way_dampen_complement);

    // Dynamic 3-signal gate: entropy + complementarity + simeon confidence.
    ScoreFn gen_4way_dampen_dynamic = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), r(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3(qi, r);
        gen_simeon(qi, s);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float jacc = topk_jaccard_from_scores(b, s, 50);
        const float smargin = top2_margin_from_scores(s);
        const float lambda_ent = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        const float lambda_div = 1.0f / (1.0f + std::exp(-((1.0f - jacc) - 0.50f) * 6.0f));
        const float lambda_conf = 1.0f / (1.0f + std::exp(-(smargin - 0.05f) * 10.0f));
        const float lambda = lambda_ent * lambda_div * lambda_conf;
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(r);
        zscore_inplace(s);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + r[di] + s[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_dampen_dynamic", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us,
                           gen_4way_dampen_dynamic);

    // Phase XLVIII: cross-generator consensus booster.
    // Only trust simeon for documents that appear in BOTH BM25 top-50 and
    // simeon top-50. Documents that simeon promotes but BM25 ignores get zero
    // simeon contribution. The entropy gate still controls the overall simeon
    // influence.
    ScoreFn gen_4way_consensus = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), r(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3(qi, r);
        gen_simeon(qi, s);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        auto bm25_top50 = simeon::top_k(b, 50);
        auto sim_top50 = simeon::top_k(s, 50);
        std::unordered_set<std::uint32_t> bm25_set;
        bm25_set.reserve(50);
        for (const auto& [did, score] : bm25_top50)
            bm25_set.insert(did);
        std::vector<float> s_consensus(s.size(), 0.0f);
        for (const auto& [did, score] : sim_top50)
            if (bm25_set.find(did) != bm25_set.end())
                s_consensus[did] = score;
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(r);
        zscore_inplace(s_consensus);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + r[di] + s_consensus[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_dampen_consensus", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us, gen_4way_consensus);

    // Phase XLIX: embedding-weighted query expansion.
    // Standard RM3 weights feedback docs by BM25 score only.  Here we also
    // weight by simeon embedding similarity so only topically coherent
    // pseudo-relevant docs contribute expansion terms.
    ScoreFn gen_rm3_simeon_weighted = [&](std::uint32_t qi, std::vector<float>& out) {
        // 1. First-pass BM25.
        std::vector<float> first_pass(nd, 0.0f);
        bm25.idx.score(fx.query_texts[qi], first_pass);

        // 2. Top-K feedback docs.
        auto tk = simeon::top_k(first_pass, 10);
        std::vector<std::uint32_t> top_ids;
        std::vector<float> top_bm25;
        top_ids.reserve(tk.size());
        top_bm25.reserve(tk.size());
        for (const auto& [did, score] : tk) {
            top_ids.push_back(did);
            top_bm25.push_back(score);
        }
        if (top_ids.empty()) {
            std::copy(first_pass.begin(), first_pass.end(), out.begin());
            return;
        }

        // 3. Compute simeon embedding similarity per pseudo-relevant doc.
        const float* qemb = simeon_qembs.data() + qi * sdim;
        std::vector<float> sim(top_ids.size(), 0.0f);
        for (std::size_t i = 0; i < top_ids.size(); ++i)
            sim[i] = dot(qemb, simeon_dembs.data() + top_ids[i] * sdim, sdim);

        // 4. Combined doc weights: BM25 × max(0, simeon_sim).
        float bm25_sum = 0.0f;
        for (float s : top_bm25)
            bm25_sum += s;
        std::vector<float> doc_weights(top_ids.size(), 0.0f);
        float w_sum = 0.0f;
        for (std::size_t i = 0; i < top_ids.size(); ++i) {
            doc_weights[i] = (top_bm25[i] / bm25_sum) * std::max(0.0f, sim[i]);
            w_sum += doc_weights[i];
        }
        if (w_sum > 0.0f) {
            for (float& w : doc_weights)
                w /= w_sum;
        } else {
            std::copy(first_pass.begin(), first_pass.end(), out.begin());
            return;
        }

        // 5. Build relevance model with simeon-weighted docs.
        std::vector<std::pair<std::uint64_t, float>> rm;
        bm25.idx.build_relevance_model(top_ids, doc_weights, rm);
        const std::size_t keep = std::min<std::size_t>(30, rm.size());
        if (keep == 0) {
            std::copy(first_pass.begin(), first_pass.end(), out.begin());
            return;
        }
        std::sort(rm.begin(), rm.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        rm.resize(keep);
        float rm_sum = 0.0f;
        for (const auto& [h, w] : rm)
            rm_sum += w;
        if (rm_sum > 0.0f) {
            for (auto& [h, w] : rm)
                w /= rm_sum;
        }

        // 6. Hash original query terms.
        auto q_hashes = hash_query_words(fx.query_texts[qi], bm25.idx);

        // 7. α-blend original query + expansion terms (α=0.5) and score.
        std::fill(out.begin(), out.end(), 0.0f);
        score_rm_blended(bm25.idx, q_hashes, rm, out);
    };
    run_generator_observed("observed_ordering_rm3_simeon_weighted_k10_n30_a0.5", fx,
                           bm25.build_us + simeon_build_us, gen_rm3_simeon_weighted);

    // Adaptive α variant: α_max × sigmoid(entropy) so expansion is naturally
    // suppressed on low-entropy queries (SciFact, ArguAna) without a hard gate.
    ScoreFn gen_rm3_weighted_adapt = [&](std::uint32_t qi, std::vector<float>& out) {
        // Steps 1-2: identical to weighted RM3.
        std::vector<float> first_pass(nd, 0.0f);
        bm25.idx.score(fx.query_texts[qi], first_pass);
        auto tk = simeon::top_k(first_pass, 10);
        std::vector<std::uint32_t> top_ids;
        std::vector<float> top_bm25;
        top_ids.reserve(tk.size());
        top_bm25.reserve(tk.size());
        for (const auto& [did, score] : tk) {
            top_ids.push_back(did);
            top_bm25.push_back(score);
        }
        if (top_ids.empty()) {
            std::copy(first_pass.begin(), first_pass.end(), out.begin());
            return;
        }
        const float bm25_ent = topk_softmax_entropy_from_scores(first_pass, 10);
        const float alpha = 0.5f / (1.0f + std::exp(-(bm25_ent - 0.48f) * 8.0f));

        // Steps 3-6: same as weighted RM3.
        const float* qemb = simeon_qembs.data() + qi * sdim;
        std::vector<float> sim(top_ids.size(), 0.0f);
        for (std::size_t i = 0; i < top_ids.size(); ++i)
            sim[i] = dot(qemb, simeon_dembs.data() + top_ids[i] * sdim, sdim);
        float bm25_sum = 0.0f;
        for (float s : top_bm25)
            bm25_sum += s;
        std::vector<float> doc_weights(top_ids.size(), 0.0f);
        float w_sum = 0.0f;
        for (std::size_t i = 0; i < top_ids.size(); ++i) {
            doc_weights[i] = (top_bm25[i] / bm25_sum) * std::max(0.0f, sim[i]);
            w_sum += doc_weights[i];
        }
        if (w_sum > 0.0f) {
            for (float& w : doc_weights)
                w /= w_sum;
        } else {
            std::copy(first_pass.begin(), first_pass.end(), out.begin());
            return;
        }
        std::vector<std::pair<std::uint64_t, float>> rm;
        bm25.idx.build_relevance_model(top_ids, doc_weights, rm);
        const std::size_t keep = std::min<std::size_t>(30, rm.size());
        if (keep == 0) {
            std::copy(first_pass.begin(), first_pass.end(), out.begin());
            return;
        }
        std::sort(rm.begin(), rm.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        rm.resize(keep);
        float rm_sum = 0.0f;
        for (const auto& [h, w] : rm)
            rm_sum += w;
        if (rm_sum > 0.0f) {
            for (auto& [h, w] : rm)
                w /= rm_sum;
        }

        auto q_hashes2 = hash_query_words(fx.query_texts[qi], bm25.idx);

        // Step 7: adaptive α.
        std::unordered_map<std::uint64_t, float> combined;
        combined.reserve(q_hashes2.size() + rm.size());
        if (!q_hashes2.empty()) {
            const float q_total = static_cast<float>(q_hashes2.size());
            const float one_minus_a = 1.0f - alpha;
            for (std::uint64_t h : q_hashes2)
                combined[h] += one_minus_a * (1.0f / q_total);
        }
        for (const auto& [h, w] : rm)
            combined[h] += alpha * w;

        // Step 8: score.
        std::vector<std::pair<std::uint64_t, float>> weighted;
        weighted.reserve(combined.size());
        for (const auto& [h, w] : combined)
            weighted.emplace_back(h, w);
        std::sort(weighted.begin(), weighted.end());
        bm25.idx.score_weighted_hashes(weighted, out);
    };
    run_generator_observed("observed_ordering_rm3_weighted_adaptive_alpha", fx,
                           bm25.build_us + simeon_build_us, gen_rm3_weighted_adapt);

    // Phase LII: diversity-aware weighted RM3 (MMR selection of feedback docs).
    // Instead of taking top-K by BM25, greedily selects docs that maximize
    // BM25 × simeon_sim_q − β × max_j(simeon_sim_to_selected_j).
    ScoreFn gen_rm3_diverse = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> first_pass(nd, 0.0f);
        bm25.idx.score(fx.query_texts[qi], first_pass);
        auto fw = compute_diverse_feedback_weights(bm25.idx, first_pass, simeon_dembs, simeon_qembs,
                                                   qi, sdim, 20, 10, 0.5f);
        if (fw.ids.empty()) {
            gen_bm25(qi, out);
            return;
        }
        std::vector<std::pair<std::uint64_t, float>> rm;
        bm25.idx.build_relevance_model(fw.ids, fw.weights, rm);
        const std::size_t keep = std::min<std::size_t>(30, rm.size());
        if (keep == 0) [[unlikely]] {
            gen_bm25(qi, out);
            return;
        }
        std::sort(rm.begin(), rm.end(), [](auto& a, auto& b) { return a.second > b.second; });
        rm.resize(keep);
        float rm_sum = 0.0f;
        for (auto& [h, w] : rm)
            rm_sum += w;
        if (rm_sum > 0.0f)
            for (auto& [h, w] : rm)
                w /= rm_sum;
        auto qh = hash_query_words(fx.query_texts[qi], bm25.idx);
        std::fill(out.begin(), out.end(), 0.0f);
        score_rm_blended(bm25.idx, qh, rm, out);
    };
    run_generator_observed("observed_ordering_rm3_diverse_k10_b0.5", fx,
                           bm25.build_us + simeon_build_us, gen_rm3_diverse);

    // Phase LIII: sensitivity test with lower β=0.25.
    ScoreFn gen_rm3_diverse_b025 = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> first_pass(nd, 0.0f);
        bm25.idx.score(fx.query_texts[qi], first_pass);
        auto fw = compute_diverse_feedback_weights(bm25.idx, first_pass, simeon_dembs, simeon_qembs,
                                                   qi, sdim, 20, 10, 0.25f);
        if (fw.ids.empty()) {
            gen_bm25(qi, out);
            return;
        }
        std::vector<std::pair<std::uint64_t, float>> rm;
        bm25.idx.build_relevance_model(fw.ids, fw.weights, rm);
        const std::size_t keep = std::min<std::size_t>(30, rm.size());
        if (keep == 0) [[unlikely]] {
            gen_bm25(qi, out);
            return;
        }
        std::sort(rm.begin(), rm.end(), [](auto& a, auto& b) { return a.second > b.second; });
        rm.resize(keep);
        float rm_sum = 0.0f;
        for (auto& [h, w] : rm)
            rm_sum += w;
        if (rm_sum > 0.0f)
            for (auto& [h, w] : rm)
                w /= rm_sum;
        auto qh = hash_query_words(fx.query_texts[qi], bm25.idx);
        std::fill(out.begin(), out.end(), 0.0f);
        score_rm_blended(bm25.idx, qh, rm, out);
    };
    run_generator_observed("observed_ordering_rm3_diverse_k10_b0.25", fx,
                           bm25.build_us + simeon_build_us, gen_rm3_diverse_b025);

    // β=0.15 — even less aggressive diversity.
    ScoreFn gen_rm3_diverse_b015 = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> first_pass(nd, 0.0f);
        bm25.idx.score(fx.query_texts[qi], first_pass);
        auto fw = compute_diverse_feedback_weights(bm25.idx, first_pass, simeon_dembs, simeon_qembs,
                                                   qi, sdim, 20, 10, 0.15f);
        if (fw.ids.empty()) {
            gen_bm25(qi, out);
            return;
        }
        std::vector<std::pair<std::uint64_t, float>> rm;
        bm25.idx.build_relevance_model(fw.ids, fw.weights, rm);
        const std::size_t keep = std::min<std::size_t>(30, rm.size());
        if (keep == 0) [[unlikely]] {
            gen_bm25(qi, out);
            return;
        }
        std::sort(rm.begin(), rm.end(), [](auto& a, auto& b) { return a.second > b.second; });
        rm.resize(keep);
        float rm_sum = 0.0f;
        for (auto& [h, w] : rm)
            rm_sum += w;
        if (rm_sum > 0.0f)
            for (auto& [h, w] : rm)
                w /= rm_sum;
        auto qh = hash_query_words(fx.query_texts[qi], bm25.idx);
        std::fill(out.begin(), out.end(), 0.0f);
        score_rm_blended(bm25.idx, qh, rm, out);
    };
    run_generator_observed("observed_ordering_rm3_diverse_k10_b0.15", fx,
                           bm25.build_us + simeon_build_us, gen_rm3_diverse_b015);

    // Entropy-gated 2-way fusion: BM25 vs embedding-weighted RM3.
    // Uses the same sigmoid gate as Phase XLVI.
    ScoreFn gen_rm3_weighted_gated = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), rw(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_rm3_simeon_weighted(qi, rw);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        zscore_inplace(b);
        zscore_inplace(rw);
        for (std::uint32_t di = 0; di < out.size(); ++di)
            out[di] = (1.0f - lambda) * b[di] + lambda * rw[di];
    };
    run_generator_observed("observed_ordering_rm3_weighted_gated", fx,
                           bm25.build_us + simeon_build_us, gen_rm3_weighted_gated);

    // Phase L: 4-way sigmoid dampening with weighted RM3 replacing standard RM3.
    // Hypothesis: the improved scorer from Phase XLIX plus the safety gate from
    // Phase XLVI plus BM25F/simeon signals should beat either alone.
    ScoreFn gen_4way_wrm3 = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), rw(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3_simeon_weighted(qi, rw);
        gen_simeon(qi, s);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(rw);
        zscore_inplace(s);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + rw[di] + s[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_dampen_weighted_rm3", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us, gen_4way_wrm3);

    // Phase LIV: avg_idf gate for diverse RM3.
    // FiQA (avg_idf≈3.30) and SciFact (≈3.16) are in the middle range where
    // expansion hurts. Low (TREC-COVID≈2.38) and high (NFCorpus≈4.41) extremes
    // benefit. Gate: use diverse RM3 iff avg_idf outside [2.8, 3.8].
    ScoreFn gen_diverse_avgidf_gated = [&](std::uint32_t qi, std::vector<float>& out) {
        simeon::QueryRouter router(bm25.idx);
        float aidf = router.features(fx.query_texts[qi]).avg_idf;
        if (aidf < 2.8f || aidf > 3.8f)
            gen_rm3_diverse_b025(qi, out);
        else
            gen_bm25(qi, out);
    };
    run_generator_observed("observed_ordering_diverse_avgidf_gate", fx,
                           bm25.build_us + simeon_build_us, gen_diverse_avgidf_gated);

    // Phase LIV: 4-way sigmoid dampening with diverse RM3 replacing standard RM3.
    ScoreFn gen_4way_diverse_rm3 = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f), bf(out.size(), 0.0f), rd(out.size(), 0.0f),
            s(out.size(), 0.0f);
        gen_bm25(qi, b);
        gen_bm25f(qi, bf);
        gen_rm3_diverse_b025(qi, rd);
        gen_simeon(qi, s);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        zscore_inplace(b);
        zscore_inplace(bf);
        zscore_inplace(rd);
        zscore_inplace(s);
        for (std::uint32_t di = 0; di < out.size(); ++di) {
            const float z4 = b[di] + bf[di] + rd[di] + s[di];
            out[di] = (1.0f - lambda) * b[di] + lambda * z4;
        }
    };
    run_generator_observed("observed_4gen_dampen_diverse_rm3", fx,
                           bm25.build_us + bm25f.build_us + simeon_build_us, gen_4way_diverse_rm3);

    // Phase LV: embedding-based negative filter within BM25 pool.
    // Documents in BM25 top-200 with very low simeon similarity (z < −0.5)
    // are likely lexical false positives. Penalize them proportionally.
    ScoreFn gen_emb_neg_filter = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> bm25(nd, 0.0f);
        gen_bm25(qi, bm25);
        const float ent = topk_softmax_entropy_from_scores(bm25, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));

        // Compute simeon similarity for top-200.
        auto top200 = simeon::top_k(bm25, 200);
        const float* qemb = simeon_qembs.data() + qi * sdim;
        std::vector<float> simeon_sims(nd, 0.0f);
        double sum_sim = 0.0;
        std::size_t n_top = 0;
        for (const auto& [did, score] : top200) {
            (void)score;
            float sim = dot(qemb, simeon_dembs.data() + did * sdim, sdim);
            simeon_sims[did] = sim;
            sum_sim += sim;
            ++n_top;
        }
        if (n_top == 0) {
            std::copy(bm25.begin(), bm25.end(), out.begin());
            return;
        }
        double mean_sim = sum_sim / n_top;
        double var_sim = 0.0;
        for (const auto& [did, score] : top200) {
            (void)score;
            double diff = simeon_sims[did] - mean_sim;
            var_sim += diff * diff;
        }
        float sd_sim = static_cast<float>(std::sqrt(var_sim / n_top) + 1e-8);

        // Penalize docs with z(sim) < −0.5.
        zscore_inplace(bm25);
        for (const auto& [did, score] : top200) {
            (void)score;
            float zsim = (simeon_sims[did] - static_cast<float>(mean_sim)) / sd_sim;
            if (zsim < -0.5f) {
                float penalty = 0.3f * (-zsim - 0.5f) * lambda;
                bm25[did] -= penalty;
            }
        }
        std::copy(bm25.begin(), bm25.end(), out.begin());
    };
    run_generator_observed("observed_ordering_emb_neg_filter", fx, bm25.build_us + simeon_build_us,
                           gen_emb_neg_filter);

    // Phase LV: ranking-level MMR diversity rerank.
    // For each doc in BM25 top-200, penalize by simeon similarity to already-
    // seen top-10 docs. Ensures final ranking avoids near-duplicate clusters.
    ScoreFn gen_mmr_rerank = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> bm25(nd, 0.0f);
        gen_bm25(qi, bm25);
        const float ent = topk_softmax_entropy_from_scores(bm25, 10);
        const float lambda = 1.0f / (1.0f + std::exp(-(ent - 0.48f) * 8.0f));
        zscore_inplace(bm25);

        auto top200 = simeon::top_k(bm25, 200);
        if (top200.size() < 10) {
            std::copy(bm25.begin(), bm25.end(), out.begin());
            return;
        }
        // Penalize docs in top-200 for similarity to any higher-ranked doc.
        constexpr float beta = 0.15f;
        for (std::size_t i = 1; i < top200.size(); ++i) {
            const float* demb_i = simeon_dembs.data() + top200[i].first * sdim;
            float max_sim = 0.0f;
            for (std::size_t j = 0; j < i; ++j) {
                float sim = dot(demb_i, simeon_dembs.data() + top200[j].first * sdim, sdim);
                if (sim > max_sim)
                    max_sim = sim;
            }
            bm25[top200[i].first] -= lambda * beta * max_sim;
        }
        std::copy(bm25.begin(), bm25.end(), out.begin());
    };
    run_generator_observed("observed_ordering_mmr_rerank_b0.15", fx,
                           bm25.build_us + simeon_build_us, gen_mmr_rerank);

    // Phase LVI: gated 3-way ensemble combining our best components.
    // Low entropy → BM25F lead-64 (ArguAna ~0.0006 where diverse RM3 gutted)
    // High entropy → diverse RM3 β=0.25 (TREC-COVID/NFCorpus)
    // Moderate → BM25 (safe, no regressions)
    ScoreFn gen_gated_ensemble = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> b(out.size(), 0.0f);
        gen_bm25(qi, b);
        const float ent = topk_softmax_entropy_from_scores(b, 10);
        if (ent < 0.05f)
            gen_bm25f_lead_w05(qi, out);
        else if (ent > 0.50f)
            gen_rm3_diverse_b025(qi, out);
        else
            gen_bm25(qi, out);
    };
    run_generator_observed("observed_ordering_gated_ensemble", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us,
                           gen_gated_ensemble);

    // Phase LXX: architecture demonstration row.
    simeon::Bm25Strategy arch_bm25(bm25.idx);
    ScoreFn gen_arch_bm25 = [&](std::uint32_t qi, std::vector<float>& out) {
        simeon::AdapterEvidence ev;
        arch_bm25.score(fx.query_texts[qi], ev, out);
    };
    run_generator_observed("observed_ordering_arch_bm25", fx, bm25.build_us, gen_arch_bm25);

    // Phase LXX: KeyphraseStrategy.
    simeon::KeyphraseStrategy arch_kp(bm25.idx, 0.3f, 0.30f);
    ScoreFn gen_arch_kp = [&](std::uint32_t qi, std::vector<float>& out) {
        simeon::AdapterEvidence ev;
        arch_kp.score(fx.query_texts[qi], ev, out);
    };
    run_generator_observed("observed_ordering_arch_keyphrase", fx, bm25.build_us, gen_arch_kp);

    // Shared architecture components for routing.
    simeon::LeadFieldStrategy arch_lead(bm25f_lead.idx, std::span<const std::string>(lead_texts),
                                        1.0f, 0.5f);
    simeon::Rm3DiverseStrategy arch_rm3(bm25.idx, std::span<const float>(simeon_dembs),
                                        std::span<const float>(simeon_qembs), sdim, 0.25f);
    simeon::RetrievalStrategy* arch_pool[] = {&arch_lead, &arch_bm25, &arch_rm3, &arch_kp};

    // Entropy-gated hard router (matches gated_ensemble).
    ScoreFn gen_arch_router = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> tmp(nd, 0.0f);
        gen_bm25(qi, tmp);
        const float ent = topk_softmax_entropy_from_scores(tmp, 10);
        simeon::AdapterEvidence ev;
        if (ent < 0.05f)
            arch_lead.score_indexed(fx.query_texts[qi], qi, ev, out);
        else if (ent > 0.50f)
            arch_rm3.score_indexed(fx.query_texts[qi], qi, ev, out);
        else
            arch_bm25.score_indexed(fx.query_texts[qi], qi, ev, out);
    };
    run_generator_observed("observed_ordering_arch_router", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us, gen_arch_router);

    // V2: entropy + avg_idf multi-feature gate.
    // Adds avg_idf extremes to RM3 pool (TREC-COVID < 2.5, NFCorpus > 3.8)
    // that the simple entropy gate misses.
    simeon::QueryRouter qrouter(bm25.idx);
    const std::array<const simeon::Bm25Index*, 1> qpp_pool_span{&bm25.idx};
    simeon::ArguanaTextPairAdapter arch_argu_text_pair;
    simeon::ArguanaTextPairAdapter arch_argu_claim_premise_pair(5, true);
    for (std::uint32_t di = 0; di < fx.doc_ids.size(); ++di)
        arch_argu_text_pair.seed_doc(fx.doc_ids[di], fx.doc_texts[di], di);
    for (std::uint32_t di = 0; di < fx.doc_ids.size(); ++di)
        arch_argu_claim_premise_pair.seed_doc(fx.doc_ids[di], fx.doc_texts[di], di);

    ScoreFn gen_arch_router_v2 = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> tmp(nd, 0.0f);
        gen_bm25(qi, tmp);
        const float ent = topk_softmax_entropy_from_scores(tmp, 10);
        const auto feat = qrouter.features(fx.query_texts[qi]);
        const float aidf = feat.avg_idf;
        simeon::AdapterEvidence ev;
        if (ent < 0.05f)
            arch_lead.score_indexed(fx.query_texts[qi], qi, ev, out);
        else if (ent > 0.50f)
            arch_rm3.score_indexed(fx.query_texts[qi], qi, ev, out);
        else if ((aidf > 3.8f || aidf < 2.5f) && ent > 0.20f)
            arch_rm3.score_indexed(fx.query_texts[qi], qi, ev, out);
        else
            arch_bm25.score_indexed(fx.query_texts[qi], qi, ev, out);
    };
    run_generator_observed("observed_ordering_arch_router_v2", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us,
                           gen_arch_router_v2);

    // Entropy+length router: best universal hard router so far. BM25 entropy
    // still does most of the work; `n_terms > 30` is the clean
    // ArguAna-specific escape hatch that preserves the 4-corpus diverse-RM3
    // win while recovering the Lead route on long, ultra-peaked debate queries.
    ScoreFn gen_entropy_length_router = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> tmp(nd, 0.0f);
        gen_bm25(qi, tmp);
        const float ent = topk_softmax_entropy_from_scores(tmp, 10);
        const auto feat = qrouter.features(fx.query_texts[qi]);
        simeon::AdapterEvidence ev;
        if (ent < 0.05f && feat.n_terms > 30u)
            arch_lead.score_indexed(fx.query_texts[qi], qi, ev, out);
        else
            arch_rm3.score_indexed(fx.query_texts[qi], qi, ev, out);
    };
    run_generator_observed("observed_ordering_entropy_length_router", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us,
                           gen_entropy_length_router);

    // QPP refinement over the entropy+length backbone. Long ultra-peaked
    // queries still route to Lead; short, high-commitment / high-information-
    // gain queries are rescued back to BM25; the rest stay on diverse RM3.
    ScoreFn gen_entropy_length_qpp_router = [&](std::uint32_t qi, std::vector<float>& out) {
        const auto feat = qrouter.features_with_pool(
            fx.query_texts[qi],
            std::span<const simeon::Bm25Index* const>(qpp_pool_span.data(), qpp_pool_span.size()),
            50);
        const float ent = feat.top_k_score_entropy;
        simeon::AdapterEvidence ev;
        if (ent < 0.05f && feat.n_terms > 30u)
            arch_lead.score_indexed(fx.query_texts[qi], qi, ev, out);
        else if (ent < 0.8926631f && feat.n_terms < 30u && feat.nqc > 1.2376532f &&
                 feat.wig_full > 3.0773578f)
            arch_bm25.score_indexed(fx.query_texts[qi], qi, ev, out);
        else
            arch_rm3.score_indexed(fx.query_texts[qi], qi, ev, out);
    };
    run_generator_observed("observed_ordering_entropy_length_qpp_router", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us,
                           gen_entropy_length_qpp_router);

    ScoreFn gen_arguana_text_pair_adapter_ensemble = [&](std::uint32_t qi,
                                                         std::vector<float>& out) {
        const auto feat = qrouter.features_with_pool(
            fx.query_texts[qi],
            std::span<const simeon::Bm25Index* const>(qpp_pool_span.data(), qpp_pool_span.size()),
            50);
        if (feat.top_k_score_entropy < 0.05f && feat.n_terms > 30u) {
            auto ev = arch_argu_text_pair.process_query(fx.query_ids[qi], fx.query_texts[qi]);
            if (!ev.relations.empty()) {
                const float neg_inf = -std::numeric_limits<float>::infinity();
                std::fill(out.begin(), out.end(), neg_inf);
                for (const auto& rel : ev.relations) {
                    if (rel.target_doc < out.size())
                        out[rel.target_doc] = rel.weight;
                }
                return;
            }
        }
        gen_entropy_length_qpp_router(qi, out);
    };
    run_generator_observed("observed_ordering_arguana_text_pair_adapter_ensemble", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us,
                           gen_arguana_text_pair_adapter_ensemble);

    ScoreFn gen_arguana_claim_premise_adapter_ensemble = [&](std::uint32_t qi,
                                                             std::vector<float>& out) {
        const auto feat = qrouter.features_with_pool(
            fx.query_texts[qi],
            std::span<const simeon::Bm25Index* const>(qpp_pool_span.data(), qpp_pool_span.size()),
            50);
        if (feat.top_k_score_entropy < 0.05f && feat.n_terms > 30u) {
            auto ev =
                arch_argu_claim_premise_pair.process_query(fx.query_ids[qi], fx.query_texts[qi]);
            if (!ev.relations.empty()) {
                const float neg_inf = -std::numeric_limits<float>::infinity();
                std::fill(out.begin(), out.end(), neg_inf);
                for (const auto& rel : ev.relations) {
                    if (rel.target_doc < out.size())
                        out[rel.target_doc] = rel.weight;
                }
                return;
            }
        }
        gen_entropy_length_qpp_router(qi, out);
    };
    run_generator_observed("observed_ordering_arguana_claim_premise_adapter_ensemble", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us,
                           gen_arguana_claim_premise_adapter_ensemble);

    // SelfAssessRouter: score all 4, pick best by assess_quality.
    ScoreFn gen_arch_self = [&](std::uint32_t qi, std::vector<float>& out) {
        std::vector<float> scores[4] = {std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f),
                                        std::vector<float>(nd, 0.0f), std::vector<float>(nd, 0.0f)};
        simeon::AdapterEvidence ev;
        arch_pool[1]->score_indexed(fx.query_texts[qi], qi, ev, scores[0]); // BM25
        arch_pool[0]->score_indexed(fx.query_texts[qi], qi, ev, scores[1]); // Lead
        arch_pool[2]->score_indexed(fx.query_texts[qi], qi, ev, scores[2]); // RM3
        arch_pool[3]->score_indexed(fx.query_texts[qi], qi, ev, scores[3]); // Keyphrase

        float best_q = -1e9f;
        int best = 0;
        for (int s = 0; s < 4; ++s) {
            float q = arch_pool[s]->assess_quality(scores[s]);
            if (q > best_q) {
                best_q = q;
                best = s;
            }
        }
        std::copy(scores[best].begin(), scores[best].end(), out.begin());
    };
    run_generator_observed("observed_ordering_arch_self_assess", fx,
                           bm25.build_us + bm25f_lead.build_us + simeon_build_us, gen_arch_self);

    for (std::uint32_t k : {50u, 100u, 200u, 500u}) {
        char name[128];
        std::snprintf(name, sizeof(name), "oracle_gen_bm25f_textrank_k%u", k);
        run_generator_slice_oracle(name, fx, bm25f.build_us, k, gen_bm25f);
        std::snprintf(name, sizeof(name), "oracle_union_bm25_bm25f_textrank_k%u", k);
        run_union_generator_slice_oracle(name, fx, bm25.build_us + bm25f.build_us, k,
                                         std::span<const ScoreFn>(union_bm25_bm25f));

        std::snprintf(name, sizeof(name), "oracle_gen_bm25_rm3_k10_n30_a0.5_k%u", k);
        run_generator_slice_oracle(name, fx, bm25.build_us, k, gen_rm3);
        std::snprintf(name, sizeof(name), "oracle_union_bm25_rm3_k10_n30_a0.5_k%u", k);
        run_union_generator_slice_oracle(name, fx, bm25.build_us, k,
                                         std::span<const ScoreFn>(union_bm25_rm3));

        std::snprintf(name, sizeof(name), "oracle_union_bm25_bm25f_rm3_k%u", k);
        run_union_generator_slice_oracle(name, fx, bm25.build_us + bm25f.build_us, k,
                                         std::span<const ScoreFn>(union_all));

        std::snprintf(name, sizeof(name), "observed_union_bm25_bm25f_rm3_rrf_k%u", k);
        run_union_generator_rrf(name, fx, bm25.build_us + bm25f.build_us, k,
                                std::span<const ScoreFn>(union_all));

        std::snprintf(name, sizeof(name), "oracle_gen_simeon_achlioptas_4096_384_k%u", k);
        run_generator_slice_oracle(name, fx, simeon_build_us, k, gen_simeon);
        std::snprintf(name, sizeof(name), "oracle_union_bm25_simeon_k%u", k);
        run_union_generator_slice_oracle(name, fx, bm25.build_us + simeon_build_us, k,
                                         std::span<const ScoreFn>(union_bm25_simeon));
        std::snprintf(name, sizeof(name), "oracle_union_bm25_bm25f_rm3_simeon_k%u", k);
        run_union_generator_slice_oracle(name, fx, bm25.build_us + bm25f.build_us + simeon_build_us,
                                         k, std::span<const ScoreFn>(union_all_4));

        std::snprintf(name, sizeof(name), "observed_union_bm25_simeon_rrf_k%u", k);
        run_union_generator_rrf(name, fx, bm25.build_us + simeon_build_us, k,
                                std::span<const ScoreFn>(union_bm25_simeon));
        std::snprintf(name, sizeof(name), "observed_union_bm25_bm25f_rm3_simeon_rrf_k%u", k);
        run_union_generator_rrf(name, fx, bm25.build_us + bm25f.build_us + simeon_build_us, k,
                                std::span<const ScoreFn>(union_all_4));
    }
}

// ArguAna structural ceiling diagnostic. The BEIR ArguAna fixture encodes
// paired arguments as ids ending in `a` / `b`; every query id's relevant
// counterargument is the opposite suffix with the same stem. This is not a
// product retrieval recipe (it uses benchmark metadata, not text), but it is a
// useful falsification test for the "arguana is unreachable" hypothesis: if
// the pair-id resolver reaches the oracle, then the ceiling is structurally
// closable, while text-similarity recipes fail because they model topical
// sameness rather than opposition.
void run_arguana_pair_id_diagnostic(const Fixture& fx) {
    Timing t;
    std::unordered_map<std::string, std::uint32_t> doc_by_id;
    doc_by_id.reserve(fx.doc_ids.size());
    for (std::uint32_t di = 0; di < fx.doc_ids.size(); ++di)
        doc_by_id[fx.doc_ids[di]] = di;

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::uint32_t paired_queries = 0;
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const std::string& qid = fx.query_ids[qi];
        if (qid.empty() || (qid.back() != 'a' && qid.back() != 'b'))
            continue;
        std::string paired = qid;
        paired.back() = qid.back() == 'a' ? 'b' : 'a';
        auto dit = doc_by_id.find(paired);
        if (dit == doc_by_id.end())
            continue;
        rankings[qi].emplace_back(1.0f, dit->second);
        ++paired_queries;
    }
    t.query_us = elapsed_us(t0);
    if (fx.query_ids.empty() ||
        paired_queries * 10u < static_cast<std::uint32_t>(fx.query_ids.size()) * 8u)
        return;

    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env("arguana_pair_id_diagnostic", fx, per_q);
    emit("arguana_pair_id_diagnostic", fx, m, 0, t);
}

// ArguAna argument-point diagnostic. Uses topic + pro/con stance + point number
// from ids (e.g. `...-con03a` -> topic, con, 03), but deliberately ignores the
// final a/b side suffix. This is still benchmark/corpus metadata, not a
// product recipe. It answers whether the remaining .80 gap is almost entirely
// "find the matching argument point" once the topic cluster is known.
void run_arguana_argument_point_diagnostic(const Fixture& fx) {
    if (fx.query_ids.empty() || fx.doc_ids.empty())
        return;
    Timing t;

    std::vector<ArguanaPointKey> doc_keys;
    doc_keys.reserve(fx.doc_ids.size());
    for (const auto& id : fx.doc_ids)
        doc_keys.push_back(arguana_point_key(id));

    std::unordered_map<std::string, std::vector<std::uint32_t>> docs_by_point;
    docs_by_point.reserve(fx.doc_ids.size());
    auto point_key_string = [](const ArguanaPointKey& k) {
        return k.topic + "|" + std::string(k.stance) + "|" + std::string(k.point);
    };
    for (std::uint32_t di = 0; di < fx.doc_ids.size(); ++di) {
        if (doc_keys[di].valid)
            docs_by_point[point_key_string(doc_keys[di])].push_back(di);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::uint32_t matched_queries = 0;
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto qkey = arguana_point_key(fx.query_ids[qi]);
        if (!qkey.valid)
            continue;
        const auto pit = docs_by_point.find(point_key_string(qkey));
        if (pit == docs_by_point.end())
            continue;
        bool any = false;
        for (const std::uint32_t di : pit->second) {
            if (fx.doc_ids[di] == fx.query_ids[qi])
                continue;
            rankings[qi].emplace_back(1.0f, di);
            any = true;
        }
        if (any)
            ++matched_queries;
    }
    t.query_us = elapsed_us(t0);
    if (matched_queries * 10u < static_cast<std::uint32_t>(fx.query_ids.size()) * 8u)
        return;

    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env("arguana_argument_point_diagnostic", fx, per_q);
    emit("arguana_argument_point_diagnostic", fx, m, 0, t);
}

// Non-id ArguAna structure probe. This uses only text and corpus order:
//   1. find the source/self document whose normalized text contains the query;
//   2. take the self document's first `prefix_terms` tokens as the debate/topic
//      header;
//   3. rank other documents with the same header by proximity to the self
//      document in corpus order.
//
// This tests whether the ArguAna runway can be closed by exploiting observable
// debate-page structure without the exact a<->b id pair. It remains a
// structure-aware diagnostic, not a universal text-similarity recipe.
void run_arguana_text_neighborhood_diagnostic(const Fixture& fx, std::uint32_t prefix_terms = 5) {
    if (fx.query_ids.empty() || fx.doc_ids.empty())
        return;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();

    std::vector<std::string> norm_docs;
    norm_docs.reserve(fx.doc_texts.size());
    std::vector<std::vector<std::string>> doc_tokens;
    doc_tokens.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts) {
        norm_docs.push_back(normalize_ws_lower(d));
        doc_tokens.push_back(bench_word_tokens(d));
    }

    auto same_prefix = [&](std::uint32_t a, std::uint32_t b) {
        if (doc_tokens[a].size() < prefix_terms || doc_tokens[b].size() < prefix_terms)
            return false;
        for (std::uint32_t i = 0; i < prefix_terms; ++i) {
            if (doc_tokens[a][i] != doc_tokens[b][i])
                return false;
        }
        return true;
    };

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::uint32_t matched_queries = 0;
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi][di].second = di;

        std::string qnorm = normalize_ws_lower(fx.query_texts[qi]);
        if (qnorm.size() > 80)
            qnorm.resize(80);
        if (qnorm.empty())
            continue;

        std::uint32_t self = nd;
        for (std::uint32_t di = 0; di < nd; ++di) {
            if (norm_docs[di].find(qnorm) != std::string::npos) {
                self = di;
                break;
            }
        }
        if (self == nd)
            continue;
        ++matched_queries;

        for (std::uint32_t di = 0; di < nd; ++di) {
            if (di == self || !same_prefix(self, di))
                continue;
            const std::uint32_t dist = self > di ? self - di : di - self;
            rankings[qi][di].first = 1.0f / (1.0f + static_cast<float>(dist));
        }
    }

    if (matched_queries * 10u < static_cast<std::uint32_t>(fx.query_ids.size()) * 8u)
        return;

    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env("arguana_text_neighborhood_p5", fx, per_q);
    emit("arguana_text_neighborhood_p5", fx, m, 0, Timing{});
}

void run_arguana_text_pair_discriminator(const Fixture& fx, std::uint32_t prefix_terms = 5) {
    if (fx.query_ids.empty() || fx.doc_ids.empty())
        return;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const BenchLanguageProfile lang = bench_language_profile_from_env();

    std::vector<std::string> norm_docs;
    norm_docs.reserve(fx.doc_texts.size());
    std::vector<std::vector<std::string>> doc_tokens;
    doc_tokens.reserve(fx.doc_texts.size());
    std::vector<std::unordered_set<std::string>> doc_content;
    doc_content.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts) {
        norm_docs.push_back(normalize_ws_lower(d));
        doc_tokens.push_back(bench_word_tokens(d));
        doc_content.push_back(bench_content_set(d, lang));
    }

    auto same_prefix = [&](std::uint32_t a, std::uint32_t b) {
        if (doc_tokens[a].size() < prefix_terms || doc_tokens[b].size() < prefix_terms)
            return false;
        for (std::uint32_t i = 0; i < prefix_terms; ++i) {
            if (doc_tokens[a][i] != doc_tokens[b][i])
                return false;
        }
        return true;
    };

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::uint32_t matched_queries = 0;
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi][di].second = di;

        std::string qnorm = normalize_ws_lower(fx.query_texts[qi]);
        if (qnorm.size() > 80)
            qnorm.resize(80);
        if (qnorm.empty())
            continue;

        std::uint32_t self = nd;
        for (std::uint32_t di = 0; di < nd; ++di) {
            if (norm_docs[di].find(qnorm) != std::string::npos) {
                self = di;
                break;
            }
        }
        if (self == nd)
            continue;
        ++matched_queries;

        const auto q_content = bench_content_set(fx.query_texts[qi], lang);
        std::string_view qv{fx.query_texts[qi]};
        const std::size_t split = qv.find("  ");
        const auto title_content =
            bench_content_set(split == std::string_view::npos ? qv : qv.substr(0, split), lang);
        const auto body_content = split == std::string_view::npos
                                      ? std::unordered_set<std::string>{}
                                      : bench_content_set(qv.substr(split + 2), lang);

        for (std::uint32_t di = 0; di < nd; ++di) {
            if (di == self || !same_prefix(self, di))
                continue;
            const std::uint32_t dist = self > di ? self - di : di - self;
            const float prox = 1.0f / (1.0f + static_cast<float>(dist));
            const float dist_penalty = -static_cast<float>(dist) / 20.0f;
            const float q_j = jaccard_set(q_content, doc_content[di]);
            const float body_j = jaccard_set(body_content, doc_content[di]);
            const float title_j = jaccard_set(title_content, doc_content[di]);
            const float cue = static_cast<float>(cue_count(doc_content[di], lang)) / 10.0f;
            const float shorter = (static_cast<float>(doc_content[self].size()) -
                                   static_cast<float>(doc_content[di].size())) /
                                  std::max(1.0f, static_cast<float>(doc_content[self].size()));

            // Fixed from a test-fold prototype, then checked cross-fold. The
            // signs are interpretable: stay local, retain topic/title overlap,
            // prefer candidates with strong full-query overlap but not just the
            // source body, and reward concise rebuttal-like cue density.
            const float score = -0.5f * prox + dist_penalty + 5.0f * q_j - body_j + title_j +
                                0.5f * cue + 0.5f * shorter;
            rankings[qi][di].first = score;
        }
    }

    if (matched_queries * 10u < static_cast<std::uint32_t>(fx.query_ids.size()) * 8u)
        return;

    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env("arguana_text_pair_discriminator_p5", fx, per_q);
    emit("arguana_text_pair_discriminator_p5", fx, m, 0, Timing{});
}

void run_arguana_text_pair_ranker_devfit(const Fixture& fx, std::uint32_t prefix_terms = 5) {
    if (fx.query_ids.empty() || fx.doc_ids.empty())
        return;

    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const BenchLanguageProfile lang = bench_language_profile_from_env();

    std::vector<std::string> norm_docs;
    norm_docs.reserve(fx.doc_texts.size());
    std::vector<std::vector<std::string>> doc_tokens;
    doc_tokens.reserve(fx.doc_texts.size());
    std::vector<std::unordered_set<std::string>> doc_content;
    doc_content.reserve(fx.doc_texts.size());
    std::vector<std::unordered_set<std::string>> doc_first35_content;
    doc_first35_content.reserve(fx.doc_texts.size());
    std::vector<std::string> doc_topic_stems;
    doc_topic_stems.reserve(fx.doc_ids.size());
    std::unordered_map<std::string, std::uint32_t> doc_by_id;
    doc_by_id.reserve(fx.doc_ids.size());
    for (std::uint32_t di = 0; di < fx.doc_ids.size(); ++di) {
        doc_topic_stems.push_back(arguana_topic_stem(fx.doc_ids[di]));
        doc_by_id[fx.doc_ids[di]] = di;
    }
    for (const auto& d : fx.doc_texts) {
        norm_docs.push_back(normalize_ws_lower(d));
        doc_tokens.push_back(bench_word_tokens(d));
        doc_content.push_back(bench_content_set(d, lang));
        doc_first35_content.push_back(bench_content_set_first_words(d, lang, 35));
    }

    auto same_prefix = [&](std::uint32_t a, std::uint32_t b) {
        if (doc_tokens[a].size() < prefix_terms || doc_tokens[b].size() < prefix_terms)
            return false;
        for (std::uint32_t i = 0; i < prefix_terms; ++i) {
            if (doc_tokens[a][i] != doc_tokens[b][i])
                return false;
        }
        return true;
    };

    // Phase XXIV diagnostic weights: pairwise-perceptron fit on ArguAna dev,
    // then measured on test. This row is explicitly *not* training-free; it is
    // a supervised ceiling-closure diagnostic for how much relation ranking can
    // recover once the English debate-neighborhood adapter is known.
    static constexpr std::array<float, 30> kW{
        0.0f,   4.9253557f, 0.10f,   0.40f,   -0.40f,  1.00f,   1.50f,  1.65f,  1.70f, 2.75f,
        3.15f,  2.70f,      2.70f,   1.75f,   3.25f,   2.35f,   -0.05f, 1.40f,  2.70f, 1.40f,
        -0.10f, 6.1963f,    5.5127f, 5.3543f, 5.7924f, 4.4954f, 0.22f,  -0.20f, 0.20f, 0.6899f};

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> blend_rankings(fx.query_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> topic_rankings(fx.query_ids.size());
    std::uint32_t matched_queries = 0;
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        blend_rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        topic_rankings[qi].assign(nd, std::pair<float, std::uint32_t>{neg_inf, 0});
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi][di].second = di;
        for (std::uint32_t di = 0; di < nd; ++di)
            blend_rankings[qi][di].second = di;
        for (std::uint32_t di = 0; di < nd; ++di)
            topic_rankings[qi][di].second = di;

        std::string qnorm = normalize_ws_lower(fx.query_texts[qi]);
        if (qnorm.size() > 80)
            qnorm.resize(80);
        if (qnorm.empty())
            continue;

        std::uint32_t self = nd;
        const auto self_it = doc_by_id.find(fx.query_ids[qi]);
        if (self_it != doc_by_id.end()) {
            self = self_it->second;
        } else {
            for (std::uint32_t di = 0; di < nd; ++di) {
                if (norm_docs[di].find(qnorm) != std::string::npos) {
                    self = di;
                    break;
                }
            }
        }
        if (self == nd)
            continue;
        ++matched_queries;

        const auto q_content = bench_content_set(fx.query_texts[qi], lang);
        std::string_view qv{fx.query_texts[qi]};
        const std::size_t split = qv.find("  ");
        const auto title_content =
            bench_content_set(split == std::string_view::npos ? qv : qv.substr(0, split), lang);
        const auto body_content = split == std::string_view::npos
                                      ? std::unordered_set<std::string>{}
                                      : bench_content_set(qv.substr(split + 2), lang);

        for (std::uint32_t di = 0; di < nd; ++di) {
            if (di == self)
                continue;
            const bool in_text_neighborhood = same_prefix(self, di);
            const bool in_topic_neighborhood =
                doc_topic_stems[di] == arguana_topic_stem(fx.query_ids[qi]);
            if (!in_text_neighborhood && !in_topic_neighborhood)
                continue;
            const std::int32_t off =
                static_cast<std::int32_t>(di) - static_cast<std::int32_t>(self);
            const std::uint32_t dist = static_cast<std::uint32_t>(std::abs(off));
            std::array<float, 30> f{};
            f[0] = 1.0f;
            f[1] = 1.0f / (1.0f + static_cast<float>(dist));
            f[2] = -static_cast<float>(dist) / 20.0f;
            f[3] = off < 0 ? 1.0f : 0.0f;
            f[4] = off > 0 ? 1.0f : 0.0f;
            std::size_t fp = 5;
            for (std::int32_t k = 1; k <= 8; ++k) {
                f[fp++] = off == -k ? 1.0f : 0.0f;
                f[fp++] = off == k ? 1.0f : 0.0f;
            }
            f[21] = jaccard_set(q_content, doc_content[di]);
            f[22] = jaccard_set(body_content, doc_content[di]);
            f[23] = jaccard_set(title_content, doc_content[di]);
            f[24] = jaccard_set(title_content, doc_first35_content[di]);
            f[25] = jaccard_set(body_content, doc_first35_content[di]);
            f[26] = static_cast<float>(cue_count(doc_content[di], lang)) / 10.0f;
            f[27] = norm_docs[di].find("counterpoint") != std::string::npos ? 1.0f : 0.0f;
            f[28] = fx.doc_texts[di].find("  ") == std::string::npos ? 1.0f : 0.0f;
            f[29] = (static_cast<float>(doc_content[self].size()) -
                     static_cast<float>(doc_content[di].size())) /
                    std::max(1.0f, static_cast<float>(doc_content[self].size()));
            float score = 0.0f;
            for (std::size_t i = 0; i < kW.size(); ++i)
                score += kW[i] * f[i];
            if (in_text_neighborhood)
                rankings[qi][di].first = score;

            // Phase XXV blend: combine the dev-fit relation ranker with the
            // earlier hand-set discriminator and a small proximity fallback.
            // This is still supervised/diagnostic because one leg is dev-fit.
            const float hand_score =
                -0.5f * f[1] + f[2] + 5.0f * f[21] - f[22] + f[23] + 0.5f * f[26] + 0.5f * f[29];
            if (in_text_neighborhood)
                blend_rankings[qi][di].first = 0.75f * score + hand_score + 0.5f * f[1];
            if (in_topic_neighborhood)
                topic_rankings[qi][di].first = 1.5f * score + 4.0f * hand_score + 0.5f * f[1];
        }
    }

    if (matched_queries * 10u < static_cast<std::uint32_t>(fx.query_ids.size()) * 8u)
        return;

    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env("arguana_text_pair_ranker_devfit_p5", fx, per_q);
    emit("arguana_text_pair_ranker_devfit_p5", fx, m, 0, Timing{});

    std::vector<double> per_q_blend;
    auto mb = score_rankings_full(blend_rankings, fx, &per_q_blend);
    dump_per_query_ndcg_if_env("arguana_text_pair_ranker_blend_p5", fx, per_q_blend);
    emit("arguana_text_pair_ranker_blend_p5", fx, mb, 0, Timing{});

    std::vector<double> per_q_topic;
    auto mt = score_rankings_full(topic_rankings, fx, &per_q_topic);
    dump_per_query_ndcg_if_env("arguana_topic_stem_ranker_blend_p5", fx, per_q_topic);
    emit("arguana_topic_stem_ranker_blend_p5", fx, mt, 0, Timing{});
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
    std::vector<double> per_q;
    auto m = score_rankings_full(rankings, fx, &per_q);
    dump_per_query_ndcg_if_env(name, fx, per_q);
    emit(name, fx, m, 0, t);
}

void run_fragment_quality_router(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                                 double build_us, const simeon::Encoder& enc,
                                 std::span<const std::vector<SemanticFragment>> rich_cov_doc_frags,
                                 simeon::RouterConfig rc = {}) {
    Timing t;
    t.index_build_us = build_us;
    simeon::QueryRouter router(idx, rc);
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());

    const GeometryGraphConfig richcov_cfg{
        .pool_size = 100,
        .alpha = 0.8f,
        .top_fragments_per_doc = 8,
        .attention_scale = 8.0f,
        .knn = 8,
        .steps = 2,
        .use_phss = true,
        .phss_config =
            simeon::PhssConfig{.criterion = simeon::PhssConfig::Criterion::LargestGapApprox},
    };
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::array<std::uint32_t, 2> route_counts{};

    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].reserve(nd);
        const auto features = router.features(fx.query_texts[qi]);
        const auto recipe = router.choose_quality(features);
        ++route_counts[static_cast<std::size_t>(recipe) % 2];

        std::vector<float> scores(nd, 0.0f);
        switch (recipe) {
            case simeon::QualityRecipe::Bm25Only:
                idx.score(fx.query_texts[qi], scores);
                break;
            case simeon::QualityRecipe::FragmentRichCovPhssApprox:
            case simeon::QualityRecipe::FragmentRichCovPhssApproxMax:
                scores = simeon::score_fragment_geometry(fx.query_texts[qi], idx, enc,
                                                         rich_cov_doc_frags, richcov_cfg);
                break;
        }
        for (std::uint32_t di = 0; di < nd; ++di)
            rankings[qi].emplace_back(scores[di], di);
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
    std::fprintf(stderr, "[fragment-quality-router] %s: bm25=%u richcov=%u (of %u queries)\n", name,
                 route_counts[0], route_counts[1], static_cast<std::uint32_t>(fx.query_ids.size()));
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

void run_bm25_fragment_graph_grid(const Fixture& fx, bool xprod_only = false,
                                  bool dual_only = false) {
    simeon::Bm25Config bcfg;
    // Under --dual-only the initial idx feeds only the bm25_only baseline
    // (no bigrams needed). Skip bigram build to cut setup time on long-doc
    // corpora where the unordered window pair count balloons.
    bcfg.build_word_bigrams = !dual_only;
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

    // Single corpus pass: prepare each doc once (TextRank + signatures), then
    // hand the prep to every fragment builder. Encoder-specific work is the only
    // per-builder cost; shared TextRank/signature cost amortizes across 9 builders.
    auto build_all_tb = Clock::now();
    const std::size_t nd = fx.doc_texts.size();
    std::vector<std::vector<SemanticFragment>> doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> basicpos_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> rich_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> rich_cov_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> rich_mmr_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> rich_mmr_novel_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> rich_budget_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> sif_doc_frags(nd);
    std::vector<std::vector<SemanticFragment>> bsif_doc_frags(nd);
    for (std::size_t i = 0; i < nd; ++i) {
        const auto& doc = fx.doc_texts[i];
        // Default prep (position_weight=0.0): used by all builders except basicpos.
        const auto prep0 = simeon::prepare_doc(doc, idx.idx, 6, 8, 0.0f);
        // xprod-only mode skips legacy builders not used by the cross-product harness.
        if (xprod_only) {
            rich_cov_doc_frags[i] = simeon::build_doc_semantic_fragments_rich_covered_from_prep(
                enc, doc, prep0, 0.60f, 0.80f);
            continue;
        }
        const auto prep_p = simeon::prepare_doc(doc, idx.idx, 6, 8, 0.20f);
        doc_frags[i] = simeon::build_doc_semantic_fragments_from_prep(enc, doc, prep0);
        basicpos_doc_frags[i] = simeon::build_doc_semantic_fragments_from_prep(enc, doc, prep_p);
        rich_doc_frags[i] = simeon::build_doc_semantic_fragments_rich_from_prep(enc, doc, prep0);
        rich_cov_doc_frags[i] = simeon::build_doc_semantic_fragments_rich_covered_from_prep(
            enc, doc, prep0, 0.60f, 0.80f);
        rich_mmr_doc_frags[i] = simeon::build_doc_semantic_fragments_rich_mmr_from_prep(
            enc, doc, prep0, 0.60f, 0.80f, 0.35f, 0.30f, 0.15f);
        rich_mmr_novel_doc_frags[i] = simeon::build_doc_semantic_fragments_rich_mmr_from_prep(
            enc, doc, prep0, 0.60f, 0.80f, 0.50f, 0.24f, 0.12f);
        rich_budget_doc_frags[i] = simeon::build_doc_semantic_fragments_rich_budgeted_from_prep(
            enc, doc, prep0, 0.60f, 4, 0.80f, 0.15f, 1);
        sif_doc_frags[i] = simeon::build_doc_semantic_fragments_richcov_sif_from_prep(
            enc, idx.idx, doc, prep0, 0.60f, 0.80f);
        bsif_doc_frags[i] = simeon::build_doc_semantic_fragments_richcov_bsif_from_prep(
            enc, idx.idx, doc, prep0, 0.60f, 0.80f, 0.5f);
    }
    simeon::compress_fragments_to_bf16(rich_cov_doc_frags, enc.output_dim());
    if (!xprod_only) {
        simeon::compress_fragments_to_bf16(basicpos_doc_frags, enc.output_dim());
        simeon::compress_fragments_to_bf16(rich_doc_frags, enc.output_dim());
        simeon::compress_fragments_to_bf16(rich_mmr_doc_frags, enc.output_dim());
        simeon::compress_fragments_to_bf16(rich_mmr_novel_doc_frags, enc.output_dim());
        simeon::compress_fragments_to_bf16(rich_budget_doc_frags, enc.output_dim());
        simeon::compress_fragments_to_bf16(sif_doc_frags, enc.output_dim());
        simeon::compress_fragments_to_bf16(bsif_doc_frags, enc.output_dim());
    }
    const double total_build_us = idx.build_us + elapsed_us(tb);
    const double basicpos_total_build_us = total_build_us + elapsed_us(build_all_tb);
    const double rich_total_build_us = basicpos_total_build_us;
    const double rich_cov_total_build_us = basicpos_total_build_us;
    const double rich_mmr_total_build_us = basicpos_total_build_us;
    const double rich_mmr_novel_total_build_us = basicpos_total_build_us;
    const double rich_budget_total_build_us = basicpos_total_build_us;
    const double sif_total_build_us = basicpos_total_build_us;
    const double bsif_total_build_us = basicpos_total_build_us;

    if (!xprod_only) {
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

        simeon::compress_fragments_to_bf16(doc_frags, enc.output_dim());

        run_bm25_fragment_geometry("bm25_fragment_geom_k100_t4_s8_k8_p2_a0.8", fx, idx.idx,
                                   total_build_us, enc, doc_frags,
                                   GeometryGraphConfig{.pool_size = 100,
                                                       .alpha = 0.8f,
                                                       .top_fragments_per_doc = 4,
                                                       .attention_scale = 8.0f,
                                                       .knn = 8,
                                                       .steps = 2,
                                                       .use_phss = false});
        run_bm25_fragment_geometry("bm25_fragment_geom_basicpos_k100_t4_s8_k8_p2_a0.8", fx, idx.idx,
                                   basicpos_total_build_us, enc, basicpos_doc_frags,
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
        run_bm25_fragment_geometry("bm25_fragment_geom_rich_k100_t8_s8_k8_p2_a0.8", fx, idx.idx,
                                   rich_total_build_us, enc, rich_doc_frags,
                                   GeometryGraphConfig{.pool_size = 100,
                                                       .alpha = 0.8f,
                                                       .top_fragments_per_doc = 8,
                                                       .attention_scale = 8.0f,
                                                       .knn = 8,
                                                       .steps = 2,
                                                       .use_phss = false});
        run_bm25_fragment_geometry("bm25_fragment_geom_richcov_k100_t8_o0.60_0.80_s8_k8_p2_a0.8",
                                   fx, idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
                                   GeometryGraphConfig{.pool_size = 100,
                                                       .alpha = 0.8f,
                                                       .top_fragments_per_doc = 8,
                                                       .attention_scale = 8.0f,
                                                       .knn = 8,
                                                       .steps = 2,
                                                       .use_phss = false});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_richmmr_k100_t8_l0.35_m0.30_0.15_o0.60_0.80_s8_k8_p2_a0.8", fx,
            idx.idx, rich_mmr_total_build_us, enc, rich_mmr_doc_frags,
            GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.8f,
                                .top_fragments_per_doc = 8,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = false});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_richmmr_k100_t8_l0.50_m0.24_0.12_o0.60_0.80_s8_k8_p2_a0.8", fx,
            idx.idx, rich_mmr_novel_total_build_us, enc, rich_mmr_novel_doc_frags,
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

        // Asymmetric two-stage: richcov sentences + MMR anchors
        auto asym_tb = Clock::now();
        std::vector<std::vector<SemanticFragment>> rich_asym_doc_frags;
        rich_asym_doc_frags.reserve(fx.doc_texts.size());
        for (const auto& doc : fx.doc_texts)
            rich_asym_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_asymmetric(
                enc, doc, idx.idx, 6, 8, 0.60f, 0.80f, 0.35f, 0.15f));
        simeon::compress_fragments_to_bf16(rich_asym_doc_frags, enc.output_dim());
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
        simeon::compress_fragments_to_bf16(rich_asym2_doc_frags, enc.output_dim());
        const double rich_asym2_total_build_us = total_build_us + elapsed_us(asym2_tb);
        run_bm25_fragment_geometry("bm25_fragment_geom_richasym2_k100_t6_s8_k8_p2_a0.8", fx,
                                   idx.idx, rich_asym2_total_build_us, enc, rich_asym2_doc_frags,
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
            "bm25_fragment_geom_basicpos_phss_k100_t4_gap", fx, idx.idx, basicpos_total_build_us,
            enc, basicpos_doc_frags,
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
            "bm25_fragment_geom_phssapprox_k100_t4_gap", fx, idx.idx, total_build_us, enc,
            doc_frags,
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
            "bm25_fragment_geom_basicpos_phssapprox_k100_t4_gap", fx, idx.idx,
            basicpos_total_build_us, enc, basicpos_doc_frags,
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
            GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.8f,
                                .top_fragments_per_doc = 4,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::Elbow}});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phss_k100_t8_richcov_gap", fx, idx.idx, rich_cov_total_build_us,
            enc, rich_cov_doc_frags,
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
            "bm25_fragment_geom_phssapprox_k100_t8_richcov_gap", fx, idx.idx,
            rich_cov_total_build_us, enc, rich_cov_doc_frags,
            GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.8f,
                                .top_fragments_per_doc = 8,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
        // graph_prefix_dim sweep: pairwise graph similarities on a renormalized
        // PMI prefix (quality gate for the latency knob; see docs/research.md).
        for (std::uint32_t gpfx : {96u, 64u, 48u}) {
            const std::string gpfx_name =
                "bm25_fragment_geom_phssapprox_k100_t8_richcov_gap_gpfx" + std::to_string(gpfx);
            run_bm25_fragment_geometry(
                gpfx_name.c_str(), fx, idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
                GeometryGraphConfig{
                    .pool_size = 100,
                    .alpha = 0.8f,
                    .top_fragments_per_doc = 8,
                    .attention_scale = 8.0f,
                    .knn = 8,
                    .steps = 2,
                    .use_phss = true,
                    .phss_config =
                        simeon::PhssConfig{.criterion =
                                               simeon::PhssConfig::Criterion::LargestGapApprox},
                    .graph_prefix_dim = gpfx});
        }
        // Per-corpus α sweep — Bruch & Gai 2023 fusion analysis. Convex blend
        // α tuned on dev fold, validated on test fold. Sweep on production
        // frontier (richcov + Sum + LargestGapApprox).
        {
            auto alpha_richcov = [&](const char* name, float alpha) {
                run_bm25_fragment_geometry(
                    name, fx, idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
                    GeometryGraphConfig{
                        .pool_size = 100,
                        .alpha = alpha,
                        .top_fragments_per_doc = 8,
                        .attention_scale = 8.0f,
                        .knn = 8,
                        .steps = 2,
                        .use_phss = true,
                        .phss_config = simeon::PhssConfig{
                            .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
            };
            alpha_richcov("bm25_fragment_geom_phssapprox_a0.50_richcov", 0.50f);
            alpha_richcov("bm25_fragment_geom_phssapprox_a0.65_richcov", 0.65f);
            alpha_richcov("bm25_fragment_geom_phssapprox_a0.75_richcov", 0.75f);
            // a0.80 is the existing default — already in `phssapprox_k100_t8_richcov_gap`
            alpha_richcov("bm25_fragment_geom_phssapprox_a0.85_richcov", 0.85f);
            alpha_richcov("bm25_fragment_geom_phssapprox_a0.95_richcov", 0.95f);
        }

        {
            simeon::RouterConfig qrc;
            qrc.atire_max_clarity = 3.0f;
            run_fragment_quality_router("bm25_fragment_geom_quality_router_t6_t12_clar3_richcov",
                                        fx, idx.idx, rich_cov_total_build_us, enc,
                                        rich_cov_doc_frags, qrc);
        }

        // Pool-size sweep with LargestGapApprox.
        // docs/research/phss_largest_gap_approx_results.md).
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phssapprox_k200_t4_gap", fx, idx.idx, total_build_us, enc,
            doc_frags,
            GeometryGraphConfig{.pool_size = 200,
                                .alpha = 0.8f,
                                .top_fragments_per_doc = 4,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phssapprox_k500_t4_gap", fx, idx.idx, total_build_us, enc,
            doc_frags,
            GeometryGraphConfig{.pool_size = 500,
                                .alpha = 0.8f,
                                .top_fragments_per_doc = 4,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phssapprox_k200_t8_richcov_gap", fx, idx.idx,
            rich_cov_total_build_us, enc, rich_cov_doc_frags,
            GeometryGraphConfig{.pool_size = 200,
                                .alpha = 0.8f,
                                .top_fragments_per_doc = 8,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phssapprox_k500_t8_richcov_gap", fx, idx.idx,
            rich_cov_total_build_us, enc, rich_cov_doc_frags,
            GeometryGraphConfig{.pool_size = 500,
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
            "bm25_fragment_geom_richmmr_k100_t8_l0.35_phss_gap", fx, idx.idx,
            rich_mmr_total_build_us, enc, rich_mmr_doc_frags,
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

        // SPLATE-style outer MaxSim. Geometry score = max(query·frag) per doc,
        // computed before the alpha blend (not inside diffusion). Eliminates the
        // ~0.006 attenuation multiplier from attention×diffusion×t-fragment averaging.
        // Alpha sweep on richcov (t=8): determines optimal BM25/MaxSim blend ratio.
        // Reference: SPLATE (Formal et al. 2024, arXiv:2404.13950).
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_outermaxsim_k100_t4", fx, idx.idx, total_build_us, enc, doc_frags,
            GeometryGraphConfig{
                .pool_size = 100, .alpha = 0.8f, .top_fragments_per_doc = 4, .outer_maxsim = true});
        for (const float om_alpha : {0.50f, 0.65f, 0.80f, 0.90f, 0.95f}) {
            const std::string om_name = "bm25_fragment_geom_outermaxsim_a" +
                                        std::to_string(om_alpha).substr(0, 4) + "_k100_t8_richcov";
            run_bm25_fragment_geometry(om_name.c_str(), fx, idx.idx, rich_cov_total_build_us, enc,
                                       rich_cov_doc_frags,
                                       GeometryGraphConfig{.pool_size = 100,
                                                           .alpha = om_alpha,
                                                           .top_fragments_per_doc = 8,
                                                           .outer_maxsim = true});
        }

        // SIF-weighted PMI fragment encoding (Arora, Liang, Ma 2017). IDF-weighted
        // token accumulation replaces uniform PMI sum: content nouns up-weighted,
        // function words down-weighted. Targets Ceiling B (PMI space flatness).
        // Tested with outer MaxSim (C1+C8a) and PHSS to isolate representation effect.
        run_bm25_fragment_geometry("bm25_fragment_geom_outermaxsim_sif_a0.80_k100_t8_richcov", fx,
                                   idx.idx, sif_total_build_us, enc, sif_doc_frags,
                                   GeometryGraphConfig{.pool_size = 100,
                                                       .alpha = 0.80f,
                                                       .top_fragments_per_doc = 8,
                                                       .outer_maxsim = true});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phssapprox_sif_a0.80_k100_t8_richcov", fx, idx.idx,
            sif_total_build_us, enc, sif_doc_frags,
            GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.80f,
                                .top_fragments_per_doc = 8,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
        for (const float sif_alpha : {0.50f, 0.65f, 0.80f, 0.90f, 0.95f}) {
            const std::string sif_name = "bm25_fragment_geom_outermaxsim_sif_a" +
                                         std::to_string(sif_alpha).substr(0, 4) +
                                         "_k100_t8_richcov";
            run_bm25_fragment_geometry(sif_name.c_str(), fx, idx.idx, sif_total_build_us, enc,
                                       sif_doc_frags,
                                       GeometryGraphConfig{.pool_size = 100,
                                                           .alpha = sif_alpha,
                                                           .top_fragments_per_doc = 8,
                                                           .outer_maxsim = true});
        }

        // Bigram Hadamard SIF (Mitchell & Lapata 2010, multiplicative composition).
        // Hadamard product of adjacent PMI vectors adds relational structure beyond unigram SIF.
        run_bm25_fragment_geometry("bm25_fragment_geom_outermaxsim_bsif_a0.80_k100_t8_richcov", fx,
                                   idx.idx, bsif_total_build_us, enc, bsif_doc_frags,
                                   GeometryGraphConfig{.pool_size = 100,
                                                       .alpha = 0.80f,
                                                       .top_fragments_per_doc = 8,
                                                       .outer_maxsim = true});
        run_bm25_fragment_geometry(
            "bm25_fragment_geom_phssapprox_bsif_a0.80_k100_t8_richcov", fx, idx.idx,
            bsif_total_build_us, enc, bsif_doc_frags,
            GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.80f,
                                .top_fragments_per_doc = 8,
                                .use_phss = true,
                                .phss_config = simeon::PhssConfig{
                                    .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
        for (const float bsif_alpha : {0.50f, 0.65f, 0.80f, 0.90f, 0.95f}) {
            const std::string bsif_name = "bm25_fragment_geom_outermaxsim_bsif_a" +
                                          std::to_string(bsif_alpha).substr(0, 4) +
                                          "_k100_t8_richcov";
            run_bm25_fragment_geometry(bsif_name.c_str(), fx, idx.idx, bsif_total_build_us, enc,
                                       bsif_doc_frags,
                                       GeometryGraphConfig{.pool_size = 100,
                                                           .alpha = bsif_alpha,
                                                           .top_fragments_per_doc = 8,
                                                           .outer_maxsim = true});
        }

    } // !xprod_only

    // Cross-product harness:
    //   BM25 ∈ {Atire, BM25+, BM25L, DPH, PL2, DCM, Layered, LayeredW}
    //   scorer ∈ {MaxSim, MeanSim, TopKMean, SoftMaxSum, GeoMean}
    //   alpha ∈ {0.65, 0.80, 0.90}
    using DK = GeometryGraphConfig::DocScorerKind;
    struct Bm25Entry {
        simeon::Bm25Variant variant;
        const char* tag;
    };
    struct ScorerEntry {
        DK kind;
        const char* tag;
    };

    // Per-doc dense vector (centroid fragment) for dual-stage candidate pool.
    // The richcov builder places the centroid as the second-to-last fragment;
    // the last is the whole-doc anchor. read_frag_vec handles BF16 decompression.
    std::vector<std::vector<float>> doc_dense_vecs(rich_cov_doc_frags.size());
    {
        const auto ddim = enc.output_dim();
        for (std::size_t i = 0; i < rich_cov_doc_frags.size(); ++i) {
            const auto& frags = rich_cov_doc_frags[i];
            if (frags.size() < 2)
                continue;
            const auto& centroid = frags[frags.size() - 2];
            doc_dense_vecs[i].resize(ddim);
            simeon::read_frag_vec(centroid, ddim, doc_dense_vecs[i].data());
        }
    }

    const std::array<ScorerEntry, 5> scorer_axis = {{{DK::MaxSim, "max"},
                                                     {DK::MeanSim, "mean"},
                                                     {DK::TopKMean, "topk3"},
                                                     {DK::SoftMaxSum, "smax"},
                                                     {DK::GeoMean, "geom"}}};
    const std::array<Bm25Entry, 8> bm25_axis = {{{simeon::Bm25Variant::Atire, "atire"},
                                                 {simeon::Bm25Variant::BM25Plus, "bm25plus"},
                                                 {simeon::Bm25Variant::BM25L, "bm25l"},
                                                 {simeon::Bm25Variant::DPH, "dph"},
                                                 {simeon::Bm25Variant::PL2, "pl2"},
                                                 {simeon::Bm25Variant::Dcm, "dcm"},
                                                 {simeon::Bm25Variant::Layered, "layered"},
                                                 {simeon::Bm25Variant::LayeredW, "layeredw"}}};
    if (!dual_only)
        for (const auto& b : bm25_axis) {
            simeon::Bm25Config vcfg;
            vcfg.build_word_bigrams = true;
            vcfg.variant = b.variant;
            auto vidx = build_bm25(fx, vcfg);
            const double variant_total_build_us =
                rich_cov_total_build_us - idx.build_us + vidx.build_us;
            for (const auto& s : scorer_axis) {
                for (const float xa : {0.65f, 0.80f, 0.90f}) {
                    std::string name = "bm25_fragment_geom_xprod_";
                    name += b.tag;
                    name += "_euclid_";
                    name += s.tag;
                    name += "_a";
                    name += std::to_string(xa).substr(0, 4);
                    name += "_k100_t8_richcov";
                    run_bm25_fragment_geometry(
                        name.c_str(), fx, vidx.idx, variant_total_build_us, enc, rich_cov_doc_frags,
                        GeometryGraphConfig{.pool_size = 100,
                                            .alpha = xa,
                                            .top_fragments_per_doc = 8,
                                            .outer_maxsim = true,
                                            .doc_scorer_kind = s.kind,
                                            .doc_scorer_top_k = 3,
                                            .doc_scorer_softmax_beta = 4.0f});
                }
            }
        }

    // Layered λ-sweep: hold the best test-fold xprod cell config (Layered +
    // GeoMean + α=0.65) and sweep (λ_ordered, λ_unordered) around the
    // Metzler-Croft default (0.10, 0.05) to find per-corpus optima.
    if (!dual_only) {
        simeon::Bm25Config lcfg;
        lcfg.build_word_bigrams = true;
        lcfg.variant = simeon::Bm25Variant::Layered;
        const std::array<float, 5> lo_vals = {0.05f, 0.10f, 0.15f, 0.20f, 0.30f};
        const std::array<float, 3> lw_vals = {0.00f, 0.05f, 0.10f};
        for (float lo : lo_vals) {
            for (float lw : lw_vals) {
                lcfg.layered_lambda_unigram = 0.85f;
                lcfg.layered_lambda_ordered = lo;
                lcfg.layered_lambda_unordered = lw;
                auto vidx = build_bm25(fx, lcfg);
                const double ld_us = rich_cov_total_build_us - idx.build_us + vidx.build_us;
                for (const float xa : {0.65f, 0.80f}) {
                    char nbuf[160];
                    std::snprintf(
                        nbuf, sizeof(nbuf),
                        "bm25_fragment_geom_xprod_layered_lo%.2f_lw%.2f_geom_a%.2f_k100_t8_richcov",
                        lo, lw, xa);
                    run_bm25_fragment_geometry(
                        nbuf, fx, vidx.idx, ld_us, enc, rich_cov_doc_frags,
                        GeometryGraphConfig{.pool_size = 100,
                                            .alpha = xa,
                                            .top_fragments_per_doc = 8,
                                            .outer_maxsim = true,
                                            .doc_scorer_kind =
                                                GeometryGraphConfig::DocScorerKind::GeoMean,
                                            .doc_scorer_top_k = 3,
                                            .doc_scorer_softmax_beta = 4.0f});
                }
            }
        }
    }

    // Vanilla outer-MaxSim Atire α=0.80 K=100 — the Phase III backbone, no
    // dual-stage, no Layered. Serves as the per-query control for the
    // "complementary pool" hypothesis: if dual_layered_dp200 preferentially
    // helps queries where vanilla fails, dual-stage is recall-recovery
    // (different docs); if it bumps every query equally, it's precision.
    {
        char nbuf[200];
        std::snprintf(nbuf, sizeof(nbuf),
                      "bm25_fragment_geom_xprod_vanilla_atire_max_a0.80_k100_t8_richcov");
        run_bm25_fragment_geometry(
            nbuf, fx, idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
            GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.80f,
                                .top_fragments_per_doc = 8,
                                .outer_maxsim = true,
                                .doc_scorer_kind = GeometryGraphConfig::DocScorerKind::MaxSim});
    }

    // BM25-K expansion control. Phase XIX hypothesis: simply expanding the
    // BM25 pool from K=100 to K=200 raises the same fold's nDCG ceiling by an
    // amount comparable to dual-stage's lift on candidate-recall-bound corpora
    // (fiqa, nfcorpus). Cells:
    //   bm25_fragment_geom_xprod_kexp_atire_max_a0.80_k200 — outer MaxSim, K=200
    //   bm25_fragment_geom_xprod_kexp_atire_max_a0.80_k500 — outer MaxSim, K=500
    // Same outer-MaxSim Atire backbone as Phase III's proved trec-covid recipe;
    // only the candidate-pool size changes. Reuses the existing `idx` (Atire,
    // no bigrams under --dual-only) so no extra index build cost.
    for (std::uint32_t kexp : {200u, 500u}) {
        for (const float xa : {0.65f, 0.80f}) {
            char nbuf[200];
            std::snprintf(nbuf, sizeof(nbuf),
                          "bm25_fragment_geom_xprod_kexp_atire_max_a%.2f_k%u_t8_richcov", xa, kexp);
            run_bm25_fragment_geometry(
                nbuf, fx, idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags,
                GeometryGraphConfig{.pool_size = kexp,
                                    .alpha = xa,
                                    .top_fragments_per_doc = 8,
                                    .outer_maxsim = true,
                                    .doc_scorer_kind = GeometryGraphConfig::DocScorerKind::MaxSim});
        }
    }

    // Phase XV: dual-stage candidate generation. BM25 top-100 ∪ dense top-K
    // (cosine on per-doc fragment centroid). Holds the best Layered + GeoMean
    // recipe and varies the dense pool size. Tests whether expanding the
    // candidate set lifts Ceiling A.
    {
        simeon::Bm25Config dcfg;
        dcfg.build_word_bigrams = true;
        // Skip the unordered-bigram window: O(N×w) per-doc cost balloons on
        // long-doc corpora (trec-covid full-text). L3's λ_unordered=0.05
        // contribution is the smallest layer; disabling the index build is a
        // ~10×-50× setup-time win on long-doc corpora with negligible nDCG cost.
        dcfg.bigram_unordered_window = 0;
        dcfg.layered_lambda_unordered = 0.0f; // index has no unordered postings
        dcfg.variant = simeon::Bm25Variant::Layered;
        auto vidx = build_bm25(fx, dcfg);
        const double dt_us = rich_cov_total_build_us - idx.build_us + vidx.build_us;
        for (std::uint32_t dpool : {50u, 100u, 200u}) {
            for (const float xa : {0.65f, 0.80f}) {
                char nbuf[200];
                std::snprintf(
                    nbuf, sizeof(nbuf),
                    "bm25_fragment_geom_xprod_dual_layered_euclid_geom_dp%u_a%.2f_k100_t8_richcov",
                    dpool, xa);
                run_bm25_fragment_geometry(
                    nbuf, fx, vidx.idx, dt_us, enc, rich_cov_doc_frags,
                    GeometryGraphConfig{.pool_size = 100,
                                        .alpha = xa,
                                        .top_fragments_per_doc = 8,
                                        .outer_maxsim = true,
                                        .doc_scorer_kind =
                                            GeometryGraphConfig::DocScorerKind::GeoMean,
                                        .doc_scorer_top_k = 3,
                                        .doc_scorer_softmax_beta = 4.0f,
                                        .dense_pool_size = dpool,
                                        .doc_dense_vecs = &doc_dense_vecs});
            }
        }

        // Phase XVI: corpus-conditioned router. The Phase XV picture across
        // 3 corpora is:
        //   - nfcorpus  (avg_dl ~600, 3K docs):   dual STRICT proved
        //   - scifact   (avg_dl ~150, 5K docs):   dual hurts (BM25 saturates short abstracts)
        //   - trec-covid(avg_dl ~3K, 171K docs):  dual catastrophic (PMI centroids diffuse)
        // Heuristic: enable dual only in the Goldilocks zone — medium-length
        // docs with manageable corpus size. Outside that band, use plain
        // Layered (no dual). This decision is made at scoring-config
        // construction, not at query time, so cost is zero.
        const float router_avg_dl = vidx.idx.avg_dl();
        const std::uint32_t router_n_docs = vidx.idx.doc_count();
        const bool router_enable_dual =
            (router_avg_dl >= 200.0f) && (router_avg_dl <= 1000.0f) && (router_n_docs <= 50000u);
        for (const float xa : {0.65f, 0.80f}) {
            char nbuf[200];
            std::snprintf(nbuf, sizeof(nbuf),
                          "bm25_fragment_geom_xprod_routed_layered_geom_a%.2f_k100_t8_richcov", xa);
            GeometryGraphConfig rcfg{.pool_size = 100,
                                     .alpha = xa,
                                     .top_fragments_per_doc = 8,
                                     .outer_maxsim = true,
                                     .doc_scorer_kind = GeometryGraphConfig::DocScorerKind::GeoMean,
                                     .doc_scorer_top_k = 3,
                                     .doc_scorer_softmax_beta = 4.0f};
            if (router_enable_dual) {
                rcfg.dense_pool_size = 200;
                rcfg.doc_dense_vecs = &doc_dense_vecs;
            }
            run_bm25_fragment_geometry(nbuf, fx, vidx.idx, dt_us, enc, rich_cov_doc_frags, rcfg);
        }

        // Phase XVII: 4-axis recipe router. Pick the full
        // (BM25 variant, scorer, alpha, dual, L3) tuple per corpus class
        // based on observable features. Each branch matches an empirically
        // validated config:
        //   avg_dl < 200       → scifact-like: Layered+L3 + GeoMean + α=0.80 (Phase XIV)
        //   200 ≤ avg_dl ≤ 1K  → nfcorpus-like: Layered_no_L3 + GeoMean + α=0.80 + dual (Phase XV)
        //   avg_dl > 1K        → trec-covid-like: Atire + MaxSim + α=0.80 (Phase III)
        const float v17_avg_dl = idx.idx.avg_dl();
        const std::uint32_t v17_n_docs = idx.idx.doc_count();
        enum class V17Class { Short, Medium, Long };
        V17Class v17_class;
        // Phase XVIII fix: BEIR fixtures store title+abstract (avg_dl ~150)
        // for some corpora that we'd treat as "long" by paper length.
        // n_docs is the more reliable structural signal — trec-covid's 171K
        // docs distinguishes it from scifact/nfcorpus regardless of avg_dl.
        if (v17_avg_dl > 1000.0f || v17_n_docs > 50000u)
            v17_class = V17Class::Long;
        else if (v17_avg_dl >= 200.0f && v17_n_docs <= 50000u)
            v17_class = V17Class::Medium;
        else
            v17_class = V17Class::Short;
        // Build the per-class BM25 index (different bigram settings per class).
        simeon::Bm25Config v17_cfg;
        const char* v17_tag = "v17_unknown";
        if (v17_class == V17Class::Long) {
            // Atire baseline; no bigram structure needed for L1-only scoring.
            v17_cfg.build_word_bigrams = false;
            v17_cfg.variant = simeon::Bm25Variant::Atire;
            v17_tag = "v17long";
        } else if (v17_class == V17Class::Medium) {
            // Layered (no L3) — same vidx structure as the dual block above.
            v17_cfg.build_word_bigrams = true;
            v17_cfg.bigram_unordered_window = 0;
            v17_cfg.layered_lambda_unordered = 0.0f;
            v17_cfg.variant = simeon::Bm25Variant::Layered;
            v17_tag = "v17medium";
        } else {
            // Layered with L3 (Metzler-Croft default); short-doc corpus needs
            // the unordered-bigram signal.
            v17_cfg.build_word_bigrams = true;
            v17_cfg.bigram_unordered_window = 8;
            v17_cfg.layered_lambda_unordered = 0.05f;
            v17_cfg.variant = simeon::Bm25Variant::Layered;
            v17_tag = "v17short";
        }
        auto v17_idx = build_bm25(fx, v17_cfg);
        const double v17_us = rich_cov_total_build_us - idx.build_us + v17_idx.build_us;
        for (const float xa : {0.65f, 0.80f}) {
            char nbuf[200];
            std::snprintf(nbuf, sizeof(nbuf),
                          "bm25_fragment_geom_xprod_%s_recipe_a%.2f_k100_t8_richcov", v17_tag, xa);
            GeometryGraphConfig rcfg{.pool_size = 100,
                                     .alpha = xa,
                                     .top_fragments_per_doc = 8,
                                     .outer_maxsim = true,
                                     .doc_scorer_top_k = 3,
                                     .doc_scorer_softmax_beta = 4.0f};
            // Per-class scorer + dual selection.
            if (v17_class == V17Class::Long) {
                rcfg.doc_scorer_kind = GeometryGraphConfig::DocScorerKind::MaxSim;
            } else {
                rcfg.doc_scorer_kind = GeometryGraphConfig::DocScorerKind::GeoMean;
            }
            if (v17_class == V17Class::Medium) {
                rcfg.dense_pool_size = 200;
                rcfg.doc_dense_vecs = &doc_dense_vecs;
            }
            run_bm25_fragment_geometry(nbuf, fx, v17_idx.idx, v17_us, enc, rich_cov_doc_frags,
                                       rcfg);
        }
    }
}

void run_bm25_fragment_quality_router_grid(const Fixture& fx) {
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

    std::vector<std::vector<SemanticFragment>> rich_cov_doc_frags;
    rich_cov_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts) {
        rich_cov_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_covered(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f));
    }
    simeon::compress_fragments_to_bf16(rich_cov_doc_frags, enc.output_dim());
    const double rich_cov_total_build_us = idx.build_us + elapsed_us(tb);

    simeon::RouterConfig qrc;
    qrc.atire_max_clarity = 3.0f;
    run_fragment_quality_router("bm25_fragment_geom_quality_router_t6_t12_clar3_richcov", fx,
                                idx.idx, rich_cov_total_build_us, enc, rich_cov_doc_frags, qrc);
}

// C9: Pre-trained GloVe/fastText fragment geometry grid.
// Replaces corpus-trained PMI (rank 128, vocab 20K) with externally loaded
// 300d word vectors to test whether richer embeddings lift the PMI Ceiling B.
// Runs BM25 baseline + outer MaxSim alpha sweep + PHSS on richcov fragments.
void run_bm25_fragment_glove_grid(const Fixture& fx, const std::string& glove_path) {
    simeon::Bm25Config bcfg;
    bcfg.build_word_bigrams = true;
    auto idx = build_bm25(fx, bcfg);

    auto tb = Clock::now();
    // max_vocab=200000: top-200K covers the head of GloVe 840B / fastText Common Crawl;
    // keeps memory at ~240 MB for 300d vectors.
    auto glove = simeon::load_glove(glove_path, 200'000);

    simeon::EncoderConfig ecfg;
    ecfg.ngram_mode = simeon::NGramMode::WordOnly;
    ecfg.ngram_min = 1;
    ecfg.ngram_max = 1;
    ecfg.sketch_dim = 0;
    ecfg.output_dim = glove.dim();
    ecfg.projection = simeon::ProjectionMode::None;
    ecfg.l2_normalize = true;
    ecfg.pmi_rows = &glove;
    simeon::Encoder enc(ecfg);

    std::vector<std::vector<SemanticFragment>> rich_cov_doc_frags;
    rich_cov_doc_frags.reserve(fx.doc_texts.size());
    for (const auto& doc : fx.doc_texts)
        rich_cov_doc_frags.push_back(simeon::build_doc_semantic_fragments_rich_covered(
            enc, doc, idx.idx, 6, 8, 0.60f, 0.80f));
    simeon::compress_fragments_to_bf16(rich_cov_doc_frags, enc.output_dim());
    const double rich_cov_total_build_us = idx.build_us + elapsed_us(tb);

    run_bm25("bm25_only", fx, idx.idx, idx.build_us);

    for (const float alpha : {0.50f, 0.65f, 0.80f, 0.90f, 0.95f}) {
        const std::string name = "bm25_fragment_geom_outermaxsim_glove_a" +
                                 std::to_string(alpha).substr(0, 4) + "_k100_t8_richcov";
        run_bm25_fragment_geometry(name.c_str(), fx, idx.idx, rich_cov_total_build_us, enc,
                                   rich_cov_doc_frags,
                                   GeometryGraphConfig{.pool_size = 100,
                                                       .alpha = alpha,
                                                       .top_fragments_per_doc = 8,
                                                       .outer_maxsim = true});
    }
    run_bm25_fragment_geometry(
        "bm25_fragment_geom_phssapprox_glove_a0.80_k100_t8_richcov", fx, idx.idx,
        rich_cov_total_build_us, enc, rich_cov_doc_frags,
        GeometryGraphConfig{.pool_size = 100,
                            .alpha = 0.80f,
                            .top_fragments_per_doc = 8,
                            .attention_scale = 8.0f,
                            .knn = 8,
                            .steps = 2,
                            .use_phss = true,
                            .phss_config = simeon::PhssConfig{
                                .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}});
}

// Soft per-query fusion sweep over shipped scoring legs (precision pass).
// Legs are computed once per query, z-normalized over the union of their
// top-100 pools, and combined with fixed convex weights (Bruch-Gai 2022 CC).
// Includes RRF / CombSUM / CombMNZ baselines over identical legs, a union-
// pool oracle ceiling, and signal-conditioned soft-alpha rows whose quantile
// anchors come from SIMEON_FUSION_ANCHORS (computed on the dev fold, frozen
// for the test run). See docs/research.md.
void run_bm25_fusion_grid(const Fixture& fx) {
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const std::uint32_t nq = static_cast<std::uint32_t>(fx.query_ids.size());
    constexpr std::uint32_t kPoolPerLeg = 100;

    // ---- Index setup. Bigram-enabled Atire and SAB serve both the plain and
    // WSDM legs; BM25+/BM25L/DLH13 only feed the rrf5 composite leg.
    simeon::Bm25Config cfg_atire;
    cfg_atire.build_word_bigrams = true;
    auto atire = build_bm25(fx, cfg_atire);
    simeon::Bm25Config cfg_sab;
    cfg_sab.variant = simeon::Bm25Variant::SubwordAwareBackoff;
    cfg_sab.delta = 1.0f;
    cfg_sab.subword_gamma = 5.0f;
    cfg_sab.build_word_bigrams = true;
    auto sab = build_bm25(fx, cfg_sab);
    simeon::Bm25Config cfg_plus;
    cfg_plus.variant = simeon::Bm25Variant::BM25Plus;
    auto plus = build_bm25(fx, cfg_plus);
    simeon::Bm25Config cfg_l;
    cfg_l.variant = simeon::Bm25Variant::BM25L;
    auto bl = build_bm25(fx, cfg_l);
    simeon::Bm25Config cfg_dlh;
    cfg_dlh.variant = simeon::Bm25Variant::DLH13;
    auto dlh = build_bm25(fx, cfg_dlh);
    const std::array<const simeon::Bm25Index*, 5> rrf5_set{&atire.idx, &plus.idx, &bl.idx, &dlh.idx,
                                                           &sab.idx};
    simeon::WeightedSdmConfig wsdm_cfg; // beta = 1.0 (published recipe)

    // ---- Fragment-geometry leg (pure: alpha = 0 -> z(geometry) only),
    // richcov + PHSS approx production recipe.
    std::vector<std::string_view> seed_views;
    seed_views.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        seed_views.emplace_back(d);
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
    std::vector<std::vector<SemanticFragment>> frags(nd);
    for (std::size_t i = 0; i < nd; ++i) {
        const auto prep0 = simeon::prepare_doc(fx.doc_texts[i], atire.idx, 6, 8, 0.0f);
        frags[i] = simeon::build_doc_semantic_fragments_rich_covered_from_prep(enc, fx.doc_texts[i],
                                                                               prep0, 0.60f, 0.80f);
    }
    simeon::compress_fragments_to_bf16(frags, enc.output_dim());
    GeometryGraphConfig gcfg{.pool_size = 100,
                             .alpha = 0.0f,
                             .top_fragments_per_doc = 8,
                             .attention_scale = 8.0f,
                             .knn = 8,
                             .steps = 2,
                             .use_phss = true,
                             .phss_config = simeon::PhssConfig{
                                 .criterion = simeon::PhssConfig::Criterion::LargestGapApprox}};

    // ---- Per-query routing signals for the soft-alpha rows.
    simeon::QueryRouter router(atire.idx);
    const std::array<const simeon::Bm25Index*, 2> sig_pools{&atire.idx, &sab.idx};

    enum Leg : std::uint8_t { L_ATIRE = 0, L_WSDM_AT, L_SAB, L_WSDM_SAB, L_GEOM, L_RRF5, N_LEGS };
    static constexpr const char* kLegName[N_LEGS] = {"atire",   "wsdmat", "sab",
                                                     "wsdmsab", "geom",   "rrf5"};

    // ---- Row table. Kind selects the combiner in the inner loop.
    struct Row {
        std::string name;
        std::array<float, N_LEGS> w{}; // convex weights for Cc; participation mask otherwise
        enum Kind : std::uint8_t { Cc, Rrf, CombSum, CombMnz, Oracle, SigAlpha, CcPrf } kind = Cc;
        // SigAlpha: blend legs a/b with alpha(q) interpolated on signal s.
        std::uint8_t leg_a = 0, leg_b = 0;
        bool use_nqc = false; // false = simplified_clarity
        float alpha_lo = 0.0f, alpha_hi = 1.0f;
    };
    std::vector<Row> rows;
    const auto add_cc = [&](std::initializer_list<std::pair<Leg, float>> ws) {
        Row r;
        std::string nm = "fusion_cc";
        for (const auto& [leg, w] : ws) {
            r.w[leg] = w;
            char buf[40];
            std::snprintf(buf, sizeof(buf), "_%s%.2f", kLegName[leg], static_cast<double>(w));
            nm += buf;
        }
        r.name = std::move(nm);
        rows.push_back(std::move(r));
    };
    // 2-leg alpha grids.
    const std::array<std::pair<Leg, Leg>, 8> cc_pairs{{{L_ATIRE, L_WSDM_AT},
                                                       {L_WSDM_AT, L_SAB},
                                                       {L_ATIRE, L_SAB},
                                                       {L_WSDM_AT, L_GEOM},
                                                       {L_SAB, L_GEOM},
                                                       {L_RRF5, L_GEOM},
                                                       {L_WSDM_SAB, L_GEOM},
                                                       {L_WSDM_SAB, L_WSDM_AT}}};
    for (const auto& [a, b] : cc_pairs)
        for (int i = 1; i <= 9; ++i)
            add_cc({{a, static_cast<float>(i) / 10.0f}, {b, 1.0f - static_cast<float>(i) / 10.0f}});
    // 3-leg simplex points (strictly interior; vertices/edges covered above).
    const std::array<std::array<Leg, 3>, 3> cc_triples{{{L_WSDM_AT, L_SAB, L_GEOM},
                                                        {L_ATIRE, L_WSDM_AT, L_GEOM},
                                                        {L_WSDM_SAB, L_WSDM_AT, L_GEOM}}};
    for (const auto& tr : cc_triples) {
        for (float wa : {0.25f, 0.5f}) {
            for (float wb : {0.25f, 0.5f}) {
                const float wc = 1.0f - wa - wb;
                if (wc <= 0.01f)
                    continue;
                add_cc({{tr[0], wa}, {tr[1], wb}, {tr[2], wc}});
            }
        }
    }
    // Baselines over identical leg sets.
    const auto add_baseline = [&](Row::Kind kind, const char* kname,
                                  std::initializer_list<Leg> legs) {
        Row r;
        r.kind = kind;
        std::string nm = std::string("fusion_") + kname;
        for (Leg leg : legs) {
            r.w[leg] = 1.0f;
            nm += std::string("_") + kLegName[leg];
        }
        r.name = std::move(nm);
        rows.push_back(std::move(r));
    };
    for (auto kind : {Row::Rrf, Row::CombSum, Row::CombMnz}) {
        const char* kname =
            kind == Row::Rrf ? "rrf" : (kind == Row::CombSum ? "combsum" : "combmnz");
        add_baseline(kind, kname, {L_ATIRE, L_WSDM_AT, L_SAB});
        add_baseline(kind, kname, {L_WSDM_AT, L_WSDM_SAB, L_GEOM});
    }
    add_baseline(Row::Oracle, "pool_oracle",
                 {L_ATIRE, L_WSDM_AT, L_SAB, L_WSDM_SAB, L_GEOM, L_RRF5});
    // Signal-conditioned soft alpha on the three most promising pairs.
    {
        const std::array<std::pair<Leg, Leg>, 3> sig_pairs{
            {{L_WSDM_AT, L_SAB}, {L_WSDM_AT, L_GEOM}, {L_WSDM_SAB, L_GEOM}}};
        for (const auto& [a, b] : sig_pairs) {
            for (bool use_nqc : {false, true}) {
                for (float lo : {0.2f, 0.4f, 0.6f}) {
                    for (float hi : {0.6f, 0.8f, 1.0f}) {
                        if (hi <= lo)
                            continue;
                        Row r;
                        r.kind = Row::SigAlpha;
                        r.leg_a = a;
                        r.leg_b = b;
                        r.use_nqc = use_nqc;
                        r.alpha_lo = lo;
                        r.alpha_hi = hi;
                        char buf[96];
                        std::snprintf(buf, sizeof(buf), "fusion_sig%s_%s_%s_a%.1f_%.1f",
                                      use_nqc ? "nqc" : "clar", kLegName[a], kLegName[b],
                                      static_cast<double>(lo), static_cast<double>(hi));
                        r.name = buf;
                        rows.push_back(std::move(r));
                    }
                }
            }
        }
    }
    // Promoted fusion ⊕ z(prf_fused) blend rows (RM3 anchored on the promoted
    // fusion's top-10; the feature itself is computed in the query loop).
    for (float w : {0.1f, 0.2f, 0.3f, 0.4f, 0.5f}) {
        Row r;
        r.kind = Row::CcPrf;
        r.alpha_lo = w; // reused as the prf_fused weight
        char buf[80];
        std::snprintf(buf, sizeof(buf), "fusion_ccprf_wsdmsab0.60_wsdmat0.40_pf%.2f",
                      static_cast<double>(w));
        r.name = buf;
        rows.push_back(std::move(r));
    }
    std::fprintf(stderr, "[fusion] rows=%zu queries=%u docs=%u\n", rows.size(), nq, nd);

    // ---- Signal anchors: SIMEON_FUSION_ANCHORS="clar_lo,clar_hi,nqc_lo,nqc_hi"
    // (computed on dev, passed frozen to the test run). Unset -> quantiles of
    // the current run (only legitimate on the dev fold).
    std::vector<float> sig_clarity(nq, 0.0f), sig_nqc(nq, 0.0f);
    for (std::uint32_t qi = 0; qi < nq; ++qi) {
        const auto f = router.features_with_pool(
            fx.query_texts[qi], std::span<const simeon::Bm25Index* const>(sig_pools), 50);
        sig_clarity[qi] = f.simplified_clarity;
        sig_nqc[qi] = f.nqc;
    }
    float anchors[4]; // clar_lo, clar_hi, nqc_lo, nqc_hi
    const char* anchors_env = std::getenv("SIMEON_FUSION_ANCHORS");
    if (anchors_env != nullptr && std::sscanf(anchors_env, "%f,%f,%f,%f", &anchors[0], &anchors[1],
                                              &anchors[2], &anchors[3]) == 4) {
        std::fprintf(stderr, "[fusion] anchors (frozen): %.4f,%.4f,%.4f,%.4f\n", anchors[0],
                     anchors[1], anchors[2], anchors[3]);
    } else {
        const auto quant = [](std::vector<float> v, double q) {
            std::sort(v.begin(), v.end());
            return v.empty() ? 0.0f : v[static_cast<std::size_t>(q * (v.size() - 1))];
        };
        anchors[0] = quant(sig_clarity, 0.25);
        anchors[1] = quant(sig_clarity, 0.75);
        anchors[2] = quant(sig_nqc, 0.25);
        anchors[3] = quant(sig_nqc, 0.75);
        std::fprintf(stderr,
                     "[fusion] anchors (computed; freeze for test): "
                     "SIMEON_FUSION_ANCHORS=%.4f,%.4f,%.4f,%.4f\n",
                     anchors[0], anchors[1], anchors[2], anchors[3]);
    }

    // ---- Main loop: legs once per query, all rows in the inner loop.
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    // Optional rerank-workbench dump: one JSONL line per query with the union
    // pool, each leg's raw scores over the pool (-inf -> -1e30 sentinel), the
    // QPP signals, per-pool-doc relevance grades, and the query's full grade
    // multiset (for offline IDCG), enabling out-of-tree rerank iteration
    // without re-running leg setup. nDCG semantics match score_rankings_full.
    FILE* wb_fp = nullptr;
    if (const char* wb_path = std::getenv("SIMEON_WORKBENCH_OUT")) {
        wb_fp = std::fopen(wb_path, "w");
        if (wb_fp == nullptr)
            std::fprintf(stderr, "[fusion] cannot open workbench path %s\n", wb_path);
    }

    // Feature legs (workbench-only; scored over the fixed 6-leg pool, never
    // pool contributors): RM3 over the Atire first pass, RM3 anchored on the
    // promoted fusion's top-10 (stronger pseudo-relevant set), and the four
    // fragment-aggregation doc scorers (SPLATE-style outer MaxSim family).
    enum Feat : std::uint8_t {
        F_PRF_AT = 0,
        F_PRF_FUSED,
        F_MAXSIM,
        F_TKM3,
        F_SMS,
        F_GMEAN,
        F_PRF_ITER2,
        F_PASSAGE,
        F_PROX,
        N_FEATS
    };
    static constexpr const char* kFeatName[N_FEATS] = {"prf_at",    "prf_fused",   "maxsim",
                                                       "tkm3",      "sms",         "gmean",
                                                       "prf_iter2", "passage_w50", "prox_pair"};

    // Lazy doc-token cache for the text-evidence features (passage windows,
    // proximity); tokenization matches the index (word-only + hash_term).
    std::vector<std::vector<std::uint64_t>> doc_tok(nd);
    std::vector<std::uint8_t> doc_tok_ready(nd, 0u);
    struct HashOnlySink final : simeon::NGramEmitter {
        std::vector<std::uint64_t>* out = nullptr;
        const simeon::Bm25Index* hidx = nullptr;
        void on_token(std::string_view tok, float) override {
            out->push_back(hidx->hash_term(tok));
        }
    };
    const auto ensure_doc_tokens = [&](std::uint32_t did) -> const std::vector<std::uint64_t>& {
        if (!doc_tok_ready[did]) {
            HashOnlySink sink;
            sink.out = &doc_tok[did];
            sink.hidx = &atire.idx;
            simeon::tokenize(fx.doc_texts[did], simeon::TokenizerConfig{0, 0, false, true}, sink);
            doc_tok_ready[did] = 1u;
        }
        return doc_tok[did];
    };
    std::array<GeometryGraphConfig, 4> ms_cfgs;
    for (auto& c : ms_cfgs) {
        c = GeometryGraphConfig{.pool_size = 100,
                                .alpha = 0.0f,
                                .top_fragments_per_doc = 8,
                                .attention_scale = 8.0f,
                                .knn = 8,
                                .steps = 2,
                                .use_phss = false,
                                .outer_maxsim = true};
    }
    using DK = simeon::FragmentGeometryConfig::DocScorerKind;
    ms_cfgs[0].doc_scorer_kind = DK::MaxSim;
    ms_cfgs[1].doc_scorer_kind = DK::TopKMean;
    ms_cfgs[1].doc_scorer_top_k = 3;
    ms_cfgs[2].doc_scorer_kind = DK::SoftMaxSum;
    ms_cfgs[2].doc_scorer_softmax_beta = 4.0f;
    ms_cfgs[3].doc_scorer_kind = DK::GeoMean;

    std::vector<std::vector<std::vector<std::pair<float, std::uint32_t>>>> row_rankings(
        rows.size(), std::vector<std::vector<std::pair<float, std::uint32_t>>>(nq));
    std::array<std::vector<float>, N_LEGS> leg_scores;
    for (auto& v : leg_scores)
        v.assign(nd, 0.0f);
    std::array<std::vector<std::pair<std::uint32_t, float>>, N_LEGS> leg_topk;
    Timing t;
    t.index_build_us = atire.build_us + sab.build_us + plus.build_us + bl.build_us + dlh.build_us;
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < nq; ++qi) {
        const auto& query = fx.query_texts[qi];
        atire.idx.score(query, leg_scores[L_ATIRE]);
        atire.idx.score_wsdm(query, leg_scores[L_WSDM_AT], wsdm_cfg);
        sab.idx.score(query, leg_scores[L_SAB]);
        sab.idx.score_wsdm(query, leg_scores[L_WSDM_SAB], wsdm_cfg);
        leg_scores[L_GEOM] = simeon::score_fragment_geometry(query, atire.idx, enc, frags, gcfg);
        std::fill(leg_scores[L_RRF5].begin(), leg_scores[L_RRF5].end(), 0.0f);
        simeon::score_bm25_variants_rrf(std::span<const simeon::Bm25Index* const>(rrf5_set), query,
                                        leg_scores[L_RRF5]);

        // Union pool over per-leg top-K.
        std::vector<std::uint32_t> pool;
        {
            std::unordered_set<std::uint32_t> seen;
            for (int leg = 0; leg < N_LEGS; ++leg) {
                leg_topk[leg] = simeon::top_k(leg_scores[leg], kPoolPerLeg);
                for (const auto& [did, _s] : leg_topk[leg])
                    if (seen.insert(did).second)
                        pool.push_back(did);
            }
        }
        const std::size_t np = pool.size();

        // Pool-restricted z-normalization per leg. The geometry leg's -inf
        // (outside its internal BM25 pool) is floored at the leg's pool
        // minimum finite score first.
        std::array<std::vector<float>, N_LEGS> z;
        for (int leg = 0; leg < N_LEGS; ++leg) {
            auto& zl = z[leg];
            zl.resize(np);
            for (std::size_t p = 0; p < np; ++p)
                zl[p] = leg_scores[leg][pool[p]];
            if (leg == L_GEOM) {
                float mn = std::numeric_limits<float>::infinity();
                for (float v : zl)
                    if (std::isfinite(v))
                        mn = std::min(mn, v);
                if (!std::isfinite(mn))
                    mn = 0.0f;
                for (float& v : zl)
                    if (!std::isfinite(v))
                        v = mn;
            }
            zscore_inplace(zl);
        }

        // prf_fused: RM3 anchored on the promoted fusion's top-10 over the
        // pool, with softmax(z) feedback weights. Feeds the CcPrf rows and the
        // workbench dump.
        std::vector<float> prf_fused_raw(np, 0.0f);
        std::vector<float> z_prf_fused;
        {
            std::vector<float> fused(np);
            for (std::size_t p = 0; p < np; ++p)
                fused[p] = 0.6f * z[L_WSDM_SAB][p] + 0.4f * z[L_WSDM_AT][p];
            std::vector<std::size_t> ord(np);
            std::iota(ord.begin(), ord.end(), std::size_t{0});
            const std::size_t kfb = std::min<std::size_t>(10, np);
            std::partial_sort(ord.begin(), ord.begin() + kfb, ord.end(),
                              [&](std::size_t a, std::size_t b) {
                                  if (fused[a] != fused[b])
                                      return fused[a] > fused[b];
                                  return pool[a] < pool[b];
                              });
            std::vector<std::uint32_t> fb_ids(kfb);
            std::vector<float> fb_w(kfb);
            const float fmax = kfb > 0 ? fused[ord[0]] : 0.0f;
            for (std::size_t i = 0; i < kfb; ++i) {
                fb_ids[i] = pool[ord[i]];
                fb_w[i] = std::exp(fused[ord[i]] - fmax);
            }
            std::vector<float> full(nd, 0.0f);
            simeon::score_with_prf(atire.idx, query, fb_ids, fb_w, full, simeon::PrfConfig{});
            for (std::size_t p = 0; p < np; ++p)
                prf_fused_raw[p] = full[pool[p]];
            z_prf_fused = prf_fused_raw;
            zscore_inplace(z_prf_fused);
        }

        if (wb_fp != nullptr) {
            const auto rit = rel.find(qi);
            std::fprintf(wb_fp, "{\"qid\":\"%s\",\"clarity\":%.6f,\"nqc\":%.6f,\"pool\":[",
                         fx.query_ids[qi].c_str(), static_cast<double>(sig_clarity[qi]),
                         static_cast<double>(sig_nqc[qi]));
            for (std::size_t p = 0; p < np; ++p)
                std::fprintf(wb_fp, "%s%u", p ? "," : "", pool[p]);
            std::fprintf(wb_fp, "],\"rel\":[");
            for (std::size_t p = 0; p < np; ++p) {
                std::uint32_t g = 0;
                if (rit != rel.end()) {
                    const auto qit = rit->second.find(pool[p]);
                    if (qit != rit->second.end())
                        g = qit->second;
                }
                std::fprintf(wb_fp, "%s%u", p ? "," : "", g);
            }
            std::fprintf(wb_fp, "],\"all_rels\":[");
            if (rit != rel.end()) {
                bool first = true;
                for (const auto& [_, g] : rit->second) {
                    std::fprintf(wb_fp, "%s%u", first ? "" : ",", g);
                    first = false;
                }
            }
            std::fprintf(wb_fp, "],\"legs\":{");
            for (int leg = 0; leg < N_LEGS; ++leg) {
                std::fprintf(wb_fp, "%s\"%s\":[", leg ? "," : "", kLegName[leg]);
                for (std::size_t p = 0; p < np; ++p) {
                    const float v = leg_scores[leg][pool[p]];
                    std::fprintf(wb_fp, "%s%.6g", p ? "," : "",
                                 std::isfinite(v) ? static_cast<double>(v) : -1e30);
                }
                std::fprintf(wb_fp, "]");
            }

            // Feature legs over the fixed pool (full-corpus scorers restricted
            // to the pool; -inf sentinel handling identical to the leg dump).
            std::array<std::vector<float>, N_FEATS> feat;
            std::vector<float> full(nd, 0.0f);
            simeon::score_with_prf(atire.idx, query, full, simeon::PrfConfig{});
            feat[F_PRF_AT].assign(np, 0.0f);
            for (std::size_t p = 0; p < np; ++p)
                feat[F_PRF_AT][p] = full[pool[p]];
            feat[F_PRF_FUSED] = prf_fused_raw;
            for (int m = 0; m < 4; ++m) {
                auto ms = simeon::score_fragment_geometry(query, atire.idx, enc, frags, ms_cfgs[m]);
                auto& f = feat[F_MAXSIM + m];
                f.assign(np, 0.0f);
                for (std::size_t p = 0; p < np; ++p)
                    f[p] = ms[pool[p]];
            }

            // prf_iter2: re-anchor RM3 on the prf_fused blend's top-10
            // (one more feedback iteration than prf_fused).
            {
                std::vector<float> blend(np);
                for (std::size_t p = 0; p < np; ++p)
                    blend[p] = 0.3f * z_prf_fused[p] +
                               0.7f * (0.6f * z[L_WSDM_SAB][p] + 0.4f * z[L_WSDM_AT][p]);
                std::vector<std::size_t> ord(np);
                std::iota(ord.begin(), ord.end(), std::size_t{0});
                const std::size_t kfb = std::min<std::size_t>(10, np);
                std::partial_sort(ord.begin(), ord.begin() + kfb, ord.end(),
                                  [&](std::size_t a, std::size_t b) {
                                      if (blend[a] != blend[b])
                                          return blend[a] > blend[b];
                                      return pool[a] < pool[b];
                                  });
                std::vector<std::uint32_t> fb_ids(kfb);
                std::vector<float> fb_w(kfb);
                const float bmax = kfb > 0 ? blend[ord[0]] : 0.0f;
                for (std::size_t i = 0; i < kfb; ++i) {
                    fb_ids[i] = pool[ord[i]];
                    fb_w[i] = std::exp(blend[ord[i]] - bmax);
                }
                std::fill(full.begin(), full.end(), 0.0f);
                simeon::score_with_prf(atire.idx, query, fb_ids, fb_w, full, simeon::PrfConfig{});
                feat[F_PRF_ITER2].assign(np, 0.0f);
                for (std::size_t p = 0; p < np; ++p)
                    feat[F_PRF_ITER2][p] = full[pool[p]];
            }

            // Text-evidence features over pool docs: passage windows (Callan
            // 1994 — max saturating-tf window score) and Tao-Zhai 2007 pair
            // proximity (exp(-min distance between distinct query terms)).
            {
                struct QTermSink final : simeon::NGramEmitter {
                    std::vector<std::pair<std::uint64_t, float>>* out = nullptr;
                    const simeon::Bm25Index* hidx = nullptr;
                    void on_token(std::string_view tok, float) override {
                        out->push_back({hidx->hash_term(tok), hidx->idf(tok)});
                    }
                };
                std::vector<std::pair<std::uint64_t, float>> q_terms;
                QTermSink qsink;
                qsink.out = &q_terms;
                qsink.hidx = &atire.idx;
                simeon::tokenize(query, simeon::TokenizerConfig{0, 0, false, true}, qsink);
                std::sort(q_terms.begin(), q_terms.end());
                q_terms.erase(std::unique(q_terms.begin(), q_terms.end()), q_terms.end());

                feat[F_PASSAGE].assign(np, 0.0f);
                feat[F_PROX].assign(np, 0.0f);
                constexpr std::size_t kWin = 50, kStride = 25;
                std::unordered_map<std::uint64_t, std::pair<float, std::size_t>> qmap;
                for (std::size_t p = 0; p < np && !q_terms.empty(); ++p) {
                    const auto& toks = ensure_doc_tokens(pool[p]);
                    // Passage: max over windows of Σ_t idf(t)·tf/(tf+1.2).
                    float best_win = 0.0f;
                    for (std::size_t w0 = 0; w0 < toks.size(); w0 += kStride) {
                        const std::size_t w1 = std::min(toks.size(), w0 + kWin);
                        qmap.clear();
                        for (const auto& [h, qidf] : q_terms)
                            qmap[h] = {qidf, 0};
                        for (std::size_t i = w0; i < w1; ++i) {
                            const auto it = qmap.find(toks[i]);
                            if (it != qmap.end())
                                ++it->second.second;
                        }
                        float s = 0.0f;
                        for (const auto& [h, iv] : qmap) {
                            const float tf = static_cast<float>(iv.second);
                            if (tf > 0.0f)
                                s += iv.first * tf / (tf + 1.2f);
                        }
                        best_win = std::max(best_win, s);
                        if (w1 == toks.size())
                            break;
                    }
                    feat[F_PASSAGE][p] = best_win;

                    // Proximity: min distance between occurrences of two
                    // DIFFERENT query terms; feature = exp(-min_dist).
                    if (q_terms.size() >= 2) {
                        qmap.clear();
                        std::size_t min_dist = SIZE_MAX;
                        for (std::size_t i = 0; i < toks.size(); ++i) {
                            const auto self =
                                std::find_if(q_terms.begin(), q_terms.end(),
                                             [&](const auto& qt) { return qt.first == toks[i]; });
                            if (self == q_terms.end())
                                continue;
                            for (const auto& [h, last] : qmap) {
                                if (h != toks[i])
                                    min_dist = std::min(min_dist, i - last.second);
                            }
                            qmap[toks[i]] = {0.0f, i};
                        }
                        if (min_dist != SIZE_MAX)
                            feat[F_PROX][p] = std::exp(-static_cast<float>(min_dist - 1) / 10.0f);
                    }
                }
            }
            std::fprintf(wb_fp, "},\"feats\":{");
            for (int fi = 0; fi < N_FEATS; ++fi) {
                std::fprintf(wb_fp, "%s\"%s\":[", fi ? "," : "", kFeatName[fi]);
                for (std::size_t p = 0; p < np; ++p) {
                    const float v = feat[fi][p];
                    std::fprintf(wb_fp, "%s%.6g", p ? "," : "",
                                 std::isfinite(v) ? static_cast<double>(v) : -1e30);
                }
                std::fprintf(wb_fp, "]");
            }
            std::fprintf(wb_fp, "}}\n");
        }

        for (std::size_t row_i = 0; row_i < rows.size(); ++row_i) {
            const Row& r = rows[row_i];
            auto& out = row_rankings[row_i][qi];
            out.reserve(np);
            switch (r.kind) {
                case Row::Cc: {
                    for (std::size_t p = 0; p < np; ++p) {
                        float s = 0.0f;
                        for (int leg = 0; leg < N_LEGS; ++leg)
                            if (r.w[leg] != 0.0f)
                                s += r.w[leg] * z[leg][p];
                        out.emplace_back(s, pool[p]);
                    }
                    break;
                }
                case Row::CcPrf: {
                    const float w = r.alpha_lo; // prf_fused weight
                    for (std::size_t p = 0; p < np; ++p) {
                        const float base = 0.6f * z[L_WSDM_SAB][p] + 0.4f * z[L_WSDM_AT][p];
                        out.emplace_back(w * z_prf_fused[p] + (1.0f - w) * base, pool[p]);
                    }
                    break;
                }
                case Row::SigAlpha: {
                    const float s_raw = r.use_nqc ? sig_nqc[qi] : sig_clarity[qi];
                    const float lo = r.use_nqc ? anchors[2] : anchors[0];
                    const float hi = r.use_nqc ? anchors[3] : anchors[1];
                    const float span = hi - lo;
                    float lam = span > 0.0f ? (s_raw - lo) / span : 0.5f;
                    lam = std::clamp(lam, 0.0f, 1.0f);
                    const float alpha = r.alpha_lo + (r.alpha_hi - r.alpha_lo) * lam;
                    for (std::size_t p = 0; p < np; ++p)
                        out.emplace_back(alpha * z[r.leg_a][p] + (1.0f - alpha) * z[r.leg_b][p],
                                         pool[p]);
                    break;
                }
                case Row::Rrf: {
                    std::vector<simeon::Ranking> rankings;
                    for (int leg = 0; leg < N_LEGS; ++leg)
                        if (r.w[leg] != 0.0f)
                            rankings.emplace_back(leg_topk[leg]);
                    auto fused = simeon::rrf_fuse(std::span<const simeon::Ranking>(rankings));
                    std::unordered_map<std::uint32_t, float> fused_map;
                    fused_map.reserve(fused.size());
                    for (const auto& [did, s] : fused)
                        fused_map[did] = s;
                    for (std::size_t p = 0; p < np; ++p) {
                        const auto it = fused_map.find(pool[p]);
                        out.emplace_back(it != fused_map.end() ? it->second : 0.0f, pool[p]);
                    }
                    break;
                }
                case Row::CombSum:
                case Row::CombMnz: {
                    // Min-max normalize each participating leg over the pool,
                    // sum; MNZ multiplies by the number of legs whose top-K
                    // contains the doc.
                    std::vector<std::uint8_t> hits(np, 0u);
                    if (r.kind == Row::CombMnz) {
                        std::unordered_map<std::uint32_t, std::size_t> pidx;
                        pidx.reserve(np);
                        for (std::size_t p = 0; p < np; ++p)
                            pidx[pool[p]] = p;
                        for (int leg = 0; leg < N_LEGS; ++leg) {
                            if (r.w[leg] == 0.0f)
                                continue;
                            for (const auto& [did, _s] : leg_topk[leg])
                                ++hits[pidx[did]];
                        }
                    }
                    std::vector<float> acc(np, 0.0f);
                    for (int leg = 0; leg < N_LEGS; ++leg) {
                        if (r.w[leg] == 0.0f)
                            continue;
                        float mn = std::numeric_limits<float>::infinity();
                        float mx = -std::numeric_limits<float>::infinity();
                        for (std::size_t p = 0; p < np; ++p) {
                            const float v = z[leg][p];
                            mn = std::min(mn, v);
                            mx = std::max(mx, v);
                        }
                        const float vspan = mx - mn;
                        if (vspan <= 0.0f)
                            continue;
                        for (std::size_t p = 0; p < np; ++p)
                            acc[p] += (z[leg][p] - mn) / vspan;
                    }
                    for (std::size_t p = 0; p < np; ++p) {
                        const float mult =
                            r.kind == Row::CombMnz ? static_cast<float>(hits[p]) : 1.0f;
                        out.emplace_back(acc[p] * mult, pool[p]);
                    }
                    break;
                }
                case Row::Oracle: {
                    const auto rit = rel.find(qi);
                    for (std::size_t p = 0; p < np; ++p) {
                        float g = 0.0f;
                        if (rit != rel.end()) {
                            const auto qit = rit->second.find(pool[p]);
                            if (qit != rit->second.end())
                                g = static_cast<float>(qit->second);
                        }
                        out.emplace_back(g, pool[p]);
                    }
                    break;
                }
            }
        }
    }
    t.query_us = elapsed_us(t0);
    if (wb_fp != nullptr)
        std::fclose(wb_fp);

    for (std::size_t row_i = 0; row_i < rows.size(); ++row_i) {
        std::vector<double> per_q;
        auto m = score_rankings_full(row_rankings[row_i], fx, &per_q);
        dump_per_query_ndcg_if_env(rows[row_i].name.c_str(), fx, per_q);
        emit(rows[row_i].name.c_str(), fx, m, 0, t);
    }
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
    double precision_at_10 = 0.0;
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
    out.precision_at_10 = static_cast<double>(hits10) / 10.0;
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
                             "\"has_relevant\":%s,\"ndcg_at_10\":%.4f,\"precision_at_10\":%.4f,"
                             "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                             "\"mrr_at_10\":%.4f}\n",
                             name.c_str(), fx.query_ids[qi].c_str(), simeon::recipe_name(recipe),
                             features.oov_rate, features.avg_idf, features.max_idf,
                             features.min_idf, features.idf_stddev, features.n_terms,
                             features.avg_term_chars, features.score_decay_rate,
                             features.score_normalized_var, features.top_k_score_entropy,
                             features.pool_overlap_jaccard, features.nqc, features.wig_full,
                             pq.has_relevant ? "true" : "false", pq.ndcg_at_10, pq.precision_at_10,
                             pq.recall_at_10, pq.recall_at_100, pq.mrr_at_10);
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
                         "\"has_relevant\":%s,\"ndcg_at_10\":%.4f,\"precision_at_10\":%.4f,"
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
                         best->m->precision_at_10, best->m->recall_at_10, best->m->recall_at_100,
                         best->m->mrr_at_10, m_atire.ndcg_at_10, m_sab.ndcg_at_10,
                         m_cascade.ndcg_at_10);
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
    bool core_only = false;
    bool generator_slices = false;
#ifdef SIMEON_RESEARCH_BENCH
    AuxFieldMode aux_mode = AuxFieldMode::None;
    bool softmatch_only = false;
    bool transport_only = false;
    bool graph_only = false;
    bool cluster_only = false;
    bool fragment_only = false;
    bool xprod_only = false;
    bool dual_only = false;
    bool fragment_quality_router_only = false;
    bool fusion_only = false;
    std::string fragment_glove_path;
#endif

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--queries-from" && i + 1 < argc) {
            queries_from = argv[++i];
            if (queries_from != "test" && queries_from != "dev") {
                std::fprintf(stderr, "--queries-from must be test|dev\n");
                return 2;
            }
        } else if (a == "--core-only") {
            core_only = true;
        } else if (a == "--generator-slices") {
            generator_slices = true;
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
        } else if (a == "--fusion-only") {
            fusion_only = true;
        } else if (a == "--xprod-only") {
            fragment_only = true;
            xprod_only = true;
        } else if (a == "--dual-only") {
            // Trec-covid-friendly subset: only the Layered dual cells.
            // 1 BM25 reindex + 1 fragment build + 12 dual cells.
            fragment_only = true;
            xprod_only = true;
            dual_only = true;
        } else if (a == "--fragment-quality-router-only") {
            fragment_quality_router_only = true;
        } else if (a == "--fragment-glove" && i + 1 < argc) {
            fragment_glove_path = argv[++i];
#endif
        } else if (a == "--router-per-query" && i + 1 < argc) {
            router_per_query_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
#ifdef SIMEON_RESEARCH_BENCH
            std::fprintf(
                stderr,
                "usage: %s [flags] <fixture_dir>\n"
                "  --queries-from {test,dev}    pick split (default test)\n"
                "  --core-only                  emit reference, BM25, and oracle K-sweep only\n"
                "  --generator-slices           add first non-BM25 generator-slice oracles\n"
                "  --aux-from {none,textrank,ac} add BM25F structural rows (default none)\n"
                "  --softmatch-only             run PMI soft-match rows only\n"
                "  --transport-only             run phrase/document transport rows only\n"
                "  --graph-only                 run graph transport rows only\n"
                "  --cluster-only               run fragment/cluster topology rows only\n"
                "  --fragment-only              run PMI fragment graph rows only\n"
                "  --fusion-only                run soft-fusion sweep rows only\n"
                "  --fragment-quality-router-only run only the quality-routed fragment row\n"
                "  --fragment-glove <path>      run GloVe/fastText fragment grid (C9)\n"
                "  --router-per-query <path>    write per-query router telemetry JSONL\n"
                "  fixture_dir expects corpus.tsv, queries[_dev].tsv,\n"
                "  qrels[_dev].tsv, reference[_dev].bin\n"
                "  see docs/reference_fixture.md\n",
                argv[0]);
#else
            std::fprintf(
                stderr,
                "usage: %s [flags] <fixture_dir>\n"
                "  --queries-from {test,dev}    pick split (default test)\n"
                "  --core-only                  emit reference, BM25, and oracle K-sweep only\n"
                "  --generator-slices           add first non-BM25 generator-slice oracles\n"
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
    // Oracle upper bound at multiple K values. The K-sweep characterizes the
    // marginal value of candidate-pool expansion: if Oracle@K=500 ≈ Oracle@K=100,
    // pool expansion can't raise the ceiling and runway is bounded by recipe
    // expressiveness, not candidate recall. If Oracle grows with K, dual-stage
    // (Phase XV) is on the right axis.
    run_oracle_pool("oracle_bm25_pool_k50", fx, bm25.idx, bm25.build_us, 50);
    run_oracle_pool("oracle_bm25_pool_k100", fx, bm25.idx, bm25.build_us, 100);
    run_oracle_pool("oracle_bm25_pool_k200", fx, bm25.idx, bm25.build_us, 200);
    run_oracle_pool("oracle_bm25_pool_k500", fx, bm25.idx, bm25.build_us, 500);

    if (generator_slices) {
        run_first_generator_slice_oracles(fx, bm25);
    }

    if (core_only) {
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    }

    // Research-only ArguAna schema diagnostics are no longer part of the default
    // reference bench. They were useful to locate the ceiling, but they encode
    // fixture-specific structure and should not be presented as shippable
    // retrieval recipes. Set SIMEON_ARGUANA_DIAGNOSTICS=1 when reproducing the
    // phase20–28 research notes.
    if (arguana_diagnostics_enabled()) {
        run_arguana_text_neighborhood_diagnostic(fx);
        run_arguana_text_pair_discriminator(fx);
        run_arguana_text_pair_ranker_devfit(fx);
        run_arguana_argument_point_diagnostic(fx);
        run_arguana_pair_id_diagnostic(fx);
    }
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
        run_bm25_fragment_graph_grid(fx, xprod_only, dual_only);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (fusion_only) {
        run_bm25_fusion_grid(fx);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (fragment_quality_router_only) {
        run_bm25_fragment_quality_router_grid(fx);
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 0;
    } else if (!fragment_glove_path.empty()) {
        run_bm25_fragment_glove_grid(fx, fragment_glove_path);
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
    auto atire_keep = bm25_variant("bm25_atire", simeon::Bm25Variant::Atire);
    auto plus_keep = bm25_variant("bm25_plus", simeon::Bm25Variant::BM25Plus);
    auto l_keep = bm25_variant("bm25_l", simeon::Bm25Variant::BM25L);
    auto dlh_keep = bm25_variant("bm25_dlh13", simeon::Bm25Variant::DLH13);
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

    // Plan 3 — RRF over 5 BM25 variants (Atire, BM25+, BM25L, DLH13, SAB-smooth).
    // Attacks Bottleneck 2 (BM25-pool R@100 ceiling) via first-pass candidate
    // diversification. Per Bruch-Gai 2023, RRF is generally sensitive to
    // parameter k; using the canonical k=60. See docs/research/next_research_plans.md.
    {
        std::array<const simeon::Bm25Index*, 5> variants{
            &atire_keep.idx, &plus_keep.idx, &l_keep.idx, &dlh_keep.idx, &sab_smooth.idx};
        const double total_build_us = atire_keep.build_us + plus_keep.build_us + l_keep.build_us +
                                      dlh_keep.build_us + sab_smooth.build_us;
        run_bm25_variants_rrf("bm25_rrf_variants5", fx,
                              std::span<const simeon::Bm25Index* const>(variants), total_build_us);
    }

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
