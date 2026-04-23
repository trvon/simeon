#include "simeon/fragment_geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "simeon/bm25.hpp"
#include "simeon/fusion.hpp"
#include "simeon/query_router.hpp"
#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"
#include "simeon/text_rank.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

// ---------------------------------------------------------------------------
// Stopwords
// ---------------------------------------------------------------------------

bool is_stopword(std::string_view token) {
    static const std::unordered_set<std::string_view> kStopwords = {
        "a",    "an",    "and",  "are", "as",  "at",   "be",   "by", "for",
        "from", "in",    "into", "is",  "it",  "of",   "on",   "or", "that",
        "the",  "their", "this", "to",  "was", "were", "with",
    };
    return kStopwords.contains(token);
}

// ---------------------------------------------------------------------------
// Tokenization helper
// ---------------------------------------------------------------------------

struct StringTokenSink final : NGramEmitter {
    std::vector<std::string>* out = nullptr;
    void on_token(std::string_view tok, float) override { out->emplace_back(tok); }
};

constexpr TokenizerConfig word_only_cfg() noexcept {
    return TokenizerConfig{0, 0, false, true};
}

void tokenize_words(std::string_view text, std::vector<std::string>& out) {
    out.clear();
    StringTokenSink sink{};
    sink.out = &out;
    tokenize(text, word_only_cfg(), sink);
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

float vector_l2_norm(const float* v, std::uint32_t dim) {
    double sq = 0.0;
    for (std::uint32_t i = 0; i < dim; ++i)
        sq += static_cast<double>(v[i]) * v[i];
    return static_cast<float>(std::sqrt(sq));
}

void l2_normalize_inplace(float* v, std::uint32_t dim) {
    const float n = vector_l2_norm(v, dim);
    if (n <= 0.0f)
        return;
    const float inv = 1.0f / n;
    for (std::uint32_t i = 0; i < dim; ++i)
        v[i] *= inv;
}

// Routes through the SIMD-dispatched dot kernel.
inline float dot(const float* a, const float* b, std::size_t n) {
    return simd::dot(a, b, static_cast<std::uint32_t>(n));
}

// ---------------------------------------------------------------------------
// Z-score and interpolation helpers
// ---------------------------------------------------------------------------

void zscore_inplace(std::vector<float>& v) {
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
}

float clamp_unit(float x) {
    if (x < 0.0f)
        return 0.0f;
    if (x > 1.0f)
        return 1.0f;
    return x;
}

float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float bm25_pool_decay(std::span<const std::pair<std::uint32_t, float>> pool) {
    if (pool.size() < 10)
        return 0.0f;
    const std::size_t i9 = pool.size() / 10;
    return clamp_unit((pool.front().second - pool[i9].second) / pool.front().second);
}

float geometry_query_confidence(std::span<const float> sims) {
    float top1 = 0.0f, top2 = 0.0f;
    for (float s : sims) {
        if (s > top1) {
            top2 = top1;
            top1 = s;
        } else if (s > top2) {
            top2 = s;
        }
    }
    const float top_conf = clamp_unit((top1 - 0.10f) / 0.25f);
    const float gap_conf = clamp_unit(((top1 - top2) - 0.02f) / 0.12f);
    float ent_norm = 0.0f;
    float sum = 0.0f;
    for (float s : sims) {
        const float p = std::max(0.0f, s);
        sum += p;
    }
    if (sum > 0.0f) {
        for (float s : sims) {
            const float p = std::max(0.0f, s) / sum;
            if (p > 0.0f)
                ent_norm -= p * std::log(p);
        }
    }
    ent_norm /= std::log(2.0f + static_cast<float>(sims.size()));
    const float ent_conf = 1.0f - clamp_unit(ent_norm);
    return clamp_unit(0.4f * top_conf + 0.35f * gap_conf + 0.25f * ent_conf);
}

// ---------------------------------------------------------------------------
// Sparse signature helpers
// ---------------------------------------------------------------------------

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

SparseSignature build_sparse_signature(const Bm25Index& idx, std::string_view text,
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

} // anonymous namespace

// ---------------------------------------------------------------------------
// Fragment builders
// ---------------------------------------------------------------------------

std::vector<SemanticFragment> build_doc_semantic_fragments(const Encoder& enc, std::string_view doc,
                                                           const Bm25Index& idx,
                                                           std::uint32_t top_sentence_fragments,
                                                           std::uint32_t fragment_signature_terms) {
    TextRank ranker;
    const auto dim = enc.output_dim();
    auto ranked = ranker.rank(doc, top_sentence_fragments);
    std::vector<SemanticFragment> out;
    out.reserve(ranked.empty() ? 1 : ranked.size());

    auto try_encode = [&](std::string_view text) {
        std::vector<float> vec(dim, 0.0f);
        enc.encode(text, vec.data());
        auto sig = build_sparse_signature(idx, text, fragment_signature_terms);
        if (vector_l2_norm(vec.data(), dim) > 0.0f && !sig.terms.empty())
            out.push_back(SemanticFragment{std::move(vec), std::move(sig)});
    };

    for (const auto& sent : ranked)
        try_encode(sent.text);
    if (out.empty())
        try_encode(doc);
    return out;
}

std::vector<SemanticFragment>
build_doc_semantic_fragments_rich(const Encoder& enc, std::string_view doc, const Bm25Index& idx,
                                  std::uint32_t top_sentence_fragments,
                                  std::uint32_t fragment_signature_terms) {
    auto out = build_doc_semantic_fragments(enc, doc, idx, top_sentence_fragments,
                                            fragment_signature_terms);
    const auto dim = enc.output_dim();
    if (out.size() >= 2) {
        std::vector<float> centroid(dim, 0.0f);
        SparseSignature merged;
        const std::uint32_t merged_terms =
            std::max<std::uint32_t>(fragment_signature_terms * 2, fragment_signature_terms + 4);
        for (const auto& frag : out) {
            for (std::uint32_t d = 0; d < dim; ++d)
                centroid[d] += frag.vec[d];
            merge_signature_max(merged, frag.signature, merged_terms);
        }
        const float inv = 1.0f / static_cast<float>(out.size());
        for (float& x : centroid)
            x *= inv;
        l2_normalize_inplace(centroid.data(), dim);
        if (vector_l2_norm(centroid.data(), dim) > 0.0f && !merged.terms.empty())
            out.push_back(SemanticFragment{std::move(centroid), std::move(merged)});
    }

    std::vector<float> doc_vec(dim, 0.0f);
    enc.encode(doc, doc_vec.data());
    auto doc_sig = build_sparse_signature(idx, doc, fragment_signature_terms * 2);
    if (vector_l2_norm(doc_vec.data(), dim) > 0.0f && !doc_sig.terms.empty())
        out.push_back(SemanticFragment{std::move(doc_vec), std::move(doc_sig)});
    return out;
}

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_covered(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap) {
    auto rich = build_doc_semantic_fragments_rich(enc, doc, idx, top_sentence_fragments,
                                                  fragment_signature_terms);
    if (rich.empty())
        return rich;

    std::vector<SemanticFragment> out;
    out.reserve(rich.size());
    const std::size_t base_count = rich.size() >= 2 ? rich.size() - 2 : rich.size();
    auto append_if_novel = [&](SemanticFragment frag, float overlap_cap) {
        float max_ov = 0.0f;
        for (const auto& kept : out)
            max_ov = std::max(max_ov, weighted_overlap_coeff(frag.signature, kept.signature));
        if (max_ov <= overlap_cap || out.empty())
            out.push_back(std::move(frag));
    };

    for (std::size_t i = 0; i < base_count; ++i)
        append_if_novel(std::move(rich[i]), sentence_overlap_cap);
    for (std::size_t i = base_count; i < rich.size(); ++i)
        append_if_novel(std::move(rich[i]), anchor_overlap_cap);

    if (out.empty()) {
        if (base_count > 0)
            out.push_back(std::move(rich[0]));
        else
            out.push_back(std::move(rich.back()));
    }
    return out;
}

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_mmr(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float redundancy_lambda,
    float sentence_min_score, float anchor_min_score) {
    struct Candidate {
        SemanticFragment frag;
        float salience = 0.0f;
        float overlap_cap = 1.0f;
        float min_score = 0.0f;
    };

    auto rich = build_doc_semantic_fragments_rich(enc, doc, idx, top_sentence_fragments,
                                                  fragment_signature_terms);
    if (rich.empty())
        return rich;

    std::vector<Candidate> candidates;
    candidates.reserve(rich.size());
    const std::size_t base_count = rich.size() >= 2 ? rich.size() - 2 : rich.size();
    const float sent_step = base_count > 1 ? 0.35f / static_cast<float>(base_count - 1) : 0.0f;
    for (std::size_t i = 0; i < base_count; ++i) {
        candidates.push_back(Candidate{.frag = std::move(rich[i]),
                                       .salience = 1.0f - sent_step * static_cast<float>(i),
                                       .overlap_cap = sentence_overlap_cap,
                                       .min_score = sentence_min_score});
    }
    if (base_count < rich.size()) {
        candidates.push_back(Candidate{.frag = std::move(rich[base_count]),
                                       .salience = 0.72f,
                                       .overlap_cap = anchor_overlap_cap,
                                       .min_score = anchor_min_score});
    }
    if (base_count + 1 < rich.size()) {
        candidates.push_back(Candidate{.frag = std::move(rich[base_count + 1]),
                                       .salience = 0.60f,
                                       .overlap_cap = anchor_overlap_cap,
                                       .min_score = anchor_min_score});
    }

    std::vector<SemanticFragment> out;
    out.reserve(candidates.size());
    while (!candidates.empty()) {
        std::size_t best_idx = candidates.size();
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::size_t ci = 0; ci < candidates.size(); ++ci) {
            float max_ov = 0.0f;
            for (const auto& kept : out)
                max_ov = std::max(
                    max_ov, weighted_overlap_coeff(candidates[ci].frag.signature, kept.signature));
            float score = candidates[ci].salience - redundancy_lambda * max_ov;
            if (max_ov > candidates[ci].overlap_cap)
                score -= (max_ov - candidates[ci].overlap_cap);
            if (score > best_score) {
                best_score = score;
                best_idx = ci;
            }
        }
        if (best_idx == candidates.size() || best_score < candidates[best_idx].min_score)
            break;
        out.push_back(std::move(candidates[best_idx].frag));
        candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(best_idx));
    }
    if (out.empty() && !rich.empty())
        out.push_back(std::move(rich[0]));
    return out;
}

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_asymmetric(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float anchor_redundancy_lambda,
    float anchor_min_score) {
    auto rich = build_doc_semantic_fragments_rich(enc, doc, idx, top_sentence_fragments,
                                                  fragment_signature_terms);
    if (rich.empty())
        return rich;

    std::vector<SemanticFragment> out;
    out.reserve(rich.size());
    const std::size_t base_count = rich.size() >= 2 ? rich.size() - 2 : rich.size();

    auto append_if_novel = [&](SemanticFragment frag, float overlap_cap) {
        float max_ov = 0.0f;
        for (const auto& kept : out)
            max_ov = std::max(max_ov, weighted_overlap_coeff(frag.signature, kept.signature));
        if (max_ov <= overlap_cap || out.empty())
            out.push_back(std::move(frag));
    };

    for (std::size_t i = 0; i < base_count; ++i)
        append_if_novel(std::move(rich[i]), sentence_overlap_cap);

    struct AnchorCandidate {
        SemanticFragment frag;
        float salience = 0.0f;
    };
    std::vector<AnchorCandidate> anchors;
    anchors.reserve(rich.size() - base_count);
    if (base_count < rich.size()) {
        anchors.push_back(AnchorCandidate{std::move(rich[base_count]), 0.72f});
    }
    if (base_count + 1 < rich.size()) {
        anchors.push_back(AnchorCandidate{std::move(rich[base_count + 1]), 0.60f});
    }

    while (!anchors.empty()) {
        std::size_t best_idx = anchors.size();
        float best_score = -std::numeric_limits<float>::infinity();
        for (std::size_t ci = 0; ci < anchors.size(); ++ci) {
            float max_ov = 0.0f;
            for (const auto& kept : out)
                max_ov = std::max(
                    max_ov, weighted_overlap_coeff(anchors[ci].frag.signature, kept.signature));
            float score = anchors[ci].salience - anchor_redundancy_lambda * max_ov;
            if (max_ov > anchor_overlap_cap)
                score -= (max_ov - anchor_overlap_cap);
            if (score > best_score) {
                best_score = score;
                best_idx = ci;
            }
        }
        if (best_idx == anchors.size() || best_score < anchor_min_score)
            break;
        out.push_back(std::move(anchors[best_idx].frag));
        anchors.erase(anchors.begin() + static_cast<std::ptrdiff_t>(best_idx));
    }

    if (out.empty()) {
        if (base_count > 0)
            out.push_back(std::move(rich[0]));
        else
            out.push_back(std::move(rich.back()));
    }
    return out;
}

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_budgeted(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, std::uint32_t max_sentence_keep, float anchor_overlap_cap,
    float anchor_novelty_floor, std::uint32_t max_anchor_keep) {
    auto rich = build_doc_semantic_fragments_rich(enc, doc, idx, top_sentence_fragments,
                                                  fragment_signature_terms);
    if (rich.empty())
        return rich;

    std::vector<SemanticFragment> out;
    out.reserve(rich.size());
    const std::size_t base_count = rich.size() >= 2 ? rich.size() - 2 : rich.size();

    auto max_overlap_with_kept = [&](const SparseSignature& sig) {
        float max_ov = 0.0f;
        for (const auto& kept : out)
            max_ov = std::max(max_ov, weighted_overlap_coeff(sig, kept.signature));
        return max_ov;
    };

    std::uint32_t kept_sentences = 0;
    for (std::size_t i = 0; i < base_count; ++i) {
        if (kept_sentences >= max_sentence_keep)
            break;
        const float max_ov = max_overlap_with_kept(rich[i].signature);
        if (out.empty() || max_ov <= sentence_overlap_cap) {
            out.push_back(std::move(rich[i]));
            ++kept_sentences;
        }
    }

    struct AnchorCandidate {
        SemanticFragment frag;
        float novelty = 0.0f;
    };
    std::vector<AnchorCandidate> anchors;
    anchors.reserve(rich.size() - base_count);
    for (std::size_t i = base_count; i < rich.size(); ++i) {
        const float max_ov = max_overlap_with_kept(rich[i].signature);
        const float novelty = 1.0f - max_ov;
        if (max_ov <= anchor_overlap_cap && novelty >= anchor_novelty_floor)
            anchors.push_back(AnchorCandidate{std::move(rich[i]), novelty});
    }
    std::sort(anchors.begin(), anchors.end(),
              [](const auto& a, const auto& b) { return a.novelty > b.novelty; });
    for (std::size_t i = 0; i < anchors.size() && i < max_anchor_keep; ++i)
        out.push_back(std::move(anchors[i].frag));

    if (out.empty() && !rich.empty())
        out.push_back(std::move(rich[0]));
    return out;
}

// ---------------------------------------------------------------------------
// Query scoring
// ---------------------------------------------------------------------------

std::vector<float> score_fragment_geometry(std::string_view query, const Bm25Index& idx,
                                           const Encoder& enc,
                                           std::span<const std::vector<SemanticFragment>> doc_frags,
                                           const FragmentGeometryConfig& cfg) {
    return score_fragment_geometry_profiled(query, idx, enc, doc_frags, cfg, nullptr);
}

std::vector<float>
score_fragment_geometry_profiled(std::string_view query, const Bm25Index& idx, const Encoder& enc,
                                 std::span<const std::vector<SemanticFragment>> doc_frags,
                                 const FragmentGeometryConfig& cfg,
                                 FragmentGeometryProfile* profile) {
    const auto t_total0 = profile ? Clock::now() : Clock::time_point{};
    if (profile)
        *profile = FragmentGeometryProfile{};

    const std::uint32_t nd = static_cast<std::uint32_t>(doc_frags.size());
    const std::uint32_t dim = enc.output_dim();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<float> qvec(dim, 0.0f);
    const QueryRouter router(idx);

    const auto t_bm250 = profile ? Clock::now() : Clock::time_point{};
    idx.score(query, bm25_scores);
    auto pool = top_k(bm25_scores, cfg.pool_size);
    const auto t_bm251 = profile ? Clock::now() : Clock::time_point{};
    if (profile) {
        profile->bm25_us = elapsed_us(t_bm250, t_bm251);
        profile->pool_docs = static_cast<std::uint32_t>(pool.size());
    }

    const auto t_qenc0 = profile ? Clock::now() : Clock::time_point{};
    enc.encode(query, qvec.data());
    const auto t_qenc1 = profile ? Clock::now() : Clock::time_point{};
    if (profile)
        profile->query_encode_us = elapsed_us(t_qenc0, t_qenc1);

    float query_alpha = cfg.alpha;
    float query_scale = cfg.attention_scale;
    std::uint32_t query_knn = cfg.knn;
    std::uint32_t query_steps = cfg.steps;
    if (cfg.adaptive) {
        const auto f = router.features(query);
        const float idf_t = cfg.adaptive_idf_hi > cfg.adaptive_idf_lo
                                ? clamp_unit((f.avg_idf - cfg.adaptive_idf_lo) /
                                             (cfg.adaptive_idf_hi - cfg.adaptive_idf_lo))
                                : 1.0f;
        const float decay_t = cfg.adaptive_decay_hi > cfg.adaptive_decay_lo
                                  ? clamp_unit((bm25_pool_decay(pool) - cfg.adaptive_decay_lo) /
                                               (cfg.adaptive_decay_hi - cfg.adaptive_decay_lo))
                                  : 1.0f;
        const float ambiguity = 1.0f - 0.5f * (idf_t + decay_t);
        query_alpha = lerp(cfg.adaptive_alpha_hi, cfg.adaptive_alpha_lo, ambiguity);
        query_scale = lerp(cfg.adaptive_scale_hi, cfg.adaptive_scale_lo, ambiguity);
        query_knn = static_cast<std::uint32_t>(
            std::lround(lerp(static_cast<float>(cfg.adaptive_knn_lo),
                             static_cast<float>(cfg.adaptive_knn_hi), ambiguity)));
        query_steps = static_cast<std::uint32_t>(
            std::lround(lerp(static_cast<float>(cfg.adaptive_steps_lo),
                             static_cast<float>(cfg.adaptive_steps_hi), ambiguity)));
    }

    std::vector<float> scores(nd, neg_inf);
    for (const auto& [did, sc] : pool)
        scores[did] = sc;

    if (pool.empty() || vector_l2_norm(qvec.data(), dim) <= 0.0f) {
        if (profile)
            profile->total_us = elapsed_us(t_total0, Clock::now());
        return scores;
    }

    struct FragRef {
        std::uint32_t pool_index = 0;
        const float* vec = nullptr;
    };

    const auto t_gather0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<FragRef> frags;
    frags.reserve(pool.size() * cfg.top_fragments_per_doc);
    for (std::size_t pi = 0; pi < pool.size(); ++pi) {
        const auto did = pool[pi].first;
        const auto limit = std::min<std::size_t>(cfg.top_fragments_per_doc, doc_frags[did].size());
        for (std::size_t fi = 0; fi < limit; ++fi)
            frags.push_back(FragRef{.pool_index = static_cast<std::uint32_t>(pi),
                                    .vec = doc_frags[did][fi].vec.data()});
    }
    if (frags.empty()) {
        if (profile) {
            profile->gather_us = elapsed_us(t_gather0, Clock::now());
            profile->pool_fragments = 0;
            profile->total_us = elapsed_us(t_total0, Clock::now());
        }
        return scores;
    }
    if (profile) {
        profile->gather_us = elapsed_us(t_gather0, Clock::now());
        profile->pool_fragments = static_cast<std::uint32_t>(frags.size());
    }

    const auto t_white0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> mean(dim, 0.0f), var(dim, 0.0f);
    for (const auto& frag : frags) {
        for (std::uint32_t d = 0; d < dim; ++d)
            mean[d] += frag.vec[d];
    }
    const float inv_n = 1.0f / static_cast<float>(frags.size());
    for (std::uint32_t d = 0; d < dim; ++d)
        mean[d] *= inv_n;
    for (const auto& frag : frags) {
        for (std::uint32_t d = 0; d < dim; ++d) {
            const float x = frag.vec[d] - mean[d];
            var[d] += x * x;
        }
    }
    for (std::uint32_t d = 0; d < dim; ++d)
        var[d] = std::sqrt(var[d] * inv_n + 1e-6f);

    std::vector<float> wq = qvec;
    for (std::uint32_t d = 0; d < dim; ++d)
        wq[d] = (wq[d] - mean[d]) / var[d];
    l2_normalize_inplace(wq.data(), dim);
    if (vector_l2_norm(wq.data(), dim) <= 0.0f) {
        if (profile) {
            profile->whiten_us = elapsed_us(t_white0, Clock::now());
            profile->total_us = elapsed_us(t_total0, Clock::now());
        }
        return scores;
    }

    std::vector<float> fvecs(frags.size() * dim, 0.0f);
    for (std::size_t i = 0; i < frags.size(); ++i) {
        float* dst = fvecs.data() + i * dim;
        for (std::uint32_t d = 0; d < dim; ++d)
            dst[d] = (frags[i].vec[d] - mean[d]) / var[d];
        l2_normalize_inplace(dst, dim);
    }
    if (profile)
        profile->whiten_us = elapsed_us(t_white0, Clock::now());

    const auto t_qatt0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> qsim(frags.size(), 0.0f);
    for (std::size_t i = 0; i < frags.size(); ++i)
        qsim[i] = dot(wq.data(), fvecs.data() + i * dim, dim);

    float phss_selected_scale = 0.0f;
    bool use_phss_for_graph = false;
    const float query_conf = (cfg.geometry_signal_adaptive || cfg.phss_adaptive)
                                 ? geometry_query_confidence(qsim)
                                 : 0.0f;
    if (cfg.use_phss) {
        if (profile)
            profile->phss_enabled = true;
        const bool should_run_phss =
            !cfg.phss_adaptive || query_conf >= cfg.phss_confidence_threshold;
        if (should_run_phss) {
            const std::uint32_t nf = static_cast<std::uint32_t>(frags.size());
            const std::uint32_t tri_count = nf * (nf - 1) / 2;
            const auto t_pair0 = profile ? Clock::now() : Clock::time_point{};
            std::vector<float> sims_tri;
            sims_tri.reserve(tri_count);
            for (std::uint32_t i = 0; i < nf; ++i) {
                for (std::uint32_t j = i + 1; j < nf; ++j) {
                    sims_tri.push_back(dot(fvecs.data() + i * dim, fvecs.data() + j * dim, dim));
                }
            }
            const auto t_pair1 = profile ? Clock::now() : Clock::time_point{};
            if (profile)
                profile->phss_pairwise_us = elapsed_us(t_pair0, t_pair1);
            PhssConfig phss_cfg = cfg.phss_config;
            phss_cfg.dim_max = 0;
            const auto t_sel0 = profile ? Clock::now() : Clock::time_point{};
            auto phss_result = phss_select_scale(sims_tri, nf, phss_cfg);
            const auto t_sel1 = profile ? Clock::now() : Clock::time_point{};
            if (profile)
                profile->phss_select_us = elapsed_us(t_sel0, t_sel1);
            phss_selected_scale = phss_result.selected_scale;
            use_phss_for_graph = (phss_selected_scale > 0.0f);
            if (profile)
                profile->phss_selected_scale = phss_selected_scale;
        }
    }

    if (cfg.geometry_signal_adaptive) {
        query_alpha = lerp(cfg.geometry_alpha_hi, cfg.geometry_alpha_lo, query_conf);
        query_scale = lerp(cfg.geometry_scale_hi, cfg.geometry_scale_lo, query_conf);
        query_knn = static_cast<std::uint32_t>(
            std::lround(lerp(static_cast<float>(cfg.geometry_knn_hi),
                             static_cast<float>(cfg.geometry_knn_lo), query_conf)));
        query_steps = static_cast<std::uint32_t>(
            std::lround(lerp(static_cast<float>(cfg.geometry_steps_lo),
                             static_cast<float>(cfg.geometry_steps_hi), query_conf)));
        if (profile)
            profile->query_confidence = query_conf;
    } else if (cfg.phss_adaptive) {
        if (profile)
            profile->query_confidence = query_conf;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < frags.size(); ++i)
        max_logit = std::max(max_logit, query_scale * qsim[i]);

    std::vector<float> mass(frags.size(), 0.0f);
    float mass_sum = 0.0f;
    for (std::size_t i = 0; i < frags.size(); ++i) {
        mass[i] = std::exp(query_scale * qsim[i] - max_logit);
        mass_sum += mass[i];
    }
    if (profile) {
        profile->query_attention_us = elapsed_us(t_qatt0, Clock::now());
        profile->phss_used = use_phss_for_graph;
    }
    if (mass_sum <= 0.0f) {
        if (profile)
            profile->total_us = elapsed_us(t_total0, Clock::now());
        return scores;
    }
    for (float& x : mass)
        x /= mass_sum;

    const auto t_adj0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<std::vector<std::pair<std::uint32_t, float>>> adj(frags.size());
    std::vector<float> ns(frags.size(), 0.0f);
    std::uint64_t graph_edges = 0;

    for (std::size_t i = 0; i < frags.size(); ++i) {
        std::vector<std::pair<float, std::uint32_t>> sims;
        sims.reserve(frags.size() > 0 ? frags.size() - 1 : 0);
        for (std::size_t j = 0; j < frags.size(); ++j) {
            if (i == j)
                continue;
            const float sim = dot(fvecs.data() + i * dim, fvecs.data() + j * dim, dim);
            if (use_phss_for_graph) {
                if (sim >= phss_selected_scale)
                    sims.emplace_back(sim, static_cast<std::uint32_t>(j));
            } else {
                sims.emplace_back(sim, static_cast<std::uint32_t>(j));
            }
        }

        if (use_phss_for_graph) {
            if (sims.empty())
                continue;
            float row_max = -std::numeric_limits<float>::infinity();
            for (const auto& [sim, _] : sims)
                row_max = std::max(row_max, query_scale * sim);
            float row_sum = 0.0f;
            adj[i].reserve(sims.size());
            for (const auto& [sim, j] : sims) {
                const float w = std::exp(query_scale * sim - row_max);
                adj[i].push_back({j, w});
                row_sum += w;
            }
            if (row_sum > 0.0f) {
                for (auto& [j, w] : adj[i])
                    w /= row_sum;
            }
            graph_edges += adj[i].size();
        } else {
            std::partial_sort(
                sims.begin(), sims.begin() + std::min<std::size_t>(query_knn, sims.size()),
                sims.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            const std::size_t keep = std::min<std::size_t>(query_knn, sims.size());
            if (keep == 0)
                continue;
            float row_max = -std::numeric_limits<float>::infinity();
            for (std::size_t k = 0; k < keep; ++k)
                row_max = std::max(row_max, query_scale * sims[k].first);
            float row_sum = 0.0f;
            adj[i].reserve(keep);
            for (std::size_t k = 0; k < keep; ++k) {
                const float w = std::exp(query_scale * sims[k].first - row_max);
                adj[i].push_back({sims[k].second, w});
                row_sum += w;
            }
            if (row_sum > 0.0f) {
                for (auto& [j, w] : adj[i])
                    w /= row_sum;
            }
            graph_edges += adj[i].size();
        }
    }
    if (profile) {
        profile->adjacency_us = elapsed_us(t_adj0, Clock::now());
        profile->graph_edges = graph_edges;
    }

    const auto t_diff0 = profile ? Clock::now() : Clock::time_point{};
    for (std::uint32_t step = 0; step < query_steps; ++step) {
        std::fill(ns.begin(), ns.end(), 0.0f);
        for (std::size_t i = 0; i < frags.size(); ++i) {
            ns[i] += 0.5f * mass[i];
            if (adj[i].empty())
                continue;
            const float carry = 0.5f * mass[i];
            for (const auto& [j, w] : adj[i])
                ns[j] += carry * w;
        }
        mass.swap(ns);
    }
    if (profile)
        profile->diffuse_us = elapsed_us(t_diff0, Clock::now());

    const auto t_blend0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> bm_pool(pool.size(), 0.0f), geom_pool(pool.size(), 0.0f);
    for (std::size_t i = 0; i < pool.size(); ++i)
        bm_pool[i] = pool[i].second;
    for (std::size_t i = 0; i < frags.size(); ++i)
        geom_pool[frags[i].pool_index] += mass[i];

    zscore_inplace(bm_pool);
    zscore_inplace(geom_pool);
    for (std::size_t i = 0; i < pool.size(); ++i)
        scores[pool[i].first] = query_alpha * bm_pool[i] + (1.0f - query_alpha) * geom_pool[i];

    if (profile) {
        profile->blend_us = elapsed_us(t_blend0, Clock::now());
        profile->total_us = elapsed_us(t_total0, Clock::now());
    }

    return scores;
}

} // namespace simeon
