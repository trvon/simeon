#include "simeon/fragment_geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "simeon/bm25.hpp"
#include "simeon/fusion.hpp"
#include "simeon/hasher.hpp"
#include "simeon/pmi.hpp"
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
// RFF (C4) — arc-cosine degree-1 random feature map
// ---------------------------------------------------------------------------

// Deterministic Box–Muller Gaussian: identical to gaussian_entry in projection.cpp.
float rff_gaussian(std::uint32_t row, std::uint32_t col, std::uint64_t seed) noexcept {
    const std::uint64_t key = (static_cast<std::uint64_t>(row) << 32) ^ col;
    const std::uint64_t h0 = splitmix64_mix(key ^ seed);
    const std::uint64_t h1 = splitmix64_mix(h0 ^ 0x9E3779B97F4A7C15ULL);
    const float u0 = static_cast<float>((h0 >> 11) | 1ULL) / static_cast<float>(1ULL << 53);
    const float u1 = static_cast<float>((h1 >> 11) | 1ULL) / static_cast<float>(1ULL << 53);
    return std::sqrt(-2.0f * std::log(u0)) * std::cos(6.2831853071795864769f * u1);
}

// Thread-local W matrix cache: regenerated only when (seed, in_dim, out_dim) changes.
struct RffMatrixCache {
    std::uint64_t seed = 0;
    std::uint32_t in_dim = 0;
    std::uint32_t out_dim = 0;
    std::vector<float> W; // out_dim × in_dim, row-major
};

thread_local RffMatrixCache g_rff_cache;

const float* rff_matrix(std::uint64_t seed, std::uint32_t in_dim, std::uint32_t out_dim) {
    auto& c = g_rff_cache;
    if (c.seed != seed || c.in_dim != in_dim || c.out_dim != out_dim) {
        c.W.resize(static_cast<std::size_t>(out_dim) * in_dim);
        for (std::uint32_t r = 0; r < out_dim; ++r)
            for (std::uint32_t col = 0; col < in_dim; ++col)
                c.W[r * in_dim + col] = rff_gaussian(r, col, seed);
        c.seed = seed;
        c.in_dim = in_dim;
        c.out_dim = out_dim;
    }
    return c.W.data();
}

// Apply arc-cosine RFF: dst[r] = max(0, W[r,:] · src) / sqrt(out_dim). Normalizes dst.
void apply_rff(const float* W, const float* src, float* dst, std::uint32_t in_dim,
               std::uint32_t out_dim) noexcept {
    const float scale = 1.0f / std::sqrt(static_cast<float>(out_dim));
    for (std::uint32_t r = 0; r < out_dim; ++r)
        dst[r] = std::max(0.0f, simd::dot(W + r * in_dim, src, in_dim)) * scale;
    simd::l2_normalize(dst, out_dim);
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

// Routes through the SIMD-dispatched dot kernel.
inline float dot(const float* a, const float* b, std::size_t n) {
    return simd::dot(a, b, static_cast<std::uint32_t>(n));
}

// BF16 (bfloat16) compression helpers.
// Truncates the lower 16 bits of a float32 mantissa; preserves exponent exactly.
inline std::uint16_t float_to_bf16(float f) noexcept {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

inline float bf16_to_float(std::uint16_t bf) noexcept {
    std::uint32_t bits = static_cast<std::uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void store_vec_bf16(const float* src, std::uint32_t dim, std::vector<std::uint16_t>& dst) {
    dst.resize(dim);
    for (std::uint32_t i = 0; i < dim; ++i)
        dst[i] = float_to_bf16(src[i]);
}

void decompress_bf16(const std::uint16_t* src, std::uint32_t dim, float* dst) {
    for (std::uint32_t i = 0; i < dim; ++i)
        dst[i] = bf16_to_float(src[i]);
}

// Read a fragment's vector, decompressing from BF16 if vec is empty.
void read_frag_vec(const SemanticFragment& frag, std::uint32_t dim, float* dst) {
    if (!frag.vec.empty()) {
        for (std::uint32_t i = 0; i < dim; ++i)
            dst[i] = frag.vec[i];
    } else if (!frag.vec_bf16.empty()) {
        decompress_bf16(frag.vec_bf16.data(), dim, dst);
    } else {
        std::memset(dst, 0, dim * sizeof(float));
    }
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

// C2: rank-then-zscore. Sort descending, assign -rank (rank-1 doc gets the
// highest value), then z-score. Distribution-free alternative to raw z-score
// for Pareto/log-normal BM25 score distributions (Bruch et al. 2022).
void rank_norm_inplace(std::vector<float>& v) {
    if (v.size() < 2) {
        if (!v.empty())
            v[0] = 0.0f;
        return;
    }
    std::vector<std::size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::sort(idx.begin(), idx.end(), [&v](std::size_t a, std::size_t b) { return v[a] > v[b]; });
    for (std::size_t r = 0; r < idx.size(); ++r)
        v[idx[r]] = -static_cast<float>(r);
    zscore_inplace(v);
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

// ---------------------------------------------------------------------------
// C8a: SIF-Weighted PMI encoding (Arora, Liang, Ma 2017).
// IDF acts as a proxy for 1/p(w): rare content nouns get high weight,
// common function words get near-zero weight. Addresses Ceiling B (flat
// PMI bag-of-words) by producing content-noun-dominant fragment vectors.
// Requires a PMI encoder (pmi_rows != nullptr); returns empty otherwise.
// ---------------------------------------------------------------------------

std::vector<float> encode_sif_weighted(const Encoder& enc, const Bm25Index& idx,
                                       std::string_view text) {
    const auto* pmi = enc.config().pmi_rows;
    if (!pmi)
        return {};
    const auto dim = enc.output_dim();
    std::vector<float> vec(dim, 0.0f);
    std::vector<std::string> toks;
    tokenize_words(text, toks);
    for (const auto& tok : toks) {
        const float* row = pmi->row(tok);
        if (!row)
            continue;
        const float w = std::max(idx.idf(tok), 0.1f);
        simd::saxpy(vec.data(), row, w, dim);
    }
    if (simd::l2_normalize(vec.data(), dim) <= 0.0f)
        vec.clear();
    return vec;
}

// ---------------------------------------------------------------------------
// C8b: Bigram-augmented SIF (B-SIF, Mitchell & Lapata 2010).
// Adds Hadamard product of consecutive PMI vectors to the SIF unigram sum.
// Each bigram term (wᵢ, wᵢ₊₁) contributes:
//   bigram_weight · √(idf(wᵢ) · idf(wᵢ₊₁)) · (pmi(wᵢ) ⊙ pmi(wᵢ₊₁))
// Approximates multiplicative composition in distributional semantics.
// ---------------------------------------------------------------------------

std::vector<float> encode_bsif_weighted(const Encoder& enc, const Bm25Index& idx,
                                        std::string_view text, float bigram_weight = 0.5f) {
    const auto* pmi = enc.config().pmi_rows;
    if (!pmi)
        return {};
    const auto dim = enc.output_dim();
    std::vector<float> vec(dim, 0.0f);
    std::vector<std::string> toks;
    tokenize_words(text, toks);
    const float* prev_row = nullptr;
    float prev_idf = 0.0f;
    for (const auto& tok : toks) {
        const float* row = pmi->row(tok);
        const float idf_val = std::max(idx.idf(tok), 0.1f);
        if (row) {
            simd::saxpy(vec.data(), row, idf_val, dim);
            if (prev_row) {
                const float bidf = bigram_weight * std::sqrt(prev_idf * idf_val);
                if (bidf > 0.0f) {
                    for (std::uint32_t d = 0; d < dim; ++d)
                        vec[d] += bidf * prev_row[d] * row[d];
                }
            }
        }
        prev_row = row;
        prev_idf = row ? idf_val : 0.0f;
    }
    if (simd::l2_normalize(vec.data(), dim) <= 0.0f)
        vec.clear();
    return vec;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Fragment builders
// ---------------------------------------------------------------------------

std::vector<SemanticFragment> build_doc_semantic_fragments(const Encoder& enc, std::string_view doc,
                                                           const Bm25Index& idx,
                                                           std::uint32_t top_sentence_fragments,
                                                           std::uint32_t fragment_signature_terms,
                                                           float position_weight) {
    TextRank ranker;
    const auto dim = enc.output_dim();
    auto ranked = ranker.rank(doc, 4 * top_sentence_fragments);
    if (position_weight > 0.0f && !ranked.empty()) {
        for (auto& s : ranked)
            s.score = (1.0f - position_weight) * s.score + position_weight / (1.0f + s.index);
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.score > b.score; });
        if (ranked.size() > top_sentence_fragments)
            ranked.resize(top_sentence_fragments);
    } else if (ranked.size() > top_sentence_fragments) {
        ranked.resize(top_sentence_fragments);
    }
    std::vector<SemanticFragment> out;
    out.reserve(ranked.empty() ? 1 : ranked.size());

    auto try_encode = [&](std::string_view text) {
        std::vector<float> vec(dim, 0.0f);
        enc.encode(text, vec.data());
        auto sig = build_sparse_signature(idx, text, fragment_signature_terms);
        if (vector_l2_norm(vec.data(), dim) > 0.0f && !sig.terms.empty())
            out.push_back(SemanticFragment{std::move(vec), {}, std::move(sig)});
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
        if (simd::l2_normalize(centroid.data(), dim) > 0.0f && !merged.terms.empty())
            out.push_back(SemanticFragment{std::move(centroid), {}, std::move(merged)});
    }

    std::vector<float> doc_vec(dim, 0.0f);
    enc.encode(doc, doc_vec.data());
    auto doc_sig = build_sparse_signature(idx, doc, fragment_signature_terms * 2);
    if (vector_l2_norm(doc_vec.data(), dim) > 0.0f && !doc_sig.terms.empty())
        out.push_back(SemanticFragment{std::move(doc_vec), {}, std::move(doc_sig)});
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

// C8a: SIF-Weighted PMI fragment builder (richcov selection variant).
// TextRank selects top sentences; each sentence + anchors are IDF-weighted
// via encode_sif_weighted(). Richcov overlap caps applied after selection.
// Falls back to standard richcov if the encoder is not PMI-based.
std::vector<SemanticFragment>
build_doc_semantic_fragments_richcov_sif(const Encoder& enc, std::string_view doc,
                                         const Bm25Index& idx, std::uint32_t top_sentence_fragments,
                                         std::uint32_t fragment_signature_terms,
                                         float sentence_overlap_cap, float anchor_overlap_cap) {
    if (!enc.config().pmi_rows)
        return build_doc_semantic_fragments_rich_covered(enc, doc, idx, top_sentence_fragments,
                                                         fragment_signature_terms,
                                                         sentence_overlap_cap, anchor_overlap_cap);
    const auto dim = enc.output_dim();

    auto try_sif = [&](std::string_view text) -> std::optional<SemanticFragment> {
        auto vec = encode_sif_weighted(enc, idx, text);
        auto sig = build_sparse_signature(idx, text, fragment_signature_terms);
        if (!vec.empty() && !sig.terms.empty())
            return SemanticFragment{std::move(vec), {}, std::move(sig)};
        return std::nullopt;
    };

    // Phase 1: TextRank sentence selection.
    TextRank ranker;
    auto ranked = ranker.rank(doc, 4 * top_sentence_fragments);
    if (ranked.size() > top_sentence_fragments)
        ranked.resize(top_sentence_fragments);

    // Phase 2: Encode selected sentences with SIF weighting.
    std::vector<SemanticFragment> sif_frags;
    sif_frags.reserve(ranked.size() + 2);
    for (const auto& sent : ranked) {
        if (auto f = try_sif(sent.text))
            sif_frags.push_back(std::move(*f));
    }
    if (sif_frags.empty()) {
        if (auto f = try_sif(doc))
            sif_frags.push_back(std::move(*f));
        return sif_frags;
    }

    // Phase 3: Add centroid and whole-doc anchors (mirrors build_doc_semantic_fragments_rich).
    if (sif_frags.size() >= 2) {
        std::vector<float> centroid(dim, 0.0f);
        SparseSignature merged;
        const std::uint32_t merged_terms =
            std::max<std::uint32_t>(fragment_signature_terms * 2, fragment_signature_terms + 4);
        for (const auto& frag : sif_frags) {
            for (std::uint32_t d = 0; d < dim; ++d)
                centroid[d] += frag.vec[d];
            merge_signature_max(merged, frag.signature, merged_terms);
        }
        const float inv = 1.0f / static_cast<float>(sif_frags.size());
        for (float& x : centroid)
            x *= inv;
        if (simd::l2_normalize(centroid.data(), dim) > 0.0f && !merged.terms.empty())
            sif_frags.push_back(SemanticFragment{std::move(centroid), {}, std::move(merged)});
        if (auto f = try_sif(doc))
            sif_frags.push_back(std::move(*f));
    }

    // Phase 4: Apply richcov overlap caps.
    std::vector<SemanticFragment> out;
    out.reserve(sif_frags.size());
    const std::size_t base_count = sif_frags.size() >= 2 ? sif_frags.size() - 2 : sif_frags.size();
    auto append_if_novel = [&](SemanticFragment frag, float cap) {
        float max_ov = 0.0f;
        for (const auto& kept : out)
            max_ov = std::max(max_ov, weighted_overlap_coeff(frag.signature, kept.signature));
        if (max_ov <= cap || out.empty())
            out.push_back(std::move(frag));
    };
    for (std::size_t i = 0; i < base_count; ++i)
        append_if_novel(std::move(sif_frags[i]), sentence_overlap_cap);
    for (std::size_t i = base_count; i < sif_frags.size(); ++i)
        append_if_novel(std::move(sif_frags[i]), anchor_overlap_cap);
    if (out.empty())
        out.push_back(std::move(sif_frags[0]));
    return out;
}

// C8b: Bigram-SIF PMI fragment builder (Mitchell-Lapata Hadamard composition).
// Adds Hadamard bigram products to the SIF unigram accumulation: adjacent token
// pairs contribute pmi(a) ⊙ pmi(b) scaled by sqrt(idf(a)*idf(b))*bigram_weight.
// Falls back to standard richcov if the encoder is not PMI-based.
std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_bsif(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float bigram_weight) {
    if (!enc.config().pmi_rows)
        return build_doc_semantic_fragments_rich_covered(enc, doc, idx, top_sentence_fragments,
                                                         fragment_signature_terms,
                                                         sentence_overlap_cap, anchor_overlap_cap);
    const auto dim = enc.output_dim();

    auto try_bsif = [&](std::string_view text) -> std::optional<SemanticFragment> {
        auto vec = encode_bsif_weighted(enc, idx, text, bigram_weight);
        auto sig = build_sparse_signature(idx, text, fragment_signature_terms);
        if (!vec.empty() && !sig.terms.empty())
            return SemanticFragment{std::move(vec), {}, std::move(sig)};
        return std::nullopt;
    };

    TextRank ranker;
    auto ranked = ranker.rank(doc, 4 * top_sentence_fragments);
    if (ranked.size() > top_sentence_fragments)
        ranked.resize(top_sentence_fragments);

    std::vector<SemanticFragment> bsif_frags;
    bsif_frags.reserve(ranked.size() + 2);
    for (const auto& sent : ranked) {
        if (auto f = try_bsif(sent.text))
            bsif_frags.push_back(std::move(*f));
    }
    if (bsif_frags.empty()) {
        if (auto f = try_bsif(doc))
            bsif_frags.push_back(std::move(*f));
        return bsif_frags;
    }

    if (bsif_frags.size() >= 2) {
        std::vector<float> centroid(dim, 0.0f);
        SparseSignature merged;
        const std::uint32_t merged_terms =
            std::max<std::uint32_t>(fragment_signature_terms * 2, fragment_signature_terms + 4);
        for (const auto& frag : bsif_frags) {
            for (std::uint32_t d = 0; d < dim; ++d)
                centroid[d] += frag.vec[d];
            merge_signature_max(merged, frag.signature, merged_terms);
        }
        const float inv = 1.0f / static_cast<float>(bsif_frags.size());
        for (float& x : centroid)
            x *= inv;
        if (simd::l2_normalize(centroid.data(), dim) > 0.0f && !merged.terms.empty())
            bsif_frags.push_back(SemanticFragment{std::move(centroid), {}, std::move(merged)});
        if (auto f = try_bsif(doc))
            bsif_frags.push_back(std::move(*f));
    }

    std::vector<SemanticFragment> out;
    out.reserve(bsif_frags.size());
    const std::size_t base_count =
        bsif_frags.size() >= 2 ? bsif_frags.size() - 2 : bsif_frags.size();
    auto append_if_novel = [&](SemanticFragment frag, float cap) {
        float max_ov = 0.0f;
        for (const auto& kept : out)
            max_ov = std::max(max_ov, weighted_overlap_coeff(frag.signature, kept.signature));
        if (max_ov <= cap || out.empty())
            out.push_back(std::move(frag));
    };
    for (std::size_t i = 0; i < base_count; ++i)
        append_if_novel(std::move(bsif_frags[i]), sentence_overlap_cap);
    for (std::size_t i = base_count; i < bsif_frags.size(); ++i)
        append_if_novel(std::move(bsif_frags[i]), anchor_overlap_cap);
    if (out.empty())
        out.push_back(std::move(bsif_frags[0]));
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

    // Plan 2 self-KB expansion: union pool with precomputed doc-doc neighbors
    // of each pool member. Phase B adds: per-doc neighbor cap,
    // BM25-relevance filter, topology (score-decay) gate.
    if (!cfg.doc_doc_neighbors.empty()) {
        bool apply_expansion = true;
        if (cfg.selfkb_gate_score_decay_min > 0.0f) {
            const float decay = bm25_pool_decay(pool);
            apply_expansion = (decay >= cfg.selfkb_gate_score_decay_min);
        }
        if (apply_expansion) {
            std::vector<std::uint8_t> in_pool(static_cast<std::size_t>(nd), 0);
            for (const auto& [did, _sc] : pool)
                in_pool[did] = 1;
            const std::size_t orig_pool_size = pool.size();
            const std::uint32_t n_cap = cfg.selfkb_neighbors_per_pool_doc;
            for (std::size_t i = 0; i < orig_pool_size; ++i) {
                const auto did = pool[i].first;
                if (did >= cfg.doc_doc_neighbors.size())
                    continue;
                const auto& neighbors = cfg.doc_doc_neighbors[did];
                const std::size_t limit = (n_cap == 0)
                                              ? neighbors.size()
                                              : std::min<std::size_t>(n_cap, neighbors.size());
                for (std::size_t k = 0; k < limit; ++k) {
                    const auto nid = neighbors[k];
                    if (nid >= nd || in_pool[nid])
                        continue;
                    const float nscore = bm25_scores[nid];
                    if (nscore <= cfg.selfkb_min_bm25_score)
                        continue;
                    pool.emplace_back(nid, nscore);
                    in_pool[nid] = 1;
                }
            }
        }
    }

    const auto t_bm251 = profile ? Clock::now() : Clock::time_point{};
    if (profile) {
        profile->bm25_us = elapsed_us(t_bm250, t_bm251);
        profile->pool_docs = static_cast<std::uint32_t>(pool.size());
    }

    // C6: IDF-weighted query-coverage. Tokenize query, build IDF-weighted term
    // hashes, score all docs, extract pool subset, z-score. Applied additively
    // at both blend exits. Computed once; shared by outer MaxSim + PHSS paths.
    std::vector<float> cov_pool_z;
    if (cfg.idf_coverage && cfg.idf_coverage_gamma != 0.0f) {
        std::vector<std::string> qtoks;
        tokenize_words(query, qtoks);
        std::sort(qtoks.begin(), qtoks.end());
        qtoks.erase(std::unique(qtoks.begin(), qtoks.end()), qtoks.end());

        std::vector<std::pair<std::uint64_t, float>> wterms;
        float idf_sum = 0.0f;
        for (const auto& tok : qtoks) {
            const float iv = idx.idf(tok);
            if (iv <= 0.0f)
                continue;
            wterms.emplace_back(idx.hash_term(tok), iv);
            idf_sum += iv;
        }
        if (!wterms.empty() && idf_sum > 0.0f) {
            for (auto& [h, w] : wterms)
                w /= idf_sum;
            std::vector<float> cov_all(nd, 0.0f);
            idx.score_weighted_hashes(wterms, cov_all);
            cov_pool_z.resize(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i)
                cov_pool_z[i] = cov_all[pool[i].first];
            zscore_inplace(cov_pool_z);
        }
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
        std::uint32_t frag_index = 0;
        const float* vec = nullptr;
    };

    const auto t_gather0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<FragRef> frags;
    frags.reserve(pool.size() * cfg.top_fragments_per_doc);
    for (std::size_t pi = 0; pi < pool.size(); ++pi) {
        const auto did = pool[pi].first;
        const auto limit = std::min<std::size_t>(cfg.top_fragments_per_doc, doc_frags[did].size());
        for (std::size_t fi = 0; fi < limit; ++fi) {
            frags.push_back(FragRef{.pool_index = static_cast<std::uint32_t>(pi),
                                    .frag_index = static_cast<std::uint32_t>(fi),
                                    .vec = nullptr});
        }
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
    std::vector<float> fvecs_raw(frags.size() * dim, 0.0f);
    std::vector<float> mean(dim, 0.0f), m2(dim, 0.0f), var(dim, 0.0f);
    std::size_t seen = 0;
    for (std::size_t i = 0; i < frags.size(); ++i) {
        const auto did = pool[frags[i].pool_index].first;
        const auto& src_frag = doc_frags[did][frags[i].frag_index];
        float* dst = fvecs_raw.data() + i * dim;
        read_frag_vec(src_frag, dim, dst);
        frags[i].vec = dst;

        ++seen;
        const float inv_seen = 1.0f / static_cast<float>(seen);
        for (std::uint32_t d = 0; d < dim; ++d) {
            const float x = dst[d];
            const float delta = x - mean[d];
            mean[d] += delta * inv_seen;
            const float delta2 = x - mean[d];
            m2[d] += delta * delta2;
        }
    }
    const float inv_n = 1.0f / static_cast<float>(frags.size());
    for (std::uint32_t d = 0; d < dim; ++d)
        var[d] = std::sqrt(m2[d] * inv_n + 1e-6f);

    std::vector<float> wq = qvec;
    for (std::uint32_t d = 0; d < dim; ++d)
        wq[d] = (wq[d] - mean[d]) / var[d];
    const float wq_inv = simd::l2_normalize(wq.data(), dim);
    if (wq_inv <= 0.0f) {
        if (profile) {
            profile->whiten_us = elapsed_us(t_white0, Clock::now());
            profile->total_us = elapsed_us(t_total0, Clock::now());
        }
        return scores;
    }

    std::vector<float> fvecs(frags.size() * dim, 0.0f);
    for (std::size_t i = 0; i < frags.size(); ++i) {
        const float* raw = fvecs_raw.data() + i * dim;
        float* dst = fvecs.data() + i * dim;
        for (std::uint32_t d = 0; d < dim; ++d)
            dst[d] = (raw[d] - mean[d]) / var[d];
        simd::l2_normalize(dst, dim);
    }
    if (profile)
        profile->whiten_us = elapsed_us(t_white0, Clock::now());

    // C4: RFF kernel augmentation — apply arc-cosine feature map to whitened
    // vectors before similarity computation. W is thread-locally cached.
    std::vector<float> fvecs_rff, wq_rff;
    const float* active_fvecs = fvecs.data();
    const float* active_wq = wq.data();
    std::uint32_t active_dim = dim;
    if (cfg.rff_augment && cfg.rff_dim > dim) {
        constexpr std::uint64_t kRffSeed = 0xB5AD4ECEDA1CE2A9ULL;
        const std::uint32_t rdim = cfg.rff_dim;
        const float* W = rff_matrix(kRffSeed, dim, rdim);
        wq_rff.resize(rdim);
        fvecs_rff.resize(frags.size() * rdim);
        apply_rff(W, wq.data(), wq_rff.data(), dim, rdim);
        for (std::size_t i = 0; i < frags.size(); ++i)
            apply_rff(W, fvecs.data() + i * dim, fvecs_rff.data() + i * rdim, dim, rdim);
        active_fvecs = fvecs_rff.data();
        active_wq = wq_rff.data();
        active_dim = rdim;
    }

    const auto t_qatt0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> qsim(frags.size(), 0.0f);
    for (std::size_t i = 0; i < frags.size(); ++i)
        qsim[i] = dot(active_wq, active_fvecs + i * active_dim, active_dim);

    // C1: SPLATE-style outer MaxSim — geometry score = max(query · frag) per
    // doc, bypassing PHSS+diffusion. MaxSim is at the outermost layer before
    // the alpha blend, eliminating the attention×diffusion×averaging attenuation.
    if (cfg.outer_maxsim) {
        std::vector<float> bm_pool(pool.size(), 0.0f);
        std::vector<float> geom_pool(pool.size(), -std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < pool.size(); ++i)
            bm_pool[i] = pool[i].second;
        for (std::size_t i = 0; i < frags.size(); ++i) {
            float& slot = geom_pool[frags[i].pool_index];
            if (qsim[i] > slot)
                slot = qsim[i];
        }
        for (float& g : geom_pool)
            if (!std::isfinite(g))
                g = 0.0f;
        zscore_inplace(bm_pool);
        zscore_inplace(geom_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            scores[pool[i].first] = query_alpha * bm_pool[i] + (1.0f - query_alpha) * geom_pool[i];
        if (!cov_pool_z.empty())
            for (std::size_t i = 0; i < pool.size(); ++i)
                scores[pool[i].first] += cfg.idf_coverage_gamma * cov_pool_z[i];
        if (profile)
            profile->total_us = elapsed_us(t_total0, Clock::now());
        return scores;
    }

    const std::uint32_t nf = static_cast<std::uint32_t>(frags.size());
    const std::uint32_t tri_count = nf * (nf - 1) / 2;
    const auto t_pair0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> sims_tri;
    sims_tri.reserve(tri_count);
    for (std::uint32_t i = 0; i < nf; ++i) {
        for (std::uint32_t j = i + 1; j < nf; ++j) {
            sims_tri.push_back(
                dot(active_fvecs + i * active_dim, active_fvecs + j * active_dim, active_dim));
        }
    }
    const auto t_pair1 = profile ? Clock::now() : Clock::time_point{};
    if (profile)
        profile->phss_pairwise_us = elapsed_us(t_pair0, t_pair1);

    auto sim_at = [&sims_tri, nf](std::uint32_t a, std::uint32_t b) -> float {
        if (a == b)
            return 1.0f;
        if (a > b)
            std::swap(a, b);
        const std::uint32_t offset = a * (2 * nf - a - 1) / 2 + (b - a - 1);
        return sims_tri[offset];
    };

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
            PhssConfig phss_cfg = cfg.phss_config;
            phss_cfg.dim_max = 0;
            const auto t_sel0 = profile ? Clock::now() : Clock::time_point{};
            auto phss_result = phss_select_scale(sims_tri, nf, phss_cfg);
            const auto t_sel1 = profile ? Clock::now() : Clock::time_point{};
            if (profile) {
                profile->phss_select_us = elapsed_us(t_sel0, t_sel1);
                profile->phss_select_edge_gather_us = phss_result.edge_gather_us;
                profile->phss_select_edge_sort_us = phss_result.edge_sort_us;
                profile->phss_select_uf_us = phss_result.uf_traversal_us;
                profile->phss_select_survivor_us = phss_result.survivor_scan_us;
                profile->phss_select_death_sort_us = phss_result.death_sort_us;
                profile->phss_select_criterion_us = phss_result.criterion_us;
            }
            phss_selected_scale = phss_result.selected_scale;
            use_phss_for_graph = (phss_selected_scale > 0.0f);
            if (profile)
                profile->phss_selected_scale = phss_selected_scale;
            if (cfg.phss_gap_adaptive && cfg.phss_gap_scale > 0.0f) {
                const float conf = std::min(phss_result.max_gap / cfg.phss_gap_scale, 1.0f);
                const float adapted = query_alpha - conf * cfg.phss_gap_delta;
                query_alpha = std::max(adapted, cfg.phss_gap_alpha_min);
            }
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

    // Plan 4 — single-fragment-per-doc filter. Per pool_index (doc), keep
    // only the argmax-qsim fragment; suppress others to -inf so the softmax
    // assigns them zero mass. Breaks multi-fragment averaging at the source.
    if (cfg.single_fragment_per_doc) {
        std::vector<std::uint32_t> best_per_pool(pool.size(),
                                                 std::numeric_limits<std::uint32_t>::max());
        std::vector<float> best_sim_per_pool(pool.size(), -std::numeric_limits<float>::infinity());
        for (std::size_t i = 0; i < frags.size(); ++i) {
            const auto pi = frags[i].pool_index;
            if (qsim[i] > best_sim_per_pool[pi]) {
                best_sim_per_pool[pi] = qsim[i];
                best_per_pool[pi] = static_cast<std::uint32_t>(i);
            }
        }
        for (std::size_t i = 0; i < frags.size(); ++i) {
            if (best_per_pool[frags[i].pool_index] != i)
                qsim[i] = -std::numeric_limits<float>::infinity();
        }
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
            const float sim = sim_at(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j));
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

    // PHSS-1D Phase A — triangle counter + per-fragment importance weight.
    // Counts triangles per node in the kNN graph: for each i, count pairs
    // (j, k) of i's neighbors where j and k are also adjacent. Each triangle
    // (i, j, k) is counted three times (once per node). Cost O(N * d^2) where
    // d is the average degree; tiny compared to phss_select edge sort.
    std::vector<float> tri_weight;
    if (cfg.use_triangle_weight && cfg.triangle_alpha != 0.0f) {
        const auto t_tri0 = profile ? Clock::now() : Clock::time_point{};
        tri_weight.assign(frags.size(), 1.0f);
        std::vector<std::uint8_t> in_adj(frags.size(), 0);
        std::uint64_t tri_total = 0;
        for (std::size_t i = 0; i < frags.size(); ++i) {
            for (const auto& [j, _w] : adj[i])
                in_adj[j] = 1;
            std::uint32_t tc = 0;
            for (const auto& [j, _wj] : adj[i]) {
                for (const auto& [k, _wk] : adj[j]) {
                    if (k != i && in_adj[k])
                        ++tc;
                }
            }
            // adj[j] traversal counts each (i,j,k) twice (once via j and once
            // via k); divide by 2 for the actual per-i triangle count.
            tri_weight[i] = 1.0f + cfg.triangle_alpha * std::log1p(static_cast<float>(tc / 2));
            tri_total += tc / 2;
            for (const auto& [j, _w] : adj[i])
                in_adj[j] = 0;
        }
        if (profile) {
            profile->triangle_count_us = elapsed_us(t_tri0, Clock::now());
            profile->triangle_count_total = tri_total;
        }
        if (cfg.triangle_placement == FragmentGeometryConfig::TrianglePlacement::QueryAttention) {
            float renorm_sum = 0.0f;
            for (std::size_t i = 0; i < frags.size(); ++i) {
                mass[i] *= tri_weight[i];
                renorm_sum += mass[i];
            }
            if (renorm_sum > 0.0f) {
                for (float& x : mass)
                    x /= renorm_sum;
            }
        }
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
    if (cfg.use_triangle_weight && cfg.triangle_alpha != 0.0f &&
        cfg.triangle_placement == FragmentGeometryConfig::TrianglePlacement::Diffusion &&
        !tri_weight.empty()) {
        float renorm_sum = 0.0f;
        for (std::size_t i = 0; i < frags.size(); ++i) {
            mass[i] *= tri_weight[i];
            renorm_sum += mass[i];
        }
        if (renorm_sum > 0.0f) {
            for (float& x : mass)
                x /= renorm_sum;
        }
    }
    std::vector<float> bm_pool(pool.size(), 0.0f), geom_pool(pool.size(), 0.0f);
    for (std::size_t i = 0; i < pool.size(); ++i)
        bm_pool[i] = pool[i].second;
    if (cfg.doc_aggregator == FragmentGeometryConfig::DocAggregator::Max) {
        for (std::size_t i = 0; i < frags.size(); ++i) {
            float& slot = geom_pool[frags[i].pool_index];
            if (mass[i] > slot)
                slot = mass[i];
        }
    } else {
        for (std::size_t i = 0; i < frags.size(); ++i)
            geom_pool[frags[i].pool_index] += mass[i];
    }

    if (cfg.bm25_rank_norm)
        rank_norm_inplace(bm_pool);
    else
        zscore_inplace(bm_pool);
    zscore_inplace(geom_pool);
    for (std::size_t i = 0; i < pool.size(); ++i)
        scores[pool[i].first] = query_alpha * bm_pool[i] + (1.0f - query_alpha) * geom_pool[i];
    if (!cov_pool_z.empty())
        for (std::size_t i = 0; i < pool.size(); ++i)
            scores[pool[i].first] += cfg.idf_coverage_gamma * cov_pool_z[i];

    if (profile) {
        profile->blend_us = elapsed_us(t_blend0, Clock::now());
        profile->total_us = elapsed_us(t_total0, Clock::now());
    }

    return scores;
}

void compress_fragments_to_bf16(std::span<std::vector<SemanticFragment>> doc_frags,
                                std::uint32_t dim) {
    for (auto& doc : doc_frags) {
        for (auto& frag : doc) {
            if (frag.vec.empty() || !frag.vec_bf16.empty())
                continue;
            store_vec_bf16(frag.vec.data(), dim, frag.vec_bf16);
            frag.vec.clear();
            frag.vec.shrink_to_fit();
        }
    }
}

} // namespace simeon
