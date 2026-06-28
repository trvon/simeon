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
#include "simeon/manifold.hpp"
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
// Vector helpers
// ---------------------------------------------------------------------------

float vector_l2_norm(const float* v, std::uint32_t dim) {
    double sq = 0.0;
    for (std::uint32_t i = 0; i < dim; ++i)
        sq += static_cast<double>(v[i]) * v[i];
    return static_cast<float>(std::sqrt(sq));
}

// Routes through the active manifold's similarity kernel. Default manifold is
// `EuclideanCosineManifold` (= cosine on unit-norm vectors). Swap the alias below to
// experiment with hyperbolic / RBF / spherical variants without touching call sites.
using ActiveManifold = EuclideanCosineManifold;

inline float dot(const float* a, const float* b, std::size_t n) {
    return ActiveManifold::similarity(a, b, static_cast<std::uint32_t>(n));
}

inline void dot4(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
                 float* out4, std::size_t n) {
    const auto dim = static_cast<std::uint32_t>(n);
    if constexpr (requires { ActiveManifold::similarity4(a, b0, b1, b2, b3, out4, dim); }) {
        ActiveManifold::similarity4(a, b0, b1, b2, b3, out4, dim);
    } else {
        out4[0] = ActiveManifold::similarity(a, b0, dim);
        out4[1] = ActiveManifold::similarity(a, b1, dim);
        out4[2] = ActiveManifold::similarity(a, b2, dim);
        out4[3] = ActiveManifold::similarity(a, b3, dim);
    }
}

inline void dot2x4(const float* a0, const float* a1, const float* b0, const float* b1,
                   const float* b2, const float* b3, float* out0, float* out1, std::size_t n) {
    const auto dim = static_cast<std::uint32_t>(n);
    if constexpr (requires {
                      ActiveManifold::similarity2x4(a0, a1, b0, b1, b2, b3, out0, out1, dim);
                  }) {
        ActiveManifold::similarity2x4(a0, a1, b0, b1, b2, b3, out0, out1, dim);
    } else {
        dot4(a0, b0, b1, b2, b3, out0, n);
        dot4(a1, b0, b1, b2, b3, out1, n);
    }
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
// Internal helper. The public `simeon::read_frag_vec` declared in the
// header is defined at the end of this TU and forwards to this one.
void read_frag_vec_impl(const SemanticFragment& frag, std::uint32_t dim, float* dst) {
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

float clamp_unit(float x) {
    if (x < 0.0f)
        return 0.0f;
    if (x > 1.0f)
        return 1.0f;
    return x;
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
// SIF-weighted PMI encoding (Arora, Liang, Ma 2017).
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
// Bigram-augmented SIF (B-SIF, Mitchell & Lapata 2010).
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
// Shared per-doc prep — TextRank ranking + per-sentence/anchor signatures.
// Built once per doc, reusable across all rich- and richcov-family builders.
// ---------------------------------------------------------------------------

DocPrep prepare_doc(std::string_view doc, const Bm25Index& idx,
                    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
                    float position_weight) {
    DocPrep prep;
    prep.fragment_signature_terms = fragment_signature_terms;
    prep.top_sentence_fragments = top_sentence_fragments;

    TextRank ranker;
    auto ranked = ranker.rank(doc, 4 * top_sentence_fragments);
    if (position_weight > 0.0f && !ranked.empty()) {
        for (auto& s : ranked)
            s.score = (1.0f - position_weight) * s.score + position_weight / (1.0f + s.index);
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.score > b.score; });
    }
    if (ranked.size() > top_sentence_fragments)
        ranked.resize(top_sentence_fragments);

    prep.sentence_sigs.reserve(ranked.size());
    for (const auto& s : ranked)
        prep.sentence_sigs.push_back(build_sparse_signature(idx, s.text, fragment_signature_terms));

    prep.anchor_sig_1x = build_sparse_signature(idx, doc, fragment_signature_terms);
    prep.anchor_sig_2x = build_sparse_signature(idx, doc, fragment_signature_terms * 2);
    prep.ranked = std::move(ranked);
    return prep;
}

// ---------------------------------------------------------------------------
// Fragment builders
// ---------------------------------------------------------------------------

std::vector<SemanticFragment> build_doc_semantic_fragments_from_prep(const Encoder& enc,
                                                                     std::string_view doc,
                                                                     const DocPrep& prep) {
    const auto dim = enc.output_dim();
    std::vector<SemanticFragment> out;
    out.reserve(prep.ranked.empty() ? 1 : prep.ranked.size());

    for (std::size_t i = 0; i < prep.ranked.size(); ++i) {
        std::vector<float> vec(dim, 0.0f);
        enc.encode(prep.ranked[i].text, vec.data());
        if (vector_l2_norm(vec.data(), dim) > 0.0f && !prep.sentence_sigs[i].terms.empty())
            out.push_back(SemanticFragment{std::move(vec), {}, prep.sentence_sigs[i]});
    }
    if (out.empty()) {
        std::vector<float> vec(dim, 0.0f);
        enc.encode(doc, vec.data());
        if (vector_l2_norm(vec.data(), dim) > 0.0f && !prep.anchor_sig_1x.terms.empty())
            out.push_back(SemanticFragment{std::move(vec), {}, prep.anchor_sig_1x});
    }
    return out;
}

std::vector<SemanticFragment> build_doc_semantic_fragments(const Encoder& enc, std::string_view doc,
                                                           const Bm25Index& idx,
                                                           std::uint32_t top_sentence_fragments,
                                                           std::uint32_t fragment_signature_terms,
                                                           float position_weight) {
    auto prep =
        prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms, position_weight);
    return build_doc_semantic_fragments_from_prep(enc, doc, prep);
}

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_from_prep(const Encoder& enc,
                                                                          std::string_view doc,
                                                                          const DocPrep& prep) {
    auto out = build_doc_semantic_fragments_from_prep(enc, doc, prep);
    const auto dim = enc.output_dim();
    if (out.size() >= 2) {
        std::vector<float> centroid(dim, 0.0f);
        SparseSignature merged;
        const std::uint32_t merged_terms = std::max<std::uint32_t>(
            prep.fragment_signature_terms * 2, prep.fragment_signature_terms + 4);
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
    if (vector_l2_norm(doc_vec.data(), dim) > 0.0f && !prep.anchor_sig_2x.terms.empty())
        out.push_back(SemanticFragment{std::move(doc_vec), {}, prep.anchor_sig_2x});
    return out;
}

std::vector<SemanticFragment>
build_doc_semantic_fragments_rich(const Encoder& enc, std::string_view doc, const Bm25Index& idx,
                                  std::uint32_t top_sentence_fragments,
                                  std::uint32_t fragment_signature_terms) {
    auto prep = prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms);
    return build_doc_semantic_fragments_rich_from_prep(enc, doc, prep);
}

std::vector<SemanticFragment>
build_doc_semantic_fragments_rich_covered_from_prep(const Encoder& enc, std::string_view doc,
                                                    const DocPrep& prep, float sentence_overlap_cap,
                                                    float anchor_overlap_cap) {
    auto rich = build_doc_semantic_fragments_rich_from_prep(enc, doc, prep);
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

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_covered(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap) {
    auto prep = prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms);
    return build_doc_semantic_fragments_rich_covered_from_prep(enc, doc, prep, sentence_overlap_cap,
                                                               anchor_overlap_cap);
}

std::vector<SemanticFragment>
build_doc_semantic_fragments_rich_mmr_from_prep(const Encoder& enc, std::string_view doc,
                                                const DocPrep& prep, float sentence_overlap_cap,
                                                float anchor_overlap_cap, float redundancy_lambda,
                                                float sentence_min_score, float anchor_min_score) {
    struct Candidate {
        SemanticFragment frag;
        float salience = 0.0f;
        float overlap_cap = 1.0f;
        float min_score = 0.0f;
    };

    auto rich = build_doc_semantic_fragments_rich_from_prep(enc, doc, prep);
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

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_mmr(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float redundancy_lambda,
    float sentence_min_score, float anchor_min_score) {
    auto prep = prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms);
    return build_doc_semantic_fragments_rich_mmr_from_prep(enc, doc, prep, sentence_overlap_cap,
                                                           anchor_overlap_cap, redundancy_lambda,
                                                           sentence_min_score, anchor_min_score);
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

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_budgeted_from_prep(
    const Encoder& enc, std::string_view doc, const DocPrep& prep, float sentence_overlap_cap,
    std::uint32_t max_sentence_keep, float anchor_overlap_cap, float anchor_novelty_floor,
    std::uint32_t max_anchor_keep) {
    auto rich = build_doc_semantic_fragments_rich_from_prep(enc, doc, prep);
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

std::vector<SemanticFragment> build_doc_semantic_fragments_rich_budgeted(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, std::uint32_t max_sentence_keep, float anchor_overlap_cap,
    float anchor_novelty_floor, std::uint32_t max_anchor_keep) {
    auto prep = prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms);
    return build_doc_semantic_fragments_rich_budgeted_from_prep(
        enc, doc, prep, sentence_overlap_cap, max_sentence_keep, anchor_overlap_cap,
        anchor_novelty_floor, max_anchor_keep);
}

// SIF-weighted PMI fragment builder (richcov overlap caps).
// IDF-weighted token accumulation via encode_sif_weighted().
// Falls back to standard richcov if the encoder is not PMI-based.
std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_sif_from_prep(
    const Encoder& enc, const Bm25Index& idx, std::string_view doc, const DocPrep& prep,
    float sentence_overlap_cap, float anchor_overlap_cap) {
    if (!enc.config().pmi_rows)
        return build_doc_semantic_fragments_rich_covered_from_prep(
            enc, doc, prep, sentence_overlap_cap, anchor_overlap_cap);
    const auto dim = enc.output_dim();

    std::vector<SemanticFragment> sif_frags;
    sif_frags.reserve(prep.ranked.size() + 2);
    for (std::size_t i = 0; i < prep.ranked.size(); ++i) {
        auto vec = encode_sif_weighted(enc, idx, prep.ranked[i].text);
        if (!vec.empty() && !prep.sentence_sigs[i].terms.empty())
            sif_frags.push_back(SemanticFragment{std::move(vec), {}, prep.sentence_sigs[i]});
    }
    if (sif_frags.empty()) {
        auto vec = encode_sif_weighted(enc, idx, doc);
        if (!vec.empty() && !prep.anchor_sig_1x.terms.empty())
            sif_frags.push_back(SemanticFragment{std::move(vec), {}, prep.anchor_sig_1x});
        return sif_frags;
    }

    if (sif_frags.size() >= 2) {
        std::vector<float> centroid(dim, 0.0f);
        SparseSignature merged;
        const std::uint32_t merged_terms = std::max<std::uint32_t>(
            prep.fragment_signature_terms * 2, prep.fragment_signature_terms + 4);
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
        auto anchor_vec = encode_sif_weighted(enc, idx, doc);
        if (!anchor_vec.empty() && !prep.anchor_sig_1x.terms.empty())
            sif_frags.push_back(SemanticFragment{std::move(anchor_vec), {}, prep.anchor_sig_1x});
    }

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

std::vector<SemanticFragment>
build_doc_semantic_fragments_richcov_sif(const Encoder& enc, std::string_view doc,
                                         const Bm25Index& idx, std::uint32_t top_sentence_fragments,
                                         std::uint32_t fragment_signature_terms,
                                         float sentence_overlap_cap, float anchor_overlap_cap) {
    auto prep = prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms);
    return build_doc_semantic_fragments_richcov_sif_from_prep(
        enc, idx, doc, prep, sentence_overlap_cap, anchor_overlap_cap);
}

// Bigram-SIF PMI fragment builder. Hadamard bigram products on top of SIF unigram
// accumulation. Falls back to standard richcov if the encoder is not PMI-based.
std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_bsif_from_prep(
    const Encoder& enc, const Bm25Index& idx, std::string_view doc, const DocPrep& prep,
    float sentence_overlap_cap, float anchor_overlap_cap, float bigram_weight) {
    if (!enc.config().pmi_rows)
        return build_doc_semantic_fragments_rich_covered_from_prep(
            enc, doc, prep, sentence_overlap_cap, anchor_overlap_cap);
    const auto dim = enc.output_dim();

    std::vector<SemanticFragment> bsif_frags;
    bsif_frags.reserve(prep.ranked.size() + 2);
    for (std::size_t i = 0; i < prep.ranked.size(); ++i) {
        auto vec = encode_bsif_weighted(enc, idx, prep.ranked[i].text, bigram_weight);
        if (!vec.empty() && !prep.sentence_sigs[i].terms.empty())
            bsif_frags.push_back(SemanticFragment{std::move(vec), {}, prep.sentence_sigs[i]});
    }
    if (bsif_frags.empty()) {
        auto vec = encode_bsif_weighted(enc, idx, doc, bigram_weight);
        if (!vec.empty() && !prep.anchor_sig_1x.terms.empty())
            bsif_frags.push_back(SemanticFragment{std::move(vec), {}, prep.anchor_sig_1x});
        return bsif_frags;
    }

    if (bsif_frags.size() >= 2) {
        std::vector<float> centroid(dim, 0.0f);
        SparseSignature merged;
        const std::uint32_t merged_terms = std::max<std::uint32_t>(
            prep.fragment_signature_terms * 2, prep.fragment_signature_terms + 4);
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
        auto anchor_vec = encode_bsif_weighted(enc, idx, doc, bigram_weight);
        if (!anchor_vec.empty() && !prep.anchor_sig_1x.terms.empty())
            bsif_frags.push_back(SemanticFragment{std::move(anchor_vec), {}, prep.anchor_sig_1x});
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

std::vector<SemanticFragment> build_doc_semantic_fragments_richcov_bsif(
    const Encoder& enc, std::string_view doc, const Bm25Index& idx,
    std::uint32_t top_sentence_fragments, std::uint32_t fragment_signature_terms,
    float sentence_overlap_cap, float anchor_overlap_cap, float bigram_weight) {
    auto prep = prepare_doc(doc, idx, top_sentence_fragments, fragment_signature_terms);
    return build_doc_semantic_fragments_richcov_bsif_from_prep(
        enc, idx, doc, prep, sentence_overlap_cap, anchor_overlap_cap, bigram_weight);
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

    // Dual-stage candidate pool. When dense_pool_size > 0 and the caller has
    // supplied per-doc dense vectors, augment the BM25 top-pool_size with the
    // top-dense_pool_size docs by cosine(qvec, doc_dense_vec). The dense-only
    // arrivals retain their (typically low) BM25 score so the BM25 leg of the
    // linear blend stays consistent. Pool grows up to pool_size +
    // dense_pool_size after dedup.
    if (cfg.dense_pool_size > 0 && cfg.doc_dense_vecs && cfg.doc_dense_vecs->size() == nd &&
        vector_l2_norm(qvec.data(), dim) > 0.0f) {
        std::vector<float> dense_scores(nd, 0.0f);
        for (std::uint32_t did = 0; did < nd; ++did) {
            const auto& dv = (*cfg.doc_dense_vecs)[did];
            if (dv.size() == dim)
                dense_scores[did] = simd::dot(qvec.data(), dv.data(), dim);
        }
        auto dense_pool = top_k(dense_scores, cfg.dense_pool_size);
        std::vector<std::uint8_t> in_pool(nd, 0);
        for (const auto& [did, _sc] : pool)
            in_pool[did] = 1;
        for (const auto& [did, _ds] : dense_pool) {
            if (did < nd && !in_pool[did]) {
                pool.emplace_back(did, bm25_scores[did]);
                in_pool[did] = 1;
            }
        }
    }

    float query_alpha = cfg.alpha;
    float query_scale = cfg.attention_scale;
    std::uint32_t query_knn = cfg.knn;
    std::uint32_t query_steps = cfg.steps;

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
    // mean=0 / var=1 makes the transform below the identity, i.e. plain
    // L2-normalized cosine (cfg.whiten=false). Per-query whitening overwrites them
    // with the pool-fragment statistics; see FragmentGeometryConfig::whiten.
    std::vector<float> mean(dim, 0.0f), var(dim, 1.0f);
    for (std::size_t i = 0; i < frags.size(); ++i) {
        const auto did = pool[frags[i].pool_index].first;
        const auto& src_frag = doc_frags[did][frags[i].frag_index];
        float* dst = fvecs_raw.data() + i * dim;
        read_frag_vec_impl(src_frag, dim, dst);
        frags[i].vec = dst;
    }
    if (cfg.whiten) {
        std::vector<float> m2(dim, 0.0f);
        std::size_t seen = 0;
        for (std::size_t i = 0; i < frags.size(); ++i) {
            const float* x = fvecs_raw.data() + i * dim;
            ++seen;
            const float inv_seen = 1.0f / static_cast<float>(seen);
            for (std::uint32_t d = 0; d < dim; ++d) {
                const float delta = x[d] - mean[d];
                mean[d] += delta * inv_seen;
                const float delta2 = x[d] - mean[d];
                m2[d] += delta * delta2;
            }
        }
        const float inv_n = 1.0f / static_cast<float>(frags.size());
        for (std::uint32_t d = 0; d < dim; ++d)
            var[d] = std::sqrt(m2[d] * inv_n + 1e-6f);
    }

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

    const auto t_qatt0 = profile ? Clock::now() : Clock::time_point{};
    std::vector<float> qsim(frags.size(), 0.0f);
    for (std::size_t i = 0; i < frags.size(); ++i)
        qsim[i] = dot(wq.data(), fvecs.data() + i * dim, dim);

    // Outer MaxSim path — geometry score = aggregate(qsim) per doc, bypassing
    // PHSS+diffusion. Aggregator selected by cfg.doc_scorer_kind: MaxSim |
    // MeanSim | TopKMean | SoftMaxSum.
    if (cfg.outer_maxsim) {
        std::vector<float> bm_pool(pool.size(), 0.0f);
        for (std::size_t i = 0; i < pool.size(); ++i)
            bm_pool[i] = pool[i].second;

        // Group fragment indices by pool_index for non-MaxSim aggregators.
        std::vector<std::vector<std::uint32_t>> frags_by_pool(pool.size());
        for (std::size_t i = 0; i < frags.size(); ++i)
            frags_by_pool[frags[i].pool_index].push_back(static_cast<std::uint32_t>(i));

        std::vector<float> geom_pool(pool.size(), 0.0f);
        using DK = FragmentGeometryConfig::DocScorerKind;
        switch (cfg.doc_scorer_kind) {
            case DK::MeanSim: {
                for (std::size_t pi = 0; pi < pool.size(); ++pi) {
                    const auto& ids = frags_by_pool[pi];
                    if (ids.empty()) {
                        geom_pool[pi] = 0.0f;
                        continue;
                    }
                    float sum = 0.0f;
                    for (auto fi : ids)
                        sum += qsim[fi];
                    geom_pool[pi] = sum / static_cast<float>(ids.size());
                }
                break;
            }
            case DK::TopKMean: {
                const std::uint32_t k = std::max<std::uint32_t>(1u, cfg.doc_scorer_top_k);
                for (std::size_t pi = 0; pi < pool.size(); ++pi) {
                    const auto& ids = frags_by_pool[pi];
                    if (ids.empty()) {
                        geom_pool[pi] = 0.0f;
                        continue;
                    }
                    std::vector<float> vals;
                    vals.reserve(ids.size());
                    for (auto fi : ids)
                        vals.push_back(qsim[fi]);
                    const std::uint32_t use_k =
                        std::min<std::uint32_t>(k, static_cast<std::uint32_t>(vals.size()));
                    std::partial_sort(vals.begin(), vals.begin() + use_k, vals.end(),
                                      std::greater<float>());
                    float sum = 0.0f;
                    for (std::uint32_t i = 0; i < use_k; ++i)
                        sum += vals[i];
                    geom_pool[pi] = sum / static_cast<float>(use_k);
                }
                break;
            }
            case DK::SoftMaxSum: {
                const float beta = cfg.doc_scorer_softmax_beta;
                for (std::size_t pi = 0; pi < pool.size(); ++pi) {
                    const auto& ids = frags_by_pool[pi];
                    if (ids.empty()) {
                        geom_pool[pi] = 0.0f;
                        continue;
                    }
                    float mx = -std::numeric_limits<float>::infinity();
                    for (auto fi : ids)
                        mx = std::max(mx, qsim[fi]);
                    float wsum = 0.0f, num = 0.0f;
                    for (auto fi : ids) {
                        const float w = std::exp(beta * (qsim[fi] - mx));
                        wsum += w;
                        num += w * qsim[fi];
                    }
                    geom_pool[pi] = wsum > 0.0f ? num / wsum : mx;
                }
                break;
            }
            case DK::GeoMean: {
                // Geometric mean over a small ε floor to handle non-positive qsims.
                // exp(mean(log(qsim_i + shift))) − shift, with shift chosen so all
                // values are strictly positive within the doc.
                for (std::size_t pi = 0; pi < pool.size(); ++pi) {
                    const auto& ids = frags_by_pool[pi];
                    if (ids.empty()) {
                        geom_pool[pi] = 0.0f;
                        continue;
                    }
                    float mn = std::numeric_limits<float>::infinity();
                    for (auto fi : ids)
                        mn = std::min(mn, qsim[fi]);
                    const float shift = mn < 1e-3f ? (1e-3f - mn) : 0.0f;
                    double log_sum = 0.0;
                    for (auto fi : ids)
                        log_sum += std::log(static_cast<double>(qsim[fi] + shift));
                    const double mean_log = log_sum / static_cast<double>(ids.size());
                    geom_pool[pi] = static_cast<float>(std::exp(mean_log) - shift);
                }
                break;
            }
            case DK::MaxSim:
            default: {
                std::vector<float> max_pool(pool.size(), -std::numeric_limits<float>::infinity());
                for (std::size_t i = 0; i < frags.size(); ++i) {
                    float& slot = max_pool[frags[i].pool_index];
                    if (qsim[i] > slot)
                        slot = qsim[i];
                }
                for (std::size_t pi = 0; pi < pool.size(); ++pi)
                    geom_pool[pi] = std::isfinite(max_pool[pi]) ? max_pool[pi] : 0.0f;
                break;
            }
        }

        for (float& g : geom_pool)
            if (!std::isfinite(g))
                g = 0.0f;
        zscore_inplace(bm_pool);
        zscore_inplace(geom_pool);
        for (std::size_t i = 0; i < pool.size(); ++i)
            scores[pool[i].first] = query_alpha * bm_pool[i] + (1.0f - query_alpha) * geom_pool[i];
        if (profile)
            profile->total_us = elapsed_us(t_total0, Clock::now());
        return scores;
    }

    const std::uint32_t nf = static_cast<std::uint32_t>(frags.size());
    const std::uint32_t tri_count = nf * (nf - 1) / 2;
    const auto t_pair0 = profile ? Clock::now() : Clock::time_point{};

    // Optional prefix-dim graph vectors: copy the leading dims of each
    // whitened fragment and renormalize, so pairwise graph similarities stay
    // cosines. Query attention (qsim, above) is untouched. Default off.
    const bool use_graph_prefix = cfg.graph_prefix_dim > 0 && cfg.graph_prefix_dim < dim;
    const std::uint32_t gdim = use_graph_prefix ? cfg.graph_prefix_dim : dim;
    std::vector<float> gvecs;
    const float* gbase = fvecs.data();
    if (use_graph_prefix) {
        gvecs.resize(static_cast<std::size_t>(nf) * gdim);
        for (std::uint32_t i = 0; i < nf; ++i) {
            float* dst = gvecs.data() + static_cast<std::size_t>(i) * gdim;
            std::memcpy(dst, fvecs.data() + static_cast<std::size_t>(i) * dim,
                        gdim * sizeof(float));
            simd::l2_normalize(dst, gdim);
        }
        gbase = gvecs.data();
    }

    std::vector<float> sims_tri(tri_count);
    PhssStats sims_stats;
    sims_stats.count = tri_count;
    float sims_min = std::numeric_limits<float>::infinity();
    float sims_max = -std::numeric_limits<float>::infinity();
    // Rows are processed in pairs so the 2x4 kernel can share each b-row load
    // across both a-rows; every entry is still an independent dot, bit-identical
    // to the row-at-a-time form. Range stats for PHSS scale selection are taken
    // while rows are cache-hot; min/max are order-independent so the values
    // match a post-hoc scan exactly.
    const auto fold_range = [&](const float* row, std::uint32_t len) {
        if (len == 0)
            return;
        float row_min = sims_min, row_max = sims_max;
        simd::range(row, len, &row_min, &row_max);
        sims_min = std::min(sims_min, row_min);
        sims_max = std::max(sims_max, row_max);
    };
    for (std::uint32_t i = 0; i + 2 <= nf; i += 2) {
        const float* vi0 = gbase + static_cast<std::size_t>(i) * gdim;
        const float* vi1 = gbase + static_cast<std::size_t>(i + 1) * gdim;
        float* row0 = sims_tri.data() + static_cast<std::size_t>(i) * (2 * nf - i - 1) / 2;
        float* row1 = sims_tri.data() + static_cast<std::size_t>(i + 1) * (2 * nf - i - 2) / 2;
        row0[0] = dot(vi0, vi1, gdim);
        std::uint32_t j = i + 2;
        for (; j + 4 <= nf; j += 4) {
            const float* vj = gbase + static_cast<std::size_t>(j) * gdim;
            dot2x4(vi0, vi1, vj, vj + gdim, vj + 2 * gdim, vj + 3 * gdim, row0 + (j - i - 1),
                   row1 + (j - i - 2), gdim);
        }
        for (; j < nf; ++j) {
            const float* vj = gbase + static_cast<std::size_t>(j) * gdim;
            row0[j - i - 1] = dot(vi0, vj, gdim);
            row1[j - i - 2] = dot(vi1, vj, gdim);
        }
        fold_range(row0, nf - i - 1);
        fold_range(row1, nf - i - 2);
    }
    sims_stats.vmin = sims_min;
    sims_stats.vmax = sims_max;
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
    const float query_conf = cfg.phss_adaptive ? geometry_query_confidence(qsim) : 0.0f;
    if (cfg.use_phss) {
        if (profile)
            profile->phss_enabled = true;
        const bool should_run_phss =
            !cfg.phss_adaptive || query_conf >= cfg.phss_confidence_threshold;
        if (should_run_phss) {
            PhssConfig phss_cfg = cfg.phss_config;
            phss_cfg.dim_max = 0;
            const auto t_sel0 = profile ? Clock::now() : Clock::time_point{};
            auto phss_result = phss_select_scale(sims_tri, nf, phss_cfg, &sims_stats);
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
        }
    }

    if (cfg.phss_adaptive) {
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

    if (use_phss_for_graph) {
        // Survivor density above the PHSS scale is typically <1%, so extract
        // surviving pairs once with a SIMD threshold scan over the contiguous
        // triangle rows, then do the softmax bookkeeping on the sparse edge
        // list. In a row-major scan, row r receives partners in ascending index
        // order — the same order the per-row scan used — so row_sum
        // accumulation and the normalized weights are bit-identical; row_max
        // is a float max, order-independent.
        struct SurvivorEdge {
            std::uint32_t i, j;
            float sim;
        };
        std::vector<SurvivorEdge> survivors;
        std::vector<std::uint32_t> hits(frags.size());
        const std::uint32_t nfr = static_cast<std::uint32_t>(frags.size());
        for (std::uint32_t i = 0; i < nfr; ++i) {
            const float* row =
                sims_tri.data() + static_cast<std::size_t>(i) * (2 * nfr - i - 1) / 2;
            const std::uint32_t len = nfr - i - 1;
            const std::uint32_t cnt = simd::scan_ge(row, len, phss_selected_scale, hits.data());
            for (std::uint32_t k = 0; k < cnt; ++k)
                survivors.push_back({i, i + 1 + hits[k], row[hits[k]]});
        }

        std::vector<float> row_maxes(frags.size(), -std::numeric_limits<float>::infinity());
        std::vector<float> row_sums(frags.size(), 0.0f);
        for (const auto& e : survivors) {
            const float qsim_ij = query_scale * e.sim;
            row_maxes[e.i] = std::max(row_maxes[e.i], qsim_ij);
            row_maxes[e.j] = std::max(row_maxes[e.j], qsim_ij);
        }
        for (const auto& e : survivors) {
            const float wi = std::exp(query_scale * e.sim - row_maxes[e.i]);
            adj[e.i].push_back({e.j, wi});
            row_sums[e.i] += wi;
            const float wj = std::exp(query_scale * e.sim - row_maxes[e.j]);
            adj[e.j].push_back({e.i, wj});
            row_sums[e.j] += wj;
        }
        for (std::size_t i = 0; i < frags.size(); ++i) {
            if (row_sums[i] > 0.0f) {
                for (auto& [j, w] : adj[i])
                    w /= row_sums[i];
            }
            graph_edges += adj[i].size();
        }
    } else {
        std::vector<std::pair<float, std::uint32_t>> sims;
        sims.reserve(frags.size() > 0 ? frags.size() - 1 : 0);
        for (std::size_t i = 0; i < frags.size(); ++i) {
            sims.clear();
            for (std::size_t j = 0; j < i; ++j) {
                const float sim =
                    sim_at(static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j));
                sims.emplace_back(sim, static_cast<std::uint32_t>(j));
            }
            // Row i of the upper triangle is contiguous: entries (i, j) for j > i
            // start at offset i*(2*nf - i - 1)/2.
            const float* row =
                sims_tri.data() + static_cast<std::size_t>(i) * (2 * frags.size() - i - 1) / 2;
            for (std::size_t j = i + 1; j < frags.size(); ++j)
                sims.emplace_back(row[j - i - 1], static_cast<std::uint32_t>(j));

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

void read_frag_vec(const SemanticFragment& frag, std::uint32_t dim, float* dst) {
    read_frag_vec_impl(frag, dim, dst);
}

} // namespace simeon
