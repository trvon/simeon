#include "simeon/bm25.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "simeon/hasher.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

struct TfSink final : NGramEmitter {
    std::unordered_map<std::uint64_t, std::uint32_t>* tf;
    HashFamily family;
    std::uint64_t seed;
    void on_token(std::string_view tok, float) override { ++(*tf)[hash64(tok, seed, family)]; }
};

struct TermSink final : NGramEmitter {
    std::vector<std::uint64_t>* terms;
    HashFamily family;
    std::uint64_t seed;
    void on_token(std::string_view tok, float) override {
        terms->push_back(hash64(tok, seed, family));
    }
};

// Captures the raw query terms as strings as well as their hashes so the
// SubwordAwareBackoff scorer can re-tokenize OOV terms into char n-grams
// without going back through the tokenizer for the entire query.
struct StringTermSink final : NGramEmitter {
    std::vector<std::string>* strs;
    std::vector<std::uint64_t>* hashes;
    HashFamily family;
    std::uint64_t seed;
    void on_token(std::string_view tok, float) override {
        strs->emplace_back(tok);
        hashes->push_back(hash64(tok, seed, family));
    }
};

constexpr TokenizerConfig word_only_cfg() noexcept {
    return TokenizerConfig{0, 0, false, true};
}

inline TokenizerConfig ngram_only_cfg(std::uint32_t lo, std::uint32_t hi) noexcept {
    return TokenizerConfig{lo, hi, true, false};
}

// Per-(term, doc) Atire BM25 contribution.
inline float score_atire(float tf, float dl, float idf, float k1, float b, float avg_dl) noexcept {
    const float denom = tf + k1 * (1.0f - b + b * dl / avg_dl);
    return idf * tf * (k1 + 1.0f) / denom;
}

// Atire BM25 with Fang-Zhai 2005 axiomatic LTD correction. Replaces the
// linear length normalization (dl/avg_dl) with a sublinear power form
// (dl/avg_dl)^alpha; alpha=1 recovers Atire byte-identically.
inline float score_atire_ltd(float tf, float dl, float idf, float k1, float b, float avg_dl,
                             float alpha) noexcept {
    const float ratio = dl / avg_dl;
    // Fast path for alpha == 1 keeps byte-identity with score_atire and
    // avoids the powf call in the common no-op case.
    const float norm = (alpha == 1.0f) ? ratio : std::pow(ratio, alpha);
    const float denom = tf + k1 * (1.0f - b + b * norm);
    return idf * tf * (k1 + 1.0f) / denom;
}

// BM25+ (Lv & Zhai, CIKM 2011): adds δ floor to fix long-doc lower-bound
// violation. δ=0 recovers Atire.
inline float score_bm25_plus(float tf, float dl, float idf, float k1, float b, float avg_dl,
                             float delta) noexcept {
    const float denom = tf + k1 * (1.0f - b + b * dl / avg_dl);
    return idf * (tf * (k1 + 1.0f) / denom + delta);
}

// BM25L (Lv & Zhai, SIGIR 2011): different functional form than BM25+. tf is
// length-normalized first, then plugged into a saturating ratio with δ floor.
inline float score_bm25_l(float tf, float dl, float idf, float k1, float b, float avg_dl,
                          float delta) noexcept {
    const float c_prime = tf / (1.0f - b + b * dl / avg_dl);
    return idf * (k1 + 1.0f) * (c_prime + delta) / (k1 + c_prime + delta);
}

// DCM / Dirichlet-LM (Madsen-Kauchak-Elkan 2005; Zhai-Lafferty 2001):
// per-(term, doc) contribution log(1 + tf * total_tokens / (α_sum * ttf)).
// Natural log. Zero-safe: returns 0 for degenerate inputs so a pathological
// term never poisons a doc score.
inline float score_dcm(float tf, std::uint64_t total_tokens, float ttf, float alpha_sum) noexcept {
    if (tf <= 0.0f || ttf <= 0.0f || alpha_sum <= 0.0f || total_tokens == 0) {
        return 0.0f;
    }
    const float denom = alpha_sum * ttf;
    if (denom <= 0.0f)
        return 0.0f;
    const float ratio = tf * static_cast<float>(total_tokens) / denom;
    return std::log1p(ratio);
}

// PL2 (Amati & van Rijsbergen 2002, Terrier reference): Poisson with Laplace
// after-effect and L2 length normalization. Returns 0 for degenerate inputs
// so a pathological term never poisons a doc score.
inline float score_pl2(float tf, float dl, float avg_dl, float n_docs, float ttf,
                       float c) noexcept {
    if (dl <= 0.0f || tf <= 0.0f || ttf <= 0.0f || n_docs <= 0.0f || c <= 0.0f)
        return 0.0f;
    constexpr float kInvLog2 = 1.4426950408889634f; // 1/ln(2)
    const float norm_arg = 1.0f + c * avg_dl / dl;
    if (norm_arg <= 0.0f)
        return 0.0f;
    const float tfn = tf * std::log(norm_arg) * kInvLog2;
    if (tfn <= 0.0f)
        return 0.0f;
    const float lambda = ttf / n_docs;
    if (lambda <= 0.0f)
        return 0.0f;
    const float ratio = tfn / lambda;
    if (ratio <= 0.0f)
        return 0.0f;
    constexpr float kLog2E = 1.4426950408889634f; // log2(e) = 1/ln(2)
    const float term1 = tfn * std::log(ratio) * kInvLog2;
    const float term2 = (lambda - tfn) * kLog2E;
    const float two_pi_tfn = 2.0f * 3.14159265358979323846f * tfn;
    if (two_pi_tfn <= 0.0f)
        return 0.0f;
    const float term3 = 0.5f * std::log(two_pi_tfn) * kInvLog2;
    return (term1 + term2 + term3) / (tfn + 1.0f);
}

// DPH (Amati 2007 hypergeometric, Terrier reference): parameter-free DFR.
// norm = (1 - tf/dl)^2 / (tf + 1) gates the Stirling log-argument
// tf * avg_dl / dl * N / ttf. Returns 0 for degenerate inputs.
inline float score_dph(float tf, float dl, float avg_dl, float n_docs, float ttf) noexcept {
    if (dl <= 0.0f || tf <= 0.0f || ttf <= 0.0f || n_docs <= 0.0f)
        return 0.0f;
    const float f = tf / dl;
    const float one_minus_f = 1.0f - f;
    if (one_minus_f <= 0.0f)
        return 0.0f;
    const float norm = (one_minus_f * one_minus_f) / (tf + 1.0f);
    const float arg1 = (tf * avg_dl / dl) * (n_docs / ttf);
    if (arg1 <= 0.0f)
        return 0.0f;
    constexpr float kInvLog2 = 1.4426950408889634f; // 1/ln(2)
    const float term1 = tf * std::log(arg1) * kInvLog2;
    const float two_pi_tf_oneminusf = 2.0f * 3.14159265358979323846f * tf * one_minus_f;
    if (two_pi_tf_oneminusf <= 0.0f)
        return 0.0f;
    const float term2 = 0.5f * std::log(two_pi_tf_oneminusf) * kInvLog2;
    return norm * (term1 + term2);
}

// DLH13 (Amati DFR, Terrier reference impl): parameter-free divergence-from-
// randomness. Returns 0 for degenerate inputs (tf>=dl, ttf=0, dl=0) so a
// pathological term never poisons a doc score.
inline float score_dlh13(float tf, float dl, float avg_dl, float n_docs, float ttf) noexcept {
    if (dl <= 0.0f || tf <= 0.0f || ttf <= 0.0f || n_docs <= 0.0f)
        return 0.0f;
    const float f = tf / dl;
    const float one_minus_f = 1.0f - f;
    if (one_minus_f <= 0.0f)
        return 0.0f;
    const float arg1 = (tf * avg_dl / dl) * (n_docs / ttf);
    if (arg1 <= 0.0f)
        return 0.0f;
    constexpr float kInvLog2 = 1.4426950408889634f; // 1/ln(2)
    const float term1 = tf * std::log(arg1) * kInvLog2;
    const float two_pi_tf_oneminusf = 2.0f * 3.14159265358979323846f * tf * one_minus_f;
    if (two_pi_tf_oneminusf <= 0.0f)
        return 0.0f;
    const float term2 = 0.5f * std::log(two_pi_tf_oneminusf) * kInvLog2;
    return (term1 + term2) / (tf + 0.5f);
}

} // namespace

Bm25Index::Bm25Index(Bm25Config cfg) noexcept : cfg_(cfg) {}

std::uint64_t Bm25Index::hash_bigram(std::uint64_t a, std::uint64_t b) const noexcept {
    std::uint64_t h = splitmix64_mix(a ^ cfg_.hash_seed);
    h = splitmix64_mix(h + b);
    return h;
}

std::uint32_t Bm25Index::df(std::string_view term) const noexcept {
    if (!finalized_ || term.empty())
        return 0;
    const std::uint64_t h = hash64(term, cfg_.hash_seed, cfg_.hash);
    auto it = postings_.find(h);
    if (it == postings_.end())
        return 0;
    return static_cast<std::uint32_t>(it->second.docs.size());
}

float Bm25Index::idf(std::string_view term) const noexcept {
    if (!finalized_ || term.empty())
        return 0.0f;
    const std::uint64_t h = hash64(term, cfg_.hash_seed, cfg_.hash);
    auto it = postings_.find(h);
    if (it == postings_.end())
        return 0.0f;
    return it->second.idf;
}

std::uint64_t Bm25Index::hash_term(std::string_view term) const noexcept {
    return hash64(term, cfg_.hash_seed, cfg_.hash);
}

std::uint64_t Bm25Index::total_tf(std::string_view term) const noexcept {
    if (!finalized_ || term.empty())
        return 0;
    const std::uint64_t h = hash64(term, cfg_.hash_seed, cfg_.hash);
    auto it = postings_.find(h);
    if (it == postings_.end())
        return 0;
    return it->second.total_tf;
}

std::uint64_t Bm25Index::total_tf_by_hash(std::uint64_t term_hash) const noexcept {
    if (!finalized_)
        return 0;
    auto it = postings_.find(term_hash);
    if (it == postings_.end())
        return 0;
    return it->second.total_tf;
}

void Bm25Index::build_relevance_model(
    std::span<const std::uint32_t> top_k_docs, std::span<const float> doc_weights,
    std::vector<std::pair<std::uint64_t, float>>& out_terms) const {
    out_terms.clear();
    if (!finalized_ || top_k_docs.empty() || top_k_docs.size() != doc_weights.size())
        return;

    // Flat doc_id → weight map via small array (top_k_docs is typically small).
    // For each term, walk its posting list and accumulate weight when doc is
    // in the feedback set. Posting lists are in ascending doc-id order.
    std::unordered_map<std::uint32_t, float> weight_by_doc;
    weight_by_doc.reserve(top_k_docs.size() * 2);
    for (std::size_t i = 0; i < top_k_docs.size(); ++i) {
        weight_by_doc.emplace(top_k_docs[i], doc_weights[i]);
    }

    for (const auto& [h, tp] : postings_) {
        float acc = 0.0f;
        for (const auto& [did, tf] : tp.docs) {
            auto wit = weight_by_doc.find(did);
            if (wit == weight_by_doc.end())
                continue;
            const float dl = static_cast<float>(doc_lengths_[did]);
            if (dl <= 0.0f)
                continue;
            acc += (static_cast<float>(tf) / dl) * wit->second;
        }
        if (acc > 0.0f)
            out_terms.emplace_back(h, acc);
    }
}

void Bm25Index::score_weighted_hashes(
    std::span<const std::pair<std::uint64_t, float>> weighted_terms,
    std::span<float> out_scores) const {
    if (!finalized_)
        throw std::runtime_error("Bm25Index::score_weighted_hashes before finalize()");
    if (out_scores.size() != doc_lengths_.size()) {
        throw std::runtime_error("Bm25Index::score_weighted_hashes out_scores size mismatch");
    }
    std::fill(out_scores.begin(), out_scores.end(), 0.0f);

    const float k1 = cfg_.k1;
    const float b = cfg_.b;
    const float avg = avg_dl_ > 0.0f ? avg_dl_ : 1.0f;
    const float delta = cfg_.delta;
    const float n_docs = static_cast<float>(doc_lengths_.size());

    for (const auto& [h, w] : weighted_terms) {
        if (w <= 0.0f)
            continue;
        const auto pit = postings_.find(h);
        if (pit == postings_.end())
            continue;
        const float idf = pit->second.idf;
        const float ttf = static_cast<float>(pit->second.total_tf);
        for (const auto& [did, tf] : pit->second.docs) {
            const float tff = static_cast<float>(tf);
            const float dl = static_cast<float>(doc_lengths_[did]);
            float contribution = 0.0f;
            switch (cfg_.variant) {
                case Bm25Variant::Atire:
                    contribution = score_atire(tff, dl, idf, k1, b, avg);
                    break;
                case Bm25Variant::AtireLTD:
                    contribution = score_atire_ltd(tff, dl, idf, k1, b, avg, cfg_.ltd_alpha);
                    break;
                case Bm25Variant::BM25Plus:
                case Bm25Variant::SubwordAwareBackoff:
                    contribution = score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                    break;
                case Bm25Variant::BM25L:
                    contribution = score_bm25_l(tff, dl, idf, k1, b, avg, delta);
                    break;
                case Bm25Variant::DLH13:
                    contribution = score_dlh13(tff, dl, avg, n_docs, ttf);
                    break;
                case Bm25Variant::PL2:
                    contribution = score_pl2(tff, dl, avg, n_docs, ttf, cfg_.pl2_c);
                    break;
                case Bm25Variant::DPH:
                    contribution = score_dph(tff, dl, avg, n_docs, ttf);
                    break;
                case Bm25Variant::Dcm:
                    contribution = score_dcm(tff, total_tokens_, ttf, alpha_sum_);
                    break;
            }
            out_scores[did] += w * contribution;
        }
    }
}

void Bm25Index::add_doc(std::string_view text) {
    add_doc(text, {});
}

void Bm25Index::add_doc(std::string_view text, std::string_view aux_text) {
    if (finalized_)
        throw std::runtime_error("Bm25Index::add_doc after finalize()");
    const std::uint32_t did = static_cast<std::uint32_t>(doc_lengths_.size());
    auto add_word_field = [&](std::string_view field_text, std::vector<std::uint32_t>& lengths,
                              FlatHashMapU64<TermPostings>& postings) {
        // Reuse the per-doc tf scratch across calls. add_doc() is single-threaded
        // (the index is built sequentially before finalize), so no synchronization
        // is needed; clear() preserves bucket capacity which avoids the per-doc
        // hash-table reallocation cost.
        static thread_local std::unordered_map<std::uint64_t, std::uint32_t> tf;
        tf.clear();
        TfSink sink{};
        sink.tf = &tf;
        sink.family = cfg_.hash;
        sink.seed = cfg_.hash_seed;
        const auto tcfg = word_only_cfg();
        tokenize(field_text, tcfg, sink);

        std::uint32_t dl = 0;
        for (const auto& [h, c] : tf)
            dl += c;
        lengths.push_back(dl);
        for (const auto& [h, c] : tf) {
            postings[h].docs.emplace_back(did, c);
        }
    };

    auto add_ngram_field = [&](std::string_view field_text, std::vector<std::uint32_t>& lengths,
                               FlatHashMapU64<TermPostings>& postings) {
        static thread_local std::unordered_map<std::uint64_t, std::uint32_t> ngram_tf;
        ngram_tf.clear();
        TfSink ngram_sink{};
        ngram_sink.tf = &ngram_tf;
        ngram_sink.family = cfg_.hash;
        ngram_sink.seed = cfg_.hash_seed;
        const auto ntcfg = ngram_only_cfg(cfg_.ngram_min, cfg_.ngram_max);
        tokenize(field_text, ntcfg, ngram_sink);

        std::uint32_t ngram_dl = 0;
        for (const auto& [h, c] : ngram_tf)
            ngram_dl += c;
        lengths.push_back(ngram_dl);
        for (const auto& [h, c] : ngram_tf) {
            postings[h].docs.emplace_back(did, c);
        }
    };

    add_word_field(text, doc_lengths_, postings_);
    add_word_field(aux_text, aux_doc_lengths_, aux_postings_);

    if (cfg_.variant == Bm25Variant::SubwordAwareBackoff) {
        // Parallel char n-gram tf for the secondary index. Same hashing,
        // same per-thread scratch pattern as the word index above. n-gram
        // doc length is the sum of n-gram counts; differs from word dl
        // because each word emits multiple overlapping n-grams.
        add_ngram_field(text, ngram_doc_lengths_, ngram_postings_);
        add_ngram_field(aux_text, aux_ngram_doc_lengths_, aux_ngram_postings_);
    }

    if (cfg_.build_word_bigrams) {
        // Re-tokenize into an ordered word-hash stream (TfSink above loses
        // order). This doubles the word-tokenize cost for bigram-enabled
        // indexes — one-time at add_doc time, not on the query path.
        static thread_local std::vector<std::uint64_t> ordered_words;
        ordered_words.clear();
        TermSink ord_sink{};
        ord_sink.terms = &ordered_words;
        ord_sink.family = cfg_.hash;
        ord_sink.seed = cfg_.hash_seed;
        const auto tcfg = word_only_cfg();
        tokenize(text, tcfg, ord_sink);

        // Ordered bigrams: adjacent (w_i, w_{i+1}) pairs.
        static thread_local std::unordered_map<std::uint64_t, std::uint32_t> ord_bgm_tf;
        ord_bgm_tf.clear();
        for (std::size_t i = 1; i < ordered_words.size(); ++i) {
            ++ord_bgm_tf[hash_bigram(ordered_words[i - 1], ordered_words[i])];
        }
        for (const auto& [h, c] : ord_bgm_tf) {
            ordered_bigram_postings_[h].docs.emplace_back(did, c);
        }

        // Unordered bigrams: every pair within `bigram_unordered_window`
        // positions of each other, canonicalized via (min, max) so (a,b) and
        // (b,a) hash identically. Skips self-pairs (i != j).
        static thread_local std::unordered_map<std::uint64_t, std::uint32_t> unord_bgm_tf;
        unord_bgm_tf.clear();
        const std::uint32_t w = cfg_.bigram_unordered_window;
        if (w > 0 && ordered_words.size() >= 2) {
            for (std::size_t i = 0; i < ordered_words.size(); ++i) {
                const std::size_t hi = std::min<std::size_t>(i + 1 + w, ordered_words.size());
                for (std::size_t j = i + 1; j < hi; ++j) {
                    const std::uint64_t a = ordered_words[i];
                    const std::uint64_t b = ordered_words[j];
                    const std::uint64_t lo_h = std::min(a, b);
                    const std::uint64_t hi_h = std::max(a, b);
                    ++unord_bgm_tf[hash_bigram(lo_h, hi_h)];
                }
            }
        }
        for (const auto& [h, c] : unord_bgm_tf) {
            unordered_bigram_postings_[h].docs.emplace_back(did, c);
        }
    }
}

void Bm25Index::finalize() {
    if (finalized_)
        return;
    auto finalize_word_field = [&](const std::vector<std::uint32_t>& lengths,
                                   FlatHashMapU64<TermPostings>& postings, float& avg_dl,
                                   std::uint64_t& total_tokens, float& alpha_sum) {
        const std::size_t n = lengths.size();
        std::uint64_t total = 0;
        if (n == 0) {
            avg_dl = 0.0f;
        } else {
            for (auto l : lengths)
                total += l;
            avg_dl = static_cast<float>(static_cast<double>(total) / static_cast<double>(n));
        }
        total_tokens = total;
        alpha_sum = cfg_.dcm_alpha_sum > 0.0f ? cfg_.dcm_alpha_sum : avg_dl;
        const float nf = static_cast<float>(n);
        for (auto& [h, tp] : postings) {
            const float df = static_cast<float>(tp.docs.size());
            tp.idf = std::log((nf - df + 0.5f) / (df + 0.5f) + 1.0f);
            std::uint64_t ttf = 0;
            for (const auto& [_did, dtf] : tp.docs)
                ttf += dtf;
            tp.total_tf = ttf;
        }
    };

    auto finalize_ngram_field = [&](const std::vector<std::uint32_t>& lengths,
                                    FlatHashMapU64<TermPostings>& postings, float& avg_dl) {
        const std::size_t n = lengths.size();
        if (n == 0) {
            avg_dl = 0.0f;
        } else {
            std::uint64_t total = 0;
            for (auto l : lengths)
                total += l;
            avg_dl = static_cast<float>(static_cast<double>(total) / static_cast<double>(n));
        }
        const float nf = static_cast<float>(n);
        for (auto& [h, tp] : postings) {
            const float df = static_cast<float>(tp.docs.size());
            tp.idf = std::log((nf - df + 0.5f) / (df + 0.5f) + 1.0f);
            std::uint64_t ttf = 0;
            for (const auto& [_did, dtf] : tp.docs)
                ttf += dtf;
            tp.total_tf = ttf;
        }
    };

    finalize_word_field(doc_lengths_, postings_, avg_dl_, total_tokens_, alpha_sum_);
    finalize_word_field(aux_doc_lengths_, aux_postings_, aux_avg_dl_, aux_total_tokens_,
                        aux_alpha_sum_);

    if (cfg_.variant == Bm25Variant::SubwordAwareBackoff) {
        finalize_ngram_field(ngram_doc_lengths_, ngram_postings_, ngram_avg_dl_);
        finalize_ngram_field(aux_ngram_doc_lengths_, aux_ngram_postings_, aux_ngram_avg_dl_);
    }

    if (cfg_.build_word_bigrams) {
        const float nf = static_cast<float>(doc_lengths_.size());
        for (auto* table : {&ordered_bigram_postings_, &unordered_bigram_postings_}) {
            for (auto& [h, tp] : *table) {
                const float df = static_cast<float>(tp.docs.size());
                tp.idf = std::log((nf - df + 0.5f) / (df + 0.5f) + 1.0f);
                std::uint64_t ttf = 0;
                for (const auto& [_did, dtf] : tp.docs)
                    ttf += dtf;
                tp.total_tf = ttf;
            }
        }
    }
    finalized_ = true;
}

void Bm25Index::score(std::string_view query, std::span<float> out_scores) const {
    if (!finalized_)
        throw std::runtime_error("Bm25Index::score before finalize()");
    if (out_scores.size() != doc_lengths_.size()) {
        throw std::runtime_error("Bm25Index::score out_scores size mismatch");
    }
    std::fill(out_scores.begin(), out_scores.end(), 0.0f);

    const float k1 = cfg_.k1;
    const float b = cfg_.b;
    const float avg = avg_dl_ > 0.0f ? avg_dl_ : 1.0f;
    const float delta = cfg_.delta;
    const float n_docs = static_cast<float>(doc_lengths_.size());

    if (cfg_.variant == Bm25Variant::SubwordAwareBackoff) {
        // Need raw query strings for the OOV n-gram fallback; store hashes
        // alongside so we still get a fast postings_ lookup for the exact
        // path. score() is const but the scratch is thread_local.
        static thread_local std::vector<std::string> q_strs;
        static thread_local std::vector<std::uint64_t> q_hashes;
        q_strs.clear();
        q_hashes.clear();
        StringTermSink ssink{};
        ssink.strs = &q_strs;
        ssink.hashes = &q_hashes;
        ssink.family = cfg_.hash;
        ssink.seed = cfg_.hash_seed;
        const auto tcfg = word_only_cfg();
        tokenize(query, tcfg, ssink);

        const float gamma = cfg_.subword_gamma;
        const float ngram_avg = ngram_avg_dl_ > 0.0f ? ngram_avg_dl_ : 1.0f;
        // Per-query reusable scratch for the n-gram tokenization of one term.
        static thread_local std::unordered_map<std::uint64_t, std::uint32_t> term_ngram_tf;
        for (std::size_t qi = 0; qi < q_hashes.size(); ++qi) {
            const std::uint64_t h = q_hashes[qi];
            const auto pit = postings_.find(h);
            const std::uint64_t df = (pit == postings_.end()) ? 0 : pit->second.docs.size();
            const float alpha =
                (df > 0) ? static_cast<float>(df) / (static_cast<float>(df) + gamma) : 0.0f;

            // Exact contribution (BM25+) — only when the term is present.
            if (alpha > 0.0f) {
                const float idf = pit->second.idf;
                for (const auto& [did, tf] : pit->second.docs) {
                    const float tff = static_cast<float>(tf);
                    const float dl = static_cast<float>(doc_lengths_[did]);
                    out_scores[did] += alpha * score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                }
            }

            // N-gram contribution: tokenize this query term into char
            // n-grams, look each up in ngram_postings_, sum BM25+ scoring
            // weighted by (1-α). Skips entirely if α==1 (γ=0 + df>0).
            if (alpha < 1.0f) {
                term_ngram_tf.clear();
                TfSink ng_sink{};
                ng_sink.tf = &term_ngram_tf;
                ng_sink.family = cfg_.hash;
                ng_sink.seed = cfg_.hash_seed;
                const auto ntcfg = ngram_only_cfg(cfg_.ngram_min, cfg_.ngram_max);
                tokenize(q_strs[qi], ntcfg, ng_sink);
                const float w = 1.0f - alpha;
                for (const auto& [gh, _qcount] : term_ngram_tf) {
                    const auto git = ngram_postings_.find(gh);
                    if (git == ngram_postings_.end())
                        continue;
                    const float idf_g = git->second.idf;
                    for (const auto& [did, tf] : git->second.docs) {
                        const float tff = static_cast<float>(tf);
                        const float dl = static_cast<float>(ngram_doc_lengths_[did]);
                        out_scores[did] +=
                            w * score_bm25_plus(tff, dl, idf_g, k1, b, ngram_avg, delta);
                    }
                }
            }
        }
        return;
    }

    // Standard variants (Atire, BM25Plus, BM25L, DLH13): tokenize once,
    // dispatch the per-term per-doc inner loop on cfg_.variant.
    static thread_local std::vector<std::uint64_t> q_terms;
    q_terms.clear();
    TermSink sink{};
    sink.terms = &q_terms;
    sink.family = cfg_.hash;
    sink.seed = cfg_.hash_seed;
    const auto tcfg = word_only_cfg();
    tokenize(query, tcfg, sink);

    for (std::uint64_t h : q_terms) {
        const auto pit = postings_.find(h);
        if (pit == postings_.end())
            continue;
        const float idf = pit->second.idf;
        const float ttf = static_cast<float>(pit->second.total_tf);
        for (const auto& [did, tf] : pit->second.docs) {
            const float tff = static_cast<float>(tf);
            const float dl = static_cast<float>(doc_lengths_[did]);
            float contribution = 0.0f;
            switch (cfg_.variant) {
                case Bm25Variant::Atire:
                    contribution = score_atire(tff, dl, idf, k1, b, avg);
                    break;
                case Bm25Variant::AtireLTD:
                    contribution = score_atire_ltd(tff, dl, idf, k1, b, avg, cfg_.ltd_alpha);
                    break;
                case Bm25Variant::BM25Plus:
                    contribution = score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                    break;
                case Bm25Variant::BM25L:
                    contribution = score_bm25_l(tff, dl, idf, k1, b, avg, delta);
                    break;
                case Bm25Variant::DLH13:
                    contribution = score_dlh13(tff, dl, avg, n_docs, ttf);
                    break;
                case Bm25Variant::PL2:
                    contribution = score_pl2(tff, dl, avg, n_docs, ttf, cfg_.pl2_c);
                    break;
                case Bm25Variant::DPH:
                    contribution = score_dph(tff, dl, avg, n_docs, ttf);
                    break;
                case Bm25Variant::Dcm:
                    contribution = score_dcm(tff, total_tokens_, ttf, alpha_sum_);
                    break;
                case Bm25Variant::SubwordAwareBackoff:
                    // Unreachable: handled by the early return above.
                    break;
            }
            out_scores[did] += contribution;
        }
    }
}

void Bm25Index::score_bm25f(std::string_view query, std::span<float> out_scores, float weight_body,
                            float weight_aux) const {
    score_bm25f(query, query, out_scores, weight_body, weight_aux);
}

void Bm25Index::score_bm25f(std::string_view body_query, std::string_view aux_query,
                            std::span<float> out_scores, float weight_body,
                            float weight_aux) const {
    if (!finalized_)
        throw std::runtime_error("Bm25Index::score_bm25f before finalize()");
    if (out_scores.size() != doc_lengths_.size()) {
        throw std::runtime_error("Bm25Index::score_bm25f out_scores size mismatch");
    }
    if (weight_body == 1.0f && weight_aux == 0.0f) {
        score(body_query, out_scores);
        return;
    }

    if (weight_body != 0.0f) {
        score(body_query, out_scores);
        if (weight_body != 1.0f) {
            for (float& s : out_scores)
                s *= weight_body;
        }
    } else {
        std::fill(out_scores.begin(), out_scores.end(), 0.0f);
    }

    if (weight_aux == 0.0f ||
        (aux_postings_.empty() &&
         (cfg_.variant != Bm25Variant::SubwordAwareBackoff || aux_ngram_postings_.empty()))) {
        return;
    }

    const float k1 = cfg_.k1;
    const float b = cfg_.b;
    const float avg = aux_avg_dl_ > 0.0f ? aux_avg_dl_ : 1.0f;
    const float delta = cfg_.delta;
    const float n_docs = static_cast<float>(aux_doc_lengths_.size());

    if (cfg_.variant == Bm25Variant::SubwordAwareBackoff) {
        static thread_local std::vector<std::string> q_strs;
        static thread_local std::vector<std::uint64_t> q_hashes;
        q_strs.clear();
        q_hashes.clear();
        StringTermSink ssink{};
        ssink.strs = &q_strs;
        ssink.hashes = &q_hashes;
        ssink.family = cfg_.hash;
        ssink.seed = cfg_.hash_seed;
        const auto tcfg = word_only_cfg();
        tokenize(aux_query, tcfg, ssink);

        const float gamma = cfg_.subword_gamma;
        const float aux_ngram_avg = aux_ngram_avg_dl_ > 0.0f ? aux_ngram_avg_dl_ : 1.0f;
        static thread_local std::unordered_map<std::uint64_t, std::uint32_t> term_ngram_tf;
        for (std::size_t qi = 0; qi < q_hashes.size(); ++qi) {
            const std::uint64_t h = q_hashes[qi];
            const auto pit = aux_postings_.find(h);
            const std::uint64_t df = (pit == aux_postings_.end()) ? 0 : pit->second.docs.size();
            const float alpha =
                (df > 0) ? static_cast<float>(df) / (static_cast<float>(df) + gamma) : 0.0f;

            if (alpha > 0.0f) {
                const float idf = pit->second.idf;
                for (const auto& [did, tf] : pit->second.docs) {
                    const float tff = static_cast<float>(tf);
                    const float dl = static_cast<float>(aux_doc_lengths_[did]);
                    out_scores[did] +=
                        weight_aux * alpha * score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                }
            }

            if (alpha < 1.0f) {
                term_ngram_tf.clear();
                TfSink ng_sink{};
                ng_sink.tf = &term_ngram_tf;
                ng_sink.family = cfg_.hash;
                ng_sink.seed = cfg_.hash_seed;
                const auto ntcfg = ngram_only_cfg(cfg_.ngram_min, cfg_.ngram_max);
                tokenize(q_strs[qi], ntcfg, ng_sink);
                const float w = 1.0f - alpha;
                for (const auto& [gh, _qcount] : term_ngram_tf) {
                    const auto git = aux_ngram_postings_.find(gh);
                    if (git == aux_ngram_postings_.end())
                        continue;
                    const float idf_g = git->second.idf;
                    for (const auto& [did, tf] : git->second.docs) {
                        const float tff = static_cast<float>(tf);
                        const float dl = static_cast<float>(aux_ngram_doc_lengths_[did]);
                        out_scores[did] +=
                            weight_aux * w *
                            score_bm25_plus(tff, dl, idf_g, k1, b, aux_ngram_avg, delta);
                    }
                }
            }
        }
        return;
    }

    static thread_local std::vector<std::uint64_t> q_terms;
    q_terms.clear();
    TermSink sink{};
    sink.terms = &q_terms;
    sink.family = cfg_.hash;
    sink.seed = cfg_.hash_seed;
    const auto tcfg = word_only_cfg();
    tokenize(aux_query, tcfg, sink);

    for (std::uint64_t h : q_terms) {
        const auto pit = aux_postings_.find(h);
        if (pit == aux_postings_.end())
            continue;
        const float idf = pit->second.idf;
        const float ttf = static_cast<float>(pit->second.total_tf);
        for (const auto& [did, tf] : pit->second.docs) {
            const float tff = static_cast<float>(tf);
            const float dl = static_cast<float>(aux_doc_lengths_[did]);
            float contribution = 0.0f;
            switch (cfg_.variant) {
                case Bm25Variant::Atire:
                    contribution = score_atire(tff, dl, idf, k1, b, avg);
                    break;
                case Bm25Variant::AtireLTD:
                    contribution = score_atire_ltd(tff, dl, idf, k1, b, avg, cfg_.ltd_alpha);
                    break;
                case Bm25Variant::BM25Plus:
                    contribution = score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                    break;
                case Bm25Variant::BM25L:
                    contribution = score_bm25_l(tff, dl, idf, k1, b, avg, delta);
                    break;
                case Bm25Variant::DLH13:
                    contribution = score_dlh13(tff, dl, avg, n_docs, ttf);
                    break;
                case Bm25Variant::PL2:
                    contribution = score_pl2(tff, dl, avg, n_docs, ttf, cfg_.pl2_c);
                    break;
                case Bm25Variant::DPH:
                    contribution = score_dph(tff, dl, avg, n_docs, ttf);
                    break;
                case Bm25Variant::Dcm:
                    contribution = score_dcm(tff, aux_total_tokens_, ttf, aux_alpha_sum_);
                    break;
                case Bm25Variant::SubwordAwareBackoff:
                    break;
            }
            out_scores[did] += weight_aux * contribution;
        }
    }
}

void Bm25Index::score_sdm(std::string_view query, std::span<float> out_scores,
                          const SdmConfig& cfg) const {
    if (!finalized_)
        throw std::runtime_error("Bm25Index::score_sdm before finalize()");
    if (out_scores.size() != doc_lengths_.size()) {
        throw std::runtime_error("Bm25Index::score_sdm out_scores size mismatch");
    }
    std::fill(out_scores.begin(), out_scores.end(), 0.0f);

    const float k1 = cfg_.k1;
    const float b = cfg_.b;
    const float avg = avg_dl_ > 0.0f ? avg_dl_ : 1.0f;
    const float delta = cfg_.delta;
    const float n_docs = static_cast<float>(doc_lengths_.size());

    // Ordered query tokenization (same path as score()). Keep the strings too
    // so the SAB unigram leg can do its OOV n-gram fallback per term.
    static thread_local std::vector<std::string> q_strs;
    static thread_local std::vector<std::uint64_t> q_hashes;
    q_strs.clear();
    q_hashes.clear();
    StringTermSink ssink{};
    ssink.strs = &q_strs;
    ssink.hashes = &q_hashes;
    ssink.family = cfg_.hash;
    ssink.seed = cfg_.hash_seed;
    const auto tcfg = word_only_cfg();
    tokenize(query, tcfg, ssink);

    // --- Unigram leg: configured Bm25Variant, scaled by lambda_unigram. ---
    if (cfg.lambda_unigram != 0.0f) {
        if (cfg_.variant == Bm25Variant::SubwordAwareBackoff) {
            const float gamma = cfg_.subword_gamma;
            const float ngram_avg = ngram_avg_dl_ > 0.0f ? ngram_avg_dl_ : 1.0f;
            static thread_local std::unordered_map<std::uint64_t, std::uint32_t> term_ngram_tf;
            for (std::size_t qi = 0; qi < q_hashes.size(); ++qi) {
                const std::uint64_t h = q_hashes[qi];
                const auto pit = postings_.find(h);
                const std::uint64_t df = (pit == postings_.end()) ? 0 : pit->second.docs.size();
                const float alpha =
                    (df > 0) ? static_cast<float>(df) / (static_cast<float>(df) + gamma) : 0.0f;
                if (alpha > 0.0f) {
                    const float idf = pit->second.idf;
                    for (const auto& [did, tf] : pit->second.docs) {
                        const float tff = static_cast<float>(tf);
                        const float dl = static_cast<float>(doc_lengths_[did]);
                        out_scores[did] += cfg.lambda_unigram * alpha *
                                           score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                    }
                }
                if (alpha < 1.0f) {
                    term_ngram_tf.clear();
                    TfSink ng_sink{};
                    ng_sink.tf = &term_ngram_tf;
                    ng_sink.family = cfg_.hash;
                    ng_sink.seed = cfg_.hash_seed;
                    const auto ntcfg = ngram_only_cfg(cfg_.ngram_min, cfg_.ngram_max);
                    tokenize(q_strs[qi], ntcfg, ng_sink);
                    const float w = 1.0f - alpha;
                    for (const auto& [gh, _qcount] : term_ngram_tf) {
                        const auto git = ngram_postings_.find(gh);
                        if (git == ngram_postings_.end())
                            continue;
                        const float idf_g = git->second.idf;
                        for (const auto& [did, tf] : git->second.docs) {
                            const float tff = static_cast<float>(tf);
                            const float dl = static_cast<float>(ngram_doc_lengths_[did]);
                            out_scores[did] +=
                                cfg.lambda_unigram * w *
                                score_bm25_plus(tff, dl, idf_g, k1, b, ngram_avg, delta);
                        }
                    }
                }
            }
        } else {
            for (std::uint64_t h : q_hashes) {
                const auto pit = postings_.find(h);
                if (pit == postings_.end())
                    continue;
                const float idf = pit->second.idf;
                const float ttf = static_cast<float>(pit->second.total_tf);
                for (const auto& [did, tf] : pit->second.docs) {
                    const float tff = static_cast<float>(tf);
                    const float dl = static_cast<float>(doc_lengths_[did]);
                    float contribution = 0.0f;
                    switch (cfg_.variant) {
                        case Bm25Variant::Atire:
                            contribution = score_atire(tff, dl, idf, k1, b, avg);
                            break;
                        case Bm25Variant::AtireLTD:
                            contribution =
                                score_atire_ltd(tff, dl, idf, k1, b, avg, cfg_.ltd_alpha);
                            break;
                        case Bm25Variant::BM25Plus:
                            contribution = score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                            break;
                        case Bm25Variant::BM25L:
                            contribution = score_bm25_l(tff, dl, idf, k1, b, avg, delta);
                            break;
                        case Bm25Variant::DLH13:
                            contribution = score_dlh13(tff, dl, avg, n_docs, ttf);
                            break;
                        case Bm25Variant::PL2:
                            contribution = score_pl2(tff, dl, avg, n_docs, ttf, cfg_.pl2_c);
                            break;
                        case Bm25Variant::DPH:
                            contribution = score_dph(tff, dl, avg, n_docs, ttf);
                            break;
                        case Bm25Variant::Dcm:
                            contribution = score_dcm(tff, total_tokens_, ttf, alpha_sum_);
                            break;
                        case Bm25Variant::SubwordAwareBackoff:
                            // Unreachable: handled above.
                            break;
                    }
                    out_scores[did] += cfg.lambda_unigram * contribution;
                }
            }
        }
    }

    // --- Ordered bigram leg: adjacent (q_i, q_{i+1}) pairs, Atire BM25. ---
    if (cfg.lambda_ordered != 0.0f && q_hashes.size() >= 2 && !ordered_bigram_postings_.empty()) {
        for (std::size_t i = 1; i < q_hashes.size(); ++i) {
            const std::uint64_t bh = hash_bigram(q_hashes[i - 1], q_hashes[i]);
            const auto pit = ordered_bigram_postings_.find(bh);
            if (pit == ordered_bigram_postings_.end())
                continue;
            const float idf = pit->second.idf;
            for (const auto& [did, tf] : pit->second.docs) {
                const float tff = static_cast<float>(tf);
                const float dl = static_cast<float>(doc_lengths_[did]);
                out_scores[did] += cfg.lambda_ordered * score_atire(tff, dl, idf, k1, b, avg);
            }
        }
    }

    // --- Unordered bigram leg: adjacent query pairs, canonicalized. ---
    // Query-side pairs are still adjacent (Metzler §3): the *window* is the
    // doc-side concept, captured when the index was built.
    if (cfg.lambda_unordered != 0.0f && q_hashes.size() >= 2 &&
        !unordered_bigram_postings_.empty()) {
        for (std::size_t i = 1; i < q_hashes.size(); ++i) {
            const std::uint64_t a = q_hashes[i - 1];
            const std::uint64_t b2 = q_hashes[i];
            if (a == b2)
                continue;
            const std::uint64_t lo_h = std::min(a, b2);
            const std::uint64_t hi_h = std::max(a, b2);
            const std::uint64_t bh = hash_bigram(lo_h, hi_h);
            const auto pit = unordered_bigram_postings_.find(bh);
            if (pit == unordered_bigram_postings_.end())
                continue;
            const float idf = pit->second.idf;
            for (const auto& [did, tf] : pit->second.docs) {
                const float tff = static_cast<float>(tf);
                const float dl = static_cast<float>(doc_lengths_[did]);
                out_scores[did] += cfg.lambda_unordered * score_atire(tff, dl, idf, k1, b, avg);
            }
        }
    }
}

void Bm25Index::score_wsdm(std::string_view query, std::span<float> out_scores,
                           const WeightedSdmConfig& cfg) const {
    if (!finalized_)
        throw std::runtime_error("Bm25Index::score_wsdm before finalize()");
    if (out_scores.size() != doc_lengths_.size()) {
        throw std::runtime_error("Bm25Index::score_wsdm out_scores size mismatch");
    }

    // β==0 path is byte-identical to fixed SDM — dispatch to score_sdm so
    // the existing well-tested codepath is exercised and we never pay the
    // bigram-IDF prepass for a no-op weighting.
    if (cfg.beta == 0.0f) {
        SdmConfig sdm_cfg{cfg.lambda_unigram, cfg.lambda_ordered, cfg.lambda_unordered};
        score_sdm(query, out_scores, sdm_cfg);
        return;
    }

    std::fill(out_scores.begin(), out_scores.end(), 0.0f);

    const float k1 = cfg_.k1;
    const float b = cfg_.b;
    const float avg = avg_dl_ > 0.0f ? avg_dl_ : 1.0f;
    const float delta = cfg_.delta;
    const float n_docs = static_cast<float>(doc_lengths_.size());

    static thread_local std::vector<std::string> q_strs;
    static thread_local std::vector<std::uint64_t> q_hashes;
    q_strs.clear();
    q_hashes.clear();
    StringTermSink ssink{};
    ssink.strs = &q_strs;
    ssink.hashes = &q_hashes;
    ssink.family = cfg_.hash;
    ssink.seed = cfg_.hash_seed;
    const auto tcfg = word_only_cfg();
    tokenize(query, tcfg, ssink);

    // --- Unigram leg: identical to score_sdm (no per-term IDF weighting on
    // unigrams; Bendersky-Croft 2010 reweights only the dependence legs). ---
    if (cfg.lambda_unigram != 0.0f) {
        if (cfg_.variant == Bm25Variant::SubwordAwareBackoff) {
            const float gamma = cfg_.subword_gamma;
            const float ngram_avg = ngram_avg_dl_ > 0.0f ? ngram_avg_dl_ : 1.0f;
            static thread_local std::unordered_map<std::uint64_t, std::uint32_t> term_ngram_tf;
            for (std::size_t qi = 0; qi < q_hashes.size(); ++qi) {
                const std::uint64_t h = q_hashes[qi];
                const auto pit = postings_.find(h);
                const std::uint64_t df = (pit == postings_.end()) ? 0 : pit->second.docs.size();
                const float alpha =
                    (df > 0) ? static_cast<float>(df) / (static_cast<float>(df) + gamma) : 0.0f;
                if (alpha > 0.0f) {
                    const float idf = pit->second.idf;
                    for (const auto& [did, tf] : pit->second.docs) {
                        const float tff = static_cast<float>(tf);
                        const float dl = static_cast<float>(doc_lengths_[did]);
                        out_scores[did] += cfg.lambda_unigram * alpha *
                                           score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                    }
                }
                if (alpha < 1.0f) {
                    term_ngram_tf.clear();
                    TfSink ng_sink{};
                    ng_sink.tf = &term_ngram_tf;
                    ng_sink.family = cfg_.hash;
                    ng_sink.seed = cfg_.hash_seed;
                    const auto ntcfg = ngram_only_cfg(cfg_.ngram_min, cfg_.ngram_max);
                    tokenize(q_strs[qi], ntcfg, ng_sink);
                    const float w = 1.0f - alpha;
                    for (const auto& [gh, _qcount] : term_ngram_tf) {
                        const auto git = ngram_postings_.find(gh);
                        if (git == ngram_postings_.end())
                            continue;
                        const float idf_g = git->second.idf;
                        for (const auto& [did, tf] : git->second.docs) {
                            const float tff = static_cast<float>(tf);
                            const float dl = static_cast<float>(ngram_doc_lengths_[did]);
                            out_scores[did] +=
                                cfg.lambda_unigram * w *
                                score_bm25_plus(tff, dl, idf_g, k1, b, ngram_avg, delta);
                        }
                    }
                }
            }
        } else {
            for (std::uint64_t h : q_hashes) {
                const auto pit = postings_.find(h);
                if (pit == postings_.end())
                    continue;
                const float idf = pit->second.idf;
                const float ttf = static_cast<float>(pit->second.total_tf);
                for (const auto& [did, tf] : pit->second.docs) {
                    const float tff = static_cast<float>(tf);
                    const float dl = static_cast<float>(doc_lengths_[did]);
                    float contribution = 0.0f;
                    switch (cfg_.variant) {
                        case Bm25Variant::Atire:
                            contribution = score_atire(tff, dl, idf, k1, b, avg);
                            break;
                        case Bm25Variant::AtireLTD:
                            contribution =
                                score_atire_ltd(tff, dl, idf, k1, b, avg, cfg_.ltd_alpha);
                            break;
                        case Bm25Variant::BM25Plus:
                            contribution = score_bm25_plus(tff, dl, idf, k1, b, avg, delta);
                            break;
                        case Bm25Variant::BM25L:
                            contribution = score_bm25_l(tff, dl, idf, k1, b, avg, delta);
                            break;
                        case Bm25Variant::DLH13:
                            contribution = score_dlh13(tff, dl, avg, n_docs, ttf);
                            break;
                        case Bm25Variant::PL2:
                            contribution = score_pl2(tff, dl, avg, n_docs, ttf, cfg_.pl2_c);
                            break;
                        case Bm25Variant::DPH:
                            contribution = score_dph(tff, dl, avg, n_docs, ttf);
                            break;
                        case Bm25Variant::Dcm:
                            contribution = score_dcm(tff, total_tokens_, ttf, alpha_sum_);
                            break;
                        case Bm25Variant::SubwordAwareBackoff:
                            break;
                    }
                    out_scores[did] += cfg.lambda_unigram * contribution;
                }
            }
        }
    }

    // Bigram-IDF prepass: per Bendersky-Croft 2010 §4, per-bigram weights are
    // a function of the bigram's discriminative power. We collect ordered and
    // unordered bigram IDFs over the query in a single pass, compute the
    // mean, then in the scoring legs scale each bigram's λ by
    // (idf_b / mean_idf)^β. Mean is across legs jointly so the per-query
    // mean weight equals the fixed-λ baseline (β=0 ⇒ no scaling, β=1 ⇒
    // canonical IDF reweighting). Empty/uniform IDF sets degrade gracefully
    // to fixed λ.
    auto compute_weights = [&](bool ordered, std::vector<float>& weights) {
        weights.clear();
        if (q_hashes.size() < 2)
            return;
        const auto& postings_map = ordered ? ordered_bigram_postings_ : unordered_bigram_postings_;
        if (postings_map.empty())
            return;
        weights.reserve(q_hashes.size() - 1);
        double idf_sum = 0.0;
        std::size_t n_present = 0;
        for (std::size_t i = 1; i < q_hashes.size(); ++i) {
            std::uint64_t bh;
            if (ordered) {
                bh = hash_bigram(q_hashes[i - 1], q_hashes[i]);
            } else {
                if (q_hashes[i - 1] == q_hashes[i]) {
                    weights.push_back(0.0f);
                    continue;
                }
                const std::uint64_t lo_h = std::min(q_hashes[i - 1], q_hashes[i]);
                const std::uint64_t hi_h = std::max(q_hashes[i - 1], q_hashes[i]);
                bh = hash_bigram(lo_h, hi_h);
            }
            const auto pit = postings_map.find(bh);
            if (pit == postings_map.end()) {
                weights.push_back(0.0f);
                continue;
            }
            const float idf = pit->second.idf;
            weights.push_back(idf);
            idf_sum += idf;
            ++n_present;
        }
        if (n_present == 0)
            return;
        const float mean = static_cast<float>(idf_sum / static_cast<double>(n_present));
        if (mean <= 0.0f)
            return;
        for (auto& w : weights) {
            if (w <= 0.0f) {
                w = 0.0f;
                continue;
            }
            const float ratio = w / mean;
            // Powf with β; small-β fast path uses exp(β*log(ratio)) which is
            // already what powf computes — no special-case needed.
            w = std::pow(ratio, cfg.beta);
        }
    };

    static thread_local std::vector<float> ord_weights;
    static thread_local std::vector<float> uw_weights;
    compute_weights(true, ord_weights);
    compute_weights(false, uw_weights);

    // --- Ordered bigram leg with per-bigram weighting. ---
    if (cfg.lambda_ordered != 0.0f && q_hashes.size() >= 2 && !ordered_bigram_postings_.empty()) {
        for (std::size_t i = 1; i < q_hashes.size(); ++i) {
            const std::uint64_t bh = hash_bigram(q_hashes[i - 1], q_hashes[i]);
            const auto pit = ordered_bigram_postings_.find(bh);
            if (pit == ordered_bigram_postings_.end())
                continue;
            const float w = (i - 1) < ord_weights.size() ? ord_weights[i - 1] : 1.0f;
            if (w == 0.0f)
                continue;
            const float idf = pit->second.idf;
            const float lam = cfg.lambda_ordered * w;
            for (const auto& [did, tf] : pit->second.docs) {
                const float tff = static_cast<float>(tf);
                const float dl = static_cast<float>(doc_lengths_[did]);
                out_scores[did] += lam * score_atire(tff, dl, idf, k1, b, avg);
            }
        }
    }

    // --- Unordered bigram leg with per-bigram weighting. ---
    if (cfg.lambda_unordered != 0.0f && q_hashes.size() >= 2 &&
        !unordered_bigram_postings_.empty()) {
        for (std::size_t i = 1; i < q_hashes.size(); ++i) {
            const std::uint64_t a = q_hashes[i - 1];
            const std::uint64_t b2 = q_hashes[i];
            if (a == b2)
                continue;
            const std::uint64_t lo_h = std::min(a, b2);
            const std::uint64_t hi_h = std::max(a, b2);
            const std::uint64_t bh = hash_bigram(lo_h, hi_h);
            const auto pit = unordered_bigram_postings_.find(bh);
            if (pit == unordered_bigram_postings_.end())
                continue;
            const float w = (i - 1) < uw_weights.size() ? uw_weights[i - 1] : 1.0f;
            if (w == 0.0f)
                continue;
            const float idf = pit->second.idf;
            const float lam = cfg.lambda_unordered * w;
            for (const auto& [did, tf] : pit->second.docs) {
                const float tff = static_cast<float>(tf);
                const float dl = static_cast<float>(doc_lengths_[did]);
                out_scores[did] += lam * score_atire(tff, dl, idf, k1, b, avg);
            }
        }
    }
}

} // namespace simeon
