#include "simeon/simeon.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simeon/hashed_idf.hpp"
#include "simeon/hasher.hpp"
#include "simeon/pmi.hpp"
#include "simeon/projection.hpp"
#include "simeon/simd.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

std::uint64_t integer_sqrt(std::uint64_t value) noexcept {
    if (value == 0)
        return 0;
    std::uint64_t root = static_cast<std::uint64_t>(std::sqrt(static_cast<double>(value)));
    while (root + 1 <= value / (root + 1))
        ++root;
    while (root > value / root)
        --root;
    return root;
}

std::int32_t signed_sqrt_fixed(std::int32_t value) noexcept {
    if (value == 0)
        return 0;
    const auto wide = static_cast<std::int64_t>(value);
    const std::uint64_t magnitude =
        wide < 0 ? static_cast<std::uint64_t>(-wide) : static_cast<std::uint64_t>(wide);
    // Q10 fixed point: floor(1024 * sqrt(magnitude)). The common factor is
    // removed by L2 normalization while retaining precision for projection.
    const auto weighted = static_cast<std::int32_t>(integer_sqrt(magnitude << 20));
    return value < 0 ? -weighted : weighted;
}

std::uint32_t sketch_bucket(std::uint64_t hash, std::uint32_t dimension) noexcept {
    const auto low = static_cast<std::uint32_t>(hash);
    return (dimension & (dimension - 1)) == 0 ? low & (dimension - 1) : low % dimension;
}

} // namespace

SimdTier active_simd_tier() noexcept {
#if defined(SIMEON_HAS_NEON)
    return SimdTier::Neon;
#elif defined(SIMEON_HAS_AVX2)
    return SimdTier::Avx2;
#else
    return SimdTier::Scalar;
#endif
}

const char* simd_tier_name(SimdTier tier) noexcept {
    switch (tier) {
        case SimdTier::Scalar:
            return "scalar";
        case SimdTier::Neon:
            return "neon";
        case SimdTier::Avx2:
            return "avx2";
    }
    return "unknown";
}

EncoderConfig simeon_v1_384_config() {
    EncoderConfig config;
    config.ngram_mode = NGramMode::CharOnly;
    config.ngram_min = 3;
    config.ngram_max = 5;
    config.hash = HashFamily::SplitMix64;
    config.hash_seed = 0xA5A5A5A5A5A5A5A5ULL;
    config.text_normalization = TextNormalization::None;
    config.char_ngram_scope = CharNGramScope::Text;
    config.feature_weighting = FeatureWeighting::Raw;
    config.sketch_weighting = SketchWeighting::Raw;
    config.sketch_dim = 4096;
    config.output_dim = 384;
    config.projection = ProjectionMode::AchlioptasSparse;
    config.projection_seed = 0xDEADBEEFCAFEBABEULL;
    config.l2_normalize = true;
    config.matryoshka = false;
    return config;
}

EncoderConfig compact_retrieval_config() {
    EncoderConfig cfg;
    cfg.ngram_mode = NGramMode::CharAndWord;
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.hash = HashFamily::SplitMix64;
    cfg.hash_seed = 0xA5A5A5A5A5A5A5A5ULL;
    cfg.text_normalization = TextNormalization::AsciiLower;
    cfg.char_ngram_scope = CharNGramScope::WordBounded;
    cfg.feature_weighting = FeatureWeighting::Raw;
    cfg.sketch_weighting = SketchWeighting::Raw;
    cfg.sketch_dim = 8192;
    cfg.output_dim = 384;
    cfg.projection = ProjectionMode::Fwht;
    cfg.projection_seed = 0xDEADBEEFCAFEBABEULL;
    cfg.l2_normalize = true;
    cfg.matryoshka = false;
    return cfg;
}

EncoderConfig compact_hashed_idf_retrieval_config(const HashedIdf& idf) {
    EncoderConfig config = compact_retrieval_config();
    if (idf.hash_dim() != 65'536 || idf.scope() != HashedIdfScope::All || !idf.compatible(config))
        throw std::invalid_argument(
            "simeon::compact_hashed_idf_retrieval_config: incompatible IDF artifact");
    config.hashed_idf = &idf;
    return config;
}

EncoderConfig quality_hashed_idf_retrieval_config(const HashedIdf& idf) {
    EncoderConfig config = compact_hashed_idf_retrieval_config(idf);
    config.output_dim = 768;
    return config;
}

class Encoder::Impl {
public:
    explicit Impl(EncoderConfig cfg) : cfg_(std::move(cfg)) {
        validate();

        has_pmi_ = (cfg_.pmi_rows != nullptr);

        const bool emit_char = cfg_.ngram_mode == NGramMode::CharOnly ||
                               cfg_.ngram_mode == NGramMode::CharAndWord ||
                               cfg_.ngram_mode == NGramMode::CharSubword;
        const bool emit_word =
            cfg_.ngram_mode == NGramMode::WordOnly || cfg_.ngram_mode == NGramMode::CharAndWord;
        const bool emit_subword =
            cfg_.ngram_mode == NGramMode::Subword || cfg_.ngram_mode == NGramMode::CharSubword;
        tok_cfg_ = TokenizerConfig{
            .ngram_min = cfg_.ngram_min,
            .ngram_max = cfg_.ngram_max,
            .emit_char = emit_char,
            .emit_word = emit_word,
            .char_ngram_scope = cfg_.char_ngram_scope,
            .bpe = emit_subword ? cfg_.bpe : nullptr,
        };

        if (has_pmi_) {
            output_dim_ = cfg_.pmi_rows->dim();
            has_projection_ = false;
        } else {
            has_projection_ = (cfg_.projection != ProjectionMode::None);
            output_dim_ = has_projection_ ? cfg_.output_dim : cfg_.sketch_dim;

            if (has_projection_) {
                projection_ =
                    std::make_unique<Projection>(cfg_.sketch_dim, cfg_.output_dim, cfg_.projection,
                                                 cfg_.projection_seed, cfg_.sparse_jl_eps);
            }
        }

        cfg_.output_dim = output_dim_;

        if (cfg_.matryoshka) {
            matryoshka_weights_.resize(output_dim_);
            if (!cfg_.matryoshka_weights.empty()) {
                // Caller-supplied data-aware weights override the analytic
                // 1/sqrt(1+r/decay) schedule. Length is checked in validate().
                for (std::uint32_t r = 0; r < output_dim_; ++r) {
                    matryoshka_weights_[r] = cfg_.matryoshka_weights[r];
                }
            } else {
                const float decay = cfg_.matryoshka_decay;
                for (std::uint32_t r = 0; r < output_dim_; ++r) {
                    matryoshka_weights_[r] = 1.0f / std::sqrt(1.0f + static_cast<float>(r) / decay);
                }
            }
        }
    }

    std::uint32_t output_dim() const noexcept { return output_dim_; }
    const EncoderConfig& config() const noexcept { return cfg_; }

    void encode_one(std::string_view text, float* out) const {
        thread_local std::string normalized_text;
        if (cfg_.text_normalization == TextNormalization::AsciiLower) {
            normalized_text.assign(text);
            for (char& byte : normalized_text) {
                if (byte >= 'A' && byte <= 'Z')
                    byte = static_cast<char>(byte + ('a' - 'A'));
            }
            text = normalized_text;
        }

        if (has_pmi_) {
            encode_one_pmi(text, out);
            return;
        }
        thread_local std::vector<std::int32_t> sketch;
        sketch.assign(cfg_.sketch_dim, 0);

        if (cfg_.feature_weighting == FeatureWeighting::SqrtTf) {
            thread_local FeatureCountMap feature_counts;
            thread_local std::vector<std::int64_t> weighted_sketch;
            feature_counts.clear();
            if (text.size() <= std::numeric_limits<std::size_t>::max() / 4) {
                const std::size_t reserve_hint = text.size() * 4;
                if (feature_counts.bucket_count() < reserve_hint)
                    feature_counts.reserve(reserve_hint);
            }
            SqrtTfSink sink(feature_counts, cfg_.hash_seed, cfg_.hash);
            tokenize(text, tok_cfg_, sink);

            weighted_sketch.assign(cfg_.sketch_dim, 0);
            sink.materialize(weighted_sketch.data(), cfg_.sketch_dim);
            for (std::uint32_t i = 0; i < cfg_.sketch_dim; ++i) {
                const auto value = weighted_sketch[i];
                if (value < std::numeric_limits<std::int32_t>::min() ||
                    value > std::numeric_limits<std::int32_t>::max()) {
                    throw std::overflow_error("simeon::Encoder: sqrt-TF sketch overflow");
                }
                sketch[i] = static_cast<std::int32_t>(value);
            }
        } else if (cfg_.hashed_idf != nullptr) {
            HashedIdfSink sink(sketch.data(), cfg_.sketch_dim, cfg_.hash_seed, cfg_.hash,
                               *cfg_.hashed_idf);
            tokenize(text, tok_cfg_, sink);
        } else {
            SketchSink sink(sketch.data(), cfg_.sketch_dim, cfg_.hash_seed, cfg_.hash);
            tokenize(text, tok_cfg_, sink);
        }

        if (cfg_.sketch_weighting == SketchWeighting::SignedSqrt) {
            for (std::int32_t& value : sketch)
                value = signed_sqrt_fixed(value);
        }

        if (has_projection_) {
            projection_->apply(sketch.data(), out);
        } else {
            for (std::uint32_t i = 0; i < cfg_.sketch_dim; ++i) {
                out[i] = static_cast<float>(sketch[i]);
            }
        }

        if (cfg_.matryoshka) {
            simd::scale_vec(out, matryoshka_weights_.data(), output_dim_);
        }

        if (cfg_.l2_normalize) {
            simd::l2_normalize(out, output_dim_);
        }
    }

    void encode_many(std::span<const std::string_view> texts, float* out) const {
        for (std::size_t i = 0; i < texts.size(); ++i) {
            encode_one(texts[i], out + i * output_dim_);
        }
    }

private:
    struct FeatureCounts {
        std::uint32_t half_weight = 0;
        std::uint32_t full_weight = 0;
    };
    using FeatureCountMap = std::unordered_map<std::uint64_t, FeatureCounts>;

    struct SqrtTfSink final : public NGramEmitter {
        FeatureCountMap& counts;
        std::uint64_t seed;
        HashFamily family;

        SqrtTfSink(FeatureCountMap& c, std::uint64_t s, HashFamily f)
            : counts(c), seed(s), family(f) {}

        void on_token(std::string_view token, float weight) override {
            const std::uint64_t h = hash64(token, seed, family);
            auto& count = counts[h];
            const auto magnitude = static_cast<std::int32_t>(weight * 2.0f + 0.5f);
            if (magnitude == 1) {
                if (count.half_weight == std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("simeon::Encoder: feature TF overflow");
                ++count.half_weight;
            } else if (magnitude == 2) {
                if (count.full_weight == std::numeric_limits<std::uint32_t>::max())
                    throw std::overflow_error("simeon::Encoder: feature TF overflow");
                ++count.full_weight;
            } else {
                throw std::invalid_argument("simeon::Encoder: unsupported token weight");
            }
        }

        void materialize(std::int64_t* sketch, std::uint32_t dim) const {
            for (const auto& [h, count] : counts) {
                std::int64_t magnitude = 0;
                if (count.half_weight != 0) {
                    magnitude += static_cast<std::int64_t>(
                        integer_sqrt(static_cast<std::uint64_t>(count.half_weight) << 20));
                }
                if (count.full_weight != 0) {
                    magnitude += 2 * static_cast<std::int64_t>(integer_sqrt(
                                         static_cast<std::uint64_t>(count.full_weight) << 20));
                }
                const std::uint32_t bucket = sketch_bucket(h, dim);
                const std::int64_t sign = ((h >> 63) & 1ULL) ? -1 : 1;
                sketch[bucket] += sign * magnitude;
            }
        }
    };

    struct PmiSumSink final : public NGramEmitter {
        const PmiEmbeddings* emb;
        float* out;
        std::uint32_t dim;

        PmiSumSink(const PmiEmbeddings* e, float* o, std::uint32_t d) : emb(e), out(o), dim(d) {}

        void on_token(std::string_view token, float /*weight*/) override {
            const float* r = emb->row(token);
            if (r == nullptr)
                return;
            simd::add_vec(out, r, dim);
        }
    };

    void encode_one_pmi(std::string_view text, float* out) const {
        for (std::uint32_t i = 0; i < output_dim_; ++i)
            out[i] = 0.0f;

        // Word-only tokenization: PMI embeddings are defined at word
        // granularity. Char n-grams and BPE subwords don't have PMI rows
        // (they weren't in the learn() vocabulary).
        TokenizerConfig pmi_tok{
            .ngram_min = 1,
            .ngram_max = 1,
            .emit_char = false,
            .emit_word = true,
            .bpe = nullptr,
        };
        PmiSumSink sink(cfg_.pmi_rows, out, output_dim_);
        tokenize(text, pmi_tok, sink);

        if (cfg_.matryoshka) {
            simd::scale_vec(out, matryoshka_weights_.data(), output_dim_);
        }
        if (cfg_.l2_normalize) {
            simd::l2_normalize(out, output_dim_);
        }
    }

    struct SketchSink final : public NGramEmitter {
        std::int32_t* data;
        std::uint32_t dim;
        std::uint64_t seed;
        HashFamily family;

        SketchSink(std::int32_t* d, std::uint32_t n, std::uint64_t s, HashFamily f)
            : data(d), dim(n), seed(s), family(f) {}

        void on_token(std::string_view token, float weight) override {
            const std::uint64_t h = hash64(token, seed, family);
            const std::uint32_t bucket = sketch_bucket(h, dim);
            const std::int32_t sign = ((h >> 63) & 1ULL) ? -1 : +1;
            // Integer-weighted count-sketch: char n-grams weight=1.0 → ±2, word tokens
            // weight=0.5 → ±1. Keeps the sketch deterministic int32 across platforms.
            const std::int32_t mag = static_cast<std::int32_t>(weight * 2.0f + 0.5f);
            data[bucket] += sign * mag;
        }
    };

    struct HashedIdfSink final : public NGramEmitter {
        std::int32_t* data;
        std::uint32_t dim;
        std::uint64_t seed;
        HashFamily family;
        const HashedIdf& idf;

        HashedIdfSink(std::int32_t* d, std::uint32_t n, std::uint64_t s, HashFamily f,
                      const HashedIdf& weights)
            : data(d), dim(n), seed(s), family(f), idf(weights) {}

        void on_token(std::string_view token, float weight) override {
            const std::uint64_t h = hash64(token, seed, family);
            const std::uint32_t bucket = sketch_bucket(h, dim);
            const std::int64_t sign = ((h >> 63) & 1ULL) ? -1 : 1;
            const std::int64_t base = static_cast<std::int64_t>(weight * 2.0f + 0.5f);
            const std::int64_t updated =
                static_cast<std::int64_t>(data[bucket]) + sign * base * idf.weight_q10(h, weight);
            if (updated < std::numeric_limits<std::int32_t>::min() ||
                updated > std::numeric_limits<std::int32_t>::max())
                throw std::overflow_error("simeon::Encoder: hashed-IDF sketch overflow");
            data[bucket] = static_cast<std::int32_t>(updated);
        }
    };

    void validate() const {
        const bool has_pmi = cfg_.pmi_rows != nullptr;
        const bool has_idf = cfg_.hashed_idf != nullptr;
        if (static_cast<std::uint8_t>(cfg_.ngram_mode) >
                static_cast<std::uint8_t>(NGramMode::CharSubword) ||
            static_cast<std::uint8_t>(cfg_.hash) >
                static_cast<std::uint8_t>(HashFamily::MixedTabulation) ||
            static_cast<std::uint8_t>(cfg_.projection) >
                static_cast<std::uint8_t>(ProjectionMode::Fwht) ||
            static_cast<std::uint8_t>(cfg_.text_normalization) >
                static_cast<std::uint8_t>(TextNormalization::AsciiLower) ||
            static_cast<std::uint8_t>(cfg_.char_ngram_scope) >
                static_cast<std::uint8_t>(CharNGramScope::WordBounded) ||
            static_cast<std::uint8_t>(cfg_.feature_weighting) >
                static_cast<std::uint8_t>(FeatureWeighting::SqrtTf) ||
            static_cast<std::uint8_t>(cfg_.sketch_weighting) >
                static_cast<std::uint8_t>(SketchWeighting::SignedSqrt)) {
            throw std::invalid_argument("simeon::Encoder: invalid enum value");
        }
        if (cfg_.feature_weighting != FeatureWeighting::Raw &&
            cfg_.sketch_weighting != SketchWeighting::Raw) {
            throw std::invalid_argument(
                "simeon::Encoder: feature_weighting and sketch_weighting cannot both be non-raw");
        }
        if (has_idf) {
            if (has_pmi)
                throw std::invalid_argument(
                    "simeon::Encoder: hashed_idf and pmi_rows are mutually exclusive");
            if (cfg_.feature_weighting != FeatureWeighting::Raw ||
                cfg_.sketch_weighting != SketchWeighting::Raw)
                throw std::invalid_argument(
                    "simeon::Encoder: hashed_idf requires raw feature and sketch weighting");
            if (!cfg_.hashed_idf->compatible(cfg_))
                throw std::invalid_argument("simeon::Encoder: hashed_idf does not match encoder "
                                            "tokenization/hash identity");
        }
        if (has_pmi) {
            if (cfg_.feature_weighting != FeatureWeighting::Raw ||
                cfg_.sketch_weighting != SketchWeighting::Raw) {
                throw std::invalid_argument(
                    "simeon::Encoder: non-raw weighting is incompatible with pmi_rows");
            }
            if (cfg_.projection != ProjectionMode::None) {
                throw std::invalid_argument(
                    "simeon::Encoder: pmi_rows requires projection == None");
            }
            if (cfg_.pmi_rows->dim() == 0) {
                throw std::invalid_argument("simeon::Encoder: pmi_rows->dim() must be > 0");
            }
            if (cfg_.output_dim != 0 && cfg_.output_dim != cfg_.pmi_rows->dim()) {
                throw std::invalid_argument(
                    "simeon::Encoder: output_dim must match pmi_rows->dim() when PMI is enabled");
            }
            if (cfg_.ngram_min == 0 || cfg_.ngram_max < cfg_.ngram_min) {
                throw std::invalid_argument("simeon::Encoder: ngram range invalid");
            }
        } else {
            if (cfg_.sketch_dim == 0) {
                throw std::invalid_argument("simeon::Encoder: sketch_dim must be > 0");
            }
            if (cfg_.ngram_min == 0 || cfg_.ngram_max < cfg_.ngram_min) {
                throw std::invalid_argument("simeon::Encoder: ngram range invalid");
            }
            if (cfg_.projection != ProjectionMode::None && cfg_.output_dim == 0) {
                throw std::invalid_argument(
                    "simeon::Encoder: output_dim must be > 0 when projection enabled");
            }
            if (cfg_.matryoshka && cfg_.projection == ProjectionMode::None) {
                throw std::invalid_argument(
                    "simeon::Encoder: matryoshka requires projection != None (or pmi_rows)");
            }
        }
        if (cfg_.matryoshka && cfg_.matryoshka_weights.empty() &&
            (!(cfg_.matryoshka_decay > 0.0f) || !std::isfinite(cfg_.matryoshka_decay))) {
            throw std::invalid_argument("simeon::Encoder: matryoshka_decay must be finite and > 0");
        }
        if (cfg_.matryoshka && !cfg_.matryoshka_weights.empty()) {
            std::uint32_t expected;
            if (has_pmi) {
                expected = cfg_.pmi_rows->dim();
            } else if (cfg_.projection != ProjectionMode::None) {
                expected = cfg_.output_dim;
            } else {
                expected = cfg_.sketch_dim;
            }
            if (cfg_.matryoshka_weights.size() != expected) {
                throw std::invalid_argument(
                    "simeon::Encoder: matryoshka_weights size must equal output_dim");
            }
            for (float w : cfg_.matryoshka_weights) {
                if (!(w >= 0.0f) || !std::isfinite(w)) {
                    throw std::invalid_argument(
                        "simeon::Encoder: matryoshka_weights must be finite and non-negative");
                }
            }
        }
        if ((cfg_.ngram_mode == NGramMode::Subword || cfg_.ngram_mode == NGramMode::CharSubword) &&
            cfg_.bpe == nullptr) {
            throw std::invalid_argument(
                "simeon::Encoder: ngram_mode requires a non-null bpe pointer");
        }
    }

    EncoderConfig cfg_;
    TokenizerConfig tok_cfg_{};
    bool has_projection_ = false;
    bool has_pmi_ = false;
    std::uint32_t output_dim_ = 0;
    std::unique_ptr<Projection> projection_;
    std::vector<float> matryoshka_weights_;
};

void matryoshka_prefix_normalize(float* vec, std::uint32_t prefix_dim) noexcept {
    if (vec == nullptr || prefix_dim == 0) {
        return;
    }
    simd::l2_normalize(vec, prefix_dim);
}

Encoder::Encoder(EncoderConfig cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}
Encoder::~Encoder() = default;
Encoder::Encoder(Encoder&&) noexcept = default;
Encoder& Encoder::operator=(Encoder&&) noexcept = default;

std::uint32_t Encoder::output_dim() const noexcept {
    return impl_->output_dim();
}
const EncoderConfig& Encoder::config() const noexcept {
    return impl_->config();
}

void Encoder::encode(std::string_view text, float* out) const {
    impl_->encode_one(text, out);
}

void Encoder::encode_batch(std::span<const std::string_view> texts, float* out) const {
    impl_->encode_many(texts, out);
}

} // namespace simeon
