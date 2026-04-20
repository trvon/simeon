#include "simeon/simeon.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "simeon/hasher.hpp"
#include "simeon/pmi.hpp"
#include "simeon/projection.hpp"
#include "simeon/simd.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

SimdTier active_simd_tier() noexcept {
#if defined(__aarch64__)
    return SimdTier::Neon;
#elif defined(__AVX2__)
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

class Encoder::Impl {
public:
    explicit Impl(EncoderConfig cfg) : cfg_(cfg) {
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
        if (has_pmi_) {
            encode_one_pmi(text, out);
            return;
        }
        thread_local std::vector<std::int32_t> sketch;
        sketch.assign(cfg_.sketch_dim, 0);

        SketchSink sink(sketch.data(), cfg_.sketch_dim, cfg_.hash_seed, cfg_.hash);
        tokenize(text, tok_cfg_, sink);

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
            const std::uint32_t bucket = static_cast<std::uint32_t>(h) % dim;
            const std::int32_t sign = ((h >> 63) & 1ULL) ? -1 : +1;
            // Integer-weighted count-sketch: char n-grams weight=1.0 → ±2, word tokens
            // weight=0.5 → ±1. Keeps the sketch deterministic int32 across platforms.
            const std::int32_t mag = static_cast<std::int32_t>(weight * 2.0f + 0.5f);
            data[bucket] += sign * mag;
        }
    };

    void validate() const {
        const bool has_pmi = cfg_.pmi_rows != nullptr;
        if (has_pmi) {
            if (cfg_.projection != ProjectionMode::None) {
                throw std::invalid_argument(
                    "simeon::Encoder: pmi_rows requires projection == None");
            }
            if (cfg_.pmi_rows->dim() == 0) {
                throw std::invalid_argument("simeon::Encoder: pmi_rows->dim() must be > 0");
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
        if (cfg_.matryoshka && cfg_.matryoshka_weights.empty() && !(cfg_.matryoshka_decay > 0.0f)) {
            throw std::invalid_argument("simeon::Encoder: matryoshka_decay must be > 0");
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

Encoder::Encoder(EncoderConfig cfg) : impl_(std::make_unique<Impl>(cfg)) {}
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
