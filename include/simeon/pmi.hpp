#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simeon {

// Training-free word embeddings via shifted positive PMI (Levy & Goldberg
// 2014). The matrix entries are corpus-statistic ratios; the factorization
// is truncated randomized SVD (Halko-Martinsson-Tropp 2011). Both steps are
// closed-form — no gradient descent. simeon ships no built-in embeddings;
// the caller learns a `PmiEmbeddings` from a domain-specific seed corpus
// and passes it to `EncoderConfig::pmi_rows`.
//
// Document encoding path (when `EncoderConfig::pmi_rows != nullptr`):
// tokenize into words -> sum PmiEmbeddings::row(tok) over in-vocab tokens
// -> apply matryoshka weights if set -> L2 normalize. Sketch and projection
// are skipped; `output_dim` must equal `PmiEmbeddings::dim()`.
struct PmiConfig {
    // Sliding context window radius in words (each side).
    std::uint32_t window_size = 5;

    // Target embedding rank (= `dim()` on the returned PmiEmbeddings).
    std::uint32_t target_rank = 256;

    // Vocab cutoffs: keep only tokens with count >= min_token_count, then
    // clamp to the top max_vocab_size by frequency.
    std::uint32_t min_token_count = 5;
    std::uint32_t max_vocab_size = 50'000;

    // Shifted positive PMI: subtract log(k) from log(p(i,j)/(p(i)p(j)))
    // before the max(., 0) clip. shift_log_k = 0 is the vanilla PPMI limit.
    float shift_log_k = 0.0f;

    // Randomized-SVD (Halko et al. 2011 Alg 4.4 / 5.1) parameters.
    // Power iterations stabilize the subspace when singular values decay
    // slowly; oversampling guards against rank-revealing failure.
    std::uint32_t svd_iters = 4;
    std::uint32_t svd_oversample = 10;
    std::uint64_t svd_seed = 0xC0FFEE00CAFEBABEULL;
};

class PmiEmbeddings {
public:
    // Learn embeddings from `seed_corpus` under `cfg`. Deterministic given
    // a fixed `(seed_corpus, cfg)`.
    static PmiEmbeddings learn(std::span<const std::string_view> seed_corpus,
                               const PmiConfig& cfg);

    // Binary round-trip format:
    //   char[8]  magic "SMEPMI01"
    //   uint32   rank
    //   uint32   vocab_size
    //   for each token: uint32 len, bytes...
    //   float32[vocab_size * rank] rows (little-endian)
    std::string serialize() const;
    static PmiEmbeddings from_bytes(std::string_view bytes);

    // Pointer to the `dim()`-long row for `tok`, or nullptr if OOV.
    const float* row(std::string_view tok) const noexcept;

    std::uint32_t dim() const noexcept { return dim_; }
    std::uint32_t vocab_size() const noexcept {
        return static_cast<std::uint32_t>(vocab_.size());
    }

private:
    PmiEmbeddings() = default;

    std::uint32_t dim_ = 0;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, std::uint32_t> index_;
    std::vector<float> rows_;  // row-major, vocab_size x dim
};

}  // namespace simeon
