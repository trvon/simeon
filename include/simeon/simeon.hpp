#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace simeon {

enum class NGramMode : std::uint8_t {
    CharOnly,
    WordOnly,
    CharAndWord,
    // Subword: emit only BPE subword tokens for each whitespace-separated
    // word. Requires `EncoderConfig::bpe != nullptr`.
    Subword,
    // CharSubword: emit char n-grams and BPE subword tokens. The combination
    // captures both morphological structure (subwords) and surface
    // similarity (char n-grams).
    CharSubword,
};

class BpeMerges;     // simeon/tokenizer_bpe.hpp
class PmiEmbeddings; // simeon/pmi.hpp

enum class HashFamily : std::uint8_t {
    SplitMix64,
    XxHash64,
    Crc32,
    MixedTabulation,
};

enum class ProjectionMode : std::uint8_t {
    None,
    AchlioptasSparse,
    DenseGaussian,
    VerySparse,
    // Sparse-JL with parameterized distortion (Kane & Nelson 2010,
    // arXiv:1006.3585). Each input column gets exactly
    // `s = ceil(eps * output_dim)` nonzero entries spread across distinct
    // output rows with ±1/sqrt(s) signs (column-sparsity, the canonical
    // Kane-Nelson construction). E[||Px||^2] = ||x||^2; smaller eps gives
    // tighter distortion at higher per-doc cost. Control parameter is
    // `EncoderConfig::sparse_jl_eps`.
    SparseJL,
    // Subsampled Randomized Hadamard Transform (Ailon-Chazelle 2009 FJLT,
    // refined by Tropp 2011). Sketch is zero-padded to the next power of 2,
    // multiplied by a random ±1 sign diagonal D, transformed by the Walsh-
    // Hadamard matrix H, then output_dim distinct rows are subsampled with
    // scale sqrt(pad_n/output_dim). Apply cost is O(pad_n log pad_n + output_dim)
    // — much cheaper than DenseGaussian's O(pad_n * output_dim) at wide
    // sketch dims, with comparable distortion.
    Fwht,
};

struct EncoderConfig {
    NGramMode ngram_mode = NGramMode::CharOnly;
    std::uint32_t ngram_min = 3;
    std::uint32_t ngram_max = 5;
    HashFamily hash = HashFamily::SplitMix64;
    std::uint64_t hash_seed = 0xA5A5A5A5A5A5A5A5ULL;

    std::uint32_t sketch_dim = 32768;
    std::uint32_t output_dim = 0;
    ProjectionMode projection = ProjectionMode::None;
    std::uint64_t projection_seed = 0xDEADBEEFCAFEBABEULL;

    bool l2_normalize = true;

    // Matryoshka-style nested output: scales row r of the projected vector by
    // 1/sqrt(1 + r/matryoshka_decay) before L2 normalization, biasing energy
    // toward early dimensions. A renormalized prefix of the result is then a
    // valid coarse cosine query — letting consumers pick a quality/cost point
    // per query without retraining. Training-free analog of Kusupati et al.
    // 2022. Requires projection != None.
    bool matryoshka = false;
    float matryoshka_decay = 32.0f;

    // Caller-supplied data-aware matryoshka weights. When non-empty, this
    // vector overrides the `1/sqrt(1+r/decay)` schedule entirely. Length must
    // equal `output_dim` (or `sketch_dim` if projection is None). Compute it
    // once from a seed corpus via `simeon::compute_matryoshka_weights()` and
    // hand it back here on subsequent encoder construction. Empty by default.
    std::vector<float> matryoshka_weights;

    // Distortion target for ProjectionMode::SparseJL. Each input column
    // gets `s = ceil(sparse_jl_eps * output_dim)` nonzero entries; smaller
    // eps -> tighter distortion bound, more work per doc. Ignored for
    // other projection modes. Default 0.10 puts ~10% of the output rows
    // on each input column.
    float sparse_jl_eps = 0.10f;

    // Caller-owned BPE merges table for ngram_mode == Subword or CharSubword.
    // Must outlive the encoder. simeon never owns or learns this — see
    // simeon/tokenizer_bpe.hpp for the learner.
    const BpeMerges* bpe = nullptr;

    // Caller-owned PMI embeddings (training-free word embeddings via
    // shifted positive PMI + randomized SVD; Levy & Goldberg 2014). When
    // non-null the encoder bypasses the sketch + projection pipeline
    // entirely: it tokenizes words, sums PmiEmbeddings::row(tok) for
    // in-vocab tokens, applies matryoshka weights if set, and L2-
    // normalizes. Constraints: `projection` must be None. If `output_dim`
    // is non-zero it must equal `pmi_rows->dim()`; otherwise the encoder
    // adopts `pmi_rows->dim()` automatically. OOV tokens are skipped (no
    // Achlioptas fallback). See simeon/pmi.hpp for the learner.
    const PmiEmbeddings* pmi_rows = nullptr;
};

// Re-normalize the first `prefix_dim` floats of a matryoshka-encoded vector to
// unit L2 length, in place. Use when extracting a coarse prefix of a vector
// produced with `EncoderConfig::matryoshka = true`.
void matryoshka_prefix_normalize(float* vec, std::uint32_t prefix_dim) noexcept;

enum class SimdTier : std::uint8_t {
    Scalar,
    Neon,
    Avx2,
};

SimdTier active_simd_tier() noexcept;
const char* simd_tier_name(SimdTier tier) noexcept;

class Encoder {
public:
    explicit Encoder(EncoderConfig cfg);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder(Encoder&&) noexcept;
    Encoder& operator=(Encoder&&) noexcept;

    std::uint32_t output_dim() const noexcept;
    // Returns the effective encoder configuration after constructor validation
    // and PMI/output-dimension normalization.
    const EncoderConfig& config() const noexcept;

    // Encode one document into `out[0..output_dim())`.
    void encode(std::string_view text, float* out) const;

    // Encode `texts.size()` documents into the row-major output buffer
    // `out[0..texts.size() * output_dim())`.
    void encode_batch(std::span<const std::string_view> texts, float* out) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace simeon
