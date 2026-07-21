#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simeon {

struct EncoderConfig;

enum class HashedIdfScope : std::uint8_t {
    All,
    Character,
    Word,
};

struct HashedIdfConfig {
    // Bounded document-frequency table. Hash collisions conservatively raise
    // observed frequency and therefore reduce, rather than inflate, IDF.
    std::uint32_t hash_dim = 1U << 20;
    HashedIdfScope scope = HashedIdfScope::All;
};

// Corpus-adaptive, label-free inverse-document-frequency artifact for the
// hashed n-gram encoder. The artifact records the complete tokenizer/hash
// identity and rejects mismatched EncoderConfig values.
class HashedIdf {
public:
    static HashedIdf learn(std::span<const std::string_view> corpus,
                           const EncoderConfig& encoder_config, HashedIdfConfig config = {});

    // Deterministic little-endian format with magic "SMEIDF01".
    std::string serialize() const;
    static HashedIdf from_bytes(std::string_view bytes);

    // Robertson-style smoothed IDF in Q10 fixed point:
    // round(1024 * log((N + 1) / (df + 0.5))).
    std::uint16_t weight_q10(std::uint64_t feature_hash, float token_weight = 1.0f) const noexcept {
        const bool is_character = token_weight > 0.75f;
        const auto selected_scope = static_cast<HashedIdfScope>(scope_);
        if ((selected_scope == HashedIdfScope::Character && !is_character) ||
            (selected_scope == HashedIdfScope::Word && is_character))
            return 1024;
        if (weights_.empty())
            return 0;
        const std::size_t size = weights_.size();
        const std::size_t bucket = (size & (size - 1)) == 0
                                       ? static_cast<std::size_t>(feature_hash) & (size - 1)
                                       : static_cast<std::size_t>(feature_hash % size);
        return weights_[bucket];
    }
    bool compatible(const EncoderConfig& encoder_config) const noexcept;

    std::uint32_t hash_dim() const noexcept { return static_cast<std::uint32_t>(weights_.size()); }
    std::uint32_t document_count() const noexcept { return document_count_; }
    HashedIdfScope scope() const noexcept { return static_cast<HashedIdfScope>(scope_); }
    std::size_t storage_bytes() const noexcept { return weights_.size() * sizeof(std::uint16_t); }
    std::string fingerprint() const;

private:
    HashedIdf() = default;

    std::uint32_t document_count_ = 0;
    std::uint32_t ngram_min_ = 0;
    std::uint32_t ngram_max_ = 0;
    std::uint64_t hash_seed_ = 0;
    std::uint8_t ngram_mode_ = 0;
    std::uint8_t hash_family_ = 0;
    std::uint8_t text_normalization_ = 0;
    std::uint8_t char_ngram_scope_ = 0;
    std::uint8_t scope_ = 0;
    std::vector<std::uint16_t> weights_;
};

} // namespace simeon
