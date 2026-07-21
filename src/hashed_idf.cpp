#include "simeon/hashed_idf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "simeon/hasher.hpp"
#include "simeon/simeon.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

constexpr char kMagic[8] = {'S', 'M', 'E', 'I', 'D', 'F', '0', '1'};
constexpr std::size_t kHeaderBytes = 52;

void write_u16(std::string& output, std::uint16_t value) {
    output.push_back(static_cast<char>(value & 0xFF));
    output.push_back(static_cast<char>((value >> 8) & 0xFF));
}

void write_u32(std::string& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xFF));
}

void write_u64(std::string& output, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.push_back(static_cast<char>((value >> shift) & 0xFF));
}

std::uint16_t read_u16(const char* input) noexcept {
    return static_cast<std::uint16_t>(static_cast<unsigned char>(input[0])) |
           static_cast<std::uint16_t>(static_cast<unsigned char>(input[1])) << 8;
}

std::uint32_t read_u32(const char* input) noexcept {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(input[shift / 8])) << shift;
    return value;
}

std::uint64_t read_u64(const char* input) noexcept {
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8)
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(input[shift / 8])) << shift;
    return value;
}

std::string_view normalized_text(std::string_view text, TextNormalization normalization,
                                 std::string& scratch) {
    if (normalization != TextNormalization::AsciiLower)
        return text;
    scratch.assign(text);
    for (char& byte : scratch) {
        if (byte >= 'A' && byte <= 'Z')
            byte = static_cast<char>(byte + ('a' - 'A'));
    }
    return scratch;
}

struct DocumentFrequencySink final : NGramEmitter {
    std::vector<std::uint32_t>& document_frequency;
    std::vector<std::uint32_t>& last_seen;
    std::uint32_t document_marker;
    std::uint64_t hash_seed;
    HashFamily hash_family;
    HashedIdfScope scope;

    DocumentFrequencySink(std::vector<std::uint32_t>& frequencies, std::vector<std::uint32_t>& seen,
                          std::uint32_t marker, std::uint64_t seed, HashFamily family,
                          HashedIdfScope selected_scope)
        : document_frequency(frequencies), last_seen(seen), document_marker(marker),
          hash_seed(seed), hash_family(family), scope(selected_scope) {}

    void on_token(std::string_view token, float token_weight) override {
        const bool is_character = token_weight > 0.75f;
        if ((scope == HashedIdfScope::Character && !is_character) ||
            (scope == HashedIdfScope::Word && is_character))
            return;
        const auto feature_hash = hash64(token, hash_seed, hash_family);
        const auto bucket = static_cast<std::uint32_t>(feature_hash % document_frequency.size());
        if (last_seen[bucket] == document_marker)
            return;
        last_seen[bucket] = document_marker;
        ++document_frequency[bucket];
    }
};

bool supported_ngram_mode(NGramMode mode) noexcept {
    return mode == NGramMode::CharOnly || mode == NGramMode::WordOnly ||
           mode == NGramMode::CharAndWord;
}

} // namespace

HashedIdf HashedIdf::learn(std::span<const std::string_view> corpus,
                           const EncoderConfig& encoder_config, HashedIdfConfig config) {
    if (corpus.empty())
        throw std::invalid_argument("HashedIdf::learn: corpus must not be empty");
    if (corpus.size() >= std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("HashedIdf::learn: corpus exceeds uint32 capacity");
    if (config.hash_dim == 0)
        throw std::invalid_argument("HashedIdf::learn: hash_dim must be greater than zero");
    if (static_cast<std::uint8_t>(config.scope) > static_cast<std::uint8_t>(HashedIdfScope::Word))
        throw std::invalid_argument("HashedIdf::learn: invalid scope");
    if (static_cast<std::uint8_t>(encoder_config.hash) >
            static_cast<std::uint8_t>(HashFamily::MixedTabulation) ||
        static_cast<std::uint8_t>(encoder_config.text_normalization) >
            static_cast<std::uint8_t>(TextNormalization::AsciiLower) ||
        static_cast<std::uint8_t>(encoder_config.char_ngram_scope) >
            static_cast<std::uint8_t>(CharNGramScope::WordBounded))
        throw std::invalid_argument("HashedIdf::learn: invalid encoder identity");
    if (!supported_ngram_mode(encoder_config.ngram_mode) || encoder_config.bpe != nullptr ||
        encoder_config.pmi_rows != nullptr)
        throw std::invalid_argument(
            "HashedIdf::learn: only char/word hashed encoders are supported");
    if (encoder_config.ngram_min == 0 || encoder_config.ngram_max < encoder_config.ngram_min)
        throw std::invalid_argument("HashedIdf::learn: ngram range invalid");
    if (encoder_config.feature_weighting != FeatureWeighting::Raw ||
        encoder_config.sketch_weighting != SketchWeighting::Raw)
        throw std::invalid_argument("HashedIdf::learn: non-raw weighting is not composable");

    HashedIdf result;
    result.document_count_ = static_cast<std::uint32_t>(corpus.size());
    result.ngram_min_ = encoder_config.ngram_min;
    result.ngram_max_ = encoder_config.ngram_max;
    result.hash_seed_ = encoder_config.hash_seed;
    result.ngram_mode_ = static_cast<std::uint8_t>(encoder_config.ngram_mode);
    result.hash_family_ = static_cast<std::uint8_t>(encoder_config.hash);
    result.text_normalization_ = static_cast<std::uint8_t>(encoder_config.text_normalization);
    result.char_ngram_scope_ = static_cast<std::uint8_t>(encoder_config.char_ngram_scope);
    result.scope_ = static_cast<std::uint8_t>(config.scope);

    std::vector<std::uint32_t> document_frequency(config.hash_dim, 0);
    std::vector<std::uint32_t> last_seen(config.hash_dim, 0);
    const bool emit_char = encoder_config.ngram_mode == NGramMode::CharOnly ||
                           encoder_config.ngram_mode == NGramMode::CharAndWord;
    const bool emit_word = encoder_config.ngram_mode == NGramMode::WordOnly ||
                           encoder_config.ngram_mode == NGramMode::CharAndWord;
    const TokenizerConfig tokenizer_config{
        .ngram_min = encoder_config.ngram_min,
        .ngram_max = encoder_config.ngram_max,
        .emit_char = emit_char,
        .emit_word = emit_word,
        .char_ngram_scope = encoder_config.char_ngram_scope,
        .bpe = nullptr,
    };

    std::string normalized;
    for (std::uint32_t document = 0; document < result.document_count_; ++document) {
        DocumentFrequencySink sink{
            document_frequency,       last_seen,           document + 1,
            encoder_config.hash_seed, encoder_config.hash, config.scope,
        };
        tokenize(normalized_text(corpus[document], encoder_config.text_normalization, normalized),
                 tokenizer_config, sink);
    }

    result.weights_.resize(config.hash_dim);
    const double document_count = static_cast<double>(result.document_count_);
    for (std::size_t bucket = 0; bucket < result.weights_.size(); ++bucket) {
        const double df = static_cast<double>(document_frequency[bucket]);
        const double idf = std::log((document_count + 1.0) / (df + 0.5));
        const double scaled = std::round(1024.0 * idf);
        result.weights_[bucket] = static_cast<std::uint16_t>(std::clamp(
            scaled, 0.0, static_cast<double>(std::numeric_limits<std::uint16_t>::max())));
    }
    return result;
}

bool HashedIdf::compatible(const EncoderConfig& encoder_config) const noexcept {
    return !weights_.empty() && encoder_config.ngram_min == ngram_min_ &&
           encoder_config.ngram_max == ngram_max_ && encoder_config.hash_seed == hash_seed_ &&
           static_cast<std::uint8_t>(encoder_config.ngram_mode) == ngram_mode_ &&
           static_cast<std::uint8_t>(encoder_config.hash) == hash_family_ &&
           static_cast<std::uint8_t>(encoder_config.text_normalization) == text_normalization_ &&
           static_cast<std::uint8_t>(encoder_config.char_ngram_scope) == char_ngram_scope_;
}

std::string HashedIdf::serialize() const {
    std::string output;
    output.reserve(kHeaderBytes + weights_.size() * sizeof(std::uint16_t));
    output.append(kMagic, sizeof(kMagic));
    write_u32(output, static_cast<std::uint32_t>(weights_.size()));
    write_u32(output, document_count_);
    write_u32(output, ngram_min_);
    write_u32(output, ngram_max_);
    write_u64(output, hash_seed_);
    write_u32(output, ngram_mode_);
    write_u32(output, hash_family_);
    write_u32(output, text_normalization_);
    write_u32(output, char_ngram_scope_);
    write_u32(output, scope_);
    for (const auto weight : weights_)
        write_u16(output, weight);
    return output;
}

HashedIdf HashedIdf::from_bytes(std::string_view bytes) {
    if (bytes.size() < kHeaderBytes)
        throw std::invalid_argument("HashedIdf::from_bytes: truncated header");
    if (std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0)
        throw std::invalid_argument("HashedIdf::from_bytes: bad magic");

    HashedIdf result;
    const auto hash_dim = read_u32(bytes.data() + 8);
    result.document_count_ = read_u32(bytes.data() + 12);
    result.ngram_min_ = read_u32(bytes.data() + 16);
    result.ngram_max_ = read_u32(bytes.data() + 20);
    result.hash_seed_ = read_u64(bytes.data() + 24);
    const auto ngram_mode = read_u32(bytes.data() + 32);
    const auto hash_family = read_u32(bytes.data() + 36);
    const auto text_normalization = read_u32(bytes.data() + 40);
    const auto char_ngram_scope = read_u32(bytes.data() + 44);
    const auto scope = read_u32(bytes.data() + 48);
    if (result.ngram_min_ == 0 || result.ngram_max_ < result.ngram_min_ ||
        ngram_mode > static_cast<std::uint32_t>(NGramMode::CharAndWord) ||
        hash_family > static_cast<std::uint32_t>(HashFamily::MixedTabulation) ||
        text_normalization > static_cast<std::uint32_t>(TextNormalization::AsciiLower) ||
        char_ngram_scope > static_cast<std::uint32_t>(CharNGramScope::WordBounded) ||
        scope > static_cast<std::uint32_t>(HashedIdfScope::Word))
        throw std::invalid_argument("HashedIdf::from_bytes: invalid encoder identity");
    result.ngram_mode_ = static_cast<std::uint8_t>(ngram_mode);
    result.hash_family_ = static_cast<std::uint8_t>(hash_family);
    result.text_normalization_ = static_cast<std::uint8_t>(text_normalization);
    result.char_ngram_scope_ = static_cast<std::uint8_t>(char_ngram_scope);
    result.scope_ = static_cast<std::uint8_t>(scope);
    constexpr std::size_t payload_offset = kHeaderBytes;
    if constexpr (sizeof(std::size_t) <= sizeof(std::uint32_t)) {
        if (hash_dim > (std::numeric_limits<std::size_t>::max() - payload_offset) / 2)
            throw std::invalid_argument("HashedIdf::from_bytes: hash dimension overflow");
    }
    const std::size_t expected = payload_offset + static_cast<std::size_t>(hash_dim) * 2;
    if (hash_dim == 0 || result.document_count_ == 0 || bytes.size() != expected)
        throw std::invalid_argument("HashedIdf::from_bytes: invalid payload size");
    result.weights_.resize(hash_dim);
    for (std::size_t i = 0; i < result.weights_.size(); ++i)
        result.weights_[i] = read_u16(bytes.data() + payload_offset + i * 2);
    return result;
}

std::string HashedIdf::fingerprint() const {
    constexpr std::uint64_t offset = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t hash = offset;
    const std::string bytes = serialize();
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= prime;
    }
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return output.str();
}

} // namespace simeon
