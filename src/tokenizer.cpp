#include "simeon/tokenizer.hpp"

#include <cctype>
#include <cstring>

#include "simeon/tokenizer_bpe.hpp"

namespace simeon {

namespace {

bool is_word_char(unsigned char c) noexcept {
    return std::isalnum(c) != 0 || c == '_';
}

void emit_char_ngrams(std::string_view text, std::uint32_t k_min, std::uint32_t k_max,
                      NGramEmitter& sink) {
    const std::size_t n = text.size();
    if (n == 0) return;
    for (std::uint32_t k = k_min; k <= k_max; ++k) {
        if (n < k) break;
        const std::size_t last = n - k;
        for (std::size_t i = 0; i <= last; ++i) {
            sink.on_token(text.substr(i, k), 1.0f);
        }
    }
}

void emit_word_tokens(std::string_view text, NGramEmitter& sink) {
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        while (i < n && !is_word_char(static_cast<unsigned char>(text[i]))) ++i;
        const std::size_t start = i;
        while (i < n && is_word_char(static_cast<unsigned char>(text[i]))) ++i;
        if (start < i) sink.on_token(text.substr(start, i - start), 0.5f);
    }
}

void emit_subword_tokens(std::string_view text, const BpeMerges& bpe, NGramEmitter& sink) {
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        while (i < n && !is_word_char(static_cast<unsigned char>(text[i]))) ++i;
        const std::size_t start = i;
        while (i < n && is_word_char(static_cast<unsigned char>(text[i]))) ++i;
        if (start < i) bpe.apply(text.substr(start, i - start), sink);
    }
}

}  // namespace

void tokenize(std::string_view text, const TokenizerConfig& cfg, NGramEmitter& sink) {
    if (cfg.emit_char) emit_char_ngrams(text, cfg.ngram_min, cfg.ngram_max, sink);
    if (cfg.emit_word) emit_word_tokens(text, sink);
    if (cfg.bpe != nullptr) emit_subword_tokens(text, *cfg.bpe, sink);
}

}  // namespace simeon
