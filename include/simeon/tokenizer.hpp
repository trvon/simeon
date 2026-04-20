#pragma once

#include <cstdint>
#include <string_view>

namespace simeon {

class BpeMerges;  // simeon/tokenizer_bpe.hpp

class NGramEmitter {
public:
    virtual ~NGramEmitter() = default;
    virtual void on_token(std::string_view token, float weight) = 0;
};

struct TokenizerConfig {
    std::uint32_t ngram_min;
    std::uint32_t ngram_max;
    bool emit_char;
    bool emit_word;
    // When non-null, each whitespace-separated word in `text` is also passed
    // through `bpe->apply(word, sink)` so that BPE subword tokens are emitted
    // alongside (or instead of) char/word n-grams. nullptr preserves the
    // pre-BPE behaviour. Caller owns the BpeMerges; it must outlive the call.
    const BpeMerges* bpe = nullptr;
};

void tokenize(std::string_view text, const TokenizerConfig& cfg, NGramEmitter& sink);

}  // namespace simeon
