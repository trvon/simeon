// Coverage for NGramMode::WordOnly in isolation, combined char+word emission
// weights, and UTF-8 boundary handling beyond what test_tokenizer.cpp and
// test_coverage.cpp already exercise.

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/tokenizer.hpp"

using simeon::NGramEmitter;
using simeon::TokenizerConfig;
using simeon::tokenize;

namespace {

struct Collector final : public NGramEmitter {
    std::vector<std::pair<std::string, float>> tokens;
    void on_token(std::string_view t, float w) override { tokens.emplace_back(t, w); }
};

// WordOnly mode (emit_char=false, emit_word=true). The Encoder maps
// NGramMode::WordOnly to this combination in src/simeon.cpp:44-45.
void test_word_only_no_char_grams() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 3, .ngram_max = 5, .emit_char = false, .emit_word = true};
    tokenize("alpha beta gamma", cfg, c);
    assert(c.tokens.size() == 3);
    assert(c.tokens[0].first == "alpha");
    assert(c.tokens[1].first == "beta");
    assert(c.tokens[2].first == "gamma");
    for (const auto& [_, w] : c.tokens) assert(w == 0.5f);
}

void test_word_only_punctuation_splits() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 0, .ngram_max = 0, .emit_char = false, .emit_word = true};
    tokenize("hello,world.foo!bar", cfg, c);
    assert(c.tokens.size() == 4);
    assert(c.tokens[0].first == "hello");
    assert(c.tokens[1].first == "world");
    assert(c.tokens[2].first == "foo");
    assert(c.tokens[3].first == "bar");
}

void test_word_only_underscore_is_word_char() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 0, .ngram_max = 0, .emit_char = false, .emit_word = true};
    tokenize("snake_case CamelCase", cfg, c);
    assert(c.tokens.size() == 2);
    assert(c.tokens[0].first == "snake_case");
    assert(c.tokens[1].first == "CamelCase");
}

void test_word_only_empty() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 0, .ngram_max = 0, .emit_char = false, .emit_word = true};
    tokenize("", cfg, c);
    assert(c.tokens.empty());
}

void test_word_only_only_punctuation() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 0, .ngram_max = 0, .emit_char = false, .emit_word = true};
    tokenize("...!!!---", cfg, c);
    assert(c.tokens.empty());
}

// Combined emission: char n-grams get weight 1.0, word tokens get 0.5. This
// is the weighting the SketchSink in src/simeon.cpp:96-104 relies on for the
// integer-quantized count-sketch.
void test_combined_weights_distinct() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 3, .ngram_max = 3, .emit_char = true, .emit_word = true};
    tokenize("foo bar", cfg, c);
    bool seen_char = false, seen_word = false;
    for (const auto& [tok, w] : c.tokens) {
        if (w == 1.0f) seen_char = true;
        if (w == 0.5f) seen_word = true;
    }
    assert(seen_char);
    assert(seen_word);
}

// Edge: text shorter than ngram_min should produce no char tokens.
void test_text_below_ngram_min() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 5, .ngram_max = 5, .emit_char = true, .emit_word = false};
    tokenize("abc", cfg, c);  // len=3 < min=5
    assert(c.tokens.empty());
}

// Edge: text exactly at ngram_min produces exactly one token.
void test_text_at_ngram_min() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 3, .ngram_max = 3, .emit_char = true, .emit_word = false};
    tokenize("abc", cfg, c);
    assert(c.tokens.size() == 1);
    assert(c.tokens[0].first == "abc");
}

// Edge: ngram_min > ngram_max should produce no tokens (loop body never runs).
// Production code validates this at the Encoder level (src/simeon.cpp:111-113);
// the bare tokenize() entry point silently does nothing.
void test_inverted_range_is_silent() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 5, .ngram_max = 3, .emit_char = true, .emit_word = false};
    tokenize("abcdefgh", cfg, c);
    assert(c.tokens.empty());
}

// Edge: single-byte input, char-3 mode.
void test_single_byte_input() {
    Collector c;
    TokenizerConfig cfg{.ngram_min = 3, .ngram_max = 3, .emit_char = true, .emit_word = false};
    tokenize("a", cfg, c);
    assert(c.tokens.empty());
}

// 3-byte UTF-8 codepoint handling. simeon tokenizes on bytes (deliberate, see
// test_coverage.cpp::test_utf8_is_byte_safe). A 3-byte CJK codepoint should
// appear as a single 3-gram when sandwiched between ASCII.
void test_utf8_3byte_byte_safe() {
    // "a日b" = 0x61 0xE6 0x97 0xA5 0x62 → 5 bytes
    const std::string text = "a\xE6\x97\xA5""b";
    assert(text.size() == 5);
    Collector c;
    TokenizerConfig cfg{.ngram_min = 3, .ngram_max = 3, .emit_char = true, .emit_word = false};
    tokenize(text, cfg, c);
    // 5 bytes, k=3 → 3 trigrams
    assert(c.tokens.size() == 3);
    for (const auto& [t, _] : c.tokens) assert(t.size() == 3);
}

// 4-byte UTF-8 codepoint (e.g. emoji) should also flow through unchanged.
void test_utf8_4byte_byte_safe() {
    // "a😀b" — emoji is U+1F600 = 0xF0 0x9F 0x98 0x80 (4 bytes), total 6 bytes
    const std::string text = "a\xF0\x9F\x98\x80""b";
    assert(text.size() == 6);
    Collector c;
    TokenizerConfig cfg{.ngram_min = 4, .ngram_max = 4, .emit_char = true, .emit_word = false};
    tokenize(text, cfg, c);
    // 6 bytes, k=4 → 3 4-grams
    assert(c.tokens.size() == 3);
    for (const auto& [t, _] : c.tokens) assert(t.size() == 4);
}

// is_word_char in src/tokenizer.cpp uses std::isalnum on each byte. UTF-8
// continuation bytes (0x80..0xBF) are not alnum, so the word extractor will
// split inside multi-byte codepoints. This is the documented byte-oriented
// behavior; the test pins it so any future ICU integration is intentional.
void test_word_extraction_splits_inside_utf8() {
    // "café" = 0x63 0x61 0x66 0xC3 0xA9 (5 bytes). Word extraction sees
    // "caf" as a word (3 alnum bytes), then 0xC3 0xA9 as non-alnum → no
    // second word emitted.
    const std::string text = "caf\xC3\xA9";
    assert(text.size() == 5);
    Collector c;
    TokenizerConfig cfg{.ngram_min = 0, .ngram_max = 0, .emit_char = false, .emit_word = true};
    tokenize(text, cfg, c);
    assert(c.tokens.size() == 1);
    assert(c.tokens[0].first == "caf");
}

}  // namespace

int main() {
    test_word_only_no_char_grams();
    test_word_only_punctuation_splits();
    test_word_only_underscore_is_word_char();
    test_word_only_empty();
    test_word_only_only_punctuation();
    test_combined_weights_distinct();
    test_text_below_ngram_min();
    test_text_at_ngram_min();
    test_inverted_range_is_silent();
    test_single_byte_input();
    test_utf8_3byte_byte_safe();
    test_utf8_4byte_byte_safe();
    test_word_extraction_splits_inside_utf8();
    return 0;
}
