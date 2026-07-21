#include <cassert>
#include <string>
#include <vector>

#include "simeon/tokenizer.hpp"

using simeon::CharNGramScope;
using simeon::NGramEmitter;
using simeon::tokenize;
using simeon::TokenizerConfig;

namespace {

struct Collector final : public NGramEmitter {
    std::vector<std::pair<std::string, float>> tokens;
    void on_token(std::string_view t, float w) override { tokens.emplace_back(t, w); }
};

void test_empty() {
    Collector c;
    TokenizerConfig cfg{3, 5, true, false};
    tokenize("", cfg, c);
    assert(c.tokens.empty());
}

void test_char_3gram() {
    Collector c;
    TokenizerConfig cfg{3, 3, true, false};
    tokenize("abcd", cfg, c);
    assert(c.tokens.size() == 2);
    assert(c.tokens[0].first == "abc");
    assert(c.tokens[1].first == "bcd");
}

void test_char_range() {
    Collector c;
    TokenizerConfig cfg{3, 5, true, false};
    tokenize("abcd", cfg, c);
    assert(c.tokens.size() == 3);
    assert(c.tokens[0].first == "abc");
    assert(c.tokens[1].first == "bcd");
    assert(c.tokens[2].first == "abcd");
}

void test_word_mode() {
    Collector c;
    TokenizerConfig cfg{0, 0, false, true};
    tokenize("foo bar-baz 123", cfg, c);
    assert(c.tokens.size() == 4);
    assert(c.tokens[0].first == "foo");
    assert(c.tokens[1].first == "bar");
    assert(c.tokens[2].first == "baz");
    assert(c.tokens[3].first == "123");
    for (const auto& [_, w] : c.tokens) {
        assert(w == 0.5f);
    }
}

void test_mixed() {
    Collector c;
    TokenizerConfig cfg{3, 3, true, true};
    tokenize("ab cd", cfg, c);
    // char-3 from "ab cd": "ab ", "b c", " cd"; plus word tokens "ab", "cd"
    assert(c.tokens.size() == 5);
}

void test_determinism() {
    Collector a, b;
    TokenizerConfig cfg{3, 5, true, true};
    tokenize("The quick brown fox", cfg, a);
    tokenize("The quick brown fox", cfg, b);
    assert(a.tokens == b.tokens);
}

void test_word_bounded_char_ngrams_add_markers_and_do_not_cross_words() {
    Collector c;
    TokenizerConfig cfg{3, 3, true, false};
    cfg.char_ngram_scope = CharNGramScope::WordBounded;
    tokenize("ab cd", cfg, c);
    const std::vector<std::pair<std::string, float>> expected = {
        {"<ab", 1.0f}, {"ab>", 1.0f}, {"<cd", 1.0f}, {"cd>", 1.0f}};
    assert(c.tokens == expected);
}

void test_word_bounded_char_ngrams_preserve_utf8_bytes() {
    Collector c;
    TokenizerConfig cfg{3, 3, true, false};
    cfg.char_ngram_scope = CharNGramScope::WordBounded;
    const std::string cafe = "caf\xC3\xA9";
    tokenize(cafe, cfg, c);
    assert(c.tokens.size() == 5);
    assert(c.tokens.front().first == "<ca");
    assert(c.tokens.back().first == std::string("\xC3\xA9>", 3));
}

void test_word_bounded_char_ngrams_ignore_punctuation_only_input() {
    Collector c;
    TokenizerConfig cfg{3, 5, true, false};
    cfg.char_ngram_scope = CharNGramScope::WordBounded;
    tokenize("... --- !!!", cfg, c);
    assert(c.tokens.empty());
}

} // namespace

int main() {
    test_empty();
    test_char_3gram();
    test_char_range();
    test_word_mode();
    test_mixed();
    test_determinism();
    test_word_bounded_char_ngrams_add_markers_and_do_not_cross_words();
    test_word_bounded_char_ngrams_preserve_utf8_bytes();
    test_word_bounded_char_ngrams_ignore_punctuation_only_input();
    return 0;
}
