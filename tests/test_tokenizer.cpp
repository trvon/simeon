#include <cassert>
#include <string>
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

}  // namespace

int main() {
    test_empty();
    test_char_3gram();
    test_char_range();
    test_word_mode();
    test_mixed();
    test_determinism();
    return 0;
}
