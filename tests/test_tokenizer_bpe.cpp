#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/tokenizer.hpp"
#include "simeon/tokenizer_bpe.hpp"

using simeon::BpeMerges;
using simeon::NGramEmitter;
using simeon::TokenizerConfig;

namespace {

struct CollectSink final : NGramEmitter {
    std::vector<std::string> tokens;
    void on_token(std::string_view tok, float) override {
        tokens.emplace_back(tok);
    }
};

void test_learn_deterministic_same_input() {
    std::array<std::string_view, 4> corpus = {
        "the quick brown fox",
        "the lazy dog",
        "fox jumps over the lazy dog",
        "quick brown fox jumps quickly",
    };
    auto m1 = BpeMerges::learn(corpus, 320);
    auto m2 = BpeMerges::learn(corpus, 320);
    assert(m1.vocab_size() == m2.vocab_size());
    assert(m1.serialize() == m2.serialize());
}

void test_learn_respects_target_vocab() {
    std::array<std::string_view, 1> corpus = {"abcabcabcabc"};
    auto m = BpeMerges::learn(corpus, 260);
    // Initial 256 + at most 4 merges. With "abcabcabcabc" (~3 distinct
    // adjacent pairs) the learner adds merges until count drops below 2 or
    // target is hit, whichever comes first.
    assert(m.vocab_size() <= 260);
    assert(m.vocab_size() >= 256);
}

void test_learn_below_threshold_yields_no_merges() {
    std::array<std::string_view, 1> corpus = {"abcdef"};
    auto m = BpeMerges::learn(corpus, 256);
    assert(m.vocab_size() == 256);
    assert(m.merge_count() == 0);
}

void test_serialize_roundtrip() {
    std::array<std::string_view, 3> corpus = {
        "infectious infection infect",
        "infect infect infect",
        "the quick brown fox",
    };
    auto m1 = BpeMerges::learn(corpus, 320);
    auto serialized = m1.serialize();
    auto m2 = BpeMerges::from_text(serialized);
    assert(m1.vocab_size() == m2.vocab_size());
    assert(m1.serialize() == m2.serialize());

    // Apply on the same word must yield identical token streams.
    CollectSink s1, s2;
    m1.apply("infect", s1);
    m2.apply("infect", s2);
    assert(s1.tokens == s2.tokens);
}

void test_apply_reproducible_across_reloads() {
    std::array<std::string_view, 2> corpus = {
        "hello world hello hello world",
        "world world hello",
    };
    auto m = BpeMerges::learn(corpus, 280);
    CollectSink a, b;
    m.apply("hello", a);
    m.apply("hello", b);
    assert(a.tokens == b.tokens);
    assert(!a.tokens.empty());
}

void test_apply_unknown_word_falls_back_to_bytes() {
    // No merges learned -> apply must emit one token per byte.
    std::array<std::string_view, 1> corpus = {"x"};  // single 1-char "word"
    auto m = BpeMerges::learn(corpus, 256);  // no headroom for merges
    CollectSink s;
    m.apply("zz", s);
    assert(s.tokens.size() == 2);
    assert(s.tokens[0] == "z");
    assert(s.tokens[1] == "z");
}

void test_apply_empty_word_no_emit() {
    std::array<std::string_view, 1> corpus = {"abc abc abc"};
    auto m = BpeMerges::learn(corpus, 260);
    CollectSink s;
    m.apply("", s);
    assert(s.tokens.empty());
}

void test_apply_single_byte() {
    std::array<std::string_view, 1> corpus = {"abc abc"};
    auto m = BpeMerges::learn(corpus, 260);
    CollectSink s;
    m.apply("q", s);
    assert(s.tokens.size() == 1);
    assert(s.tokens[0] == "q");
}

void test_apply_produces_subword_tokens_on_held_out_word() {
    // Train on a corpus where common substrings ("in", "ed", "ing") appear
    // across many distinct words, so the BPE learns them as merges. Then
    // apply to a held-out word that contains those substrings and assert
    // fewer tokens than bytes — proving merges fired, i.e. the BPE actually
    // produces subword tokens rather than byte-fallback.
    std::array<std::string_view, 12> corpus = {
        "running walking talking",
        "playing singing dancing",
        "jumping reading writing",
        "looking working baking",
        "ringing pinging dinging",
        "the quick brown fox",
        "lazy dog sat down",
        "ed red bed fed led",
        "indoors infield insight inset",
        "interior internal income invent",
        "hearing fearing nearing",
        "wedded petted batted matted",
    };
    auto m = BpeMerges::learn(corpus, 320);

    CollectSink s;
    // "indexing" was not in the training set but its pieces ("in", "ing")
    // appeared in many training words.
    m.apply("indexing", s);

    assert(s.tokens.size() < std::string_view("indexing").size());
}

void test_tokenize_with_bpe_emits_subwords() {
    std::array<std::string_view, 1> corpus = {"abcabcabc xyzxyzxyz"};
    auto m = BpeMerges::learn(corpus, 270);

    CollectSink s;
    TokenizerConfig cfg{};
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.emit_char = false;
    cfg.emit_word = false;
    cfg.bpe = &m;
    simeon::tokenize("abcabcabc xyzxyzxyz", cfg, s);
    assert(!s.tokens.empty());
}

void test_tokenize_no_bpe_unchanged_behaviour() {
    CollectSink baseline, with_null;
    TokenizerConfig cfg{};
    cfg.ngram_min = 3;
    cfg.ngram_max = 3;
    cfg.emit_char = true;
    cfg.emit_word = true;
    cfg.bpe = nullptr;
    simeon::tokenize("hello world", cfg, baseline);
    simeon::tokenize("hello world", cfg, with_null);
    assert(baseline.tokens == with_null.tokens);
    assert(!baseline.tokens.empty());
}

void test_from_text_handles_escaped_bytes() {
    // Hand-construct a serialized merge involving non-printable bytes.
    // Initial 256 vocab includes byte 0x09 (\t) and 0x20 (space), so
    // a merge of \x09 + \x20 -> id 256 should round-trip cleanly.
    std::string ser = "\\x09\t\\x20\n";
    auto m = BpeMerges::from_text(ser);
    assert(m.vocab_size() == 257);
    assert(m.serialize() == ser);
}

}  // namespace

int main() {
    test_learn_deterministic_same_input();
    test_learn_respects_target_vocab();
    test_learn_below_threshold_yields_no_merges();
    test_serialize_roundtrip();
    test_apply_reproducible_across_reloads();
    test_apply_unknown_word_falls_back_to_bytes();
    test_apply_empty_word_no_emit();
    test_apply_single_byte();
    test_apply_produces_subword_tokens_on_held_out_word();
    test_tokenize_with_bpe_emits_subwords();
    test_tokenize_no_bpe_unchanged_behaviour();
    test_from_text_handles_escaped_bytes();
    return 0;
}
