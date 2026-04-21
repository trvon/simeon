#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "simeon/aho_corasick.hpp"

using simeon::AcMatch;
using simeon::AhoCorasick;
using simeon::AhoCorasickConfig;

namespace {

AhoCorasick make_ac(std::span<const std::string_view> patterns,
                    std::span<const std::uint16_t> types, AhoCorasickConfig cfg = {}) {
    AhoCorasick ac(cfg);
    auto err = ac.build(patterns, types);
    assert(!err.has_value());
    return ac;
}

void test_empty_automaton_emits_nothing() {
    AhoCorasick ac;
    const auto hits = ac.match("any text");
    assert(hits.empty());
}

void test_single_pattern_exact_hit() {
    const std::string_view pats[] = {"disease"};
    const std::uint16_t tys[] = {7};
    auto ac = make_ac(pats, tys);
    const auto hits = ac.match("the disease spread");
    assert(hits.size() == 1);
    assert(hits[0].pattern_id == 0);
    assert(hits[0].type_id == 7);
    assert(hits[0].begin_utf8 == 4);
    assert(hits[0].end_utf8 == 11);
}

void test_case_insensitive_default() {
    const std::string_view pats[] = {"postgres"};
    const std::uint16_t tys[] = {1};
    auto ac = make_ac(pats, tys);
    const auto hits = ac.match("POSTGRES is running on Postgres");
    assert(hits.size() == 2);
}

void test_case_sensitive_mode() {
    AhoCorasickConfig cfg;
    cfg.case_insensitive = false;
    const std::string_view pats[] = {"Rust"};
    const std::uint16_t tys[] = {1};
    auto ac = make_ac(pats, tys, cfg);
    const auto hits = ac.match("Rust and rust are different");
    assert(hits.size() == 1);
    assert(hits[0].begin_utf8 == 0);
}

void test_whole_word_boundary_default_suppresses_substring() {
    // "cat" inside "catalog" must not match under whole_word=true.
    const std::string_view pats[] = {"cat"};
    const std::uint16_t tys[] = {3};
    auto ac = make_ac(pats, tys);
    const auto hits = ac.match("catalog of cat and cathedral");
    // Only the standalone "cat" should match.
    assert(hits.size() == 1);
    assert(hits[0].begin_utf8 == 11);
    assert(hits[0].end_utf8 == 14);
}

void test_whole_word_off_matches_substrings() {
    AhoCorasickConfig cfg;
    cfg.whole_word = false;
    const std::string_view pats[] = {"cat"};
    const std::uint16_t tys[] = {3};
    auto ac = make_ac(pats, tys, cfg);
    const auto hits = ac.match("catalog cat cathedral");
    // cat @ 0, cat @ 8, cat @ 12 → 3 substring hits (longest-match still
    // non-overlapping, so no duplicates at same position).
    assert(hits.size() == 3);
}

void test_longest_match_reduction() {
    // Both "value" and "time value" terminate; at "time value", longest
    // wins (single match covering both tokens) under longest_match.
    const std::string_view pats[] = {"value", "time value"};
    const std::uint16_t tys[] = {1, 2};
    auto ac = make_ac(pats, tys);
    const auto hits = ac.match("the time value of money");
    // Longest at pos 4: "time value" wins; no overlapping "value" at 9.
    assert(hits.size() == 1);
    assert(hits[0].pattern_id == 1);
    assert(hits[0].type_id == 2);
}

void test_overlapping_emission_when_longest_off() {
    AhoCorasickConfig cfg;
    cfg.longest_match = false;
    cfg.whole_word = false;
    const std::string_view pats[] = {"she", "he", "hers"};
    const std::uint16_t tys[] = {1, 2, 3};
    auto ac = make_ac(pats, tys, cfg);
    const auto hits = ac.match("ushers");
    // Classic Aho-Corasick example: "ushers" contains "she", "he",
    // "hers" all overlapping.
    std::unordered_set<std::string> got;
    for (const auto& h : hits) {
        got.insert(std::string("p") + std::to_string(h.pattern_id));
    }
    assert(got.count("p0")); // she
    assert(got.count("p1")); // he
    assert(got.count("p2")); // hers
}

void test_utf8_multibyte_passthrough() {
    // "café" contains non-ASCII; the automaton folds ASCII only.
    // Searching for "café" should still succeed via byte-level match.
    const std::string_view pats[] = {"café"};
    const std::uint16_t tys[] = {1};
    auto ac = make_ac(pats, tys);
    const auto hits = ac.match("the café is open");
    assert(hits.size() == 1);
}

void test_duplicate_patterns_both_terminate_in_same_state() {
    // Two identical patterns land at the same terminal; both emit (their
    // pattern_ids differ) when longest_match is on since they have the
    // same begin/end so longest-match reduction keeps only the
    // first-sorted one. We just assert that build() doesn't fail and
    // matching doesn't crash.
    const std::string_view pats[] = {"gene", "gene"};
    const std::uint16_t tys[] = {1, 2};
    auto ac = make_ac(pats, tys);
    const auto hits = ac.match("the gene is expressed");
    assert(!hits.empty());
}

void test_many_patterns_high_count_build() {
    // Build ~50k random lowercase patterns of length 4..12; verify the
    // automaton constructs cleanly and matches a sampled pattern.
    std::mt19937 rng(0xABCD1234);
    std::uniform_int_distribution<int> len(4, 12);
    std::uniform_int_distribution<int> ch('a', 'z');
    const std::size_t N = 50'000;
    std::vector<std::string> owned(N);
    std::vector<std::string_view> views(N);
    std::vector<std::uint16_t> types(N, 1);
    for (std::size_t i = 0; i < N; ++i) {
        const int k = len(rng);
        owned[i].resize(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j) {
            owned[i][j] = static_cast<char>(ch(rng));
        }
        views[i] = owned[i];
    }
    AhoCorasick ac;
    auto err = ac.build(views, types);
    assert(!err.has_value());
    assert(ac.pattern_count() == N);
    // Sample a known pattern and confirm it matches.
    const std::string text = " " + std::string(owned[42]) + " ";
    const auto hits = ac.match(text);
    assert(!hits.empty());
    bool found = false;
    for (const auto& h : hits) {
        if (h.pattern_id == 42) {
            found = true;
            break;
        }
    }
    assert(found);
}

void test_empty_pattern_rejected() {
    const std::string_view pats[] = {""};
    const std::uint16_t tys[] = {1};
    AhoCorasick ac;
    auto err = ac.build(pats, tys);
    assert(err.has_value());
    assert(ac.pattern_count() == 0);
}

void test_max_patterns_cap_enforced() {
    AhoCorasickConfig cfg;
    cfg.max_patterns = 2;
    const std::string_view pats[] = {"a", "b", "c"};
    const std::uint16_t tys[] = {1, 2, 3};
    AhoCorasick ac(cfg);
    auto err = ac.build(pats, tys);
    assert(err.has_value());
}

} // namespace

int main() {
    test_empty_automaton_emits_nothing();
    test_single_pattern_exact_hit();
    test_case_insensitive_default();
    test_case_sensitive_mode();
    test_whole_word_boundary_default_suppresses_substring();
    test_whole_word_off_matches_substrings();
    test_longest_match_reduction();
    test_overlapping_emission_when_longest_off();
    test_utf8_multibyte_passthrough();
    test_duplicate_patterns_both_terminate_in_same_state();
    test_many_patterns_high_count_build();
    test_empty_pattern_rejected();
    test_max_patterns_cap_enforced();
    return 0;
}
