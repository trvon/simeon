#include <cassert>
#include <string>
#include <string_view>

#include "simeon/text_rank.hpp"

using simeon::RankedSentence;
using simeon::TextRank;
using simeon::TextRankConfig;

namespace {

void test_empty_input_returns_empty() {
    TextRank tr;
    const auto r = tr.rank("", 1);
    assert(r.empty());
    assert(tr.top_sentence("").empty());
}

void test_single_sentence_returns_itself() {
    TextRank tr;
    const auto top = tr.top_sentence("The quick brown fox jumps over the lazy dog.");
    assert(!top.empty());
    assert(top.find("quick brown fox") != std::string_view::npos);
}

void test_short_fragments_filtered_by_min_tokens() {
    // "Hi." only has 1 token — below default min_sentence_tokens=3.
    TextRankConfig cfg;
    cfg.min_sentence_tokens = 3;
    TextRank tr(cfg);
    const auto r = tr.rank("Hi.", 1);
    assert(r.empty());
}

void test_multiple_sentences_rank_by_connectivity() {
    // Sentences that share tokens with others score higher than isolated
    // sentences. Here S0 and S1 share "records"/"database"; S2 and S3
    // are lexically disjoint from everything.
    const std::string text = "Databases store records efficiently. "
                             "Indexes on records speed up database queries for records in a table. "
                             "Weather is sunny today. "
                             "Planets orbit the sun.";
    TextRank tr;
    const auto r = tr.rank(text, 4);
    assert(r.size() == 4);
    // Top two must be the connected pair (S0 and S1) in some order.
    auto is_connected = [](std::string_view s) {
        return s.find("records") != std::string_view::npos ||
               s.find("database") != std::string_view::npos;
    };
    assert(is_connected(r[0].text));
    assert(is_connected(r[1].text));
    // Bottom two are the isolated fragments.
    assert(!is_connected(r[2].text));
    assert(!is_connected(r[3].text));
}

void test_degenerate_all_same_sentence_converges() {
    TextRank tr;
    const std::string text =
        "alpha beta gamma delta. alpha beta gamma delta. alpha beta gamma delta.";
    // All identical — power iteration should converge to equal scores,
    // and partial_sort should keep the first sentence at the top
    // (tie-break on index ascending).
    const auto r = tr.rank(text, 3);
    assert(r.size() == 3);
    assert(r[0].index == 0);
}

void test_convergence_cap_respected() {
    TextRankConfig cfg;
    cfg.max_iters = 1;
    cfg.convergence_epsilon = 0.0f; // force full iteration cap
    TextRank tr(cfg);
    const std::string text = "one two three four five. "
                             "six seven eight nine ten. "
                             "eleven twelve thirteen fourteen fifteen.";
    // Just verify it doesn't crash with a degenerate iteration cap.
    const auto r = tr.rank(text, 3);
    assert(r.size() == 3);
}

void test_top_k_zero_returns_all() {
    TextRank tr;
    const std::string text = "Alpha beta gamma delta. "
                             "Alpha beta gamma epsilon. "
                             "Alpha beta zeta eta.";
    const auto r = tr.rank(text, 0);
    assert(r.size() == 3);
}

void test_paragraph_break_splits_sentence() {
    TextRank tr;
    const std::string text = "First paragraph here with several tokens to pass the filter\n\n"
                             "Second paragraph here with several tokens to pass the filter";
    const auto r = tr.rank(text, 0);
    assert(r.size() == 2);
}

void test_case_folding_treats_tokens_equal() {
    // Case folding should make "Database" == "database" for overlap
    // scoring. Verify by setting up two identical sentences modulo
    // case and confirming they tie.
    TextRank tr;
    const std::string text = "Database queries are common. "
                             "DATABASE QUERIES ARE COMMON. "
                             "Unrelated words fill this sentence entirely.";
    const auto r = tr.rank(text, 2);
    // The two near-duplicate sentences should outrank the unrelated one.
    assert(r.size() == 2);
    bool first_two_win = true;
    for (const auto& rs : r) {
        if (rs.text.find("Unrelated") != std::string_view::npos) {
            first_two_win = false;
        }
    }
    assert(first_two_win);
}

} // namespace

int main() {
    test_empty_input_returns_empty();
    test_single_sentence_returns_itself();
    test_short_fragments_filtered_by_min_tokens();
    test_multiple_sentences_rank_by_connectivity();
    test_degenerate_all_same_sentence_converges();
    test_convergence_cap_respected();
    test_top_k_zero_returns_all();
    test_paragraph_break_splits_sentence();
    test_case_folding_treats_tokens_equal();
    return 0;
}
