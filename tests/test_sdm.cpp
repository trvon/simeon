#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "simeon/bm25.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::Bm25Variant;
using simeon::SdmConfig;

namespace {

const std::vector<std::string>& toy_corpus() {
    static const std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "interest rate derivative pricing model",
        "the rate of interest on the loan",
        "mortgage backed security yields",
        "the cat sat on the mat and purred softly",
        "scientific papers cite prior work in references",
        "interest is due on the loan",
        "derivatives interest rate swap market",
        "foreign exchange rate forecast",
        "the human immune system fights off pathogens daily",
    };
    return corpus;
}

Bm25Index build_with_bigrams() {
    Bm25Config cfg;
    cfg.variant = Bm25Variant::Atire;
    cfg.build_word_bigrams = true;
    Bm25Index idx(cfg);
    for (const auto& d : toy_corpus())
        idx.add_doc(d);
    idx.finalize();
    return idx;
}

void test_sdm_recovers_plain_bm25_when_bigram_lambdas_zero() {
    auto idx = build_with_bigrams();
    const std::size_t n = idx.doc_count();
    std::vector<float> s_plain(n, 0.0f), s_sdm(n, 0.0f);
    idx.score("interest rate", s_plain);

    SdmConfig cfg;
    cfg.lambda_unigram = 1.0f;
    cfg.lambda_ordered = 0.0f;
    cfg.lambda_unordered = 0.0f;
    idx.score_sdm("interest rate", s_sdm, cfg);

    for (std::size_t i = 0; i < n; ++i) {
        assert(std::fabs(s_plain[i] - s_sdm[i]) < 1e-5f);
    }
}

void test_sdm_scores_adjacent_phrase_higher_than_nonadjacent() {
    // Docs: d0 has "interest rate" adjacent; d1 has both words but not
    // adjacent ("interest is due on the loan" — no "rate"; use a different
    // pair). Use "interest" and "loan" which appear in d2 adjacent
    // ("interest on the loan" is separated) and...
    //
    // Rather than rely on careful corpus phrasing, build a synthetic 2-doc
    // corpus where bigram ordering is controllable.
    Bm25Config cfg;
    cfg.variant = Bm25Variant::Atire;
    cfg.build_word_bigrams = true;
    Bm25Index idx(cfg);
    idx.add_doc("alpha beta gamma delta epsilon zeta eta theta iota kappa");
    idx.add_doc("beta alpha gamma delta epsilon zeta eta theta iota kappa");
    idx.finalize();

    // Plain BM25 ties the two docs on "alpha beta" — same tfs, same dl.
    std::vector<float> s_plain(2, 0.0f);
    idx.score("alpha beta", s_plain);
    assert(std::fabs(s_plain[0] - s_plain[1]) < 1e-5f);

    // SDM's ordered-bigram leg rewards adjacency in the doc matching the
    // query's adjacency. d0 has "alpha beta" adjacent, d1 does not.
    std::vector<float> s_sdm(2, 0.0f);
    SdmConfig scfg; // Metzler defaults.
    idx.score_sdm("alpha beta", s_sdm, scfg);
    assert(s_sdm[0] > s_sdm[1]);
}

void test_sdm_deterministic() {
    auto a = build_with_bigrams();
    auto b = build_with_bigrams();
    const std::size_t n = a.doc_count();
    std::vector<float> sa(n, 0.0f), sb(n, 0.0f);
    SdmConfig cfg;
    a.score_sdm("interest rate derivative", sa, cfg);
    b.score_sdm("interest rate derivative", sb, cfg);
    assert(sa == sb);
}

void test_sdm_without_bigrams_degenerates_to_unigram_leg() {
    // With build_word_bigrams=false, score_sdm() must not crash and must
    // produce the unigram leg only (bigram postings are empty).
    Bm25Config cfg;
    cfg.variant = Bm25Variant::Atire;
    cfg.build_word_bigrams = false; // explicit
    Bm25Index idx(cfg);
    for (const auto& d : toy_corpus())
        idx.add_doc(d);
    idx.finalize();

    const std::size_t n = idx.doc_count();
    std::vector<float> s_plain(n, 0.0f), s_sdm(n, 0.0f);
    idx.score("interest rate", s_plain);

    SdmConfig scfg; // default Metzler weights.
    idx.score_sdm("interest rate", s_sdm, scfg);
    // Unigram leg scaled by lambda_unigram (0.85); bigram legs contribute 0
    // because the postings tables are empty. So s_sdm ≈ 0.85 * s_plain.
    for (std::size_t i = 0; i < n; ++i) {
        [[maybe_unused]] const float expected = scfg.lambda_unigram * s_plain[i];
        assert(std::fabs(expected - s_sdm[i]) < 1e-5f);
    }
}

void test_sdm_single_term_query_has_no_bigram_contribution() {
    auto idx = build_with_bigrams();
    const std::size_t n = idx.doc_count();
    std::vector<float> s_plain(n, 0.0f), s_sdm(n, 0.0f);
    idx.score("interest", s_plain);

    SdmConfig cfg;
    cfg.lambda_unigram = 1.0f;
    idx.score_sdm("interest", s_sdm, cfg);
    // Single-term query: no bigrams possible. SDM with lambda_unigram=1 must
    // equal plain score() exactly.
    for (std::size_t i = 0; i < n; ++i) {
        assert(std::fabs(s_plain[i] - s_sdm[i]) < 1e-5f);
    }
}

} // namespace

int main() {
    test_sdm_recovers_plain_bm25_when_bigram_lambdas_zero();
    test_sdm_scores_adjacent_phrase_higher_than_nonadjacent();
    test_sdm_deterministic();
    test_sdm_without_bigrams_degenerates_to_unigram_leg();
    test_sdm_single_term_query_has_no_bigram_contribution();
    std::printf("test_sdm: OK\n");
    return 0;
}
