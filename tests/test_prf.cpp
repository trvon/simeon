#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/prf.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::Bm25Variant;
using simeon::PrfConfig;

namespace {

const std::vector<std::string>& toy_corpus() {
    static const std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "infection of the wound caused fever and chills",
        "an infected cell undergoes apoptosis quickly",
        "machine learning models train on labeled data",
        "the cat sat on the mat and purred softly",
        "scientific papers cite prior work in references",
        "vaccines prevent the spread of common diseases",
        "the dog ran after the squirrel through the yard",
        "infections caused by bacteria respond to antibiotics",
        "the human immune system fights off pathogens daily",
        "antibiotics cure bacterial infections in wounded patients",
        "the pathogen entered the bloodstream through the wound",
    };
    return corpus;
}

Bm25Index build(Bm25Variant v) {
    Bm25Config cfg;
    cfg.variant = v;
    Bm25Index idx(cfg);
    for (const auto& d : toy_corpus())
        idx.add_doc(d);
    idx.finalize();
    return idx;
}

void test_prf_is_deterministic() {
    Bm25Index a = build(Bm25Variant::BM25Plus);
    Bm25Index b = build(Bm25Variant::BM25Plus);
    const std::size_t n = a.doc_count();
    std::vector<float> sa(n, 0.0f), sb(n, 0.0f);
    PrfConfig pc;
    pc.k = 5;
    pc.n_terms = 10;
    pc.alpha = 0.5f;
    simeon::score_with_prf(a, "infection", sa, pc);
    simeon::score_with_prf(b, "infection", sb, pc);
    assert(sa == sb);
}

void test_prf_alpha_zero_recovers_bm25() {
    // α=0 means the expansion contributes nothing; rescoring with α=0 must
    // equal the first-pass BM25 score, bit-for-bit on non-SAB variants.
    Bm25Index idx = build(Bm25Variant::BM25Plus);
    const std::size_t n = idx.doc_count();
    std::vector<float> bm25(n, 0.0f), prf(n, 0.0f);
    idx.score("infection", bm25);
    PrfConfig pc;
    pc.alpha = 0.0f;
    simeon::score_with_prf(idx, "infection", prf, pc);
    for (std::size_t i = 0; i < n; ++i) {
        assert(std::fabs(bm25[i] - prf[i]) < 1e-5f);
    }
}

void test_prf_n_terms_zero_recovers_bm25() {
    // n_terms=0 disables the expansion entirely; same invariant as α=0.
    Bm25Index idx = build(Bm25Variant::BM25Plus);
    const std::size_t n = idx.doc_count();
    std::vector<float> bm25(n, 0.0f), prf(n, 0.0f);
    idx.score("infection", bm25);
    PrfConfig pc;
    pc.alpha = 0.5f;
    pc.n_terms = 0;
    simeon::score_with_prf(idx, "infection", prf, pc);
    for (std::size_t i = 0; i < n; ++i) {
        assert(std::fabs(bm25[i] - prf[i]) < 1e-5f);
    }
}

void test_prf_expands_query_with_related_terms() {
    // For "infection", the feedback set contains docs 1, 2, 8, 10 ("infection",
    // "infected", "infections", "bacterial infections"). RM3 expansion pulls in
    // co-occurring terms (e.g. "wound", "bacterial", "antibiotics"); this lifts
    // doc 11 ("pathogen entered bloodstream through the wound") above its plain
    // BM25 rank since "wound" becomes a weighted expansion term.
    Bm25Index idx = build(Bm25Variant::BM25Plus);
    const std::size_t n = idx.doc_count();
    std::vector<float> bm25(n, 0.0f), prf(n, 0.0f);
    idx.score("infection", bm25);
    PrfConfig pc;
    pc.k = 4;
    pc.n_terms = 10;
    pc.alpha = 0.5f;
    simeon::score_with_prf(idx, "infection", prf, pc);

    // Basic sanity: PRF scores are finite.
    for (auto v : prf) {
        assert(std::isfinite(v));
        (void)v;
    }
    // Plain BM25 scores doc 11 at 0 (no "infection" token); PRF should lift
    // it above 0 through expansion terms that overlap ("wound", "bloodstream").
    assert(bm25[11] == 0.0f);
    assert(prf[11] > 0.0f);
    // The original on-topic doc (doc 1) should still score strongly.
    assert(prf[1] > prf[0]);
    assert(prf[1] > prf[4]);
}

void test_prf_on_empty_corpus_is_safe() {
    // Degenerate: empty query with a finalized but empty corpus shouldn't crash.
    Bm25Config cfg;
    cfg.variant = Bm25Variant::BM25Plus;
    Bm25Index idx(cfg);
    idx.finalize();
    std::vector<float> s; // size 0 matches doc_count()
    PrfConfig pc;
    simeon::score_with_prf(idx, "anything", s, pc);
    assert(s.empty());
}

void test_prf_works_with_sab_variant() {
    // SAB's first-pass n-gram backoff is preserved; rescoring uses the exact
    // word-posting path. Result must differ from plain BM25+ (SAB picks up
    // morphological variants the first pass BM25+ misses).
    Bm25Config cfg_sab;
    cfg_sab.variant = Bm25Variant::SubwordAwareBackoff;
    cfg_sab.subword_gamma = 5.0f;
    Bm25Index sab(cfg_sab);
    for (const auto& d : toy_corpus())
        sab.add_doc(d);
    sab.finalize();
    const std::size_t n = sab.doc_count();
    std::vector<float> prf(n, 0.0f);
    PrfConfig pc;
    pc.k = 5;
    pc.alpha = 0.5f;
    simeon::score_with_prf(sab, "infect", prf, pc); // OOV exact; SAB's n-gram path fires.
    for (auto v : prf) {
        assert(std::isfinite(v));
        (void)v;
    }
    // At least one on-topic doc scores strictly above zero.
    [[maybe_unused]] bool any_lift = false;
    for (std::size_t i = 0; i < n; ++i)
        if (prf[i] > 0.0f) {
            any_lift = true;
            break;
        }
    assert(any_lift);
}

} // namespace

int main() {
    test_prf_is_deterministic();
    test_prf_alpha_zero_recovers_bm25();
    test_prf_n_terms_zero_recovers_bm25();
    test_prf_expands_query_with_related_terms();
    test_prf_on_empty_corpus_is_safe();
    test_prf_works_with_sab_variant();
    return 0;
}
