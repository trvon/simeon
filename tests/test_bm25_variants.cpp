#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "simeon/bm25.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::Bm25Variant;

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
    };
    return corpus;
}

Bm25Index build(Bm25Variant v) {
    Bm25Config cfg;
    cfg.variant = v;
    cfg.delta = 1.0f;
    Bm25Index idx(cfg);
    for (const auto& d : toy_corpus())
        idx.add_doc(d);
    idx.finalize();
    return idx;
}

void test_each_variant_deterministic() {
    for (auto v : {Bm25Variant::Atire, Bm25Variant::BM25Plus, Bm25Variant::BM25L,
                   Bm25Variant::DLH13, Bm25Variant::PL2, Bm25Variant::DPH, Bm25Variant::Dcm,
                   Bm25Variant::SubwordAwareBackoff}) {
        auto a = build(v);
        auto b = build(v);
        const std::size_t n = a.doc_count();
        std::vector<float> sa(n, 0.0f), sb(n, 0.0f);
        a.score("infection", sa);
        b.score("infection", sb);
        assert(sa == sb);
    }
}

void test_bm25_plus_recovers_atire_when_delta_zero() {
    Bm25Config cfg_atire;
    cfg_atire.variant = Bm25Variant::Atire;
    Bm25Config cfg_plus;
    cfg_plus.variant = Bm25Variant::BM25Plus;
    cfg_plus.delta = 0.0f;

    Bm25Index a(cfg_atire), p(cfg_plus);
    for (const auto& d : toy_corpus()) {
        a.add_doc(d);
        p.add_doc(d);
    }
    a.finalize();
    p.finalize();

    const std::size_t n = a.doc_count();
    std::vector<float> sa(n, 0.0f), sp(n, 0.0f);
    a.score("the dog ran", sa);
    p.score("the dog ran", sp);
    for (std::size_t i = 0; i < n; ++i) {
        // BM25+ with δ=0 recovers Atire's per-(doc,term) score *only when
        // the term is present*. For absent terms, BM25+ contributes 0
        // (because it's not in postings) — same as Atire. So scores match.
        assert(std::fabs(sa[i] - sp[i]) < 1e-5f);
    }
}

void test_bm25_plus_floor_lifts_long_doc_score_above_atire() {
    // Lv & Zhai's motivating observation: long documents that contain a
    // query term still get a small floor contribution from BM25+ where
    // Atire's saturation can drive their score arbitrarily close to 0.
    // We don't assert the absolute floor, just that BM25+ score >= Atire
    // score for every doc with the term present (since δ>0 only adds).
    Bm25Index a = build(Bm25Variant::Atire);
    Bm25Index p = build(Bm25Variant::BM25Plus);
    const std::size_t n = a.doc_count();
    std::vector<float> sa(n, 0.0f), sp(n, 0.0f);
    a.score("infection", sa);
    p.score("infection", sp);
    for (std::size_t i = 0; i < n; ++i) {
        if (sa[i] > 0.0f) {
            assert(sp[i] >= sa[i] - 1e-5f);
        }
    }
}

void test_bm25l_finite_and_increasing_in_tf() {
    // Sanity for BM25L: scores are finite, and a doc with more occurrences
    // of the query term outscores a doc with fewer (within the same toy
    // corpus context). Constructed so we can compare two specific docs.
    Bm25Config cfg;
    cfg.variant = Bm25Variant::BM25L;
    Bm25Index idx(cfg);
    idx.add_doc("alpha");                   // tf=1
    idx.add_doc("alpha alpha alpha alpha"); // tf=4
    idx.add_doc("beta");                    // tf=0 for "alpha"
    idx.finalize();
    std::vector<float> s(3, 0.0f);
    idx.score("alpha", s);
    assert(std::isfinite(s[0]));
    assert(std::isfinite(s[1]));
    assert(std::isfinite(s[2]));
    assert(s[1] > s[0]);
    assert(s[2] == 0.0f);
}

void test_dlh13_produces_finite_scores_and_ranks_relevant_doc() {
    // DLH13 is parameter-free; just assert it produces finite scores and
    // ranks a directly-on-topic doc above unrelated docs.
    Bm25Index idx = build(Bm25Variant::DLH13);
    const std::size_t n = idx.doc_count();
    std::vector<float> s(n, 0.0f);
    idx.score("infection", s);
    for (auto v : s)
        assert(std::isfinite(v));

    // Doc 1 ("infection of the wound...") should outscore unrelated docs
    // 0 ("the quick brown fox") and 4 ("the cat sat on the mat").
    assert(s[1] > s[0]);
    assert(s[1] > s[4]);
}

void test_subword_aware_oov_query_term_gets_nonzero_score() {
    // Index has "infection", "infected", "infections" but never "infect".
    // The exact-only path returns 0 for all docs; SubwordAwareBackoff's
    // n-gram fallback should give nonzero scores to docs containing
    // morphological variants.
    Bm25Config cfg;
    cfg.variant = Bm25Variant::SubwordAwareBackoff;
    cfg.subword_gamma = 0.0f; // strict OOV fallback
    Bm25Index idx(cfg);
    for (const auto& d : toy_corpus())
        idx.add_doc(d);
    idx.finalize();

    const std::size_t n = idx.doc_count();
    std::vector<float> s(n, 0.0f);
    idx.score("infect", s);

    // No exact "infect" anywhere in the corpus — every contribution comes
    // from n-gram fallback. Docs 1, 2, 8, 9 contain the "infect" trigram
    // morphologically; they should score strictly above unrelated docs.
    assert(s[1] > 0.0f);
    assert(s[2] > 0.0f);
    assert(s[8] > 0.0f);
    // Doc 0 ("the quick brown fox") shares no "infect" n-grams.
    assert(s[0] == 0.0f);
}

void test_subword_aware_strict_collapses_to_bm25plus_on_known_query() {
    // With γ=0 and a query term that IS in the corpus, α=1 → only the
    // exact BM25+ path runs. Score must equal BM25+ alone.
    Bm25Config cfg_sab;
    cfg_sab.variant = Bm25Variant::SubwordAwareBackoff;
    cfg_sab.subword_gamma = 0.0f;
    Bm25Config cfg_plus;
    cfg_plus.variant = Bm25Variant::BM25Plus;

    Bm25Index sab(cfg_sab);
    Bm25Index plus(cfg_plus);
    for (const auto& d : toy_corpus()) {
        sab.add_doc(d);
        plus.add_doc(d);
    }
    sab.finalize();
    plus.finalize();

    const std::size_t n = sab.doc_count();
    std::vector<float> ss(n, 0.0f), sp(n, 0.0f);
    sab.score("dog", ss); // "dog" appears in docs 0, 7
    plus.score("dog", sp);
    for (std::size_t i = 0; i < n; ++i) {
        assert(std::fabs(ss[i] - sp[i]) < 1e-4f);
    }
}

void test_pl2_finite_and_ranks_relevant_doc() {
    // PL2 is Poisson-Laplace DFR; should produce finite scores, rank a
    // directly-on-topic doc above unrelated ones, and reject the all-zero
    // degenerate case.
    Bm25Index idx = build(Bm25Variant::PL2);
    const std::size_t n = idx.doc_count();
    std::vector<float> s(n, 0.0f);
    idx.score("infection", s);
    for (auto v : s)
        assert(std::isfinite(v));
    assert(s[1] > s[0]);
    assert(s[1] > s[4]);
}

void test_pl2_c_parameter_changes_score() {
    // c scales the length-normalization exponent; two different c values on
    // the same corpus/query must diverge for at least one doc.
    Bm25Config cfg_a;
    cfg_a.variant = Bm25Variant::PL2;
    cfg_a.pl2_c = 1.0f;
    Bm25Config cfg_b;
    cfg_b.variant = Bm25Variant::PL2;
    cfg_b.pl2_c = 5.0f;
    Bm25Index a(cfg_a), b(cfg_b);
    for (const auto& d : toy_corpus()) {
        a.add_doc(d);
        b.add_doc(d);
    }
    a.finalize();
    b.finalize();
    const std::size_t n = a.doc_count();
    std::vector<float> sa(n, 0.0f), sb(n, 0.0f);
    a.score("infection", sa);
    b.score("infection", sb);
    bool differs = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(sa[i] - sb[i]) > 1e-5f) {
            differs = true;
            break;
        }
    }
    assert(differs);
}

void test_dph_finite_and_ranks_relevant_doc() {
    // DPH is parameter-free hypergeometric DFR; finite and relevant-ranking
    // sanity like DLH13. The normalization (1 - tf/dl)^2 / (tf + 1) forces
    // single-token docs (tf==dl) to score 0, which the test exercises via
    // doc 1 containing "infection" among many other tokens.
    Bm25Index idx = build(Bm25Variant::DPH);
    const std::size_t n = idx.doc_count();
    std::vector<float> s(n, 0.0f);
    idx.score("infection", s);
    for (auto v : s)
        assert(std::isfinite(v));
    assert(s[1] > s[0]);
    assert(s[1] > s[4]);
}

void test_dph_single_token_doc_scores_zero() {
    // DPH norm (1 - tf/dl)^2 = 0 when tf==dl, which happens when a doc is
    // the bare query term and nothing else. Verifies the degenerate guard.
    Bm25Config cfg;
    cfg.variant = Bm25Variant::DPH;
    Bm25Index idx(cfg);
    idx.add_doc("alpha");
    idx.add_doc("alpha beta gamma delta");
    idx.finalize();
    std::vector<float> s(2, 0.0f);
    idx.score("alpha", s);
    assert(std::isfinite(s[0]));
    assert(std::isfinite(s[1]));
    // Single-token doc (tf==dl) triggers the degenerate-input guard.
    assert(s[0] == 0.0f);
    assert(s[1] > 0.0f);
}

void test_dcm_produces_finite_scores_and_ranks_relevant_doc() {
    // DCM uses a Zhai-Lafferty Dirichlet-smoothed LM contribution:
    // log(1 + tf * total_tokens / (α_sum * ttf)). Should be finite and
    // rank a relevant doc above unrelated docs, like DLH13.
    Bm25Index idx = build(Bm25Variant::Dcm);
    const std::size_t n = idx.doc_count();
    std::vector<float> s(n, 0.0f);
    idx.score("infection", s);
    for (auto v : s)
        assert(std::isfinite(v));

    assert(s[1] > s[0]);
    assert(s[1] > s[4]);
}

void test_dcm_monotone_in_tf() {
    // Sanity: a doc with more occurrences of the query term outscores a
    // doc with fewer, under DCM. log(1 + k*x) is strictly increasing in x.
    Bm25Config cfg;
    cfg.variant = Bm25Variant::Dcm;
    Bm25Index idx(cfg);
    idx.add_doc("alpha");                   // tf=1
    idx.add_doc("alpha alpha alpha alpha"); // tf=4
    idx.add_doc("beta");                    // tf=0 for "alpha"
    idx.finalize();
    std::vector<float> s(3, 0.0f);
    idx.score("alpha", s);
    assert(std::isfinite(s[0]));
    assert(std::isfinite(s[1]));
    assert(std::isfinite(s[2]));
    assert(s[1] > s[0]);
    assert(s[2] == 0.0f);
}

void test_dcm_alpha_sum_override_is_deterministic() {
    // With dcm_alpha_sum > 0 the config-supplied value is used directly;
    // two identical builds with the same override must produce identical
    // scores, and a different override must change the score (α enters
    // the denominator of the log argument).
    Bm25Config cfg_default;
    cfg_default.variant = Bm25Variant::Dcm;
    cfg_default.dcm_alpha_sum = 0.0f; // derive from corpus (avg_dl)

    Bm25Config cfg_override;
    cfg_override.variant = Bm25Variant::Dcm;
    cfg_override.dcm_alpha_sum = 50.0f;

    Bm25Index a(cfg_override), b(cfg_override), c(cfg_default);
    for (const auto& d : toy_corpus()) {
        a.add_doc(d);
        b.add_doc(d);
        c.add_doc(d);
    }
    a.finalize();
    b.finalize();
    c.finalize();

    const std::size_t n = a.doc_count();
    std::vector<float> sa(n, 0.0f), sb(n, 0.0f), sc(n, 0.0f);
    a.score("infection", sa);
    b.score("infection", sb);
    c.score("infection", sc);
    assert(sa == sb);

    bool differs = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(sa[i] - sc[i]) > 1e-5f) {
            differs = true;
            break;
        }
    }
    assert(differs);
}

void test_subword_aware_smooth_blends_exact_and_ngram() {
    // With γ>0 and an in-corpus term, both paths contribute. The blend
    // result must differ from pure BM25+ (smaller exact weight + nonzero
    // n-gram contribution).
    Bm25Config cfg_smooth;
    cfg_smooth.variant = Bm25Variant::SubwordAwareBackoff;
    cfg_smooth.subword_gamma = 5.0f;
    Bm25Config cfg_strict;
    cfg_strict.variant = Bm25Variant::SubwordAwareBackoff;
    cfg_strict.subword_gamma = 0.0f;

    Bm25Index smooth(cfg_smooth);
    Bm25Index strict(cfg_strict);
    for (const auto& d : toy_corpus()) {
        smooth.add_doc(d);
        strict.add_doc(d);
    }
    smooth.finalize();
    strict.finalize();

    const std::size_t n = smooth.doc_count();
    std::vector<float> ss(n, 0.0f), st(n, 0.0f);
    smooth.score("dog", ss);
    strict.score("dog", st);
    bool any_diff = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::fabs(ss[i] - st[i]) > 1e-4f) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);
}

} // namespace

int main() {
    test_each_variant_deterministic();
    test_bm25_plus_recovers_atire_when_delta_zero();
    test_bm25_plus_floor_lifts_long_doc_score_above_atire();
    test_bm25l_finite_and_increasing_in_tf();
    test_dlh13_produces_finite_scores_and_ranks_relevant_doc();
    test_pl2_finite_and_ranks_relevant_doc();
    test_pl2_c_parameter_changes_score();
    test_dph_finite_and_ranks_relevant_doc();
    test_dph_single_token_doc_scores_zero();
    test_dcm_produces_finite_scores_and_ranks_relevant_doc();
    test_dcm_monotone_in_tf();
    test_dcm_alpha_sum_override_is_deterministic();
    test_subword_aware_oov_query_term_gets_nonzero_score();
    test_subword_aware_strict_collapses_to_bm25plus_on_known_query();
    test_subword_aware_smooth_blends_exact_and_ngram();
    return 0;
}
