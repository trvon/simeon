#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "simeon/bm25.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;

namespace {

void test_empty_index_score_throws() {
    Bm25Index idx;
    bool threw = false;
    try {
        std::vector<float> s;
        idx.score("anything", s);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_basic_ranking() {
    Bm25Index idx;
    idx.add_doc("the quick brown fox jumps");
    idx.add_doc("a slow brown turtle walks");
    idx.add_doc("the fox is quick and brown");
    idx.finalize();
    assert(idx.doc_count() == 3);

    std::vector<float> s(3, 0.0f);
    idx.score("quick fox", s);

    // Doc 0 and 2 contain both query terms; doc 1 contains neither.
    assert(s[0] > 0.0f);
    assert(s[2] > 0.0f);
    assert(s[1] == 0.0f);
}

void test_idf_rare_beats_common() {
    Bm25Index idx;
    // "the" appears in all three docs => low IDF.
    // "rare" appears in only doc 1 => high IDF.
    idx.add_doc("the cat sat on the mat");
    idx.add_doc("the rare bird flew away");
    idx.add_doc("the dog ran fast");
    idx.finalize();

    std::vector<float> s_the(3, 0.0f);
    idx.score("the", s_the);
    std::vector<float> s_rare(3, 0.0f);
    idx.score("rare", s_rare);

    // "rare" hit on doc 1 should outscore any "the" hit.
    const float max_the = *std::max_element(s_the.begin(), s_the.end());
    assert(s_rare[1] > max_the);
}

void test_tf_saturation() {
    // BM25 with k1=1.2 saturates: doubling tf does NOT double the contribution.
    Bm25Index idx;
    idx.add_doc("alpha");                         // tf=1
    idx.add_doc("alpha alpha alpha alpha alpha"); // tf=5
    idx.finalize();

    std::vector<float> s(2, 0.0f);
    idx.score("alpha", s);

    // tf=5 contributes more than tf=1, but less than 5x (saturation).
    assert(s[1] > s[0]);
    assert(s[1] < 5.0f * s[0]);
}

void test_determinism() {
    Bm25Config cfg;
    Bm25Index a(cfg), b(cfg);
    const std::vector<std::string> docs = {
        "alpha beta gamma",
        "delta epsilon zeta",
        "alpha epsilon iota",
    };
    for (const auto& d : docs) {
        a.add_doc(d);
        b.add_doc(d);
    }
    a.finalize();
    b.finalize();

    std::vector<float> sa(3, 0.0f), sb(3, 0.0f);
    a.score("alpha epsilon", sa);
    b.score("alpha epsilon", sb);
    assert(sa == sb);
}

void test_unknown_term_zero() {
    Bm25Index idx;
    idx.add_doc("alpha beta gamma");
    idx.finalize();
    std::vector<float> s(1, 0.0f);
    idx.score("zzz_not_in_index_zzz", s);
    assert(s[0] == 0.0f);
}

void test_score_size_mismatch_throws() {
    Bm25Index idx;
    idx.add_doc("alpha");
    idx.finalize();
    bool threw = false;
    try {
        std::vector<float> s(99, 0.0f);
        idx.score("alpha", s);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_add_after_finalize_throws() {
    Bm25Index idx;
    idx.add_doc("alpha");
    idx.finalize();
    bool threw = false;
    try {
        idx.add_doc("beta");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_bm25f_aux_weight_zero_recovers_plain_bm25() {
    Bm25Index idx;
    idx.add_doc("gene therapy reduced inflammation", "gene therapy breakthrough");
    idx.add_doc("inflammation markers remained elevated", "biomarker update");
    idx.finalize();

    std::vector<float> plain(2, 0.0f), fused(2, 0.0f);
    idx.score("gene therapy", plain);
    idx.score_bm25f("gene therapy", fused, 1.0f, 0.0f);
    assert(plain == fused);
}

void test_bm25f_aux_field_adds_signal() {
    Bm25Index idx;
    idx.add_doc("general overview of methods", "gene therapy breakthrough");
    idx.add_doc("general overview of methods", "market outlook summary");
    idx.finalize();

    std::vector<float> s(2, 0.0f);
    idx.score_bm25f("gene therapy", s, 0.0f, 1.0f);
    assert(s[0] > 0.0f);
    assert(s[1] == 0.0f);
}

void test_add_doc_without_aux_keeps_bm25f_safe() {
    Bm25Index idx;
    idx.add_doc("alpha beta gamma");
    idx.finalize();

    std::vector<float> plain(1, 0.0f), fused(1, 0.0f);
    idx.score("alpha", plain);
    idx.score_bm25f("alpha", fused, 1.0f, 1.0f);
    assert(plain == fused);
}

void test_mixed_aux_presence_stays_aligned() {
    Bm25Index idx;
    idx.add_doc("general overview");
    idx.add_doc("general overview", "gene therapy breakthrough");
    idx.add_doc("general overview");
    idx.finalize();

    std::vector<float> s(3, 0.0f);
    idx.score_bm25f("missing body term", "gene therapy", s, 0.0f, 1.0f);
    assert(s[0] == 0.0f);
    assert(s[1] > 0.0f);
    assert(s[2] == 0.0f);
}

void test_bm25f_accepts_distinct_aux_query_text() {
    Bm25Index idx;
    idx.add_doc("general overview", "ent7");
    idx.add_doc("general overview", "");
    idx.finalize();

    std::vector<float> s(2, 0.0f);
    idx.score_bm25f("missing body term", "ent7", s, 0.0f, 1.0f);
    assert(s[0] > 0.0f);
    assert(s[1] == 0.0f);
}

} // namespace

int main() {
    test_empty_index_score_throws();
    test_basic_ranking();
    test_idf_rare_beats_common();
    test_tf_saturation();
    test_determinism();
    test_unknown_term_zero();
    test_score_size_mismatch_throws();
    test_add_after_finalize_throws();
    test_bm25f_aux_weight_zero_recovers_plain_bm25();
    test_bm25f_aux_field_adds_signal();
    test_add_doc_without_aux_keeps_bm25f_safe();
    test_mixed_aux_presence_stays_aligned();
    test_bm25f_accepts_distinct_aux_query_text();
    return 0;
}
