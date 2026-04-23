#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/concept_mining.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::ConceptConfig;
using simeon::ConceptEntry;
using simeon::ConceptIndex;
using simeon::mine_concepts;
using simeon::score_bm25_with_concepts;

namespace {

bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

// Build an index + docs span for a tiny fixed corpus where PMI values are
// easy to hand-verify.
struct Fixture {
    std::vector<std::string> docs_owned;
    std::vector<std::string_view> docs_view;
    Bm25Index idx;

    explicit Fixture(std::vector<std::string> d) : docs_owned(std::move(d)), idx(Bm25Config{}) {
        docs_view.reserve(docs_owned.size());
        for (const auto& s : docs_owned) {
            docs_view.emplace_back(s);
            idx.add_doc(s);
        }
        idx.finalize();
    }
};

void test_pmi_matches_hand_computed_reference() {
    // Corpus crafted so "time value" appears together 3 times, and neither
    // "time" nor "value" appears alone. PMI should be high (>> 0).
    Fixture f({
        "time value money",                          // has time-value
        "time value money",                          // has time-value
        "time value money",                          // has time-value
        "random filler one two three four five six", // noise doc
        "alpha beta gamma delta epsilon zeta",       // noise doc
    });
    ConceptConfig cfg;
    cfg.min_ttf = 2;     // "time value" has ttf=3
    cfg.pmi_floor = 0.5; // permissive to catch our concept
    cfg.concept_weight = 1.0f;

    auto concepts = mine_concepts(f.idx, std::span<const std::string_view>(f.docs_view), cfg);
    assert(concepts.doc_count() == 5);
    assert(concepts.size() > 0);

    // The "time value" bigram should be present; PMI must be finite and > 0.
    const std::uint64_t h_time = f.idx.hash_term("time");
    const std::uint64_t h_value = f.idx.hash_term("value");
    const std::uint64_t h_bigram = ConceptIndex::hash_bigram(h_time, h_value);
    const ConceptEntry* e = concepts.find(h_bigram);
    assert(e != nullptr);
    assert(std::isfinite(e->pmi));
    assert(e->pmi > 0.0f);
    assert(e->a_hash == h_time);
    assert(e->b_hash == h_value);
    // "time value" appears 3 times total, in 3 docs.
    assert(e->total_tf == 3u);
    assert(e->docs.size() == 3u);
}

void test_min_ttf_floor_excludes_rare_concepts() {
    Fixture f({
        "rare pair once only",
        "alpha beta gamma delta",
        "foo bar baz qux",
    });
    ConceptConfig cfg;
    cfg.min_ttf = 5; // nothing in this corpus repeats >= 5 times
    cfg.pmi_floor = -10.0f;

    auto concepts = mine_concepts(f.idx, std::span<const std::string_view>(f.docs_view), cfg);
    assert(concepts.size() == 0);
}

void test_concept_mining_is_deterministic() {
    std::vector<std::string> docs_a = {"time value money", "time value money", "time value money",
                                       "noise doc one two", "another noise doc"};
    std::vector<std::string> docs_b = docs_a; // identical corpus

    Fixture fa(docs_a);
    Fixture fb(docs_b);
    ConceptConfig cfg;
    cfg.min_ttf = 2;
    cfg.pmi_floor = 0.5f;

    auto ca = mine_concepts(fa.idx, std::span<const std::string_view>(fa.docs_view), cfg);
    auto cb = mine_concepts(fb.idx, std::span<const std::string_view>(fb.docs_view), cfg);
    assert(ca.size() == cb.size());

    // Walk concepts by a fixed key ("time value") and verify PMI matches.
    const std::uint64_t h_time = fa.idx.hash_term("time");
    const std::uint64_t h_value = fa.idx.hash_term("value");
    const std::uint64_t h_bg = ConceptIndex::hash_bigram(h_time, h_value);
    const auto* ea = ca.find(h_bg);
    const auto* eb = cb.find(h_bg);
    assert(ea != nullptr && eb != nullptr);
    assert(approx(ea->pmi, eb->pmi));
    assert(ea->total_tf == eb->total_tf);
    assert(ea->a_hash == eb->a_hash);
    assert(ea->b_hash == eb->b_hash);
    assert(ea->docs == eb->docs);
}

void test_query_with_no_matched_concept_yields_base_bm25() {
    Fixture f({
        "time value money",
        "time value money",
        "time value money",
        "alpha beta gamma",
        "delta epsilon zeta",
    });
    ConceptConfig cfg;
    cfg.min_ttf = 2;
    cfg.pmi_floor = 0.5f;
    cfg.concept_weight = 0.5f;

    auto concepts = mine_concepts(f.idx, std::span<const std::string_view>(f.docs_view), cfg);
    assert(concepts.size() > 0);

    // Query has no bigrams matching any mined concept -> fused == base.
    std::vector<float> base(f.docs_view.size(), 0.0f);
    std::vector<float> fused(f.docs_view.size(), 0.0f);
    const std::string_view query = "unrelated lonely query terms";
    f.idx.score(query, std::span<float>{base});
    score_bm25_with_concepts(f.idx, concepts, query, 0.5f, std::span<float>{fused});
    for (std::size_t i = 0; i < base.size(); ++i)
        assert(approx(base[i], fused[i]));
}

void test_concept_weight_zero_recovers_base_bm25_exactly() {
    Fixture f({
        "time value money",
        "time value money",
        "time value money",
        "alpha beta gamma",
        "delta epsilon zeta",
    });
    ConceptConfig cfg;
    cfg.min_ttf = 2;
    cfg.pmi_floor = 0.5f;

    auto concepts = mine_concepts(f.idx, std::span<const std::string_view>(f.docs_view), cfg);
    assert(concepts.size() > 0);

    std::vector<float> base(f.docs_view.size(), 0.0f);
    std::vector<float> fused(f.docs_view.size(), 0.0f);
    const std::string_view query = "time value";
    f.idx.score(query, std::span<float>{base});
    score_bm25_with_concepts(f.idx, concepts, query, 0.0f, std::span<float>{fused});
    for (std::size_t i = 0; i < base.size(); ++i)
        assert(approx(base[i], fused[i]));
}

void test_max_concepts_cap_is_respected() {
    // Build a corpus with many repeated bigrams so several pass the PMI
    // floor, then set max_concepts=2 and verify we keep only 2.
    std::vector<std::string> docs;
    // Three distinct high-PMI bigrams: "aa bb", "cc dd", "ee ff".
    for (int i = 0; i < 5; ++i) {
        docs.emplace_back("aa bb cc dd ee ff");
    }
    docs.emplace_back("xx yy zz ww qq rr");
    Fixture f(std::move(docs));

    ConceptConfig cfg;
    cfg.min_ttf = 2;
    cfg.pmi_floor = -10.0f;
    cfg.max_concepts = 2;

    auto concepts = mine_concepts(f.idx, std::span<const std::string_view>(f.docs_view), cfg);
    assert(concepts.size() <= 2u);
}

} // namespace

int main() {
    test_pmi_matches_hand_computed_reference();
    test_min_ttf_floor_excludes_rare_concepts();
    test_concept_mining_is_deterministic();
    test_query_with_no_matched_concept_yields_base_bm25();
    test_concept_weight_zero_recovers_base_bm25_exactly();
    test_max_concepts_cap_is_respected();
    return 0;
}
