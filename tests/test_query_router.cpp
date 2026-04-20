#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/query_router.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::QueryFeatures;
using simeon::QueryRouter;
using simeon::Recipe;
using simeon::RouterConfig;

namespace {

const std::vector<std::string>& corpus() {
    static const std::vector<std::string> c = {
        "the quick brown fox jumps over the lazy dog",
        "infection of the wound caused fever and chills",
        "an infected cell undergoes apoptosis quickly",
        "the cat sat on the mat",
        "vaccines prevent the spread of common diseases",
        "the dog ran after the squirrel",
        "infections caused by bacteria respond to antibiotics",
        "the human immune system fights off pathogens daily",
    };
    return c;
}

Bm25Index build_idx() {
    Bm25Index idx{Bm25Config{}};
    for (const auto& d : corpus()) idx.add_doc(d);
    idx.finalize();
    return idx;
}

void test_features_basic() {
    auto idx = build_idx();
    QueryRouter r(idx);
    auto f = r.features("the dog");
    assert(f.n_terms == 2);
    assert(f.oov_rate == 0.0f);
    assert(f.avg_idf > 0.0f);
    assert(f.max_idf >= f.avg_idf);
    assert(f.avg_term_chars > 0.0f);
}

void test_features_oov() {
    auto idx = build_idx();
    QueryRouter r(idx);
    // "blockchain" never appears in the corpus.
    auto f = r.features("blockchain dog");
    assert(f.n_terms == 2);
    assert(f.oov_rate == 0.5f);
}

void test_routes_oov_to_sab() {
    auto idx = build_idx();
    QueryRouter r(idx);
    // Pure-OOV query: oov_rate = 1.0 > default threshold (0).
    assert(r.choose("blockchain quantum") == Recipe::Bm25SabSmooth);
}

void test_routes_high_idf_to_atire() {
    auto idx = build_idx();
    RouterConfig cfg;
    cfg.high_idf_threshold = 1.5f;  // tuned for the toy corpus
    QueryRouter r(idx, cfg);
    // "apoptosis" appears in 1/8 docs → high IDF; single-term, no OOV.
    auto f = r.features("apoptosis");
    assert(f.oov_rate == 0.0f);
    assert(f.avg_idf > 1.5f);
    assert(r.choose("apoptosis") == Recipe::Bm25Atire);
}

void test_routes_multi_term_low_idf_to_cascade() {
    auto idx = build_idx();
    RouterConfig cfg;
    cfg.cascade_min_terms = 4;
    cfg.cascade_max_idf = 5.0f;
    QueryRouter r(idx, cfg);
    // Common terms ("the", "cat", "dog", "ran"): low IDF, 4+ terms, no OOV.
    auto f = r.features("the cat dog ran");
    assert(f.oov_rate == 0.0f);
    assert(f.n_terms == 4);
    assert(r.choose("the cat dog ran") == Recipe::CascadeLinearAlpha);
}

void test_default_falls_back_to_sab() {
    auto idx = build_idx();
    RouterConfig cfg;
    cfg.high_idf_threshold = 100.0f;  // never hits Atire
    cfg.cascade_min_terms = 100;      // never hits cascade
    QueryRouter r(idx, cfg);
    assert(r.choose("the dog") == Recipe::Bm25SabSmooth);
}

void test_empty_query_is_safe() {
    auto idx = build_idx();
    QueryRouter r(idx);
    auto f = r.features("");
    assert(f.n_terms == 0);
    assert(r.choose("") == Recipe::Bm25SabSmooth);
}

void test_choose_is_deterministic() {
    auto idx = build_idx();
    QueryRouter r1(idx);
    QueryRouter r2(idx);
    for (const char* q : {"the dog", "blockchain", "apoptosis", "the cat dog ran"}) {
        assert(r1.choose(q) == r2.choose(q));
    }
}

void test_features_min_idf_and_stddev() {
    auto idx = build_idx();
    QueryRouter r(idx);
    // "the" appears in many docs (low IDF); "apoptosis" in 1/8 (high IDF).
    auto f = r.features("the apoptosis");
    assert(f.n_terms == 2);
    assert(f.oov_rate == 0.0f);
    // min_idf must be the IDF of "the", strictly less than max_idf.
    assert(f.min_idf > 0.0f);
    assert(f.min_idf < f.max_idf);
    // Population stddev with two distinct values is half their distance.
    const float expected = (f.max_idf - f.min_idf) / 2.0f;
    assert(std::fabs(f.idf_stddev - expected) < 1e-4f);
}

void test_features_min_idf_zero_when_all_oov() {
    auto idx = build_idx();
    QueryRouter r(idx);
    auto f = r.features("blockchain quantum");
    assert(f.n_terms == 2);
    assert(f.oov_rate == 1.0f);
    assert(f.min_idf == 0.0f);
    assert(f.idf_stddev == 0.0f);
}

void test_features_stddev_zero_for_single_present_term() {
    auto idx = build_idx();
    QueryRouter r(idx);
    auto f = r.features("apoptosis");
    assert(f.n_terms == 1);
    assert(f.idf_stddev == 0.0f);
    assert(f.min_idf == f.max_idf);
}

void test_atire_min_terms_blocks_short_high_idf() {
    auto idx = build_idx();
    RouterConfig cfg;
    cfg.high_idf_threshold = 1.5f;
    cfg.atire_min_terms = 5u;  // single-term query no longer eligible for Atire
    QueryRouter r(idx, cfg);
    // Without the new gate, "apoptosis" routes to Atire (test_routes_high_idf_to_atire).
    // With atire_min_terms=5, it must drop to the SAB default.
    assert(r.choose("apoptosis") == Recipe::Bm25SabSmooth);
}

void test_atire_min_idf_floor_blocks_uneven_query() {
    auto idx = build_idx();
    QueryRouter probe(idx, RouterConfig{});
    // Pick the threshold dynamically from the toy corpus so the test is
    // independent of the BM25 IDF scale.
    auto f = probe.features("the apoptosis");
    assert(f.min_idf < f.max_idf);
    RouterConfig cfg;
    cfg.high_idf_threshold = (f.avg_idf - 0.01f);  // avg passes
    cfg.atire_min_idf_floor = (f.max_idf);          // min < floor blocks
    QueryRouter r(idx, cfg);
    // Sanity: without the floor, the same query would route to Atire.
    RouterConfig open_cfg = cfg;
    open_cfg.atire_min_idf_floor = 0.0f;
    QueryRouter r_open(idx, open_cfg);
    assert(r_open.choose("the apoptosis") == Recipe::Bm25Atire);
    // With the floor in place, it must drop off the Atire route.
    assert(r.choose("the apoptosis") != Recipe::Bm25Atire);
}

void test_default_router_unchanged_by_step1f_gates() {
    auto idx = build_idx();
    RouterConfig cfg;  // default: atire_min_terms=0, atire_min_idf_floor=0
    cfg.high_idf_threshold = 1.5f;
    QueryRouter r(idx, cfg);
    // With default zero gates, Step 1f is a no-op: "apoptosis" still routes
    // to Atire as in the pre-Step-1f behavior.
    assert(r.choose("apoptosis") == Recipe::Bm25Atire);
}

void test_recipe_name_round_trip() {
    using simeon::recipe_name;
    assert(std::string(recipe_name(Recipe::Bm25Atire)) == "Bm25Atire");
    assert(std::string(recipe_name(Recipe::Bm25SabSmooth)) == "Bm25SabSmooth");
    assert(std::string(recipe_name(Recipe::CascadeLinearAlpha)) == "CascadeLinearAlpha");
}

}  // namespace

int main() {
    test_features_basic();
    test_features_oov();
    test_routes_oov_to_sab();
    test_routes_high_idf_to_atire();
    test_routes_multi_term_low_idf_to_cascade();
    test_default_falls_back_to_sab();
    test_empty_query_is_safe();
    test_choose_is_deterministic();
    test_features_min_idf_and_stddev();
    test_features_min_idf_zero_when_all_oov();
    test_features_stddev_zero_for_single_present_term();
    test_atire_min_terms_blocks_short_high_idf();
    test_atire_min_idf_floor_blocks_uneven_query();
    test_default_router_unchanged_by_step1f_gates();
    test_recipe_name_round_trip();
    return 0;
}
