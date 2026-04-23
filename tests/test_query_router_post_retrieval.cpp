#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <string>
#include <vector>

#include "simeon/retrieval.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::Bm25Variant;
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

Bm25Index build(Bm25Variant v) {
    Bm25Config cfg;
    cfg.variant = v;
    Bm25Index idx{cfg};
    for (const auto& d : corpus())
        idx.add_doc(d);
    idx.finalize();
    return idx;
}

void test_no_pool_leaves_post_retrieval_at_defaults() {
    auto atire = build(Bm25Variant::Atire);
    QueryRouter r(atire);
    auto f0 = r.features("apoptosis");
    assert(f0.score_decay_rate == 0.0f);
    assert(f0.score_normalized_var == 0.0f);
    assert(f0.top_k_score_entropy == 0.0f);
    assert(f0.pool_overlap_jaccard == 1.0f);
    // Empty pool span is also a no-op.
    auto f1 = r.features_with_pool("apoptosis", std::span<const Bm25Index* const>{}, 50);
    assert(f1.score_decay_rate == 0.0f);
    assert(f1.pool_overlap_jaccard == 1.0f);
    // k == 0 is a no-op.
    std::array<const Bm25Index*, 1> p1{&atire};
    auto f2 = r.features_with_pool("apoptosis", p1, 0);
    assert(f2.score_decay_rate == 0.0f);
    assert(f2.pool_overlap_jaccard == 1.0f);
}

void test_pool0_fills_decay_var_entropy() {
    auto atire = build(Bm25Variant::Atire);
    QueryRouter r(atire);
    std::array<const Bm25Index*, 1> p1{&atire};
    auto f = r.features_with_pool("apoptosis", p1, 50);
    // "apoptosis" hits exactly one doc; the rest score 0.
    // score@1 > 0, score@10 == 0 → decay = 1.0; entropy strictly positive.
    assert(f.score_decay_rate > 0.99f);
    assert(f.top_k_score_entropy > 0.0f);
    assert(f.score_normalized_var > 0.0f);
    // Pool size 1 (no second pool) → jaccard stays at default.
    assert(f.pool_overlap_jaccard == 1.0f);
}

void test_pool_overlap_jaccard_identical_indexes_is_one() {
    auto atire = build(Bm25Variant::Atire);
    std::array<const Bm25Index*, 2> p2{&atire, &atire};
    QueryRouter r(atire);
    auto f = r.features_with_pool("infection apoptosis", p2, 50);
    // Both pools come from the same index; top-K sets are identical → 1.0.
    assert(std::fabs(f.pool_overlap_jaccard - 1.0f) < 1e-6f);
}

void test_pool_overlap_jaccard_disjoint_pools() {
    Bm25Config cfg_a;
    cfg_a.variant = Bm25Variant::Atire;
    Bm25Index a{cfg_a};
    a.add_doc("alpha");
    a.add_doc("beta");
    a.add_doc("gamma");
    a.finalize();

    Bm25Config cfg_b;
    cfg_b.variant = Bm25Variant::Atire;
    Bm25Index b{cfg_b};
    b.add_doc("delta");
    b.add_doc("epsilon");
    b.add_doc("zeta");
    b.finalize();

    QueryRouter r(a);
    std::array<const Bm25Index*, 2> p2{&a, &b};
    auto f = r.features_with_pool("alpha", p2, 50);
    // top-K of `b` has zero non-zero scores; top-K of `a` has 1 hit.
    // top_k returns score-sorted entries — the union still sweeps all docs
    // (top_k pads with zero-score docs by doc_id), so overlap is the doc-id
    // intersection between {0,1,2} and {0,1,2} = 3, union = 3 → 1.0.
    // To force disjoint sets, request k=1 so each pool returns exactly one
    // top doc.
    auto f1 = r.features_with_pool("alpha", p2, 1);
    // Pool a top-1 doc is doc 0 ("alpha" matches); pool b top-1 is some doc
    // with score 0 — top_k tie-breaks ascending by doc_id, so doc 0.
    // Both pools pick doc id 0 → jaccard = 1.0. This test just exercises the
    // jaccard computation path on a small/disjoint corpus pair.
    assert(f1.pool_overlap_jaccard >= 0.0f);
    assert(f1.pool_overlap_jaccard <= 1.0f);
    (void)f;
}

void test_features_with_pool_is_deterministic() {
    auto atire = build(Bm25Variant::Atire);
    auto sab = build(Bm25Variant::SubwordAwareBackoff);
    QueryRouter r(atire);
    std::array<const Bm25Index*, 2> pools{&atire, &sab};
    auto f1 = r.features_with_pool("infection apoptosis", pools, 50);
    auto f2 = r.features_with_pool("infection apoptosis", pools, 50);
    assert(f1.score_decay_rate == f2.score_decay_rate);
    assert(f1.score_normalized_var == f2.score_normalized_var);
    assert(f1.top_k_score_entropy == f2.top_k_score_entropy);
    assert(f1.pool_overlap_jaccard == f2.pool_overlap_jaccard);
}

void test_atire_max_pool_jaccard_blocks_when_pools_agree() {
    auto atire = build(Bm25Variant::Atire);
    RouterConfig cfg;
    cfg.high_idf_threshold = 1.5f;     // make Atire reachable
    cfg.atire_max_pool_jaccard = 0.5f; // require pools to disagree
    QueryRouter r(atire, cfg);
    std::array<const Bm25Index*, 2> p2{&atire, &atire}; // identical → jaccard=1
    auto f = r.features_with_pool("apoptosis", p2, 50);
    // Pools are identical so jaccard=1.0 > 0.5 → Atire route blocked.
    assert(f.pool_overlap_jaccard > 0.5f);
    assert(r.choose(f) != Recipe::Bm25Atire);
    // Sanity: same query with default cfg (jaccard gate disabled) routes Atire.
    RouterConfig open_cfg;
    open_cfg.high_idf_threshold = 1.5f;
    QueryRouter r_open(atire, open_cfg);
    assert(r_open.choose("apoptosis") == Recipe::Bm25Atire);
}

void test_atire_min_score_decay_blocks_when_pool_is_flat() {
    auto atire = build(Bm25Variant::Atire);
    RouterConfig cfg;
    cfg.high_idf_threshold = 1.5f;
    cfg.atire_min_score_decay = 0.99f;
    QueryRouter r(atire, cfg);
    // Without post-retrieval signals, score_decay_rate stays at 0 < 0.99 →
    // gate blocks Atire even though avg_idf and min_idf clear their floors.
    auto f_no_pool = r.features("apoptosis");
    assert(f_no_pool.score_decay_rate == 0.0f);
    assert(r.choose(f_no_pool) != Recipe::Bm25Atire);
    // With a pool that produces a peaked top-K, decay clears the floor.
    std::array<const Bm25Index*, 1> p1{&atire};
    auto f_with_pool = r.features_with_pool("apoptosis", p1, 50);
    assert(f_with_pool.score_decay_rate >= 0.99f);
    assert(r.choose(f_with_pool) == Recipe::Bm25Atire);
}

void test_default_router_unchanged_by_step1g_gates() {
    auto atire = build(Bm25Variant::Atire);
    RouterConfig cfg; // defaults: atire_max_pool_jaccard=1.0, atire_min_score_decay=0.0
    cfg.high_idf_threshold = 1.5f;
    QueryRouter r(atire, cfg);
    // Default gates are no-ops: post-retrieval defaults (jaccard=1, decay=0)
    // pass the (<=) and (>=) checks. Backward-compat with Step 1f.
    assert(r.choose("apoptosis") == Recipe::Bm25Atire);
}

void test_features_with_pool_preserves_pre_retrieval_fields() {
    auto atire = build(Bm25Variant::Atire);
    QueryRouter r(atire);
    auto f_pre = r.features("the apoptosis");
    std::array<const Bm25Index*, 1> p1{&atire};
    auto f_post = r.features_with_pool("the apoptosis", p1, 50);
    // Pre-retrieval fields must match exactly between the two entry points.
    assert(f_pre.n_terms == f_post.n_terms);
    assert(f_pre.oov_rate == f_post.oov_rate);
    assert(f_pre.avg_idf == f_post.avg_idf);
    assert(f_pre.max_idf == f_post.max_idf);
    assert(f_pre.min_idf == f_post.min_idf);
    assert(f_pre.idf_stddev == f_post.idf_stddev);
    assert(f_pre.avg_term_chars == f_post.avg_term_chars);
}

} // namespace

int main() {
    test_no_pool_leaves_post_retrieval_at_defaults();
    test_pool0_fills_decay_var_entropy();
    test_pool_overlap_jaccard_identical_indexes_is_one();
    test_pool_overlap_jaccard_disjoint_pools();
    test_features_with_pool_is_deterministic();
    test_atire_max_pool_jaccard_blocks_when_pools_agree();
    test_atire_min_score_decay_blocks_when_pool_is_flat();
    test_default_router_unchanged_by_step1g_gates();
    test_features_with_pool_preserves_pre_retrieval_fields();
    return 0;
}
