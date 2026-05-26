#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "simeon/bm25.hpp"

namespace simeon {

// Router config. Defaults are scifact-tuned; override per corpus by sweeping
// on a held-out fold (see docs/research.md).
struct RouterConfig {
    // (>) → Bm25SabSmooth. Even one OOV term in the query justifies the
    // SAB n-gram backoff path.
    float oov_threshold = 0.0f;
    // (>) → Bm25Atire. Rare-term-heavy queries are at the BM25 ceiling;
    // SAB's n-gram blend adds noise without lift. Default 3.0 is the
    // empirically-tuned scifact value (Pass A grid; see docs/router_design.md).
    float high_idf_threshold = 3.0f;
    // (>=) AND-gate on the Atire route alongside high_idf_threshold. Default
    // 0 keeps Step 1e behavior; raise to discourage Atire on short queries
    // where avg_idf is misleading. See docs/router_design.md "Step 1f".
    std::uint32_t atire_min_terms = 0u;
    // (>=) AND-gate on the Atire route. Default 0 keeps Step 1e behavior;
    // raise to require every term to clear an IDF floor (filters Atire-route
    // queries whose avg is dragged up by a single rare term).
    float atire_min_idf_floor = 0.0f;
    // (<=) Step 1g.1 post-retrieval AND-gate on the Atire route. Route only
    // when the BM25-Atire and BM25-SAB top-K pools disagree enough to make
    // the routing decision matter. Default 1.0 = pass (no constraint); only
    // consulted when QueryFeatures::pool_overlap_jaccard was filled by
    // features_with_pool().
    float atire_max_pool_jaccard = 1.0f;
    // (>=) Step 1g.1 post-retrieval AND-gate on the Atire route. Route only
    // when the BM25 top is sufficiently peaked (Atire's rare-term advantage
    // shows up as a sharp drop-off). Default 0.0 = pass; only consulted when
    // QueryFeatures::score_decay_rate was filled by features_with_pool().
    float atire_min_score_decay = 0.0f;
    // (>=) Step 1k B2 AND-gate on the Atire route. Route only when the sum-SCQ
    // (Zhao 2008) exceeds the floor — high SCQ indicates the query terms are
    // well-supported in the collection, favoring Atire's exact-term scoring.
    // Default 0.0 = pass (no constraint).
    float atire_min_scq = 0.0f;
    // (<=) Step 1k B2 AND-gate on the Atire route. Route only when simplified
    // clarity (Cronen-Townsend 2002) is low — low clarity = query terms close
    // to collection background, which Atire handles well without expansion.
    // Default infinity = pass.
    float atire_max_clarity = std::numeric_limits<float>::infinity();
    // (>=) → CascadeLinearAlpha gate (combined with cascade_max_idf).
    std::uint32_t cascade_min_terms = 4;
    // (<=) → CascadeLinearAlpha gate. Multi-term low-IDF queries benefit
    // most from simeon's semantic signal alongside BM25.
    float cascade_max_idf = 5.0f;

    // Quality-router extension. Short queries stayed BM25-best on nfcorpus,
    // while longer semantic queries benefited from fragment geometry on
    // scifact / fiqa. Defaults come from the three-fixture recall/precision
    // sweep: keep short lexical queries on BM25, use richcov+approx for
    // medium semantic queries, and richcov+approx+max for the longest ones.
    std::uint32_t quality_geometry_min_terms = 6u;
    std::uint32_t quality_max_min_terms = 12u;
    float quality_max_max_idf = 3.0f;
};

// Recipe selected by the router. Caller maps the recipe to a concrete
// scoring/cascade pipeline; the router itself does not run retrieval.
enum class Recipe : std::uint8_t {
    Bm25Atire,
    Bm25SabSmooth,
    CascadeLinearAlpha,
};

// Quality-oriented recipe selected by the router. Caller maps the recipe to a
// concrete retrieval pipeline. The defaults are deliberately minimal:
//  - Bm25Only: lexical ceiling / short-query regime
//  - FragmentRichCovPhssApprox: medium-length semantic queries
//  - FragmentRichCovPhssApproxMax: long semantic queries
enum class QualityRecipe : std::uint8_t {
    Bm25Only,
    FragmentRichCovPhssApprox,
    FragmentRichCovPhssApproxMax,
};

// Pre-retrieval features (Carmel & Yom-Tov 2010 family). Computed in
// O(n_query_terms) given a finalized Bm25Index.
struct QueryFeatures {
    float oov_rate = 0.0f;
    float avg_idf = 0.0f;
    float max_idf = 0.0f;
    // Smallest IDF over present (non-OOV) terms, 0 when every term is OOV.
    // Cheap discriminator that subdivides the Atire/SAB region where avg_idf
    // overlaps (Step 1f).
    float min_idf = 0.0f;
    // Population stddev of IDF over present terms, 0 when n_present <= 1.
    // High stddev = mixed-informativeness query (a rare term dragging the
    // average up). Step 1f predictor.
    float idf_stddev = 0.0f;
    std::uint32_t n_terms = 0;
    float avg_term_chars = 0.0f;

    // Step 1g.1 post-retrieval-lite predictors. Filled only by
    // features_with_pool(); features() leaves them at the no-signal defaults
    // below so that RouterConfig's Atire AND-gates default to no-ops.
    //
    // (score@1 - score@10) / score@1 over pools[0]. Higher = top is sharply
    // peaked (favors Atire's rare-term scoring). 0 when pool empty or score@1<=0.
    float score_decay_rate = 0.0f;
    // Population variance / mean of the top-K pools[0] BM25 scores. Proxy
    // for clarity (Cronen-Townsend & Croft 2002). 0 when mean <= 0.
    float score_normalized_var = 0.0f;
    // Shannon entropy (nats) of softmax(top-K pools[0] scores). Lower =
    // more peaked. 0 when pool empty.
    float top_k_score_entropy = 0.0f;
    // Jaccard of top-K doc-id sets across pools[0] vs pools[1] (Atire vs SAB
    // by convention). 1.0 = pools fully agree (routing doesn't matter); lower
    // = pools disagree (router should pick carefully). Default 1.0 keeps the
    // Atire pool-Jaccard gate a no-op when post-retrieval signals are absent.
    float pool_overlap_jaccard = 1.0f;

    // T1 — Normalized Query Commitment (Shtok, Kurland, Carmel 2012, TOIS):
    //   NQC = σ(S_top_K) / μ(S_corpus)
    // where σ is population stddev of top-K pool-0 BM25 scores and μ is the
    // mean over all corpus scores. Commitment of the top-K as a fraction of
    // the collection baseline; higher = more committed / easier query. 0 when
    // pool empty or μ ≤ 0. Filled only by features_with_pool().
    float nqc = 0.0f;
    // T2 — Full Weighted Information Gain (Zhou & Croft 2007, SIGIR §3.2),
    // BM25-adapted form:
    //   WIG = (μ(S_top_K) - μ(S_corpus)) / sqrt(|Q|)
    // Per-term information gain of top-K over collection baseline, normalized
    // by query length. Distinct from score_decay_rate ("WIG-lite") which only
    // measures intra-top decay. Filled only by features_with_pool().
    float wig_full = 0.0f;

    // Step 1k B2 pre-retrieval predictors. Computed in features() from
    // collection statistics (total_tf, total_tokens) — no first-pass scoring
    // required, keeping them cheap.
    //
    // Sum-SCQ (Zhao, Scholer, Tsegay 2008): Σ_t (1 + log(tf_C(t))) · idf(t)
    // over present query terms. High SCQ = query terms well-supported in the
    // collection. 0 when every term is OOV.
    float scq_sum = 0.0f;
    // Simplified clarity score (Cronen-Townsend & Croft 2002 simplified form):
    //   Σ_t p(t|Q) · log(p(t|Q) / p(t|C))
    // where p(t|Q) = count_Q(t) / |Q|, p(t|C) = tf_C(t) / total_tokens.
    // High clarity = query is topically distinctive from the background LM.
    // 0 when every term is OOV or |Q|=0.
    float simplified_clarity = 0.0f;
};

// Selects a Recipe per query from cheap pre-retrieval predictors. Holds a
// const reference to a finalized Bm25Index for the per-term df / idf lookup;
// caller owns the index. Routing is deterministic given (config, idx, query).
class QueryRouter {
public:
    QueryRouter(const Bm25Index& idx, RouterConfig cfg = {}) noexcept;

    QueryFeatures features(std::string_view query) const;
    // Step 1g.1: features() + post-retrieval-lite predictors derived from
    // the BM25 top-K pools the cascade already computes. `pools` is a span
    // of finalized Bm25Index pointers; convention is pools[0] = Atire,
    // pools[1] = SAB so that pool_overlap_jaccard measures the Atire/SAB
    // routing-relevant disagreement. Empty span or k == 0 leaves the
    // post-retrieval fields at their no-signal defaults (see QueryFeatures).
    QueryFeatures features_with_pool(std::string_view query,
                                     std::span<const Bm25Index* const> pools,
                                     std::uint32_t k = 50) const;
    Recipe choose(std::string_view query) const;
    Recipe choose(const QueryFeatures& f) const noexcept;
    QualityRecipe choose_quality(std::string_view query) const;
    QualityRecipe choose_quality(const QueryFeatures& f) const noexcept;

    const RouterConfig& config() const noexcept { return cfg_; }

private:
    const Bm25Index& idx_;
    RouterConfig cfg_;
};

const char* recipe_name(Recipe r) noexcept;
const char* quality_recipe_name(QualityRecipe r) noexcept;

} // namespace simeon
