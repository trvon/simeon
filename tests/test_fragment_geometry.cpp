// Correctness test for score_fragment_geometry — the rerank had no ranking test
// (only PHSS-scalar / timing coverage), which let two real defects ship:
//   D1: consumers buried the reranked pool by mixing z-blend and raw-BM25 scales.
//   D2: per-query whitening was suspected to invert the similarity signal.
// This test locks the invariants a correct geometry rerank must hold:
//   - out-of-pool docs are -inf (the documented contract);
//   - a planted query-matching doc is in-pool (finite);
//   - geometry does NOT bury that doc — it ranks #1 among in-pool docs
//     (catches signal inversion / scale collapse).
// Uses explicit checks (not assert) so it is meaningful under NDEBUG/release.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/pmi.hpp"
#include "simeon/simeon.hpp"

namespace {

int g_failures = 0;
void expect(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_failures;
    }
}

// doc[0] is the planted relevant doc (dominates the query terms); docs [1..2]
// partially overlap (in pool, must rank below the target); the rest are
// query-disjoint fillers (BM25 0 -> out of pool -> -inf).
const std::vector<std::string>& corpus() {
    static const std::vector<std::string> c = {
        "neural network training gradient descent backpropagation neural network "
        "training gradient descent neural network training gradient",
        "neural network architecture layer design and tuning",
        "gradient boosting decision trees ensemble method",
        "the quick brown fox jumps over the lazy dog",
        "weather forecast sunny with a chance of rain tomorrow",
        "sourdough bread recipe flour water salt and yeast",
        "mountain hiking trail map elevation and switchbacks",
        "ocean tides moon gravity coastal currents and waves",
        "jazz saxophone improvisation scales and chord changes",
        "garden vegetables tomato basil compost and watering",
        "bicycle drivetrain chain cassette and derailleur tuning",
        "volcano magma eruption lava ash and tectonic plates",
        "library catalog index reference shelf and circulation",
        "telescope lens aperture focal length and star fields",
    };
    return c;
}

constexpr const char* kQuery = "neural network training gradient descent";
constexpr std::size_t kTarget = 0;

std::vector<float> run_geom(bool whiten) {
    const auto& docs = corpus();
    simeon::Bm25Config bcfg;
    bcfg.variant = simeon::Bm25Variant::SubwordAwareBackoff;
    bcfg.subword_gamma = 5.0f;
    simeon::Bm25Index idx(bcfg);
    for (const auto& d : docs)
        idx.add_doc(d);
    idx.finalize();

    std::vector<std::string_view> views;
    views.reserve(docs.size());
    for (const auto& d : docs)
        views.emplace_back(d);
    simeon::PmiConfig pcfg;
    pcfg.target_rank = 64;
    pcfg.min_token_count = 1;
    pcfg.max_vocab_size = 5000;
    auto pmi = simeon::PmiEmbeddings::learn(std::span<const std::string_view>(views), pcfg);

    simeon::EncoderConfig ecfg;
    ecfg.ngram_mode = simeon::NGramMode::WordOnly;
    ecfg.ngram_min = 1;
    ecfg.ngram_max = 1;
    ecfg.sketch_dim = 0;
    ecfg.output_dim = pmi.dim();
    ecfg.projection = simeon::ProjectionMode::None;
    ecfg.l2_normalize = true;
    ecfg.pmi_rows = &pmi;
    simeon::Encoder enc(ecfg);

    std::vector<std::vector<simeon::SemanticFragment>> frags;
    frags.reserve(docs.size());
    for (const auto& d : docs)
        frags.push_back(simeon::build_doc_semantic_fragments(enc, d, idx, 6, 8));

    simeon::FragmentGeometryConfig cfg;
    cfg.pool_size = 5; // force an out-of-pool tail so the -inf contract is exercised
    cfg.use_phss = true;
    cfg.outer_maxsim = true;
    cfg.doc_scorer_kind = simeon::FragmentGeometryConfig::DocScorerKind::MaxSim;
    cfg.whiten = whiten;
    return simeon::score_fragment_geometry(kQuery, idx, enc, frags, cfg);
}

void check(bool whiten) {
    const auto scores = run_geom(whiten);
    const std::size_t nd = corpus().size();
    expect(scores.size() == nd, "score vector size == doc count");
    if (scores.size() != nd)
        return;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::size_t n_inf = 0;
    for (float s : scores)
        if (s == neg_inf)
            ++n_inf;
    expect(n_inf > 0, "query-disjoint fillers fall out of pool (-inf)");
    expect(std::isfinite(scores[kTarget]), "planted target is in-pool (finite)");

    float best = neg_inf;
    std::size_t best_doc = nd;
    for (std::size_t i = 0; i < nd; ++i) {
        if (std::isfinite(scores[i]) && scores[i] > best) {
            best = scores[i];
            best_doc = i;
        }
    }
    expect(best_doc == kTarget, "geometry ranks the planted target #1 among in-pool docs");

    std::printf("  [whiten=%d] target=%.4f best_doc=%zu in_pool=%zu out_of_pool=%zu\n",
                whiten ? 1 : 0, static_cast<double>(scores[kTarget]), best_doc, nd - n_inf, n_inf);
}

} // namespace

int main() {
    std::printf("test_fragment_geometry: planted-relevant doc ranks #1 after rerank\n");
    check(/*whiten=*/true);
    check(/*whiten=*/false);
    if (g_failures > 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
