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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
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

template <class Function> void expect_invalid_argument(Function&& function, const char* msg) {
    try {
        function();
        expect(false, msg);
    } catch (const std::invalid_argument&) {
    } catch (...) {
        expect(false, msg);
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

void check_invalid_inputs_fail_closed() {
    simeon::SemanticFragment malformed;
    malformed.vec = {1.0f};
    float destination[2]{};
    expect_invalid_argument([&] { simeon::read_frag_vec(malformed, 2, destination); },
                            "short fragment vector is rejected");

    std::vector<std::vector<simeon::SemanticFragment>> malformed_docs(1);
    malformed_docs[0].push_back(malformed);
    expect_invalid_argument([&] { simeon::compress_fragments_to_bf16(malformed_docs, 2); },
                            "short fragment vector cannot be compressed");

    simeon::Bm25Index index;
    index.add_doc("small validation document");
    index.finalize();
    simeon::EncoderConfig encoder_config;
    encoder_config.sketch_dim = 16;
    encoder_config.output_dim = 8;
    encoder_config.projection = simeon::ProjectionMode::AchlioptasSparse;
    simeon::Encoder encoder(encoder_config);
    std::vector<std::vector<simeon::SemanticFragment>> valid_docs(1);
    valid_docs[0].resize(1);
    valid_docs[0][0].vec.resize(encoder.output_dim());
    encoder.encode("small validation document", valid_docs[0][0].vec.data());
    simeon::FragmentGeometryConfig geometry_config;
    geometry_config.alpha = std::numeric_limits<float>::quiet_NaN();
    expect_invalid_argument(
        [&] {
            (void)simeon::score_fragment_geometry("validation", index, encoder, valid_docs,
                                                  geometry_config);
        },
        "non-finite blend weight is rejected");
}

void check_precomputed_bm25_matches_bit_identically() {
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
    cfg.pool_size = 5;
    cfg.use_phss = true;
    cfg.outer_maxsim = true;
    cfg.doc_scorer_kind = simeon::FragmentGeometryConfig::DocScorerKind::MaxSim;

    std::vector<float> precomputed(docs.size(), 0.0f);
    idx.score(kQuery, precomputed);

    const auto baseline = simeon::score_fragment_geometry(kQuery, idx, enc, frags, cfg);
    const auto reused = simeon::score_fragment_geometry(kQuery, idx, enc, frags, cfg, precomputed);
    expect(baseline.size() == reused.size(), "precomputed BM25 score vector size matches");
    bool identical = (baseline.size() == reused.size());
    for (std::size_t i = 0; i < baseline.size() && identical; ++i) {
        std::uint32_t a_bits = 0, b_bits = 0;
        std::memcpy(&a_bits, &baseline[i], sizeof(a_bits));
        std::memcpy(&b_bits, &reused[i], sizeof(b_bits));
        if (a_bits != b_bits)
            identical = false;
    }
    expect(identical, "precomputed BM25 produces bit-identical fragment geometry scores");

    std::vector<float> bad_size(docs.size() - 1, 0.0f);
    expect_invalid_argument(
        [&] { (void)simeon::score_fragment_geometry(kQuery, idx, enc, frags, cfg, bad_size); },
        "mismatched precomputed BM25 length is rejected");
}

struct GeometryFixture {
    simeon::Bm25Index idx;
    std::unique_ptr<simeon::PmiEmbeddings> pmi;
    simeon::Encoder enc;
    std::vector<std::vector<simeon::SemanticFragment>> frags;

    static GeometryFixture build() {
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
        auto pmi = std::make_unique<simeon::PmiEmbeddings>(
            simeon::PmiEmbeddings::learn(std::span<const std::string_view>(views), pcfg));

        simeon::EncoderConfig ecfg;
        ecfg.ngram_mode = simeon::NGramMode::WordOnly;
        ecfg.ngram_min = 1;
        ecfg.ngram_max = 1;
        ecfg.sketch_dim = 0;
        ecfg.output_dim = pmi->dim();
        ecfg.projection = simeon::ProjectionMode::None;
        ecfg.l2_normalize = true;
        ecfg.pmi_rows = pmi.get();
        simeon::Encoder enc(ecfg);

        std::vector<std::vector<simeon::SemanticFragment>> frags;
        frags.reserve(docs.size());
        for (const auto& d : docs)
            frags.push_back(simeon::build_doc_semantic_fragments(enc, d, idx, 6, 8));

        return GeometryFixture{std::move(idx), std::move(pmi), std::move(enc), std::move(frags)};
    }
};

// Invariant 1: alpha = 1.0 gives exact BM25 pool parity (z-scored BM25 values and
// identical relative ranking) for both diffusion and outer_maxsim paths.
void check_alpha_one_bm25_parity() {
    auto fx = GeometryFixture::build();
    const std::size_t nd = corpus().size();
    std::vector<float> raw_bm25(nd, 0.0f);
    fx.idx.score(kQuery, raw_bm25);

    for (bool outer_maxsim : {false, true}) {
        simeon::FragmentGeometryConfig cfg;
        cfg.pool_size = 5;
        cfg.alpha = 1.0f;
        cfg.outer_maxsim = outer_maxsim;
        cfg.use_phss = true;

        const auto scores =
            simeon::score_fragment_geometry(kQuery, fx.idx, fx.enc, fx.frags, cfg, raw_bm25);
        expect(scores.size() == nd, "alpha=1.0 score size matches doc count");

        std::vector<std::size_t> in_pool;
        for (std::size_t i = 0; i < nd; ++i) {
            if (std::isfinite(scores[i]))
                in_pool.push_back(i);
        }
        expect(!in_pool.empty(), "alpha=1.0 has non-empty in-pool docs");

        double mean = 0.0;
        for (std::size_t id : in_pool)
            mean += raw_bm25[id];
        mean /= static_cast<double>(in_pool.size());
        double var = 0.0;
        for (std::size_t id : in_pool) {
            const double d = raw_bm25[id] - mean;
            var += d * d;
        }
        const float sd = static_cast<float>(std::sqrt(var / in_pool.size()) + 1e-12);

        for (std::size_t id : in_pool) {
            const float expected = static_cast<float>((raw_bm25[id] - mean) / sd);
            expect(std::fabs(scores[id] - expected) < 1e-5f,
                   "alpha=1.0 in-pool score equals z-scored BM25");
        }
        for (std::size_t a : in_pool) {
            for (std::size_t b : in_pool) {
                if (raw_bm25[a] > raw_bm25[b] + 1e-6f)
                    expect(scores[a] > scores[b], "alpha=1.0 preserves strict BM25 ordering");
            }
        }
    }
}

// Invariant 2: Diffusion mass conservation (sum(mass) == 1.0 within tolerance) on a
// connected fragment graph (fixed kNN with knn >= 1) across multiple step counts.
void check_diffusion_mass_conservation() {
    auto fx = GeometryFixture::build();
    for (std::uint32_t steps : {0u, 1u, 2u, 5u}) {
        simeon::FragmentGeometryConfig cfg;
        cfg.pool_size = 5;
        cfg.outer_maxsim = false;
        cfg.use_phss = false;
        cfg.knn = 3;
        cfg.steps = steps;

        simeon::FragmentGeometryProfile profile;
        const auto scores = simeon::score_fragment_geometry_profiled(kQuery, fx.idx, fx.enc,
                                                                     fx.frags, cfg, &profile);
        expect(!scores.empty(), "diffusion mass conservation produced scores");
        expect(profile.pool_fragments >= 2, "diffusion graph has multiple fragments");
        expect(std::fabs(profile.diffused_mass_sum - 1.0) < 1e-5,
               "diffusion conserves total probability mass (sum == 1)");
    }
}

// Invariant 3: outer_maxsim ranking monotonicity — with alpha=0, whiten=false, MaxSim,
// documents whose best fragment has strictly higher cosine with the query vector must
// rank strictly higher, and increasing a doc's best fragment cosine monotonically
// increases its score.
void check_outer_maxsim_monotonicity() {
    simeon::Bm25Index idx;
    for (int i = 0; i < 4; ++i)
        idx.add_doc("alpha beta gamma delta");
    idx.finalize();

    simeon::EncoderConfig ecfg;
    ecfg.sketch_dim = 32;
    ecfg.output_dim = 8;
    ecfg.projection = simeon::ProjectionMode::AchlioptasSparse;
    ecfg.l2_normalize = true;
    simeon::Encoder enc(ecfg);

    const std::uint32_t dim = enc.output_dim();
    std::vector<float> qvec(dim, 0.0f);
    enc.encode("alpha beta", qvec.data());
    double qnorm = 0.0;
    for (float x : qvec)
        qnorm += static_cast<double>(x) * x;
    expect(qnorm > 0.5, "monotonicity query vector is unit-norm");

    std::vector<float> u_perp(dim, 0.0f);
    u_perp[0] = 1.0f;
    double proj = static_cast<double>(qvec[0]);
    if (std::fabs(proj) > 0.9) {
        u_perp[0] = 0.0f;
        u_perp[1] = 1.0f;
        proj = static_cast<double>(qvec[1]);
    }
    double pnorm = 0.0;
    for (std::uint32_t d = 0; d < dim; ++d) {
        u_perp[d] -= static_cast<float>(proj) * qvec[d];
        pnorm += static_cast<double>(u_perp[d]) * u_perp[d];
    }
    const float inv_p = static_cast<float>(1.0 / std::sqrt(pnorm));
    for (float& x : u_perp)
        x *= inv_p;

    auto make_vec_with_cos = [&](float c) {
        const float s = std::sqrt(std::max(0.0f, 1.0f - c * c));
        std::vector<float> v(dim);
        for (std::uint32_t d = 0; d < dim; ++d)
            v[d] = c * qvec[d] + s * u_perp[d];
        return v;
    };

    const float target_cos[4] = {0.95f, 0.70f, 0.40f, 0.10f};
    std::vector<std::vector<simeon::SemanticFragment>> frags(4);
    for (int i = 0; i < 4; ++i) {
        frags[i].push_back(simeon::SemanticFragment{make_vec_with_cos(0.05f), {}, {}});
        frags[i].push_back(simeon::SemanticFragment{make_vec_with_cos(target_cos[i]), {}, {}});
    }

    simeon::FragmentGeometryConfig cfg;
    cfg.pool_size = 4;
    cfg.alpha = 0.0f;
    cfg.outer_maxsim = true;
    cfg.whiten = false;
    cfg.doc_scorer_kind = simeon::FragmentGeometryConfig::DocScorerKind::MaxSim;

    const auto scores = simeon::score_fragment_geometry("alpha beta", idx, enc, frags, cfg);
    expect(scores.size() == 4, "monotonicity score size == 4");
    for (int i = 0; i < 3; ++i) {
        expect(scores[i] > scores[i + 1], "outer_maxsim ranks strictly by max cosine");
    }

    frags[2][1].vec = make_vec_with_cos(0.85f);
    const auto boosted = simeon::score_fragment_geometry("alpha beta", idx, enc, frags, cfg);
    expect(boosted[0] > boosted[2] && boosted[2] > boosted[1] && boosted[1] > boosted[3],
           "boosting fragment cosine monotonically promotes outer_maxsim rank");
}

// Verify reusable thread-local scratch workspace has zero cross-query leakage and is
// thread-safe under concurrent calls.
void check_scratch_reuse_and_thread_safety() {
    auto fx = GeometryFixture::build();
    simeon::FragmentGeometryConfig cfg_diff;
    cfg_diff.pool_size = 5;
    cfg_diff.outer_maxsim = false;
    cfg_diff.use_phss = true;

    const auto ref = simeon::score_fragment_geometry(kQuery, fx.idx, fx.enc, fx.frags, cfg_diff);

    simeon::FragmentGeometryConfig cfg_other;
    cfg_other.pool_size = 10;
    cfg_other.outer_maxsim = true;
    cfg_other.doc_scorer_kind = simeon::FragmentGeometryConfig::DocScorerKind::TopKMean;
    (void)simeon::score_fragment_geometry("gradient boosting decision trees", fx.idx, fx.enc,
                                          fx.frags, cfg_other);
    const auto again = simeon::score_fragment_geometry(kQuery, fx.idx, fx.enc, fx.frags, cfg_diff);
    expect(again == ref, "reused scratch produces bit-identical scores after interleaved query");

#if !defined(__MINGW32__)
    // MinGW-w64/libwinpthread has a known teardown heap-corruption bug (0xc0000374) when
    // transient std::thread workers destroy thread_local std::vector/std::string instances
    // (used inside Bm25Index::score and Encoder::encode). Covered on Linux/macOS/TSAN/MSVC.
    constexpr int kThreads = 4;
    std::vector<std::vector<float>> thread_scores(kThreads);
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            for (int iter = 0; iter < 5; ++iter) {
                thread_scores[t] =
                    simeon::score_fragment_geometry(kQuery, fx.idx, fx.enc, fx.frags, cfg_diff);
            }
        });
    }
    for (auto& th : threads)
        th.join();
    for (int t = 0; t < kThreads; ++t)
        expect(thread_scores[t] == ref, "concurrent pooled scratch matches reference");
#endif
}

} // namespace

int main() {
    std::printf("test_fragment_geometry: planted-relevant doc ranks #1 after rerank\n");
    check(/*whiten=*/true);
    check(/*whiten=*/false);
    check_invalid_inputs_fail_closed();
    check_precomputed_bm25_matches_bit_identically();
    check_alpha_one_bm25_parity();
    check_diffusion_mass_conservation();
    check_outer_maxsim_monotonicity();
    check_scratch_reuse_and_thread_safety();
    if (g_failures > 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
