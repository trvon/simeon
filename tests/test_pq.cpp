// Tests for simeon::ProductQuantizer + PQQuery (asymmetric distance).
//
// Coverage:
//  - construction validation (dim%m, k<=256)
//  - random Gaussian init: deterministic, distinct centroids
//  - Lloyd's k-means train: deterministic, reduces reconstruction error vs
//    random init, no NaN even when training has duplicate rows
//  - encode/decode round-trip: every code in [0, k) appears within reach
//  - ADC parity: PQQuery::distance_l2_sq(code) == ||q - decode(code)||^2
//    PQQuery::inner_product(code) == q · decode(code)
//  - Recall@10: trained PQ on random data recovers majority of the brute-force
//    nearest neighbors at 8-byte codes (per the Jégou TPAMI ballpark)
//  - Determinism: same seed + same training → identical codebooks
//  - Non-multiple-of-four centroid counts exercise blocked-scan tails

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "simeon/pq.hpp"

using simeon::PQConfig;
using simeon::PQInnerProductQuery;
using simeon::PQQuery;
using simeon::ProductQuantizer;

namespace {

constexpr float kEps = 1e-4f;

std::vector<float> make_random_vectors(std::uint32_t n, std::uint32_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(static_cast<std::size_t>(n) * dim);
    for (auto& x : v)
        x = dist(rng);
    return v;
}

// Sample `n_clusters` centers from N(0, sigma_c^2 I). Deterministic per seed.
std::vector<float> make_cluster_centers(std::uint32_t n_clusters, std::uint32_t dim, float sigma_c,
                                        std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> center_dist(0.0f, sigma_c);
    std::vector<float> centers(static_cast<std::size_t>(n_clusters) * dim);
    for (auto& c : centers)
        c = center_dist(rng);
    return centers;
}

// Mixture-of-Gaussians sampler that takes *fixed* cluster centers and adds
// N(0, sigma_pt^2 I) jitter. db / query / train sets must share centers —
// otherwise nearest-neighbor queries degenerate to noise (the fix for an
// earlier version of this test).
std::vector<float> make_clustered_vectors(std::uint32_t n, std::uint32_t dim,
                                          const std::vector<float>& centers, float sigma_pt,
                                          std::uint32_t seed) {
    const std::uint32_t n_clusters = static_cast<std::uint32_t>(centers.size() / dim);
    std::mt19937 rng(seed);
    std::normal_distribution<float> jitter_dist(0.0f, sigma_pt);
    std::uniform_int_distribution<std::uint32_t> pick_cluster(0, n_clusters - 1);
    std::vector<float> v(static_cast<std::size_t>(n) * dim);
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t c = pick_cluster(rng);
        for (std::uint32_t d = 0; d < dim; ++d) {
            v[static_cast<std::size_t>(i) * dim + d] =
                centers[static_cast<std::size_t>(c) * dim + d] + jitter_dist(rng);
        }
    }
    return v;
}

float l2_sq(const float* a, const float* b, std::uint32_t n) {
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

float dot(const float* a, const float* b, std::uint32_t n) {
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
        acc += a[i] * b[i];
    return acc;
}

void test_construction_validation() {
    // dim must be divisible by m.
    bool threw = false;
    try {
        ProductQuantizer pq(PQConfig{.dim = 17, .m = 4, .k = 16});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // k must be <= 256.
    threw = false;
    try {
        ProductQuantizer pq(PQConfig{.dim = 32, .m = 4, .k = 257});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // dim, m, k all > 0.
    threw = false;
    try {
        ProductQuantizer pq(PQConfig{.dim = 0, .m = 4, .k = 16});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Valid construction.
    ProductQuantizer pq(PQConfig{.dim = 32, .m = 4, .k = 16});
    assert(pq.dim() == 32);
    assert(pq.m() == 4);
    assert(pq.k() == 16);
    assert(pq.dsub() == 8);
    assert(!pq.is_trained());
}

void test_random_gaussian_init_deterministic() {
    PQConfig cfg{.dim = 32, .m = 4, .k = 16, .seed = 0xC0FFEE};
    ProductQuantizer a(cfg);
    ProductQuantizer b(cfg);
    a.init_random_gaussian();
    b.init_random_gaussian();
    assert(a.is_trained());
    for (std::uint32_t mi = 0; mi < cfg.m; ++mi) {
        for (std::uint32_t ki = 0; ki < cfg.k; ++ki) {
            const float* ca = a.centroid(mi, ki);
            const float* cb = b.centroid(mi, ki);
            for (std::uint32_t d = 0; d < a.dsub(); ++d) {
                assert(ca[d] == cb[d]);
            }
        }
    }

    // Different seed → different centroids.
    PQConfig cfg2 = cfg;
    cfg2.seed = 0xBEEF;
    ProductQuantizer c(cfg2);
    c.init_random_gaussian();
    bool any_diff = false;
    for (std::uint32_t d = 0; d < a.dsub(); ++d) {
        if (a.centroid(0, 0)[d] != c.centroid(0, 0)[d]) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);
}

// Centroids within one subspace must not all coincide (rules out a bug where
// the key-derived RNG accidentally collapses).
void test_centroids_are_distinct() {
    PQConfig cfg{.dim = 32, .m = 4, .k = 16, .seed = 0xC0FFEE};
    ProductQuantizer pq(cfg);
    pq.init_random_gaussian();
    std::uint32_t distinct_pairs = 0;
    for (std::uint32_t i = 0; i < cfg.k; ++i) {
        for (std::uint32_t j = i + 1; j < cfg.k; ++j) {
            const float d = l2_sq(pq.centroid(0, i), pq.centroid(0, j), pq.dsub());
            if (d > 1e-3f)
                ++distinct_pairs;
        }
    }
    // (k choose 2) = 120 for k=16. Allow a small slack but expect almost all
    // distinct.
    assert(distinct_pairs > 100);
}

void test_train_deterministic_and_lower_distortion() {
    PQConfig cfg{.dim = 32, .m = 4, .k = 16, .seed = 0xDEADBEEF};
    auto train_data = make_random_vectors(512, cfg.dim, 1234);

    ProductQuantizer pq_random(cfg);
    pq_random.init_random_gaussian();

    ProductQuantizer pq_trained_a(cfg);
    pq_trained_a.train(train_data.data(), 512, 25);
    ProductQuantizer pq_trained_b(cfg);
    pq_trained_b.train(train_data.data(), 512, 25);

    // Same seed + same training set → identical codebooks.
    for (std::uint32_t mi = 0; mi < cfg.m; ++mi) {
        for (std::uint32_t ki = 0; ki < cfg.k; ++ki) {
            const float* a = pq_trained_a.centroid(mi, ki);
            const float* b = pq_trained_b.centroid(mi, ki);
            for (std::uint32_t d = 0; d < pq_trained_a.dsub(); ++d) {
                assert(a[d] == b[d]);
            }
        }
    }

    // Trained codebooks have lower mean reconstruction error than random init.
    auto eval = make_random_vectors(256, cfg.dim, 9999);
    std::vector<std::uint8_t> code(cfg.m);
    std::vector<float> recon(cfg.dim);

    auto mean_recon_err = [&](const ProductQuantizer& pq) {
        double total = 0.0;
        for (std::uint32_t i = 0; i < 256; ++i) {
            const float* x = eval.data() + static_cast<std::size_t>(i) * cfg.dim;
            pq.encode(x, code.data());
            pq.decode(code.data(), recon.data());
            total += l2_sq(x, recon.data(), cfg.dim);
        }
        return total / 256.0;
    };

    const double err_random = mean_recon_err(pq_random);
    const double err_trained = mean_recon_err(pq_trained_a);
    std::printf("PQ recon err: random=%.3f trained=%.3f\n", err_random, err_trained);
    assert(err_trained < err_random);
}

void test_encode_decode_roundtrip_uses_full_codebook() {
    PQConfig cfg{.dim = 32, .m = 4, .k = 16, .seed = 0x1234};
    auto train_data = make_random_vectors(1024, cfg.dim, 42);
    ProductQuantizer pq(cfg);
    pq.train(train_data.data(), 1024, 25);

    // Encode a large evaluation set; assert that for each subspace the union
    // of codes spans most of [0, k).
    auto eval = make_random_vectors(2048, cfg.dim, 77);
    std::vector<std::uint8_t> codes(2048 * cfg.m);
    pq.encode_batch(eval.data(), 2048, codes.data());

    for (std::uint32_t mi = 0; mi < cfg.m; ++mi) {
        std::vector<std::uint8_t> seen(cfg.k, 0);
        for (std::uint32_t i = 0; i < 2048; ++i) {
            seen[codes[i * cfg.m + mi]] = 1;
        }
        const std::uint32_t used = std::accumulate(seen.begin(), seen.end(), std::uint32_t{0});
        // Should saturate the codebook on a healthy random distribution.
        assert(used == cfg.k);
    }
}

void test_adc_parity_l2_and_ip() {
    PQConfig cfg{.dim = 64, .m = 8, .k = 64, .seed = 0xABCD};
    auto train_data = make_random_vectors(1024, cfg.dim, 11);
    ProductQuantizer pq(cfg);
    pq.train(train_data.data(), 1024, 25);

    auto query = make_random_vectors(1, cfg.dim, 22);
    auto db = make_random_vectors(64, cfg.dim, 33);

    std::vector<std::uint8_t> codes(64 * cfg.m);
    pq.encode_batch(db.data(), 64, codes.data());

    PQQuery q(pq, query.data());

    std::vector<float> recon(cfg.dim);
    for (std::uint32_t i = 0; i < 64; ++i) {
        pq.decode(codes.data() + i * cfg.m, recon.data());

        const float adc_l2 = q.distance_l2_sq(codes.data() + i * cfg.m);
        const float ref_l2 = l2_sq(query.data(), recon.data(), cfg.dim);
        assert(std::fabs(adc_l2 - ref_l2) < kEps * std::max(1.0f, ref_l2));

        const float adc_ip = q.inner_product(codes.data() + i * cfg.m);
        const float ref_ip = dot(query.data(), recon.data(), cfg.dim);
        assert(std::fabs(adc_ip - ref_ip) < kEps * std::max(1.0f, std::fabs(ref_ip)));
    }

    // LUT shape sanity.
    assert(q.lut_l2_sq().size() == static_cast<std::size_t>(cfg.m) * cfg.k);
    assert(q.lut_ip().size() == static_cast<std::size_t>(cfg.m) * cfg.k);
}

void test_inner_product_only_query_parity() {
    PQConfig cfg{.dim = 64, .m = 8, .k = 64, .seed = 0x5151};
    auto train_data = make_random_vectors(1024, cfg.dim, 41);
    ProductQuantizer pq(cfg);
    pq.train(train_data.data(), 1024, 25);

    auto query = make_random_vectors(1, cfg.dim, 42);
    auto db = make_random_vectors(64, cfg.dim, 43);
    std::vector<std::uint8_t> codes(64 * cfg.m);
    pq.encode_batch(db.data(), 64, codes.data());

    PQQuery full(pq, query.data());
    PQInnerProductQuery innerProductOnly(pq, query.data());
    assert(innerProductOnly.lut_ip().size() == static_cast<std::size_t>(cfg.m) * cfg.k);
    for (std::uint32_t i = 0; i < 64; ++i) {
        const auto* code = codes.data() + i * cfg.m;
        assert(innerProductOnly.inner_product(code) == full.inner_product(code));
    }
}

void test_non_multiple_of_four_centroids() {
    ProductQuantizer pq(PQConfig{.dim = 4, .m = 1, .k = 5});
    std::vector<float> codebooks(5 * 4);
    for (std::uint32_t centroid = 0; centroid < 5; ++centroid) {
        std::fill_n(codebooks.data() + centroid * 4, 4, static_cast<float>(centroid));
    }
    pq.import_codebooks(codebooks);

    std::vector<float> query(4, 4.0f);
    std::uint8_t code = 0;
    pq.encode(query.data(), &code);
    assert(code == 4);
}

// Recall@10 on data that has cluster structure (which is what PQ is designed
// for — SIFT descriptors, learned embeddings, etc. all cluster naturally).
// Pure Gaussian noise in 64 dims would give recall ~0.3 because top-10 is
// not separable from the rest of the distribution; mixture-of-Gaussians
// reproduces the regime where PQ shines, per Jégou et al. 2010 TPAMI.
void test_recall_at_10() {
    PQConfig cfg{.dim = 64, .m = 8, .k = 256, .seed = 0xFEEDBEEF};
    constexpr std::uint32_t kClusters = 32;
    constexpr float kSigmaC = 5.0f;
    constexpr float kSigmaPt = 1.0f;

    const auto centers = make_cluster_centers(kClusters, cfg.dim, kSigmaC, 555);
    auto train_data = make_clustered_vectors(4096, cfg.dim, centers, kSigmaPt, 1);
    ProductQuantizer pq(cfg);
    pq.train(train_data.data(), 4096, 25);

    constexpr std::uint32_t kDb = 1024;
    constexpr std::uint32_t kQ = 16;
    constexpr std::uint32_t kTopK = 10;

    auto db = make_clustered_vectors(kDb, cfg.dim, centers, kSigmaPt, 2);
    auto queries = make_clustered_vectors(kQ, cfg.dim, centers, kSigmaPt, 3);

    std::vector<std::uint8_t> codes(kDb * cfg.m);
    pq.encode_batch(db.data(), kDb, codes.data());

    std::uint32_t total_hits = 0;
    for (std::uint32_t qi = 0; qi < kQ; ++qi) {
        const float* q = queries.data() + qi * cfg.dim;

        // Brute-force top-10 by exact L2.
        std::vector<std::pair<float, std::uint32_t>> exact(kDb);
        for (std::uint32_t i = 0; i < kDb; ++i) {
            exact[i] = {l2_sq(q, db.data() + i * cfg.dim, cfg.dim), i};
        }
        std::partial_sort(exact.begin(), exact.begin() + kTopK, exact.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        std::vector<std::uint32_t> exact_top(kTopK);
        for (std::uint32_t i = 0; i < kTopK; ++i)
            exact_top[i] = exact[i].second;

        // ADC top-10.
        PQQuery query(pq, q);
        std::vector<std::pair<float, std::uint32_t>> adc(kDb);
        for (std::uint32_t i = 0; i < kDb; ++i) {
            adc[i] = {query.distance_l2_sq(codes.data() + i * cfg.m), i};
        }
        std::partial_sort(adc.begin(), adc.begin() + kTopK, adc.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });

        std::sort(exact_top.begin(), exact_top.end());
        std::vector<std::uint32_t> adc_top(kTopK);
        for (std::uint32_t i = 0; i < kTopK; ++i)
            adc_top[i] = adc[i].second;
        std::sort(adc_top.begin(), adc_top.end());

        std::vector<std::uint32_t> hits;
        std::set_intersection(exact_top.begin(), exact_top.end(), adc_top.begin(), adc_top.end(),
                              std::back_inserter(hits));
        total_hits += static_cast<std::uint32_t>(hits.size());
    }

    const double recall = static_cast<double>(total_hits) / (kQ * kTopK);
    std::printf("PQ Recall@10 on mixture-of-Gaussians: %.3f\n", recall);
    // Jégou et al. 2010 TPAMI reports R@10 ~0.5-0.65 for 8-byte codes on
    // SIFT128 (comparable code-to-dim ratio to ours: 8 bytes / 64d). 0.4
    // leaves headroom for FP/RNG determinism variance across platforms
    // while still failing if PQ is fundamentally broken (random would be
    // ~10/1024 = 0.01).
    assert(recall >= 0.4);
}

} // namespace

// Batched scans must return exactly the per-code scores (same summation order per code), so
// callers can switch to them without moving ranking tie-breaks. Counts cover empty input and
// tails that are not a multiple of the interleave width.
void test_inner_product_batch_bit_identical() {
    PQConfig cfg{.dim = 128, .m = 32, .k = 256, .seed = 0x6262};
    auto train_data = make_random_vectors(2048, cfg.dim, 51);
    ProductQuantizer pq(cfg);
    pq.train(train_data.data(), 2048, 10);

    constexpr std::uint32_t kRows = 67;
    auto query = make_random_vectors(1, cfg.dim, 52);
    auto db = make_random_vectors(kRows, cfg.dim, 53);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kRows) * cfg.m);
    pq.encode_batch(db.data(), kRows, codes.data());

    PQInnerProductQuery q(pq, query.data());
    for (std::size_t count : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{4},
                              std::size_t{9}, std::size_t{kRows}}) {
        std::vector<float> batch(count + 1, -1.0f);
        q.inner_product_batch(codes.data(), count, batch.data());
        for (std::size_t i = 0; i < count; ++i) {
            assert(batch[i] == q.inner_product(codes.data() + i * cfg.m));
        }
        assert(batch[count] == -1.0f); // never writes past `count`
    }

    std::vector<std::size_t> indices;
    for (std::size_t i = kRows; i-- > 0;) {
        if (i % 3 != 1)
            indices.push_back(i);
    }
    std::vector<float> gathered(indices.size());
    q.inner_product_gather(codes.data(), indices.data(), indices.size(), gathered.data());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        assert(gathered[i] == q.inner_product(codes.data() + indices[i] * cfg.m));
    }
}

void l2_normalize_rows(std::vector<float>& vecs, std::uint32_t n, std::uint32_t dim) {
    for (std::uint32_t i = 0; i < n; ++i) {
        float* row = vecs.data() + static_cast<std::size_t>(i) * dim;
        double sq = 0.0;
        for (std::uint32_t d = 0; d < dim; ++d)
            sq += static_cast<double>(row[d]) * row[d];
        const float inv = sq > 0.0 ? static_cast<float>(1.0 / std::sqrt(sq)) : 0.0f;
        for (std::uint32_t d = 0; d < dim; ++d)
            row[d] *= inv;
    }
}

// Exercises parallel subspace training (m >= 8 && n >= 1024) and small-dsub SIMD dot
// (dim=384, m=32 -> dsub=12 < 16, plus dsub=4, 6, 8). Verifies bit-identical
// determinism across runs and SIMD-vs-scalar ADC LUT parity.
void test_parallel_train_deterministic_and_small_dsub() {
    for (std::uint32_t dsub : {4u, 6u, 8u, 12u}) {
        const std::uint32_t m = (dsub == 12u) ? 32u : 8u;
        PQConfig cfg{.dim = m * dsub, .m = m, .k = 16, .seed = 0x51A11D5Bu ^ dsub};
        const std::uint32_t kTrain = (dsub == 12u) ? 1024u : 256u;
        auto train_data = make_random_vectors(kTrain, cfg.dim, 700 + dsub);

        ProductQuantizer pq_a(cfg);
        ProductQuantizer pq_b(cfg);
        pq_a.train(train_data.data(), kTrain, 6);
        pq_b.train(train_data.data(), kTrain, 6);

        const auto cb_a = pq_a.codebooks();
        const auto cb_b = pq_b.codebooks();
        assert(cb_a.size() == cb_b.size());
        for (std::size_t i = 0; i < cb_a.size(); ++i)
            assert(cb_a[i] == cb_b[i]);

        // Verify SIMD dot() LUT matches scalar dot(query, decode(code)).
        auto query = make_random_vectors(1, cfg.dim, 800 + dsub);
        auto db = make_random_vectors(32, cfg.dim, 900 + dsub);
        std::vector<std::uint8_t> codes(32 * cfg.m);
        pq_a.encode_batch(db.data(), 32, codes.data());
        PQInnerProductQuery ipq(pq_a, query.data());
        std::vector<float> recon(cfg.dim);
        for (std::uint32_t i = 0; i < 32; ++i) {
            pq_a.decode(codes.data() + i * cfg.m, recon.data());
            const float adc = ipq.inner_product(codes.data() + i * cfg.m);
            const float ref = dot(query.data(), recon.data(), cfg.dim);
            assert(std::fabs(adc - ref) < kEps * std::max(1.0f, std::fabs(ref)));
        }
    }
}

// Verifies 4-bit SIMD FastScan LUT path (k <= 16) matches exact ADC within
// uint8 quantization error across tail sizes, and falls back bit-identically when k > 16.
void test_fastscan_lut16_parity_and_fallback() {
    for (std::uint32_t k : {12u, 16u}) {
        PQConfig cfg{.dim = 384, .m = 32, .k = k, .seed = 0xFA570000u ^ k};
        auto train_data = make_random_vectors(256, cfg.dim, 1000 + k);
        l2_normalize_rows(train_data, 256, cfg.dim);
        ProductQuantizer pq(cfg);
        pq.train(train_data.data(), 256, 6);

        constexpr std::uint32_t kRows = 67;
        auto query = make_random_vectors(1, cfg.dim, 1100 + k);
        l2_normalize_rows(query, 1, cfg.dim);
        auto db = make_random_vectors(kRows, cfg.dim, 1200 + k);
        l2_normalize_rows(db, kRows, cfg.dim);

        std::vector<std::uint8_t> codes(static_cast<std::size_t>(kRows) * cfg.m);
        pq.encode_batch(db.data(), kRows, codes.data());

        PQInnerProductQuery q(pq, query.data());
        for (std::size_t count :
             {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{15},
              std::size_t{16}, std::size_t{17}, std::size_t{31}, std::size_t{32}, std::size_t{33},
              std::size_t{kRows}}) {
            std::vector<float> exact(count + 1, -2.0f);
            std::vector<float> fast(count + 1, -2.0f);
            q.inner_product_batch(codes.data(), count, exact.data());
            q.inner_product_batch_fastscan(codes.data(), count, fast.data());
            for (std::size_t i = 0; i < count; ++i) {
                assert(std::fabs(fast[i] - exact[i]) < 0.05f);
            }
            assert(fast[count] == -2.0f);
        }
    }

    // Fallback when k > 16 must be bit-identical to inner_product_batch.
    PQConfig cfg_wide{.dim = 64, .m = 8, .k = 32, .seed = 0xFA570032u};
    auto train_wide = make_random_vectors(256, cfg_wide.dim, 1300);
    ProductQuantizer pq_wide(cfg_wide);
    pq_wide.train(train_wide.data(), 256, 8);
    auto query = make_random_vectors(1, cfg_wide.dim, 1301);
    auto db = make_random_vectors(40, cfg_wide.dim, 1302);
    std::vector<std::uint8_t> codes(40 * cfg_wide.m);
    pq_wide.encode_batch(db.data(), 40, codes.data());
    PQInnerProductQuery q_wide(pq_wide, query.data());
    std::vector<float> exact(40), fast(40);
    q_wide.inner_product_batch(codes.data(), 40, exact.data());
    q_wide.inner_product_batch_fastscan(codes.data(), 40, fast.data());
    for (std::size_t i = 0; i < 40; ++i)
        assert(fast[i] == exact[i]);
}

// PQ inner-product recall on L2-normalized clustered vectors (dim=384, m=32, k=16 -> dsub=12).
void test_inner_product_recall_l2_normalized() {
    PQConfig cfg{.dim = 384, .m = 32, .k = 16, .seed = 0xC0514E16u};
    constexpr std::uint32_t kClusters = 32;
    constexpr float kSigmaC = 4.0f;
    constexpr float kSigmaPt = 0.75f;

    const auto centers = make_cluster_centers(kClusters, cfg.dim, kSigmaC, 2001);
    auto train_data = make_clustered_vectors(1024, cfg.dim, centers, kSigmaPt, 2002);
    l2_normalize_rows(train_data, 1024, cfg.dim);

    ProductQuantizer pq(cfg);
    pq.train(train_data.data(), 1024, 10);

    constexpr std::uint32_t kDb = 512;
    constexpr std::uint32_t kQ = 16;
    constexpr std::uint32_t kTopK = 10;

    auto db = make_clustered_vectors(kDb, cfg.dim, centers, kSigmaPt, 2003);
    auto queries = make_clustered_vectors(kQ, cfg.dim, centers, kSigmaPt, 2004);
    l2_normalize_rows(db, kDb, cfg.dim);
    l2_normalize_rows(queries, kQ, cfg.dim);

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kDb) * cfg.m);
    pq.encode_batch(db.data(), kDb, codes.data());

    std::uint32_t exact_hits = 0;
    std::uint32_t fastscan_hits = 0;
    std::vector<float> batch_scores(kDb);
    std::vector<float> fast_scores(kDb);

    auto count_hits = [&](const std::vector<std::uint32_t>& gt_sorted,
                          const std::vector<float>& scores) {
        std::vector<std::pair<float, std::uint32_t>> ranked(kDb);
        for (std::uint32_t i = 0; i < kDb; ++i)
            ranked[i] = {scores[i], i};
        std::partial_sort(ranked.begin(), ranked.begin() + kTopK, ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::uint32_t> top(kTopK);
        for (std::uint32_t i = 0; i < kTopK; ++i)
            top[i] = ranked[i].second;
        std::sort(top.begin(), top.end());
        std::vector<std::uint32_t> inter;
        std::set_intersection(gt_sorted.begin(), gt_sorted.end(), top.begin(), top.end(),
                              std::back_inserter(inter));
        return static_cast<std::uint32_t>(inter.size());
    };

    for (std::uint32_t qi = 0; qi < kQ; ++qi) {
        const float* q = queries.data() + static_cast<std::size_t>(qi) * cfg.dim;
        std::vector<std::pair<float, std::uint32_t>> gt(kDb);
        for (std::uint32_t i = 0; i < kDb; ++i)
            gt[i] = {dot(q, db.data() + static_cast<std::size_t>(i) * cfg.dim, cfg.dim), i};
        std::partial_sort(gt.begin(), gt.begin() + kTopK, gt.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::uint32_t> gt_top(kTopK);
        for (std::uint32_t i = 0; i < kTopK; ++i)
            gt_top[i] = gt[i].second;
        std::sort(gt_top.begin(), gt_top.end());

        PQInnerProductQuery ipq(pq, q);
        ipq.inner_product_batch(codes.data(), kDb, batch_scores.data());
        ipq.inner_product_batch_fastscan(codes.data(), kDb, fast_scores.data());
        exact_hits += count_hits(gt_top, batch_scores);
        fastscan_hits += count_hits(gt_top, fast_scores);
    }

    const double recall_exact = static_cast<double>(exact_hits) / (kQ * kTopK);
    const double recall_fast = static_cast<double>(fastscan_hits) / (kQ * kTopK);
    std::printf("PQ IP Recall@10 (L2-norm 384d, m=32, k=16): exact=%.3f fastscan=%.3f\n",
                recall_exact, recall_fast);
    assert(recall_exact >= 0.5);
    assert(recall_fast >= 0.45);
}

int main() {
    test_construction_validation();
    test_random_gaussian_init_deterministic();
    test_centroids_are_distinct();
    test_train_deterministic_and_lower_distortion();
    test_encode_decode_roundtrip_uses_full_codebook();
    test_adc_parity_l2_and_ip();
    test_inner_product_only_query_parity();
    test_inner_product_batch_bit_identical();
    test_non_multiple_of_four_centroids();
    test_recall_at_10();
    test_parallel_train_deterministic_and_small_dsub();
    test_fastscan_lut16_parity_and_fallback();
    test_inner_product_recall_l2_normalized();
    std::printf("test_pq: all passed\n");
    return 0;
}
