#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/simeon.hpp"

using simeon::Encoder;
using simeon::EncoderConfig;
using simeon::matryoshka_prefix_normalize;
using simeon::NGramMode;
using simeon::ProjectionMode;

namespace {

EncoderConfig matryoshka_cfg(bool enable) {
    EncoderConfig cfg;
    cfg.ngram_mode = NGramMode::CharOnly;
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.sketch_dim = 4096;
    cfg.output_dim = 256;
    cfg.projection = ProjectionMode::AchlioptasSparse;
    cfg.projection_seed = 0xA11CE0001ULL;
    cfg.l2_normalize = true;
    cfg.matryoshka = enable;
    cfg.matryoshka_decay = 32.0f;
    return cfg;
}

double prefix_l2_mass(const float* v, std::uint32_t prefix) {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < prefix; ++i) {
        acc += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    }
    return acc;
}

void test_validation() {
    EncoderConfig bad = matryoshka_cfg(true);
    bad.projection = ProjectionMode::None;
    bool threw = false;
    try {
        Encoder e(bad);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    EncoderConfig bad_decay = matryoshka_cfg(true);
    bad_decay.matryoshka_decay = 0.0f;
    threw = false;
    try {
        Encoder e(bad_decay);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_full_vector_still_unit_norm() {
    Encoder e(matryoshka_cfg(true));
    std::vector<float> v(e.output_dim(), 0.0f);
    e.encode("the quick brown fox jumps over the lazy dog", v.data());
    const double mass = prefix_l2_mass(v.data(), e.output_dim());
    assert(std::fabs(mass - 1.0) < 1e-5);
}

void test_prefix_concentrates_energy() {
    // Matryoshka biases energy into early dims; a uniform projection should
    // distribute energy evenly. Compare prefix L2 mass at the same prefix.
    Encoder uniform(matryoshka_cfg(false));
    Encoder nested(matryoshka_cfg(true));

    const std::vector<std::string> docs = {
        "alpha bravo charlie delta",
        "the quick brown fox jumps",
        "machine learning vector embeddings",
        "compact deterministic text encoder",
        "matryoshka representation learning prefix",
    };

    const std::uint32_t d = uniform.output_dim();
    const std::uint32_t prefix = 64;

    double sum_uni = 0.0;
    double sum_nest = 0.0;
    std::vector<float> u(d, 0.0f);
    std::vector<float> n(d, 0.0f);
    for (const auto& doc : docs) {
        uniform.encode(doc, u.data());
        nested.encode(doc, n.data());
        sum_uni += prefix_l2_mass(u.data(), prefix);
        sum_nest += prefix_l2_mass(n.data(), prefix);
    }
    const double avg_uni = sum_uni / docs.size();
    const double avg_nest = sum_nest / docs.size();

    // Uniform expectation is roughly prefix/d = 64/256 = 0.25.
    // Nested expectation under decay=32: sum_{r<64} 1/(1+r/32) divided by
    // sum_{r<256} of the same is ~0.50 (harmonic). Demand a clear gap above
    // uniform; thresholds leave headroom for FP variance.
    assert(avg_uni < 0.40);
    assert(avg_nest > 0.45);
    assert(avg_nest > avg_uni + 0.15);
}

void test_prefix_renormalize_is_unit() {
    Encoder e(matryoshka_cfg(true));
    const std::uint32_t d = e.output_dim();
    std::vector<float> v(d, 0.0f);
    e.encode("renormalize this prefix to unit length please", v.data());

    for (std::uint32_t prefix : {16u, 32u, 64u, 128u}) {
        std::vector<float> p(v.begin(), v.begin() + prefix);
        matryoshka_prefix_normalize(p.data(), prefix);
        const double mass = prefix_l2_mass(p.data(), prefix);
        assert(std::fabs(mass - 1.0) < 1e-5);
    }
}

float cosine(const float* a, const float* b, std::uint32_t d) {
    double acc = 0.0;
    for (std::uint32_t i = 0; i < d; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(acc);
}

void test_prefix_ranking_aligns_with_full() {
    // The matryoshka claim: ranking by cosine on a renormalized prefix
    // recovers most of the ranking from the full vector. Build a small
    // corpus + queries, rank by full and by prefix, and assert overlap of
    // top-K is high.
    Encoder e(matryoshka_cfg(true));
    const std::uint32_t d = e.output_dim();

    const std::vector<std::string> corpus = {
        "introduction to product quantization for ann search",
        "hash based text embedding without learned weights",
        "asymmetric distance computation lookup table",
        "k-means clustering and lloyd iteration convergence",
        "johnson lindenstrauss random projection lemma",
        "count sketch and feature hashing for nlp",
        "matryoshka representation learning nested embeddings",
        "boyer moore string search algorithm",
        "merkle tree content addressable storage",
        "lock free queue michael scott algorithm",
        "raft consensus protocol leader election",
        "differential privacy laplace noise calibration",
        "transformer attention mechanism scaled dot product",
        "graph neural networks message passing",
        "elliptic curve cryptography ed25519 signature",
        "rsync rolling hash chunk dedup",
    };
    const std::vector<std::string> queries = {
        "product quantization ann",
        "random projection johnson lindenstrauss",
        "merkle tree storage dedup",
        "raft leader election consensus",
    };

    std::vector<std::vector<float>> docs(corpus.size(), std::vector<float>(d, 0.0f));
    for (std::size_t i = 0; i < corpus.size(); ++i) {
        e.encode(corpus[i], docs[i].data());
    }

    const std::uint32_t prefix = 64;
    const std::size_t topk = 4;

    std::size_t overlap_total = 0;
    for (const auto& qtext : queries) {
        std::vector<float> q(d, 0.0f);
        e.encode(qtext, q.data());

        std::vector<std::pair<float, std::size_t>> full;
        std::vector<std::pair<float, std::size_t>> pre;
        full.reserve(corpus.size());
        pre.reserve(corpus.size());

        std::vector<float> qp(q.begin(), q.begin() + prefix);
        matryoshka_prefix_normalize(qp.data(), prefix);

        for (std::size_t i = 0; i < corpus.size(); ++i) {
            full.emplace_back(cosine(q.data(), docs[i].data(), d), i);

            std::vector<float> dp(docs[i].begin(), docs[i].begin() + prefix);
            matryoshka_prefix_normalize(dp.data(), prefix);
            pre.emplace_back(cosine(qp.data(), dp.data(), prefix), i);
        }

        std::partial_sort(full.begin(), full.begin() + topk, full.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        std::partial_sort(pre.begin(), pre.begin() + topk, pre.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });

        std::vector<std::size_t> full_ids;
        for (std::size_t i = 0; i < topk; ++i)
            full_ids.push_back(full[i].second);
        for (std::size_t i = 0; i < topk; ++i) {
            for (auto id : full_ids) {
                if (pre[i].second == id) {
                    ++overlap_total;
                    break;
                }
            }
        }
    }

    const double overlap_ratio = static_cast<double>(overlap_total) / (queries.size() * topk);
    // 4 queries * 4 hits = 16 possible overlaps. With matryoshka we expect
    // strong agreement; demand at least 50% to leave headroom for FP variance.
    if (overlap_ratio < 0.50) {
        std::fprintf(stderr, "matryoshka prefix top-K overlap ratio %.3f < 0.50\n", overlap_ratio);
    }
    assert(overlap_ratio >= 0.50);
}

void test_determinism() {
    Encoder e(matryoshka_cfg(true));
    std::vector<float> a(e.output_dim(), 0.0f);
    std::vector<float> b(e.output_dim(), 0.0f);
    e.encode("deterministic across calls", a.data());
    e.encode("deterministic across calls", b.data());
    for (std::uint32_t i = 0; i < e.output_dim(); ++i) {
        assert(a[i] == b[i]);
    }
}

void test_prefix_normalize_no_op_on_zero() {
    // Edge cases: nullptr and zero-length should be no-ops, not crashes.
    matryoshka_prefix_normalize(nullptr, 16);
    std::vector<float> v = {1.0f, 2.0f, 3.0f};
    matryoshka_prefix_normalize(v.data(), 0);
    assert(v[0] == 1.0f && v[1] == 2.0f && v[2] == 3.0f);
}

} // namespace

int main() {
    test_validation();
    test_full_vector_still_unit_norm();
    test_prefix_concentrates_energy();
    test_prefix_renormalize_is_unit();
    test_prefix_ranking_aligns_with_full();
    test_determinism();
    test_prefix_normalize_no_op_on_zero();
    return 0;
}
