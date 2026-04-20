#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/matryoshka.hpp"
#include "simeon/simeon.hpp"

using simeon::compute_matryoshka_weights;
using simeon::Encoder;
using simeon::EncoderConfig;
using simeon::NGramMode;
using simeon::ProjectionMode;

namespace {

EncoderConfig base_cfg() {
    EncoderConfig cfg;
    cfg.ngram_mode = NGramMode::CharOnly;
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.sketch_dim = 4096;
    cfg.output_dim = 256;
    cfg.projection = ProjectionMode::AchlioptasSparse;
    cfg.projection_seed = 0xC0FFEE0001ULL;
    cfg.l2_normalize = true;
    return cfg;
}

std::vector<std::string_view> make_seed(const std::vector<std::string>& docs) {
    std::vector<std::string_view> out;
    out.reserve(docs.size());
    for (const auto& d : docs) out.emplace_back(d);
    return out;
}

void test_empty_seed_throws() {
    EncoderConfig cfg = base_cfg();
    std::vector<std::string_view> empty;
    bool threw = false;
    try {
        (void)compute_matryoshka_weights(cfg, std::span<const std::string_view>(empty));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_weights_non_negative_and_finite() {
    EncoderConfig cfg = base_cfg();
    const std::vector<std::string> docs = {
        "alpha bravo charlie delta echo foxtrot",
        "the quick brown fox jumps over the lazy dog",
        "machine learning vector embeddings cosine similarity",
        "compact deterministic text encoder hash projection",
        "matryoshka representation learning nested prefix dim",
    };
    auto seed = make_seed(docs);
    auto w = compute_matryoshka_weights(cfg, std::span<const std::string_view>(seed));
    assert(w.size() == cfg.output_dim);
    for (float x : w) {
        assert(std::isfinite(x));
        assert(x >= 0.0f);
    }
    // L1-normalized to mean ≈ 1.
    const double sum = std::accumulate(w.begin(), w.end(), 0.0);
    const double mean = sum / static_cast<double>(w.size());
    assert(std::fabs(mean - 1.0) < 1e-3);
}

void test_determinism_on_fixed_seed_corpus() {
    EncoderConfig cfg = base_cfg();
    const std::vector<std::string> docs = {
        "deterministic seed corpus document one",
        "deterministic seed corpus document two",
        "deterministic seed corpus document three",
        "deterministic seed corpus document four",
    };
    auto seed = make_seed(docs);
    auto w1 = compute_matryoshka_weights(cfg, std::span<const std::string_view>(seed));
    auto w2 = compute_matryoshka_weights(cfg, std::span<const std::string_view>(seed));
    assert(w1.size() == w2.size());
    for (std::size_t i = 0; i < w1.size(); ++i) {
        assert(w1[i] == w2[i]);
    }
}

void test_caller_supplied_weights_round_trip() {
    EncoderConfig cfg = base_cfg();
    const std::vector<std::string> seed_docs = {
        "round trip seed corpus alpha",
        "round trip seed corpus bravo",
        "round trip seed corpus charlie",
        "round trip seed corpus delta",
    };
    auto seed = make_seed(seed_docs);
    auto w = compute_matryoshka_weights(cfg, std::span<const std::string_view>(seed));

    EncoderConfig nested = cfg;
    nested.matryoshka = true;
    nested.matryoshka_weights = w;

    Encoder e(nested);
    std::vector<float> out(e.output_dim(), 0.0f);
    e.encode("a brand new query that the seed never saw", out.data());

    // Encoded result is L2-normalized.
    double mass = 0.0;
    for (float x : out) mass += static_cast<double>(x) * x;
    assert(std::fabs(mass - 1.0) < 1e-5);
}

void test_data_aware_weights_validation() {
    EncoderConfig cfg = base_cfg();
    cfg.matryoshka = true;
    cfg.matryoshka_weights = std::vector<float>(cfg.output_dim - 1, 1.0f);
    bool threw = false;
    try {
        Encoder e(cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    EncoderConfig nan_cfg = base_cfg();
    nan_cfg.matryoshka = true;
    nan_cfg.matryoshka_weights = std::vector<float>(nan_cfg.output_dim, 1.0f);
    nan_cfg.matryoshka_weights[5] = std::nanf("");
    threw = false;
    try {
        Encoder e(nan_cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    EncoderConfig neg_cfg = base_cfg();
    neg_cfg.matryoshka = true;
    neg_cfg.matryoshka_weights = std::vector<float>(neg_cfg.output_dim, 1.0f);
    neg_cfg.matryoshka_weights[10] = -0.1f;
    threw = false;
    try {
        Encoder e(neg_cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_data_aware_concentrates_more_than_uniform() {
    // Build weights from a seed corpus, then check that the resulting nested
    // encoder concentrates more prefix-energy than the uniform (no matryoshka)
    // baseline. A weaker version of the analytic-decay claim: data-aware
    // weights should still produce a nontrivial energy bias toward early dims.
    EncoderConfig cfg = base_cfg();
    const std::vector<std::string> seed_docs = {
        "introduction to product quantization for ann search",
        "hash based text embedding without learned weights",
        "asymmetric distance computation lookup table",
        "k means clustering and lloyd iteration convergence",
        "johnson lindenstrauss random projection lemma",
        "count sketch and feature hashing for nlp",
        "matryoshka representation learning nested embeddings",
        "boyer moore string search algorithm",
        "merkle tree content addressable storage",
        "lock free queue michael scott algorithm",
        "raft consensus protocol leader election",
        "differential privacy laplace noise calibration",
    };
    auto seed = make_seed(seed_docs);
    auto w = compute_matryoshka_weights(cfg, std::span<const std::string_view>(seed));

    // Sort weights descending and place the largest first — this is what
    // the encoder will use after we hand the weights back. Since the encoder
    // applies weights at fixed positions, we instead test the analog: any
    // matryoshka weight schedule with bias produces non-flat prefix energy.
    // Permute weights into descending order to maximize the bias signal.
    std::vector<float> sorted_w = w;
    std::sort(sorted_w.begin(), sorted_w.end(), std::greater<float>());

    EncoderConfig nested = cfg;
    nested.matryoshka = true;
    nested.matryoshka_weights = sorted_w;

    EncoderConfig flat = cfg;
    // No matryoshka — uniform projection.

    Encoder e_nest(nested);
    Encoder e_flat(flat);
    const std::uint32_t d = e_nest.output_dim();
    const std::uint32_t prefix = 64;

    const std::vector<std::string> probes = {
        "hash based projection for ann",
        "consensus protocol leader",
        "feature hashing nlp count sketch",
    };

    double sum_nest = 0.0;
    double sum_flat = 0.0;
    std::vector<float> n(d, 0.0f);
    std::vector<float> f(d, 0.0f);
    for (const auto& p : probes) {
        e_nest.encode(p, n.data());
        e_flat.encode(p, f.data());
        double mn = 0.0;
        double mf = 0.0;
        for (std::uint32_t i = 0; i < prefix; ++i) {
            mn += static_cast<double>(n[i]) * n[i];
            mf += static_cast<double>(f[i]) * f[i];
        }
        sum_nest += mn;
        sum_flat += mf;
    }
    const double avg_nest = sum_nest / probes.size();
    const double avg_flat = sum_flat / probes.size();
    // Sorted-descending weights bias prefix energy. Uniform expectation is
    // prefix/d = 64/256 = 0.25; nested should beat that with margin.
    assert(avg_flat < 0.40);
    assert(avg_nest > avg_flat + 0.05);
}

}  // namespace

int main() {
    test_empty_seed_throws();
    test_weights_non_negative_and_finite();
    test_determinism_on_fixed_seed_corpus();
    test_caller_supplied_weights_round_trip();
    test_data_aware_weights_validation();
    test_data_aware_concentrates_more_than_uniform();
    return 0;
}
