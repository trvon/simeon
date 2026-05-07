#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "simeon/projection.hpp"
#include "simeon/simeon.hpp"

using simeon::Projection;
using simeon::ProjectionMode;

namespace {

std::vector<std::int32_t> random_sketch(std::uint32_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> dist(-32, 32);
    std::vector<std::int32_t> v(dim);
    for (auto& x : v)
        x = dist(rng);
    return v;
}

float l2_f(const float* p, std::uint32_t n) {
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i)
        s += static_cast<double>(p[i]) * p[i];
    return static_cast<float>(std::sqrt(s));
}

float l2_i(const std::int32_t* p, std::uint32_t n) {
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const double f = static_cast<double>(p[i]);
        s += f * f;
    }
    return static_cast<float>(std::sqrt(s));
}

void test_fwht_distortion_within_bound() {
    constexpr std::uint32_t sketch_dim = 1024; // already a power of 2
    constexpr std::uint32_t output_dim = 256;
    constexpr std::uint32_t pairs = 80;

    Projection p(sketch_dim, output_dim, ProjectionMode::Fwht, /*seed*/ 42);

    std::vector<float> proj_diff(output_dim);
    double sum_sq_distortion = 0.0;
    std::uint32_t in_bound = 0;
    constexpr float bound = 0.20f;
    for (std::uint32_t k = 0; k < pairs; ++k) {
        auto a = random_sketch(sketch_dim, 1234 + k);
        auto b = random_sketch(sketch_dim, 9876 + k);
        std::vector<std::int32_t> d(sketch_dim);
        for (std::uint32_t i = 0; i < sketch_dim; ++i)
            d[i] = a[i] - b[i];

        const float in_norm = l2_i(d.data(), sketch_dim);
        if (in_norm == 0.0f)
            continue;
        p.apply(d.data(), proj_diff.data());
        const float out_norm = l2_f(proj_diff.data(), output_dim);

        const double ratio = static_cast<double>(out_norm) / static_cast<double>(in_norm);
        const double dist = std::fabs(ratio - 1.0);
        sum_sq_distortion += dist * dist;
        if (dist <= static_cast<double>(bound))
            ++in_bound;
    }
    const double rms = std::sqrt(sum_sq_distortion / pairs);
    // Subsampled-WHT distortion at k=256 over n=1024: textbook bound is
    // O(sqrt(log d / k)). Empirical RMS should land well under 0.20 and
    // most pairs within ±0.20.
    assert(rms < 0.20);
    assert(in_bound >= pairs * 7 / 10);
}

void test_fwht_pads_to_power_of_two() {
    // sketch_dim=600 should be padded internally to 1024 without crashing or
    // returning nonsense norms.
    constexpr std::uint32_t sketch_dim = 600;
    constexpr std::uint32_t output_dim = 128;
    Projection p(sketch_dim, output_dim, ProjectionMode::Fwht, 7);

    auto v = random_sketch(sketch_dim, 1);
    std::vector<float> out(output_dim);
    p.apply(v.data(), out.data());
    const float in_norm = l2_i(v.data(), sketch_dim);
    const float out_norm = l2_f(out.data(), output_dim);
    assert(in_norm > 0.0f);
    // Loose sanity bound — exact distortion depends on sample/sign; just
    // assert the output isn't degenerate.
    const float ratio = out_norm / in_norm;
    assert(ratio > 0.5f && ratio < 2.0f);
}

void test_fwht_determinism() {
    constexpr std::uint32_t sketch_dim = 512;
    constexpr std::uint32_t output_dim = 64;
    Projection p1(sketch_dim, output_dim, ProjectionMode::Fwht, 99);
    Projection p2(sketch_dim, output_dim, ProjectionMode::Fwht, 99);
    auto a = random_sketch(sketch_dim, 5);
    std::vector<float> o1(output_dim), o2(output_dim);
    p1.apply(a.data(), o1.data());
    p2.apply(a.data(), o2.data());
    for (std::uint32_t i = 0; i < output_dim; ++i) {
        assert(o1[i] == o2[i]);
    }
}

void test_fwht_seed_changes_output() {
    constexpr std::uint32_t sketch_dim = 512;
    constexpr std::uint32_t output_dim = 64;
    Projection p1(sketch_dim, output_dim, ProjectionMode::Fwht, 1);
    Projection p2(sketch_dim, output_dim, ProjectionMode::Fwht, 2);
    auto a = random_sketch(sketch_dim, 3);
    std::vector<float> o1(output_dim), o2(output_dim);
    p1.apply(a.data(), o1.data());
    p2.apply(a.data(), o2.data());
    bool any_diff = false;
    for (std::uint32_t i = 0; i < output_dim; ++i) {
        if (o1[i] != o2[i]) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);
}

void test_fwht_dense_matrix_matches_apply() {
    // The materialized dense view (entry()) and the FWHT apply path must
    // agree element-wise. This is the structural correctness check that
    // the closed-form P[r,c] = D[c] * (-1)^popcount(sample_[r] & c) /
    // sqrt(output_dim_) matches what the buf-based apply computes.
    constexpr std::uint32_t sketch_dim = 256;
    constexpr std::uint32_t output_dim = 32;
    Projection p(sketch_dim, output_dim, ProjectionMode::Fwht, 13);

    auto v = random_sketch(sketch_dim, 17);
    std::vector<float> via_apply(output_dim);
    p.apply(v.data(), via_apply.data());

    std::vector<float> via_dense(output_dim, 0.0f);
    for (std::uint32_t r = 0; r < output_dim; ++r) {
        double acc = 0.0;
        for (std::uint32_t c = 0; c < sketch_dim; ++c) {
            acc += static_cast<double>(p.entry(r, c)) * static_cast<double>(v[c]);
        }
        via_dense[r] = static_cast<float>(acc);
    }

    for (std::uint32_t r = 0; r < output_dim; ++r) {
        const float diff = std::fabs(via_apply[r] - via_dense[r]);
        // FP tolerance: FWHT does pad_n_-1 add/subs per output, scalar
        // dense path accumulates in double then casts back. Allow a small
        // absolute diff scaled to magnitude.
        const float mag = std::fabs(via_dense[r]) + 1e-3f;
        assert(diff < 1e-3f * mag);
    }
}

void test_fwht_rejects_oversized_output() {
    bool threw = false;
    try {
        // sketch_dim=8 → pad_n=8; output_dim=16 > 8 → reject.
        Projection p(8, 16, ProjectionMode::Fwht, 1);
        (void)p;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_fwht_distortion_within_bound();
    test_fwht_pads_to_power_of_two();
    test_fwht_determinism();
    test_fwht_seed_changes_output();
    test_fwht_dense_matrix_matches_apply();
    test_fwht_rejects_oversized_output();
    return 0;
}
