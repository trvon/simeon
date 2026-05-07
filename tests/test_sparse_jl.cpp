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

// Build a random integer "sketch" of given dimension. Mirrors the int32
// sketch the encoder feeds Projection::apply().
std::vector<std::int32_t> random_sketch(std::uint32_t dim, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> dist(-32, 32);
    std::vector<std::int32_t> v(dim);
    for (auto& x : v)
        x = dist(rng);
    return v;
}

float l2(const float* p, std::uint32_t n) {
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i)
        s += static_cast<double>(p[i]) * p[i];
    return static_cast<float>(std::sqrt(s));
}

float l2_sketch(const std::int32_t* p, std::uint32_t n) {
    double s = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const double f = static_cast<double>(p[i]);
        s += f * f;
    }
    return static_cast<float>(std::sqrt(s));
}

void test_sparse_jl_distortion_within_eps() {
    constexpr std::uint32_t sketch_dim = 1024;
    constexpr std::uint32_t output_dim = 256;
    constexpr float eps = 0.10f;
    constexpr std::uint32_t pairs = 60;

    Projection p(sketch_dim, output_dim, ProjectionMode::SparseJL, /*seed*/ 42, eps);

    std::vector<float> proj_a(output_dim), proj_b(output_dim);
    std::vector<float> diff_in(sketch_dim); // float view of input difference
    std::vector<float> proj_diff(output_dim);

    double sum_sq_distortion = 0.0;
    std::uint32_t in_bound = 0;
    for (std::uint32_t k = 0; k < pairs; ++k) {
        auto a = random_sketch(sketch_dim, 1234 + k);
        auto b = random_sketch(sketch_dim, 9876 + k);
        std::vector<std::int32_t> d(sketch_dim);
        for (std::uint32_t i = 0; i < sketch_dim; ++i)
            d[i] = a[i] - b[i];

        const float in_norm = l2_sketch(d.data(), sketch_dim);
        if (in_norm == 0.0f)
            continue;

        p.apply(d.data(), proj_diff.data());
        const float out_norm = l2(proj_diff.data(), output_dim);

        const double ratio = static_cast<double>(out_norm) / static_cast<double>(in_norm);
        const double dist = std::fabs(ratio - 1.0);
        sum_sq_distortion += dist * dist;
        if (dist <= static_cast<double>(eps))
            ++in_bound;
    }
    // JL is a probabilistic guarantee — most pairs (≥ 80% with eps=0.10 and
    // s=ceil(0.10 * 1024) = 103 nonzeros per row) should land within bound.
    // RMS distortion should also be O(eps).
    const double rms = std::sqrt(sum_sq_distortion / pairs);
    assert(rms < 0.20);
    assert(in_bound >= pairs * 7 / 10);
}

void test_sparse_jl_column_has_exact_s_nonzeros() {
    constexpr std::uint32_t sketch_dim = 256;
    constexpr std::uint32_t output_dim = 64;
    constexpr float eps = 0.25f;
    // Kane–Nelson column-sparsity: each input column gets s = ceil(eps*k).
    constexpr std::uint32_t expected_s = 16; // ceil(0.25 * 64)

    Projection p(sketch_dim, output_dim, ProjectionMode::SparseJL, /*seed*/ 7, eps);

    for (std::uint32_t col = 0; col < sketch_dim; ++col) {
        std::uint32_t nonzero = 0;
        for (std::uint32_t row = 0; row < output_dim; ++row) {
            if (p.entry(row, col) != 0.0f)
                ++nonzero;
        }
        assert(nonzero == expected_s);
    }
}

void test_sparse_jl_determinism() {
    constexpr std::uint32_t sketch_dim = 512;
    constexpr std::uint32_t output_dim = 64;
    Projection p1(sketch_dim, output_dim, ProjectionMode::SparseJL, 99, 0.10f);
    Projection p2(sketch_dim, output_dim, ProjectionMode::SparseJL, 99, 0.10f);
    auto a = random_sketch(sketch_dim, 5);
    std::vector<float> o1(output_dim), o2(output_dim);
    p1.apply(a.data(), o1.data());
    p2.apply(a.data(), o2.data());
    for (std::uint32_t i = 0; i < output_dim; ++i) {
        assert(o1[i] == o2[i]);
    }
}

void test_sparse_jl_seed_changes_output() {
    constexpr std::uint32_t sketch_dim = 512;
    constexpr std::uint32_t output_dim = 64;
    Projection p1(sketch_dim, output_dim, ProjectionMode::SparseJL, 1, 0.10f);
    Projection p2(sketch_dim, output_dim, ProjectionMode::SparseJL, 2, 0.10f);
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

void test_sparse_jl_rejects_invalid_eps() {
    bool threw = false;
    try {
        Projection p(64, 16, ProjectionMode::SparseJL, 1, 0.0f);
        (void)p;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        Projection p(64, 16, ProjectionMode::SparseJL, 1, 1.5f);
        (void)p;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_sparse_jl_distortion_within_eps();
    test_sparse_jl_column_has_exact_s_nonzeros();
    test_sparse_jl_determinism();
    test_sparse_jl_seed_changes_output();
    test_sparse_jl_rejects_invalid_eps();
    return 0;
}
