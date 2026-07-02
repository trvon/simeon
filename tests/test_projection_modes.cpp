// Coverage for ProjectionMode::DenseGaussian and ProjectionMode::VerySparse.
// The Achlioptas mode is covered by test_projection.cpp; this file targets the
// other two and the entry()/apply() consistency they share.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "simeon/projection.hpp"

using simeon::Projection;
using simeon::ProjectionMode;

namespace {

std::vector<std::int32_t> make_sketch(std::uint32_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::int32_t> dist(-32, 32);
    std::vector<std::int32_t> v(n);
    for (auto& x : v)
        x = dist(rng);
    return v;
}

// Reference matrix-vector multiply using entry(); used to validate apply().
// For modes that scale by inv_scale_ = 1/sqrt(output_dim) the test multiplies
// the reference by that factor explicitly.
std::vector<float> reference_apply(const Projection& p, const std::int32_t* sketch,
                                   std::uint32_t sketch_dim, std::uint32_t output_dim,
                                   bool apply_inv_scale) {
    std::vector<float> out(output_dim, 0.0f);
    const float inv_scale =
        apply_inv_scale ? (1.0f / std::sqrt(static_cast<float>(output_dim))) : 1.0f;
    for (std::uint32_t r = 0; r < output_dim; ++r) {
        double acc = 0.0;
        for (std::uint32_t c = 0; c < sketch_dim; ++c) {
            acc += static_cast<double>(p.entry(r, c)) * static_cast<double>(sketch[c]);
        }
        out[r] = static_cast<float>(acc) * inv_scale;
    }
    return out;
}

void assert_close(const std::vector<float>& a, const std::vector<float>& b, float rel_tol) {
    assert(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float scale = std::max(1.0f, std::max(std::fabs(a[i]), std::fabs(b[i])));
        const float diff = std::fabs(a[i] - b[i]);
        assert(diff <= rel_tol * scale);
    }
}

void test_dense_gaussian_zero_input() {
    Projection p(64, 16, ProjectionMode::DenseGaussian, 0xBEEF);
    std::vector<std::int32_t> sketch(64, 0);
    std::vector<float> out(16, 7.0f);
    p.apply(sketch.data(), out.data());
    for (float v : out)
        assert(v == 0.0f);
}

void test_dense_gaussian_determinism_apply() {
    Projection a(128, 32, ProjectionMode::DenseGaussian, 12345);
    Projection b(128, 32, ProjectionMode::DenseGaussian, 12345);
    auto sketch = make_sketch(128, 1);
    std::vector<float> oa(32, 0.0f), ob(32, 0.0f);
    a.apply(sketch.data(), oa.data());
    b.apply(sketch.data(), ob.data());
    for (std::size_t i = 0; i < 32; ++i)
        assert(oa[i] == ob[i]);
}

void test_dense_gaussian_seed_changes_output() {
    auto sketch = make_sketch(128, 1);
    Projection a(128, 32, ProjectionMode::DenseGaussian, 1);
    Projection b(128, 32, ProjectionMode::DenseGaussian, 2);
    std::vector<float> oa(32, 0.0f), ob(32, 0.0f);
    a.apply(sketch.data(), oa.data());
    b.apply(sketch.data(), ob.data());
    bool any_diff = false;
    for (std::size_t i = 0; i < 32; ++i)
        if (oa[i] != ob[i]) {
            any_diff = true;
            break;
        }
    assert(any_diff);
}

// Validate the NEON 4-row unroll in projection.cpp:170-207 against the entry()
// reference. Uses sketch_dim that is *not* a multiple of 8 to also exercise
// the trailing scalar tail.
void test_dense_gaussian_apply_matches_reference_small() {
    const std::uint32_t sketch_dim = 67; // not a multiple of 8 → tail loop runs
    const std::uint32_t output_dim = 12; // < 4 → ensure single-row path covered
    Projection p(sketch_dim, output_dim, ProjectionMode::DenseGaussian, 0xC0DE);
    auto sketch = make_sketch(sketch_dim, 99);
    std::vector<float> got(output_dim, 0.0f);
    p.apply(sketch.data(), got.data());
    auto ref = reference_apply(p, sketch.data(), sketch_dim, output_dim, true);
    assert_close(got, ref, 1e-4f);
}

// Larger case to exercise the NEON 4-row unroll (output_dim divisible by 4).
// Tolerance is looser because float accumulation order differs between the
// reference (sequential double accumulator) and the SIMD path.
void test_dense_gaussian_apply_matches_reference_large() {
    const std::uint32_t sketch_dim = 1024;
    const std::uint32_t output_dim = 16;
    Projection p(sketch_dim, output_dim, ProjectionMode::DenseGaussian, 0xFEED);
    auto sketch = make_sketch(sketch_dim, 17);
    std::vector<float> got(output_dim, 0.0f);
    p.apply(sketch.data(), got.data());
    auto ref = reference_apply(p, sketch.data(), sketch_dim, output_dim, true);
    assert_close(got, ref, 1e-3f);
}

void test_very_sparse_zero_input() {
    Projection p(256, 32, ProjectionMode::VerySparse, 0xABCD);
    std::vector<std::int32_t> sketch(256, 0);
    std::vector<float> out(32, 99.0f);
    p.apply(sketch.data(), out.data());
    for (float v : out)
        assert(v == 0.0f);
}

void test_very_sparse_determinism() {
    Projection a(512, 64, ProjectionMode::VerySparse, 7);
    Projection b(512, 64, ProjectionMode::VerySparse, 7);
    for (std::uint32_t r = 0; r < 64; ++r) {
        for (std::uint32_t c = 0; c < 512; ++c) {
            assert(a.entry(r, c) == b.entry(r, c));
        }
    }
}

void test_very_sparse_apply_matches_reference() {
    const std::uint32_t sketch_dim = 1024;
    const std::uint32_t output_dim = 64;
    Projection p(sketch_dim, output_dim, ProjectionMode::VerySparse, 0xF00D);
    auto sketch = make_sketch(sketch_dim, 5);
    std::vector<float> got(output_dim, 0.0f);
    p.apply(sketch.data(), got.data());
    auto ref = reference_apply(p, sketch.data(), sketch_dim, output_dim, true);
    assert_close(got, ref, 1e-4f);
}

// Per Li (2006): density 1/s where s = sqrt(sketch_dim). With sketch_dim=4096
// that's ~1/64 ≈ 1.6%. Bound generously to account for stochastic variation
// across the projection matrix.
void test_very_sparse_density() {
    const std::uint32_t sketch_dim = 4096;
    const std::uint32_t output_dim = 64;
    Projection p(sketch_dim, output_dim, ProjectionMode::VerySparse, 99);
    std::size_t nonzero = 0;
    for (std::uint32_t r = 0; r < output_dim; ++r) {
        for (std::uint32_t c = 0; c < sketch_dim; ++c) {
            if (p.entry(r, c) != 0.0f)
                ++nonzero;
        }
    }
    const double total = static_cast<double>(output_dim) * sketch_dim;
    const double observed = static_cast<double>(nonzero) / total;
    const double expected = 1.0 / std::sqrt(static_cast<double>(sketch_dim));
    // Allow ±50% relative slack — small matrix, very sparse → high variance.
    assert(observed > expected * 0.5);
    assert(observed < expected * 1.5);
}

// VerySparse entries should only take values from { -sqrt(s), 0, +sqrt(s) }
// where s = sqrt(sketch_dim).
void test_very_sparse_value_set() {
    const std::uint32_t sketch_dim = 256;
    const std::uint32_t output_dim = 16;
    Projection p(sketch_dim, output_dim, ProjectionMode::VerySparse, 42);
    const float s = std::sqrt(static_cast<float>(sketch_dim));
    const float magnitude = std::sqrt(s);
    for (std::uint32_t r = 0; r < output_dim; ++r) {
        for (std::uint32_t c = 0; c < sketch_dim; ++c) {
            const float v = p.entry(r, c);
            assert(v == 0.0f || v == magnitude || v == -magnitude);
        }
    }
}

// All projection modes reject output_dim==0 at construction (covered for
// Achlioptas in test_projection.cpp). Repeat for the other two so the
// validation path doesn't quietly drop a mode.
void test_other_modes_reject_zero_output_dim() {
    bool threw_g = false;
    try {
        Projection p(16, 0, ProjectionMode::DenseGaussian, 1);
    } catch (const std::invalid_argument&) {
        threw_g = true;
    }
    assert(threw_g);

    bool threw_v = false;
    try {
        Projection p(16, 0, ProjectionMode::VerySparse, 1);
    } catch (const std::invalid_argument&) {
        threw_v = true;
    }
    assert(threw_v);
}

// Projection::None should ignore output_dim (constructor allows 0) and act as
// an identity copy; entry(r,c) is the Kronecker delta. Covered briefly in
// test_projection.cpp test_none_identity, but assert the entry() path here.
void test_none_entry_is_identity() {
    Projection p(8, 0, ProjectionMode::None, 0);
    for (std::uint32_t r = 0; r < 8; ++r) {
        for (std::uint32_t c = 0; c < 8; ++c) {
            const float expected = (r == c) ? 1.0f : 0.0f;
            assert(p.entry(r, c) == expected);
        }
    }
}

} // namespace

int main() {
    test_dense_gaussian_zero_input();
    test_dense_gaussian_determinism_apply();
    test_dense_gaussian_seed_changes_output();
    test_dense_gaussian_apply_matches_reference_small();
    test_dense_gaussian_apply_matches_reference_large();
    test_very_sparse_zero_input();
    test_very_sparse_determinism();
    test_very_sparse_apply_matches_reference();
    test_very_sparse_density();
    test_very_sparse_value_set();
    test_other_modes_reject_zero_output_dim();
    test_none_entry_is_identity();
    return 0;
}
