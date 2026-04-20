#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <string_view>
#include <vector>

#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"

namespace {

std::vector<float> make_random(std::uint32_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (std::uint32_t i = 0; i < n; ++i)
        v[i] = dist(rng);
    return v;
}

[[maybe_unused]] void check_close(const std::vector<float>& a, const std::vector<float>& b,
                                  float tol) {
    assert(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        assert(d <= tol);
    }
}

void test_normalize_parity_dim(std::uint32_t n) {
    auto src = make_random(n, 0xC0FFEE ^ n);

    auto a = src;
    simeon::simd::l2_normalize_scalar(a.data(), n);

#if defined(SIMEON_HAS_NEON)
    auto b = src;
    simeon::simd::l2_normalize_neon(b.data(), n);
    check_close(a, b, 1e-5f);
#endif

#if defined(SIMEON_HAS_AVX2)
    auto c = src;
    simeon::simd::l2_normalize_avx2(c.data(), n);
    check_close(a, c, 1e-5f);
#endif
}

void test_normalize_dimensions() {
    // Includes dim < 4 (tail-only path) and dims off the SIMD width
    // (4 floats for NEON, 8 for AVX2) to exercise scalar tail loops.
    for (std::uint32_t n :
         {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 11u, 15u, 16u, 17u, 33u, 128u, 257u, 1024u, 4096u}) {
        test_normalize_parity_dim(n);
    }
}

void test_zero_vector() {
    std::vector<float> v(16, 0.0f);
    const float inv = simeon::simd::l2_normalize_scalar(v.data(), 16);
    assert(inv == 0.0f);
    for (float f : v)
        assert(f == 0.0f);
}

// All SIMD variants must also handle the zero vector identically (no NaN/Inf,
// no division-by-zero, output stays zero). Hits the early-return branch.
void test_zero_vector_all_tiers() {
    for (std::uint32_t n : {1u, 8u, 16u, 17u}) {
        std::vector<float> z(n, 0.0f);

        auto a = z;
        const float ia = simeon::simd::l2_normalize_scalar(a.data(), n);
        assert(ia == 0.0f);
        for (float f : a)
            assert(f == 0.0f);

#if defined(SIMEON_HAS_NEON)
        auto b = z;
        const float ib = simeon::simd::l2_normalize_neon(b.data(), n);
        assert(ib == 0.0f);
        for (float f : b)
            assert(f == 0.0f);
#endif
#if defined(SIMEON_HAS_AVX2)
        auto c = z;
        const float ic = simeon::simd::l2_normalize_avx2(c.data(), n);
        assert(ic == 0.0f);
        for (float f : c)
            assert(f == 0.0f);
#endif
    }
}

void test_unit_norm_postcondition() {
    auto v = make_random(64, 1234);
    simeon::simd::l2_normalize(v.data(), 64);
    double acc = 0.0;
    for (float f : v)
        acc += static_cast<double>(f) * static_cast<double>(f);
    assert(std::fabs(std::sqrt(acc) - 1.0) < 1e-5);
}

// Mixed-magnitude vectors stress the accumulator: small values can be lost
// when summed with large ones in single precision. The scalar reference uses
// a double accumulator; the SIMD variants accumulate in float lanes. Use a
// looser relative tolerance per-element.
void test_normalize_extreme_magnitudes() {
    std::vector<std::uint32_t> sizes = {16u, 17u, 33u, 128u, 257u};
    for (std::uint32_t n : sizes) {
        std::mt19937 rng(0xDEADBEEF ^ n);
        std::uniform_real_distribution<float> sign(-1.0f, 1.0f);
        std::vector<float> src(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            // Alternate large and small to stress lane-wise summation.
            const float magnitude = (i % 2 == 0) ? 1e6f : 1e-3f;
            src[i] = (sign(rng) > 0 ? 1.0f : -1.0f) * magnitude;
        }

        auto a = src;
        simeon::simd::l2_normalize_scalar(a.data(), n);
        // Postcondition: unit norm via double accumulation.
        double acc = 0.0;
        for (float f : a)
            acc += static_cast<double>(f) * static_cast<double>(f);
        assert(std::fabs(std::sqrt(acc) - 1.0) < 1e-4);

#if defined(SIMEON_HAS_NEON)
        auto b = src;
        simeon::simd::l2_normalize_neon(b.data(), n);
        // Compare rescaled: each variant produces its own unit-norm vector,
        // so element-wise diff is the right metric. Tolerance is generous
        // because float-lane summation order differs.
        for (std::uint32_t i = 0; i < n; ++i) {
            const float scale = std::max(1e-6f, std::max(std::fabs(a[i]), std::fabs(b[i])));
            assert(std::fabs(a[i] - b[i]) <= 1e-3f * scale);
        }
#endif
#if defined(SIMEON_HAS_AVX2)
        auto c = src;
        simeon::simd::l2_normalize_avx2(c.data(), n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const float scale = std::max(1e-6f, std::max(std::fabs(a[i]), std::fabs(c[i])));
            assert(std::fabs(a[i] - c[i]) <= 1e-3f * scale);
        }
#endif
    }
}

void test_dot_parity_dim(std::uint32_t n) {
    auto a = make_random(n, 0xBADCAFEu ^ n);
    auto b = make_random(n, 0xFEEDFACEu ^ n);
    [[maybe_unused]] const float ref = simeon::simd::dot_scalar(a.data(), b.data(), n);
#if defined(SIMEON_HAS_NEON)
    const float v = simeon::simd::dot_neon(a.data(), b.data(), n);
    const float scale = std::max(1.0f, std::fabs(ref));
    assert(std::fabs(v - ref) <= 1e-4f * scale);
#endif
#if defined(SIMEON_HAS_AVX2)
    const float v = simeon::simd::dot_avx2(a.data(), b.data(), n);
    const float scale = std::max(1.0f, std::fabs(ref));
    assert(std::fabs(v - ref) <= 1e-4f * scale);
#endif
}

void test_dot_dimensions() {
    for (std::uint32_t n :
         {1u, 2u, 3u, 4u, 7u, 8u, 9u, 15u, 16u, 17u, 33u, 128u, 257u, 384u, 768u, 1024u, 4096u}) {
        test_dot_parity_dim(n);
    }
}

void test_dot_zero() {
    std::vector<float> z(384, 0.0f);
    auto v = make_random(384, 42);
    assert(simeon::simd::dot(z.data(), v.data(), 384) == 0.0f);
    assert(simeon::simd::dot(v.data(), z.data(), 384) == 0.0f);
    assert(simeon::simd::dot(z.data(), z.data(), 384) == 0.0f);
}

void test_dot_self_unit_norm() {
    auto v = make_random(384, 7);
    simeon::simd::l2_normalize(v.data(), 384);
    const float self = simeon::simd::dot(v.data(), v.data(), 384);
    assert(std::fabs(self - 1.0f) < 1e-4f);
}

// add_vec / scale_vec / saxpy parity.
//
// add_vec and scale_vec have no reduction — scalar and SIMD impls must agree
// bit-for-bit on IEEE-754 hardware (same ops, same order). We still apply a
// small tolerance because FMA fusion (saxpy) skips the intermediate rounding
// that the scalar `dst[i] += alpha * src[i]` performs; for add/scale the
// tolerance is effectively zero.
void test_add_vec_parity_dim(std::uint32_t n) {
    auto src = make_random(n, 0xADD1ADD1u ^ n);
    auto init = make_random(n, 0xFADECAFEu ^ n);

    auto a = init;
    simeon::simd::add_vec_scalar(a.data(), src.data(), n);

#if defined(SIMEON_HAS_NEON)
    auto b = init;
    simeon::simd::add_vec_neon(b.data(), src.data(), n);
    check_close(a, b, 1e-6f);
#endif
#if defined(SIMEON_HAS_AVX2)
    auto c = init;
    simeon::simd::add_vec_avx2(c.data(), src.data(), n);
    check_close(a, c, 1e-6f);
#endif
}

void test_scale_vec_parity_dim(std::uint32_t n) {
    auto w = make_random(n, 0x5CA1E000u ^ n);
    auto init = make_random(n, 0xBEE5u ^ n);

    auto a = init;
    simeon::simd::scale_vec_scalar(a.data(), w.data(), n);

#if defined(SIMEON_HAS_NEON)
    auto b = init;
    simeon::simd::scale_vec_neon(b.data(), w.data(), n);
    check_close(a, b, 1e-6f);
#endif
#if defined(SIMEON_HAS_AVX2)
    auto c = init;
    simeon::simd::scale_vec_avx2(c.data(), w.data(), n);
    check_close(a, c, 1e-6f);
#endif
}

void test_saxpy_parity_dim(std::uint32_t n) {
    auto src = make_random(n, 0x5A5A5A5Au ^ n);
    auto init = make_random(n, 0xA5A5A5A5u ^ n);
    const float alpha = 0.375f; // exactly representable

    auto a = init;
    simeon::simd::saxpy_scalar(a.data(), src.data(), alpha, n);

#if defined(SIMEON_HAS_NEON)
    auto b = init;
    simeon::simd::saxpy_neon(b.data(), src.data(), alpha, n);
    // FMA rounds once; scalar rounds twice. Use a small relative tolerance.
    for (std::uint32_t i = 0; i < n; ++i) {
        const float scale = std::max(1.0f, std::fabs(a[i]));
        assert(std::fabs(a[i] - b[i]) <= 1e-5f * scale);
    }
#endif
#if defined(SIMEON_HAS_AVX2)
    auto c = init;
    simeon::simd::saxpy_avx2(c.data(), src.data(), alpha, n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float scale = std::max(1.0f, std::fabs(a[i]));
        assert(std::fabs(a[i] - c[i]) <= 1e-5f * scale);
    }
#endif
}

void test_elementwise_dimensions() {
    // Tail exercise: 4-lane (NEON), 8-lane (AVX2), 16-lane (AVX2 unrolled).
    for (std::uint32_t n : {1u,  2u,  3u,  4u,  5u,   7u,   8u,   9u,   15u,   16u,  17u,
                            31u, 32u, 63u, 64u, 127u, 128u, 256u, 513u, 1024u, 4096u}) {
        test_add_vec_parity_dim(n);
        test_scale_vec_parity_dim(n);
        test_saxpy_parity_dim(n);
    }
}

// Dispatcher round-trip: the public simd::add_vec / scale_vec / saxpy should
// produce the same output the per-tier impls do on the active host.
void test_dispatchers_round_trip() {
    const std::uint32_t n = 257;
    auto src = make_random(n, 0xD15D150Au);
    auto init = make_random(n, 0xC001C001u);

    auto ref = init;
    simeon::simd::add_vec_scalar(ref.data(), src.data(), n);
    auto via = init;
    simeon::simd::add_vec(via.data(), src.data(), n);
    check_close(ref, via, 1e-6f);

    auto w = make_random(n, 0x5CA1E05Eu);
    auto ref2 = init;
    simeon::simd::scale_vec_scalar(ref2.data(), w.data(), n);
    auto via2 = init;
    simeon::simd::scale_vec(via2.data(), w.data(), n);
    check_close(ref2, via2, 1e-6f);

    auto ref3 = init;
    simeon::simd::saxpy_scalar(ref3.data(), src.data(), 0.5f, n);
    auto via3 = init;
    simeon::simd::saxpy(via3.data(), src.data(), 0.5f, n);
    // FMA vs scalar: small tolerance.
    for (std::uint32_t i = 0; i < n; ++i) {
        const float scale = std::max(1.0f, std::fabs(ref3[i]));
        assert(std::fabs(ref3[i] - via3[i]) <= 1e-5f * scale);
    }
}

// Pin active_simd_tier() to what the host actually compiled with. simd.hpp
// dispatches on this value, so a wrong answer would silently route to the
// scalar path even on a SIMD-capable host.
void test_active_tier_matches_host() {
    const auto tier = simeon::active_simd_tier();
#if defined(SIMEON_HAS_NEON)
    assert(tier == simeon::SimdTier::Neon);
#elif defined(SIMEON_HAS_AVX2)
    assert(tier == simeon::SimdTier::Avx2);
#else
    assert(tier == simeon::SimdTier::Scalar);
#endif

#if defined(SIMEON_HAS_NEON)
    assert(tier == simeon::SimdTier::Neon);
#elif defined(SIMEON_HAS_AVX2)
    assert(tier == simeon::SimdTier::Avx2);
#else
    assert(tier == simeon::SimdTier::Scalar);
#endif
    // simd_tier_name must round-trip every defined tier.
    assert(std::string_view(simeon::simd_tier_name(simeon::SimdTier::Scalar)) == "scalar");
    assert(std::string_view(simeon::simd_tier_name(simeon::SimdTier::Neon)) == "neon");
    assert(std::string_view(simeon::simd_tier_name(simeon::SimdTier::Avx2)) == "avx2");
}

void test_dot_dispatch_matches_public_tier() {
    auto a = make_random(384, 0xABCD1234u);
    auto b = make_random(384, 0x1234ABCDu);
    const float via_public = simeon::simd::dot(a.data(), b.data(), 384);
#if defined(SIMEON_HAS_NEON)
    const float via_tier = simeon::simd::dot_neon(a.data(), b.data(), 384);
#elif defined(SIMEON_HAS_AVX2)
    const float via_tier = simeon::simd::dot_avx2(a.data(), b.data(), 384);
#else
    const float via_tier = simeon::simd::dot_scalar(a.data(), b.data(), 384);
#endif
    assert(std::fabs(via_public - via_tier) <= 1e-5f * std::max(1.0f, std::fabs(via_tier)));
}

} // namespace

int main() {
    test_normalize_dimensions();
    test_zero_vector();
    test_zero_vector_all_tiers();
    test_unit_norm_postcondition();
    test_normalize_extreme_magnitudes();
    test_dot_dimensions();
    test_dot_zero();
    test_dot_self_unit_norm();
    test_elementwise_dimensions();
    test_dispatchers_round_trip();
    test_active_tier_matches_host();
    test_dot_dispatch_matches_public_tier();
    return 0;
}
