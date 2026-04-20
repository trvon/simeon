#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "simeon/projection.hpp"

using simeon::Projection;
using simeon::ProjectionMode;

namespace {

void test_none_identity() {
    Projection p(8, 8, ProjectionMode::None, 0);
    std::vector<std::int32_t> sketch = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> out(8, 0.0f);
    p.apply(sketch.data(), out.data());
    for (std::size_t i = 0; i < 8; ++i) {
        assert(out[i] == static_cast<float>(sketch[i]));
    }
}

void test_achlioptas_seed_determinism() {
    Projection a(64, 16, ProjectionMode::AchlioptasSparse, 42);
    Projection b(64, 16, ProjectionMode::AchlioptasSparse, 42);
    for (std::uint32_t r = 0; r < 16; ++r) {
        for (std::uint32_t c = 0; c < 64; ++c) {
            assert(a.entry(r, c) == b.entry(r, c));
        }
    }
}

void test_achlioptas_values() {
    Projection p(1024, 32, ProjectionMode::AchlioptasSparse, 7);
    const float scale = std::sqrt(3.0f);
    int count_neg = 0, count_zero = 0, count_pos = 0;
    for (std::uint32_t r = 0; r < 32; ++r) {
        for (std::uint32_t c = 0; c < 1024; ++c) {
            const float v = p.entry(r, c);
            assert(v == 0.0f || v == scale || v == -scale);
            if (v < 0)
                ++count_neg;
            else if (v > 0)
                ++count_pos;
            else
                ++count_zero;
        }
    }
    // Expected: 1/6, 2/3, 1/6. With 32768 samples, wide tolerance.
    const int total = 32 * 1024;
    assert(std::abs(count_zero - total * 2 / 3) < total / 10);
    assert(std::abs(count_neg - total / 6) < total / 10);
    assert(std::abs(count_pos - total / 6) < total / 10);
}

void test_apply_nonzero() {
    Projection p(16, 4, ProjectionMode::AchlioptasSparse, 99);
    std::vector<std::int32_t> sketch(16, 1);
    std::vector<float> out(4, 0.0f);
    p.apply(sketch.data(), out.data());
    // Not all outputs should be exactly zero.
    float max_abs = 0.0f;
    for (float v : out) max_abs = std::max(max_abs, std::fabs(v));
    assert(max_abs > 0.0f);
}

void test_output_dim_validation() {
    bool threw = false;
    try {
        Projection p(16, 0, ProjectionMode::AchlioptasSparse, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    test_none_identity();
    test_achlioptas_seed_determinism();
    test_achlioptas_values();
    test_apply_nonzero();
    test_output_dim_validation();
    return 0;
}
