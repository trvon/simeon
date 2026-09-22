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
    for (float v : out)
        max_abs = std::max(max_abs, std::fabs(v));
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

// apply() must equal the exact integer sum (pos - neg) over the matrix's +/-1 entries, scaled once,
// for both short-text sketches (few non-zero buckets) and dense ones. Any traversal order is fine
// because the sum is exact, so this pins the output bit for bit.
void test_achlioptas_apply_matches_exact_reference() {
    constexpr std::uint32_t kSketch = 4096;
    constexpr std::uint32_t kOut = 384;
    Projection p(kSketch, kOut, ProjectionMode::AchlioptasSparse, 0x5eed);
    const float scale = std::sqrt(3.0f) * (1.0f / std::sqrt(static_cast<float>(kOut)));

    std::mt19937 rng(11);
    std::uniform_int_distribution<std::uint32_t> pick_col(0, kSketch - 1);
    std::uniform_int_distribution<std::int32_t> pick_count(-3, 9);
    for (const std::uint32_t nonzeros : {0u, 1u, 350u, kSketch}) {
        std::vector<std::int32_t> sketch(kSketch, 0);
        if (nonzeros == kSketch) {
            for (auto& v : sketch)
                v = pick_count(rng);
        } else {
            for (std::uint32_t i = 0; i < nonzeros; ++i)
                sketch[pick_col(rng)] += pick_count(rng) | 1;
        }
        std::vector<float> out(kOut, -1.0f);
        p.apply(sketch.data(), out.data());
        for (std::uint32_t r = 0; r < kOut; ++r) {
            std::int64_t acc = 0;
            for (std::uint32_t c = 0; c < kSketch; ++c) {
                const float w = p.entry(r, c);
                if (w > 0.0f)
                    acc += sketch[c];
                else if (w < 0.0f)
                    acc -= sketch[c];
            }
            assert(out[r] == static_cast<float>(acc) * scale);
        }
    }
}

} // namespace

int main() {
    test_none_identity();
    test_achlioptas_seed_determinism();
    test_achlioptas_values();
    test_apply_nonzero();
    test_achlioptas_apply_matches_exact_reference();
    test_output_dim_validation();
    return 0;
}
