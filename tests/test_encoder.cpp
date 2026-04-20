#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/simeon.hpp"

using simeon::Encoder;
using simeon::EncoderConfig;
using simeon::NGramMode;
using simeon::ProjectionMode;

namespace {

EncoderConfig highdim_cfg() {
    EncoderConfig cfg;
    cfg.ngram_mode = NGramMode::CharOnly;
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.sketch_dim = 1024;
    cfg.projection = ProjectionMode::None;
    cfg.l2_normalize = true;
    return cfg;
}

EncoderConfig projected_cfg() {
    EncoderConfig cfg = highdim_cfg();
    cfg.sketch_dim = 2048;
    cfg.output_dim = 128;
    cfg.projection = ProjectionMode::AchlioptasSparse;
    return cfg;
}

void test_output_dim_selection() {
    Encoder a(highdim_cfg());
    assert(a.output_dim() == 1024);

    Encoder b(projected_cfg());
    assert(b.output_dim() == 128);
}

void test_determinism() {
    Encoder e(highdim_cfg());
    std::vector<float> x(e.output_dim(), 0.0f);
    std::vector<float> y(e.output_dim(), 0.0f);
    e.encode("The quick brown fox jumps over the lazy dog", x.data());
    e.encode("The quick brown fox jumps over the lazy dog", y.data());
    for (std::uint32_t i = 0; i < e.output_dim(); ++i) {
        assert(x[i] == y[i]);
    }
}

void test_l2_norm() {
    Encoder e(projected_cfg());
    std::vector<float> v(e.output_dim(), 0.0f);
    e.encode("hello world, this is a moderately long piece of text to embed", v.data());
    double acc = 0.0;
    for (float f : v) acc += static_cast<double>(f) * static_cast<double>(f);
    const double norm = std::sqrt(acc);
    assert(std::fabs(norm - 1.0) < 1e-5);
}

void test_different_text_different_embedding() {
    Encoder e(projected_cfg());
    std::vector<float> a(e.output_dim(), 0.0f);
    std::vector<float> b(e.output_dim(), 0.0f);
    e.encode("completely different content alpha", a.data());
    e.encode("entirely unrelated other content beta", b.data());
    bool any_diff = false;
    for (std::uint32_t i = 0; i < e.output_dim(); ++i) {
        if (a[i] != b[i]) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);
}

void test_batch_matches_loop() {
    Encoder e(projected_cfg());
    const std::uint32_t d = e.output_dim();
    const std::vector<std::string> texts_s = {"alpha bravo", "charlie delta echo", "foxtrot"};
    std::vector<std::string_view> texts;
    for (auto& s : texts_s) texts.emplace_back(s);

    std::vector<float> ref(texts.size() * d, 0.0f);
    for (std::size_t i = 0; i < texts.size(); ++i) {
        e.encode(texts[i], ref.data() + i * d);
    }

    std::vector<float> batched(texts.size() * d, 0.0f);
    e.encode_batch(std::span<const std::string_view>(texts.data(), texts.size()), batched.data());

    for (std::size_t i = 0; i < ref.size(); ++i) {
        assert(ref[i] == batched[i]);
    }
}

void test_zero_text_is_zero_vector() {
    Encoder e(projected_cfg());
    std::vector<float> v(e.output_dim(), 0.0f);
    e.encode("", v.data());
    for (float f : v) {
        assert(f == 0.0f);
    }
}

}  // namespace

int main() {
    test_output_dim_selection();
    test_determinism();
    test_l2_norm();
    test_different_text_different_embedding();
    test_batch_matches_loop();
    test_zero_text_is_zero_vector();
    return 0;
}
