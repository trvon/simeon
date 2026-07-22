#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "simeon/hasher.hpp"
#include "simeon/simeon.hpp"

using simeon::Encoder;
using simeon::EncoderConfig;
using simeon::FeatureWeighting;
using simeon::NGramMode;
using simeon::ProjectionMode;
using simeon::SketchWeighting;
using simeon::TextNormalization;

namespace {

void expect_invalid_config(EncoderConfig config) {
    try {
        Encoder encoder(std::move(config));
        assert(false && "invalid encoder config must throw");
    } catch (const std::invalid_argument&) {
    }
}

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
    for (float f : v)
        acc += static_cast<double>(f) * static_cast<double>(f);
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
    for (auto& s : texts_s)
        texts.emplace_back(s);

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

void test_ascii_lower_normalization_is_opt_in_and_deterministic() {
    EncoderConfig raw_cfg = highdim_cfg();
    Encoder raw(raw_cfg);
    std::vector<float> raw_upper(raw.output_dim(), 0.0f);
    std::vector<float> raw_lower(raw.output_dim(), 0.0f);
    raw.encode("Alpha BETA", raw_upper.data());
    raw.encode("alpha beta", raw_lower.data());
    assert(raw_upper != raw_lower);

    EncoderConfig folded_cfg = highdim_cfg();
    folded_cfg.text_normalization = TextNormalization::AsciiLower;
    Encoder folded(folded_cfg);
    std::vector<float> folded_upper(folded.output_dim(), 0.0f);
    std::vector<float> folded_lower(folded.output_dim(), 0.0f);
    folded.encode("Alpha BETA", folded_upper.data());
    folded.encode("alpha beta", folded_lower.data());
    assert(folded_upper == folded_lower);
    assert(folded.config().text_normalization == TextNormalization::AsciiLower);
}

void test_signed_sqrt_sketch_weighting_is_exact_and_projectable() {
    EncoderConfig raw_cfg = highdim_cfg();
    raw_cfg.ngram_min = 3;
    raw_cfg.ngram_max = 3;
    raw_cfg.l2_normalize = false;
    Encoder raw(raw_cfg);

    EncoderConfig sqrt_cfg = raw_cfg;
    sqrt_cfg.sketch_weighting = SketchWeighting::SignedSqrt;
    Encoder weighted(sqrt_cfg);

    std::vector<float> raw_values(raw.output_dim(), 0.0f);
    std::vector<float> weighted_values(weighted.output_dim(), 0.0f);
    raw.encode("abcabc abcabc", raw_values.data());
    weighted.encode("abcabc abcabc", weighted_values.data());

    bool saw_repeated_feature = false;
    for (std::size_t i = 0; i < raw_values.size(); ++i) {
        const float value = raw_values[i];
        const float expected =
            std::copysign(std::floor(1024.0f * std::sqrt(std::fabs(value))), value);
        assert(weighted_values[i] == expected);
        if (std::fabs(value) > 2.0f) {
            saw_repeated_feature = true;
            assert(std::fabs(weighted_values[i]) < std::fabs(value) * 1024.0f);
        }
    }
    assert(saw_repeated_feature);

    EncoderConfig projected = projected_cfg();
    projected.sketch_weighting = SketchWeighting::SignedSqrt;
    Encoder projected_encoder(projected);
    std::vector<float> first(projected_encoder.output_dim(), 0.0f);
    std::vector<float> second(projected_encoder.output_dim(), 0.0f);
    projected_encoder.encode("repeat repeat repeat", first.data());
    projected_encoder.encode("repeat repeat repeat", second.data());
    assert(first == second);
    assert(projected_encoder.config().sketch_weighting == SketchWeighting::SignedSqrt);
}

void test_sqrt_tf_is_applied_before_sketch_collisions() {
    const std::vector<std::string> candidates = {"alpha", "beta", "gamma", "delta"};
    std::string first;
    std::string second;
    int sign = 0;
    for (std::size_t i = 0; i < candidates.size() && first.empty(); ++i) {
        const auto left =
            simeon::hash64(candidates[i], highdim_cfg().hash_seed, highdim_cfg().hash);
        const int left_sign = ((left >> 63) & 1ULL) ? -1 : 1;
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            const auto right =
                simeon::hash64(candidates[j], highdim_cfg().hash_seed, highdim_cfg().hash);
            const int right_sign = ((right >> 63) & 1ULL) ? -1 : 1;
            if (left_sign == right_sign) {
                first = candidates[i];
                second = candidates[j];
                sign = left_sign;
                break;
            }
        }
    }
    assert(!first.empty());

    EncoderConfig raw_cfg = highdim_cfg();
    raw_cfg.ngram_mode = NGramMode::WordOnly;
    raw_cfg.sketch_dim = 1;
    raw_cfg.l2_normalize = false;
    const std::string text = first + " " + first + " " + second;

    EncoderConfig feature_cfg = raw_cfg;
    feature_cfg.feature_weighting = FeatureWeighting::SqrtTf;
    Encoder feature_encoder(feature_cfg);
    std::vector<float> feature_value(1, 0.0f);
    feature_encoder.encode(text, feature_value.data());
    const float sqrt_two = std::floor(1024.0f * std::sqrt(2.0f));
    assert(feature_value[0] == static_cast<float>(sign) * (sqrt_two + 1024.0f));

    EncoderConfig post_sketch_cfg = raw_cfg;
    post_sketch_cfg.sketch_weighting = SketchWeighting::SignedSqrt;
    Encoder post_sketch(post_sketch_cfg);
    std::vector<float> post_sketch_value(1, 0.0f);
    post_sketch.encode(text, post_sketch_value.data());
    assert(post_sketch_value[0] ==
           static_cast<float>(sign) * std::floor(1024.0f * std::sqrt(3.0f)));
    assert(feature_value != post_sketch_value);

    EncoderConfig invalid = feature_cfg;
    invalid.sketch_weighting = SketchWeighting::SignedSqrt;
    try {
        Encoder encoder(invalid);
        assert(false && "feature and sketch sqrt weighting must not be combined");
    } catch (const std::invalid_argument&) {
    }
}

void test_compact_retrieval_config_is_frozen_and_deterministic() {
    const EncoderConfig cfg = simeon::compact_retrieval_config();
    assert(cfg.ngram_mode == NGramMode::CharAndWord);
    assert(cfg.ngram_min == 3);
    assert(cfg.ngram_max == 5);
    assert(cfg.text_normalization == TextNormalization::AsciiLower);
    assert(cfg.char_ngram_scope == simeon::CharNGramScope::WordBounded);
    assert(cfg.feature_weighting == FeatureWeighting::Raw);
    assert(cfg.sketch_weighting == SketchWeighting::Raw);
    assert(cfg.sketch_dim == 8192);
    assert(cfg.output_dim == 384);
    assert(cfg.projection == ProjectionMode::Fwht);
    assert(cfg.hash_seed == 0xA5A5A5A5A5A5A5A5ULL);
    assert(cfg.projection_seed == 0xDEADBEEFCAFEBABEULL);
    assert(cfg.l2_normalize);
    assert(!cfg.matryoshka);

    Encoder encoder(cfg);
    assert(encoder.output_dim() == 384);
    std::vector<float> first(encoder.output_dim(), 0.0f);
    std::vector<float> second(encoder.output_dim(), 0.0f);
    encoder.encode("Training-Free Retrieval", first.data());
    encoder.encode("training-free retrieval", second.data());
    assert(first == second);
}

void test_simeon_v1_384_config_is_frozen() {
    const EncoderConfig cfg = simeon::simeon_v1_384_config();
    assert(cfg.ngram_mode == NGramMode::CharOnly);
    assert(cfg.ngram_min == 3);
    assert(cfg.ngram_max == 5);
    assert(cfg.sketch_dim == 4096);
    assert(cfg.output_dim == 384);
    assert(cfg.projection == ProjectionMode::AchlioptasSparse);
    assert(cfg.hash_seed == 0xA5A5A5A5A5A5A5A5ULL);
    assert(cfg.projection_seed == 0xDEADBEEFCAFEBABEULL);
    assert(cfg.l2_normalize);
}

void test_word_bounded_char_ngrams_are_opt_in_and_reach_encoder() {
    EncoderConfig legacy = highdim_cfg();
    legacy.ngram_min = 3;
    legacy.ngram_max = 3;
    Encoder legacy_encoder(legacy);

    EncoderConfig bounded = legacy;
    bounded.char_ngram_scope = simeon::CharNGramScope::WordBounded;
    Encoder bounded_encoder(bounded);
    assert(bounded_encoder.config().char_ngram_scope == simeon::CharNGramScope::WordBounded);

    std::vector<float> legacy_value(legacy_encoder.output_dim(), 0.0f);
    std::vector<float> bounded_value(bounded_encoder.output_dim(), 0.0f);
    legacy_encoder.encode("ab cd", legacy_value.data());
    bounded_encoder.encode("ab cd", bounded_value.data());
    assert(legacy_value != bounded_value);
}

void test_invalid_enum_values_are_rejected() {
    auto config = highdim_cfg();
    config.ngram_mode = static_cast<NGramMode>(255);
    expect_invalid_config(config);

    config = highdim_cfg();
    config.hash = static_cast<simeon::HashFamily>(255);
    expect_invalid_config(config);

    config = highdim_cfg();
    config.projection = static_cast<ProjectionMode>(255);
    config.output_dim = 64;
    expect_invalid_config(config);

    config = highdim_cfg();
    config.text_normalization = static_cast<TextNormalization>(255);
    expect_invalid_config(config);

    config = highdim_cfg();
    config.char_ngram_scope = static_cast<simeon::CharNGramScope>(255);
    expect_invalid_config(config);

    config = highdim_cfg();
    config.feature_weighting = static_cast<FeatureWeighting>(255);
    expect_invalid_config(config);

    config = highdim_cfg();
    config.sketch_weighting = static_cast<SketchWeighting>(255);
    expect_invalid_config(config);
}

} // namespace

int main() {
    test_output_dim_selection();
    test_determinism();
    test_l2_norm();
    test_different_text_different_embedding();
    test_batch_matches_loop();
    test_zero_text_is_zero_vector();
    test_ascii_lower_normalization_is_opt_in_and_deterministic();
    test_signed_sqrt_sketch_weighting_is_exact_and_projectable();
    test_sqrt_tf_is_applied_before_sketch_collisions();
    test_compact_retrieval_config_is_frozen_and_deterministic();
    test_simeon_v1_384_config_is_frozen();
    test_word_bounded_char_ngrams_are_opt_in_and_reach_encoder();
    test_invalid_enum_values_are_rejected();
    return 0;
}
