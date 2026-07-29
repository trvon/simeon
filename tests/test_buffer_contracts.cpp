#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "simeon/minhash.hpp"
#include "simeon/projection.hpp"
#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"

namespace {

template <typename Function> void expect_invalid_argument(Function&& function) {
    bool threw = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_encoder_contracts() {
    simeon::Encoder encoder(simeon::compact_retrieval_config());
    std::vector<float> pointer_output(encoder.output_dim());
    std::vector<float> span_output(encoder.output_dim());

    encoder.encode("buffer contract", pointer_output.data());
    encoder.encode("buffer contract", std::span<float>(span_output));
    assert(pointer_output == span_output);

    expect_invalid_argument([&] { encoder.encode("buffer contract", nullptr); });
    expect_invalid_argument(
        [&] { encoder.encode("buffer contract", std::span<float>(span_output).first(1)); });

    const std::array<std::string_view, 2> texts = {"first", "second"};
    std::vector<float> batch(texts.size() * encoder.output_dim());
    encoder.encode_batch(texts, std::span<float>(batch));
    expect_invalid_argument([&] { encoder.encode_batch(texts, nullptr); });
    expect_invalid_argument(
        [&] { encoder.encode_batch(texts, std::span<float>(batch).first(batch.size() - 1)); });

    encoder.encode_batch({}, static_cast<float*>(nullptr));
    encoder.encode_batch({}, std::span<float>{});
}

void test_projection_contracts() {
    simeon::Projection projection(4, 4, simeon::ProjectionMode::None, 1);
    const std::array<std::int32_t, 4> sketch = {1, 2, 3, 4};
    std::array<float, 4> pointer_output{};
    std::array<float, 4> span_output{};

    projection.apply(sketch.data(), pointer_output.data());
    projection.apply(std::span<const std::int32_t>(sketch), std::span<float>(span_output));
    assert(pointer_output == span_output);

    expect_invalid_argument([&] { projection.apply(nullptr, pointer_output.data()); });
    expect_invalid_argument([&] { projection.apply(sketch.data(), nullptr); });
    expect_invalid_argument([&] {
        projection.apply(std::span<const std::int32_t>(sketch).first(3),
                         std::span<float>(span_output));
    });
    expect_invalid_argument([&] {
        projection.apply(std::span<const std::int32_t>(sketch),
                         std::span<float>(span_output).first(3));
    });
}

void test_minhash_contracts() {
    simeon::MinHashConfig config;
    config.k = 8;
    simeon::MinHashEncoder encoder(config);
    std::array<std::uint32_t, 8> pointer_output{};
    std::array<std::uint32_t, 8> span_output{};

    encoder.encode("minhash contract", pointer_output.data());
    encoder.encode("minhash contract", std::span<std::uint32_t>(span_output));
    assert(pointer_output == span_output);

    expect_invalid_argument([&] { encoder.encode("minhash contract", nullptr); });
    expect_invalid_argument([&] {
        encoder.encode("minhash contract", std::span<std::uint32_t>(span_output).first(7));
    });

    assert(simeon::jaccard_estimate(nullptr, nullptr, 0) == 0.0f);
    expect_invalid_argument(
        [&] { (void)simeon::jaccard_estimate(nullptr, span_output.data(), 8); });
    expect_invalid_argument(
        [&] { (void)simeon::jaccard_estimate(span_output.data(), nullptr, 8); });
    expect_invalid_argument([&] {
        (void)simeon::jaccard_estimate(std::span<const std::uint32_t>(span_output).first(7),
                                       std::span<const std::uint32_t>(span_output));
    });
    assert(simeon::jaccard_estimate(std::span<const std::uint32_t>(pointer_output),
                                    std::span<const std::uint32_t>(span_output)) == 1.0f);
}

void test_simd_span_parity() {
    const std::array<float, 4> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> b = {4.0f, 3.0f, 2.0f, 1.0f};

    assert(simeon::simd::dot(std::span<const float>(a), std::span<const float>(b)) ==
           simeon::simd::dot(a.data(), b.data(), 4));

    std::array<float, 4> out4{};
    simeon::simd::dot4(std::span<const float>(a), std::span<const float>(b),
                       std::span<const float>(a), std::span<const float>(b),
                       std::span<const float>(a), std::span<float>(out4));
    assert(out4[0] == simeon::simd::dot(a.data(), b.data(), 4));
    assert(out4[1] == simeon::simd::dot(a.data(), a.data(), 4));

    std::array<float, 4> out0{};
    std::array<float, 4> out1{};
    simeon::simd::dot2x4(std::span<const float>(a), std::span<const float>(b),
                         std::span<const float>(a), std::span<const float>(b),
                         std::span<const float>(a), std::span<const float>(b),
                         std::span<float>(out0), std::span<float>(out1));
    assert(out0[0] == simeon::simd::dot(a.data(), a.data(), 4));
    assert(out1[0] == simeon::simd::dot(b.data(), a.data(), 4));

    float minimum = 0.0f;
    float maximum = 0.0f;
    simeon::simd::range(std::span<const float>(a), minimum, maximum);
    assert(minimum == 1.0f);
    assert(maximum == 4.0f);

    std::array<std::uint32_t, 4> indices{};
    assert(simeon::simd::scan_ge(std::span<const float>(a), 3.0f,
                                 std::span<std::uint32_t>(indices)) == 2);
    assert(indices[0] == 2);
    assert(indices[1] == 3);

    auto add = a;
    simeon::simd::add_vec(std::span<float>(add), std::span<const float>(b));
    assert((add == std::array<float, 4>{5.0f, 5.0f, 5.0f, 5.0f}));

    auto scale = a;
    simeon::simd::scale_vec(std::span<float>(scale), std::span<const float>(b));
    assert((scale == std::array<float, 4>{4.0f, 6.0f, 6.0f, 4.0f}));

    auto saxpy = a;
    simeon::simd::saxpy(std::span<float>(saxpy), std::span<const float>(b), 0.5f);
    assert((saxpy == std::array<float, 4>{3.0f, 3.5f, 4.0f, 4.5f}));

    const std::array<float, 4> mean = {1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 4> deviation = {1.0f, 2.0f, 1.0f, 2.0f};
    std::array<float, 4> normalized{};
    simeon::simd::affine_norm(std::span<const float>(a), std::span<const float>(mean),
                              std::span<const float>(deviation), std::span<float>(normalized));
    assert((normalized == std::array<float, 4>{0.0f, 0.5f, 2.0f, 1.5f}));

    std::array<std::uint16_t, 4> packed{};
    std::array<float, 4> unpacked{};
    simeon::simd::bf16_pack(std::span<const float>(a), std::span<std::uint16_t>(packed));
    simeon::simd::bf16_unpack(std::span<const std::uint16_t>(packed), std::span<float>(unpacked));
    for (std::size_t i = 0; i < a.size(); ++i)
        assert(std::fabs(unpacked[i] - a[i]) < 0.02f);

    auto unit = a;
    const float inverse = simeon::simd::l2_normalize(std::span<float>(unit));
    assert(inverse > 0.0f);
}

void test_simd_span_misuse() {
    std::array<float, 4> values{};
    std::array<float, 3> short_values{};
    std::array<float, 4> output{};
    std::array<float, 3> short_output{};
    std::array<std::uint32_t, 3> short_indices{};
    std::array<std::uint16_t, 3> short_packed{};

    expect_invalid_argument([&] {
        (void)simeon::simd::dot(std::span<const float>(values),
                                std::span<const float>(short_values));
    });
    expect_invalid_argument([&] {
        simeon::simd::dot4(std::span<const float>(values), std::span<const float>(values),
                           std::span<const float>(values), std::span<const float>(values),
                           std::span<const float>(values), std::span<float>(short_output));
    });
    expect_invalid_argument([&] {
        simeon::simd::dot2x4(std::span<const float>(values), std::span<const float>(short_values),
                             std::span<const float>(values), std::span<const float>(values),
                             std::span<const float>(values), std::span<const float>(values),
                             std::span<float>(output), std::span<float>(output));
    });
    expect_invalid_argument([&] {
        (void)simeon::simd::scan_ge(std::span<const float>(values), 0.0f,
                                    std::span<std::uint32_t>(short_indices));
    });
    expect_invalid_argument([&] {
        simeon::simd::add_vec(std::span<float>(values), std::span<const float>(short_values));
    });
    expect_invalid_argument([&] {
        simeon::simd::scale_vec(std::span<float>(values), std::span<const float>(short_values));
    });
    expect_invalid_argument([&] {
        simeon::simd::saxpy(std::span<float>(values), std::span<const float>(short_values), 1.0f);
    });
    expect_invalid_argument([&] {
        simeon::simd::affine_norm(std::span<const float>(values),
                                  std::span<const float>(short_values),
                                  std::span<const float>(values), std::span<float>(output));
    });
    expect_invalid_argument([&] {
        simeon::simd::bf16_pack(std::span<const float>(values),
                                std::span<std::uint16_t>(short_packed));
    });
    expect_invalid_argument([&] {
        simeon::simd::bf16_unpack(std::span<const std::uint16_t>(short_packed),
                                  std::span<float>(output));
    });
}

void test_simd_zero_length_contracts() {
    std::span<float> mutable_empty;
    std::span<const float> empty;
    std::span<std::uint32_t> index_empty;
    std::span<std::uint16_t> packed_empty;
    std::span<const std::uint16_t> const_packed_empty;

    assert(simeon::simd::l2_normalize(mutable_empty) == 0.0f);
    assert(simeon::simd::dot(empty, empty) == 0.0f);

    float minimum = 7.0f;
    float maximum = 9.0f;
    simeon::simd::range(empty, minimum, maximum);
    assert(minimum == 7.0f);
    assert(maximum == 9.0f);
    assert(simeon::simd::scan_ge(empty, 0.0f, index_empty) == 0);

    simeon::simd::add_vec(mutable_empty, empty);
    simeon::simd::scale_vec(mutable_empty, empty);
    simeon::simd::saxpy(mutable_empty, empty, 1.0f);
    simeon::simd::affine_norm(empty, empty, empty, mutable_empty);
    simeon::simd::bf16_pack(empty, packed_empty);
    simeon::simd::bf16_unpack(const_packed_empty, mutable_empty);

    std::array<float, 4> out0{};
    std::array<float, 4> out1{};
    simeon::simd::dot4(empty, empty, empty, empty, empty, std::span<float>(out0));
    simeon::simd::dot2x4(empty, empty, empty, empty, empty, empty, std::span<float>(out0),
                         std::span<float>(out1));
    assert((out0 == std::array<float, 4>{}));
    assert((out1 == std::array<float, 4>{}));
}

} // namespace

int main() {
    test_encoder_contracts();
    test_projection_contracts();
    test_minhash_contracts();
    test_simd_span_parity();
    test_simd_span_misuse();
    test_simd_zero_length_contracts();
    return 0;
}
