#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "simeon_ffi.h"

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

} // namespace

int main() {
    const int dimension = simeon_ffi_dim();
    expect(dimension == 384, "FFI encoder identity remains 384-dimensional");
    std::vector<float> embedding(static_cast<std::size_t>(dimension));
    const char* text = "neural network training";
    expect(simeon_ffi_encode(text, static_cast<int32_t>(std::strlen(text)), embedding.data()) == 0,
           "FFI encoder succeeds");
    expect(
        std::any_of(embedding.begin(), embedding.end(), [](float value) { return value != 0.0f; }),
        "FFI encoder produces a non-zero vector");
    expect(simeon_ffi_encode(nullptr, 0, embedding.data()) == 1,
           "FFI encoder rejects a null input");

    const std::array<const char*, 3> documents = {
        "neural network training gradient descent neural network training",
        "bread recipe flour water salt and starter",
        "mountain trail weather hiking map and equipment",
    };
    std::array<int32_t, documents.size()> lengths{};
    for (std::size_t i = 0; i < documents.size(); ++i)
        lengths[i] = static_cast<int32_t>(std::strlen(documents[i]));
    std::array<float, documents.size()> scores{};
    const char* query = "neural network training gradient";
    expect(simeon_ffi_rerank(query, static_cast<int32_t>(std::strlen(query)), documents.data(),
                             lengths.data(), static_cast<int32_t>(documents.size()),
                             scores.data()) == 0,
           "FFI reranker succeeds");
    expect(
        std::all_of(scores.begin(), scores.end(), [](float value) { return std::isfinite(value); }),
        "FFI reranker returns finite scores for its complete candidate set");
    expect(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())) == 0,
           "FFI reranker places the planted relevant document first");

    auto invalid_lengths = lengths;
    invalid_lengths[1] = -1;
    expect(simeon_ffi_rerank(query, static_cast<int32_t>(std::strlen(query)), documents.data(),
                             invalid_lengths.data(), static_cast<int32_t>(documents.size()),
                             scores.data()) == 1,
           "FFI reranker rejects a negative document length");
    return failures == 0 ? 0 : 1;
}
