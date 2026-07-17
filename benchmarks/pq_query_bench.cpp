#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "simeon/pq.hpp"

namespace {

template <typename Query>
double measure(const simeon::ProductQuantizer& pq, const std::vector<float>& query,
               const std::vector<std::uint8_t>& code, std::size_t iterations, float& checksum) {
    for (std::size_t i = 0; i < 16; ++i) {
        Query warmup(pq, query.data());
        checksum += warmup.inner_product(code.data());
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        Query pqQuery(pq, query.data());
        checksum += pqQuery.inner_product(code.data());
    }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
        .count();
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t iterations =
        argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 2000;
    constexpr std::uint32_t kDim = 1024;
    constexpr std::uint32_t kSubquantizers = 32;
    constexpr std::uint32_t kCentroids = 256;

    simeon::ProductQuantizer pq({.dim = kDim, .m = kSubquantizers, .k = kCentroids});
    pq.init_random_gaussian();

    std::mt19937 rng(42);
    std::normal_distribution<float> distribution;
    std::vector<float> query(kDim);
    for (auto& value : query) {
        value = distribution(rng);
    }
    std::vector<std::uint8_t> code(kSubquantizers);
    for (auto& value : code) {
        value = static_cast<std::uint8_t>(rng() % kCentroids);
    }

    float fullChecksum = 0.0F;
    float innerProductChecksum = 0.0F;
    const double firstFullUs = measure<simeon::PQQuery>(pq, query, code, iterations, fullChecksum);
    const double firstInnerProductUs =
        measure<simeon::PQInnerProductQuery>(pq, query, code, iterations, innerProductChecksum);
    const double secondInnerProductUs =
        measure<simeon::PQInnerProductQuery>(pq, query, code, iterations, innerProductChecksum);
    const double secondFullUs = measure<simeon::PQQuery>(pq, query, code, iterations, fullChecksum);
    if (fullChecksum != innerProductChecksum) {
        std::fprintf(stderr, "PQ query checksum mismatch\n");
        return 1;
    }

    const double fullUs = (firstFullUs + secondFullUs) / 2.0;
    const double innerProductUs = (firstInnerProductUs + secondInnerProductUs) / 2.0;

    std::printf("{\"benchmark\":\"pq_query_lut\",\"dim\":%u,\"m\":%u,\"k\":%u,"
                "\"iterations\":%zu,\"full_us_per_query\":%.3f,"
                "\"inner_product_us_per_query\":%.3f,\"speedup\":%.3f}\n",
                kDim, kSubquantizers, kCentroids, iterations,
                fullUs / static_cast<double>(iterations),
                innerProductUs / static_cast<double>(iterations), fullUs / innerProductUs);
    return 0;
}
