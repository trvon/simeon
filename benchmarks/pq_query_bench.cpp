#include <algorithm>
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

// Scores `rows` codes with one query: per-code calls vs the batched and gathered scans.
int measure_scan(const simeon::ProductQuantizer& pq, const std::vector<float>& query,
                 std::uint32_t m, std::uint32_t k, std::size_t rows, std::size_t repeats) {
    std::mt19937 rng(7);
    std::vector<std::uint8_t> codes(rows * m);
    for (auto& value : codes) {
        value = static_cast<std::uint8_t>(rng() % k);
    }
    std::vector<std::size_t> subset;
    for (std::size_t i = 0; i < rows; i += 3) {
        subset.push_back(i);
    }
    std::shuffle(subset.begin(), subset.end(), rng);

    simeon::PQInnerProductQuery q(pq, query.data());
    std::vector<float> single(rows);
    std::vector<float> batch(rows);
    std::vector<float> gathered(subset.size());
    const auto time_us = [repeats](auto&& fn) {
        fn();
        double best = 1e300;
        for (std::size_t r = 0; r < repeats; ++r) {
            const auto start = std::chrono::steady_clock::now();
            fn();
            best = std::min(best, std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() - start)
                                      .count());
        }
        return best;
    };
    const double singleUs = time_us([&] {
        for (std::size_t i = 0; i < rows; ++i) {
            single[i] = q.inner_product(codes.data() + i * m);
        }
    });
    const double batchUs =
        time_us([&] { q.inner_product_batch(codes.data(), rows, batch.data()); });
    const double singleSubsetUs = time_us([&] {
        for (std::size_t i = 0; i < subset.size(); ++i) {
            gathered[i] = q.inner_product(codes.data() + subset[i] * m);
        }
    });
    const double gatherUs = time_us([&] {
        q.inner_product_gather(codes.data(), subset.data(), subset.size(), gathered.data());
    });
    if (single != batch) {
        std::fprintf(stderr, "PQ batch scan differs from per-code scores\n");
        return 1;
    }
    for (std::size_t i = 0; i < subset.size(); ++i) {
        if (gathered[i] != single[subset[i]]) {
            std::fprintf(stderr, "PQ gather scan differs from per-code scores\n");
            return 1;
        }
    }
    std::printf("{\"benchmark\":\"pq_scan\",\"m\":%u,\"k\":%u,\"rows\":%zu,"
                "\"single_ns_per_code\":%.3f,\"batch_ns_per_code\":%.3f,\"batch_speedup\":%.3f,"
                "\"gather_rows\":%zu,\"single_subset_ns_per_code\":%.3f,"
                "\"gather_ns_per_code\":%.3f,\"gather_speedup\":%.3f}\n",
                m, k, rows, singleUs * 1000.0 / static_cast<double>(rows),
                batchUs * 1000.0 / static_cast<double>(rows), singleUs / batchUs, subset.size(),
                singleSubsetUs * 1000.0 / static_cast<double>(subset.size()),
                gatherUs * 1000.0 / static_cast<double>(subset.size()), singleSubsetUs / gatherUs);
    return 0;
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
    return measure_scan(pq, query, kSubquantizers, kCentroids, 100000, 20);
}
