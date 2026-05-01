#pragma once

#include <cstdint>

#include "simeon/simd.hpp"

namespace simeon {

// A Manifold defines how vectors are compared (similarity), how distances are derived,
// and how vectors are normalized into the manifold's canonical form. The fragment
// geometry pipeline calls Manifold::similarity at the query-fragment and pairwise-frag
// hot paths; the Encoder calls Manifold::normalize after token-vector accumulation.

struct EuclideanCosineManifold {
    static constexpr const char* name = "euclid_cos";

    // Inner-product similarity over L2-normalized vectors (= cosine on the sphere).
    // Inputs that are not L2-normalized return raw dot product.
    static inline float similarity(const float* a, const float* b, std::uint32_t dim) noexcept {
        return simd::dot(a, b, dim);
    }

    // Bounded similarity-derived distance. Range [0, 2] for unit-norm vectors.
    static inline float distance(const float* a, const float* b, std::uint32_t dim) noexcept {
        return 1.0f - similarity(a, b, dim);
    }

    // L2-normalize in place. Returns 1/||v|| or 0 for the zero vector.
    static inline float normalize(float* v, std::uint32_t dim) noexcept {
        return simd::l2_normalize(v, dim);
    }
};

} // namespace simeon
