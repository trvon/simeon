#include "simeon/pq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "simeon/hasher.hpp"
#include "simeon/simd.hpp"

namespace simeon {

namespace {

// Box–Muller from two splitmix64 outputs. Same construction as projection.cpp's
// gaussian_entry but parameterized by an arbitrary key so we can index by
// (subspace, centroid, dimension) instead of (row, col).
float gaussian_from_key(std::uint64_t key, std::uint64_t seed) noexcept {
    const std::uint64_t h0 = splitmix64_mix(key ^ seed);
    const std::uint64_t h1 = splitmix64_mix(h0 ^ 0x9E3779B97F4A7C15ULL);
    const float u0 = static_cast<float>((h0 >> 11) | 1ULL) / static_cast<float>(1ULL << 53);
    const float u1 = static_cast<float>((h1 >> 11) | 1ULL) / static_cast<float>(1ULL << 53);
    const float r = std::sqrt(-2.0f * std::log(u0));
    const float theta = 6.2831853071795864769f * u1;
    return r * std::cos(theta);
}

// Squared L2 between two `n`-dim float vectors (scalar; PQ subspaces are tiny —
// dsub typically 4..32 — so this is not on a hot path that benefits from SIMD).
float l2_sq(const float* a, const float* b, std::uint32_t n) noexcept {
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

float dot(const float* a, const float* b, std::uint32_t n) noexcept {
    if (n >= 16)
        return simd::dot(a, b, n);
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
        acc += a[i] * b[i];
    return acc;
}

} // namespace

class ProductQuantizer::Impl {
public:
    explicit Impl(PQConfig cfg) : cfg_(cfg) { validate(); }

    const PQConfig& config() const noexcept { return cfg_; }
    std::uint32_t dim() const noexcept { return cfg_.dim; }
    std::uint32_t m() const noexcept { return cfg_.m; }
    std::uint32_t k() const noexcept { return cfg_.k; }
    std::uint32_t dsub() const noexcept { return cfg_.dim / cfg_.m; }
    bool is_trained() const noexcept { return trained_; }

    void init_random_gaussian() {
        const std::uint32_t dsub_ = dsub();
        const std::uint32_t k = cfg_.k;
        codebooks_.assign(static_cast<std::size_t>(cfg_.m) * k * dsub_, 0.0f);
        for (std::uint32_t mi = 0; mi < cfg_.m; ++mi) {
            for (std::uint32_t ki = 0; ki < k; ++ki) {
                float* c = centroid_mut(mi, ki);
                for (std::uint32_t d = 0; d < dsub_; ++d) {
                    const std::uint64_t key = (static_cast<std::uint64_t>(mi) << 40) ^
                                              (static_cast<std::uint64_t>(ki) << 16) ^
                                              static_cast<std::uint64_t>(d);
                    c[d] = gaussian_from_key(key, cfg_.seed);
                }
            }
        }
        trained_ = true;
    }

    void train(const float* training, std::uint32_t n_train, std::uint32_t n_iters) {
        if (n_train < cfg_.k) {
            throw std::invalid_argument(
                "simeon::ProductQuantizer: n_train must be >= k centroids per subspace");
        }
        const std::uint32_t dsub_ = dsub();
        const std::uint32_t k = cfg_.k;
        codebooks_.assign(static_cast<std::size_t>(cfg_.m) * k * dsub_, 0.0f);

        // Per-subspace working buffers (reused across subspaces).
        std::vector<float> sub(static_cast<std::size_t>(n_train) * dsub_);
        std::vector<std::uint32_t> assign(n_train);
        std::vector<float> new_centroids(static_cast<std::size_t>(k) * dsub_);
        std::vector<std::uint32_t> counts(k);

        for (std::uint32_t mi = 0; mi < cfg_.m; ++mi) {
            // Slice subspace `mi` from each training vector.
            for (std::uint32_t i = 0; i < n_train; ++i) {
                std::memcpy(sub.data() + static_cast<std::size_t>(i) * dsub_,
                            training + static_cast<std::size_t>(i) * cfg_.dim + mi * dsub_,
                            dsub_ * sizeof(float));
            }

            // k-means++ init, deterministic via splitmix64 stream seeded by
            // (cfg_.seed, mi). Picks the first centroid as training row 0
            // (deterministic; data ordering is the caller's responsibility).
            float* cb = codebooks_.data() + static_cast<std::size_t>(mi) * k * dsub_;
            std::memcpy(cb, sub.data(), dsub_ * sizeof(float));

            std::vector<float> d2(n_train, std::numeric_limits<float>::infinity());
            std::uint64_t rng_state =
                splitmix64_mix(cfg_.seed ^ (static_cast<std::uint64_t>(mi) + 1));
            for (std::uint32_t ki = 1; ki < k; ++ki) {
                // Update d2 with the just-added centroid.
                const float* last = cb + static_cast<std::size_t>(ki - 1) * dsub_;
                double total = 0.0;
                for (std::uint32_t i = 0; i < n_train; ++i) {
                    const float dist =
                        l2_sq(sub.data() + static_cast<std::size_t>(i) * dsub_, last, dsub_);
                    if (dist < d2[i])
                        d2[i] = dist;
                    total += d2[i];
                }
                rng_state = splitmix64_mix(rng_state);
                std::uint32_t pick = 0;
                if (total > 0.0) {
                    const double u =
                        static_cast<double>(rng_state >> 11) / static_cast<double>(1ULL << 53);
                    double target = u * total;
                    double cum = 0.0;
                    for (std::uint32_t i = 0; i < n_train; ++i) {
                        cum += d2[i];
                        if (cum >= target) {
                            pick = i;
                            break;
                        }
                    }
                } else {
                    // All training points coincide with existing centroids —
                    // pick a deterministic fallback.
                    pick = (rng_state % n_train);
                }
                std::memcpy(cb + static_cast<std::size_t>(ki) * dsub_,
                            sub.data() + static_cast<std::size_t>(pick) * dsub_,
                            dsub_ * sizeof(float));
            }

            // Lloyd iterations.
            for (std::uint32_t it = 0; it < n_iters; ++it) {
                // Assignment step.
                std::uint32_t changes = 0;
                for (std::uint32_t i = 0; i < n_train; ++i) {
                    const float* x = sub.data() + static_cast<std::size_t>(i) * dsub_;
                    std::uint32_t best = 0;
                    float best_d = std::numeric_limits<float>::infinity();
                    for (std::uint32_t ki = 0; ki < k; ++ki) {
                        const float d = l2_sq(x, cb + static_cast<std::size_t>(ki) * dsub_, dsub_);
                        if (d < best_d) {
                            best_d = d;
                            best = ki;
                        }
                    }
                    if (it == 0 || assign[i] != best)
                        ++changes;
                    assign[i] = best;
                }

                // Update step.
                std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0u);
                for (std::uint32_t i = 0; i < n_train; ++i) {
                    const std::uint32_t ki = assign[i];
                    const float* x = sub.data() + static_cast<std::size_t>(i) * dsub_;
                    float* c = new_centroids.data() + static_cast<std::size_t>(ki) * dsub_;
                    for (std::uint32_t d = 0; d < dsub_; ++d)
                        c[d] += x[d];
                    ++counts[ki];
                }
                for (std::uint32_t ki = 0; ki < k; ++ki) {
                    if (counts[ki] == 0) {
                        // Empty cluster: reseed from a deterministic pick.
                        rng_state = splitmix64_mix(rng_state);
                        const std::uint32_t pick = static_cast<std::uint32_t>(rng_state % n_train);
                        std::memcpy(cb + static_cast<std::size_t>(ki) * dsub_,
                                    sub.data() + static_cast<std::size_t>(pick) * dsub_,
                                    dsub_ * sizeof(float));
                    } else {
                        const float inv = 1.0f / static_cast<float>(counts[ki]);
                        float* c = new_centroids.data() + static_cast<std::size_t>(ki) * dsub_;
                        float* dst = cb + static_cast<std::size_t>(ki) * dsub_;
                        for (std::uint32_t d = 0; d < dsub_; ++d)
                            dst[d] = c[d] * inv;
                    }
                }

                if (changes == 0)
                    break;
            }
        }

        trained_ = true;
    }

    void encode(const float* vec, std::uint8_t* code) const noexcept {
        const std::uint32_t dsub_ = dsub();
        for (std::uint32_t mi = 0; mi < cfg_.m; ++mi) {
            const float* x = vec + mi * dsub_;
            const float* cb = codebooks_.data() + static_cast<std::size_t>(mi) * cfg_.k * dsub_;
            std::uint32_t best = 0;
            float best_d = std::numeric_limits<float>::infinity();
            for (std::uint32_t ki = 0; ki < cfg_.k; ++ki) {
                const float d = l2_sq(x, cb + static_cast<std::size_t>(ki) * dsub_, dsub_);
                if (d < best_d) {
                    best_d = d;
                    best = ki;
                }
            }
            code[mi] = static_cast<std::uint8_t>(best);
        }
    }

    void decode(const std::uint8_t* code, float* vec) const noexcept {
        const std::uint32_t dsub_ = dsub();
        for (std::uint32_t mi = 0; mi < cfg_.m; ++mi) {
            const float* c = centroid(mi, code[mi]);
            std::memcpy(vec + mi * dsub_, c, dsub_ * sizeof(float));
        }
    }

    const float* centroid(std::uint32_t mi, std::uint32_t ki) const noexcept {
        return codebooks_.data() + static_cast<std::size_t>(mi) * cfg_.k * dsub() +
               static_cast<std::size_t>(ki) * dsub();
    }

    // Used internally by PQQuery; safe to expose to friend.
    const float* codebooks_data() const noexcept { return codebooks_.data(); }

private:
    float* centroid_mut(std::uint32_t mi, std::uint32_t ki) noexcept {
        return codebooks_.data() + static_cast<std::size_t>(mi) * cfg_.k * dsub() +
               static_cast<std::size_t>(ki) * dsub();
    }

    void validate() const {
        if (cfg_.dim == 0 || cfg_.m == 0 || cfg_.k == 0) {
            throw std::invalid_argument("simeon::ProductQuantizer: dim, m, k must be > 0");
        }
        if (cfg_.k > 256) {
            throw std::invalid_argument("simeon::ProductQuantizer: k must be <= 256");
        }
        if (cfg_.dim % cfg_.m != 0) {
            throw std::invalid_argument("simeon::ProductQuantizer: dim must be divisible by m");
        }
    }

    PQConfig cfg_;
    std::vector<float> codebooks_; // m * k * dsub
    bool trained_ = false;
};

ProductQuantizer::ProductQuantizer(PQConfig cfg) : impl_(std::make_unique<Impl>(cfg)) {}
ProductQuantizer::~ProductQuantizer() = default;
ProductQuantizer::ProductQuantizer(ProductQuantizer&&) noexcept = default;
ProductQuantizer& ProductQuantizer::operator=(ProductQuantizer&&) noexcept = default;

const PQConfig& ProductQuantizer::config() const noexcept {
    return impl_->config();
}
std::uint32_t ProductQuantizer::dim() const noexcept {
    return impl_->dim();
}
std::uint32_t ProductQuantizer::m() const noexcept {
    return impl_->m();
}
std::uint32_t ProductQuantizer::k() const noexcept {
    return impl_->k();
}
std::uint32_t ProductQuantizer::dsub() const noexcept {
    return impl_->dsub();
}
bool ProductQuantizer::is_trained() const noexcept {
    return impl_->is_trained();
}

void ProductQuantizer::init_random_gaussian() {
    impl_->init_random_gaussian();
}
void ProductQuantizer::train(const float* training, std::uint32_t n_train, std::uint32_t n_iters) {
    impl_->train(training, n_train, n_iters);
}
void ProductQuantizer::encode(const float* vec, std::uint8_t* code) const noexcept {
    impl_->encode(vec, code);
}
void ProductQuantizer::encode_batch(const float* vecs, std::uint32_t n,
                                    std::uint8_t* codes) const noexcept {
    for (std::uint32_t i = 0; i < n; ++i) {
        impl_->encode(vecs + static_cast<std::size_t>(i) * impl_->dim(),
                      codes + static_cast<std::size_t>(i) * impl_->m());
    }
}
void ProductQuantizer::decode(const std::uint8_t* code, float* vec) const noexcept {
    impl_->decode(code, vec);
}
const float* ProductQuantizer::centroid(std::uint32_t mi, std::uint32_t ki) const noexcept {
    return impl_->centroid(mi, ki);
}

class PQQuery::Impl {
public:
    Impl(const ProductQuantizer& pq, const float* query) : m_(pq.m()), k_(pq.k()) {
        const std::uint32_t dsub = pq.dsub();
        lut_l2_.resize(static_cast<std::size_t>(m_) * k_);
        lut_ip_.resize(static_cast<std::size_t>(m_) * k_);
        for (std::uint32_t mi = 0; mi < m_; ++mi) {
            const float* q = query + mi * dsub;
            for (std::uint32_t ki = 0; ki < k_; ++ki) {
                const float* c = pq.centroid(mi, ki);
                lut_l2_[static_cast<std::size_t>(mi) * k_ + ki] = l2_sq(q, c, dsub);
                lut_ip_[static_cast<std::size_t>(mi) * k_ + ki] = dot(q, c, dsub);
            }
        }
    }

    float distance_l2_sq(const std::uint8_t* code) const noexcept {
        float acc = 0.0f;
        for (std::uint32_t mi = 0; mi < m_; ++mi) {
            acc += lut_l2_[static_cast<std::size_t>(mi) * k_ + code[mi]];
        }
        return acc;
    }

    float inner_product(const std::uint8_t* code) const noexcept {
        float acc = 0.0f;
        for (std::uint32_t mi = 0; mi < m_; ++mi) {
            acc += lut_ip_[static_cast<std::size_t>(mi) * k_ + code[mi]];
        }
        return acc;
    }

    std::span<const float> lut_l2_sq() const noexcept { return {lut_l2_.data(), lut_l2_.size()}; }
    std::span<const float> lut_ip() const noexcept { return {lut_ip_.data(), lut_ip_.size()}; }

private:
    std::uint32_t m_;
    std::uint32_t k_;
    std::vector<float> lut_l2_; // m * k, squared L2 from query subspace to each centroid
    std::vector<float> lut_ip_; // m * k, inner product from query subspace to each centroid
};

PQQuery::PQQuery(const ProductQuantizer& pq, const float* query)
    : impl_(std::make_unique<Impl>(pq, query)) {}
PQQuery::~PQQuery() = default;
PQQuery::PQQuery(PQQuery&&) noexcept = default;
PQQuery& PQQuery::operator=(PQQuery&&) noexcept = default;

float PQQuery::distance_l2_sq(const std::uint8_t* code) const noexcept {
    return impl_->distance_l2_sq(code);
}
float PQQuery::inner_product(const std::uint8_t* code) const noexcept {
    return impl_->inner_product(code);
}
std::span<const float> PQQuery::lut_l2_sq() const noexcept {
    return impl_->lut_l2_sq();
}
std::span<const float> PQQuery::lut_ip() const noexcept {
    return impl_->lut_ip();
}

} // namespace simeon
