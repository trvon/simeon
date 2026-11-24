#include "simeon/pq.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
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

// Squared L2 is the dominant PQ training kernel: tiny subspaces still produce
// millions of calls during assignment and k-means++ initialization.
// NEON is the AArch64 baseline; AVX2 must be confirmed on the running CPU
// (cached once, so the hot path pays one predictable branch).
#if defined(SIMEON_HAS_AVX2)
bool use_avx2() noexcept {
    static const bool enabled = active_simd_tier() == SimdTier::Avx2;
    return enabled;
}
#endif

float l2_sq(const float* a, const float* b, std::uint32_t n) noexcept {
#if defined(SIMEON_HAS_NEON)
    return simd::l2_squared_neon(a, b, n);
#elif defined(SIMEON_HAS_AVX2)
    return use_avx2() ? simd::l2_squared_avx2(a, b, n) : simd::l2_squared_scalar(a, b, n);
#else
    return simd::l2_squared_scalar(a, b, n);
#endif
}

void l2_sq4(const float* a, const float* b0, const float* b1, const float* b2, const float* b3,
            float* out4, std::uint32_t n) noexcept {
#if defined(SIMEON_HAS_NEON)
    simd::l2_squared4_neon(a, b0, b1, b2, b3, out4, n);
#elif defined(SIMEON_HAS_AVX2)
    if (use_avx2())
        simd::l2_squared4_avx2(a, b0, b1, b2, b3, out4, n);
    else
        simd::l2_squared4_scalar(a, b0, b1, b2, b3, out4, n);
#else
    simd::l2_squared4_scalar(a, b0, b1, b2, b3, out4, n);
#endif
}

std::uint32_t nearest_centroid(const float* x, const float* centroids, std::uint32_t k,
                               std::uint32_t dsub) noexcept {
    std::uint32_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity();
    std::uint32_t ki = 0;
    for (; ki + 4 <= k; ki += 4) {
        const float* batch = centroids + static_cast<std::size_t>(ki) * dsub;
        float distances[4];
        l2_sq4(x, batch, batch + dsub, batch + 2 * dsub, batch + 3 * dsub, distances, dsub);
        for (std::uint32_t lane = 0; lane < 4; ++lane) {
            if (distances[lane] < best_distance) {
                best_distance = distances[lane];
                best = ki + lane;
            }
        }
    }
    for (; ki < k; ++ki) {
        const float distance = l2_sq(x, centroids + static_cast<std::size_t>(ki) * dsub, dsub);
        if (distance < best_distance) {
            best_distance = distance;
            best = ki;
        }
    }
    return best;
}

float dot(const float* a, const float* b, std::uint32_t n) noexcept {
    if (n >= 4) {
#if defined(SIMEON_HAS_NEON)
        return simd::dot_neon(a, b, n);
#elif defined(SIMEON_HAS_AVX2)
        return use_avx2() ? simd::dot_avx2(a, b, n) : simd::dot_scalar(a, b, n);
#else
        return simd::dot_scalar(a, b, n);
#endif
    }
    float acc = 0.0f;
    for (std::uint32_t i = 0; i < n; ++i)
        acc += a[i] * b[i];
    return acc;
}

inline void prefetch_read(const void* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 1);
#else
    (void)ptr;
#endif
}

float inner_product_from_lut(std::span<const float> lut, std::uint32_t m, std::uint32_t k,
                             const std::uint8_t* code) noexcept {
    float acc = 0.0f;
    for (std::uint32_t mi = 0; mi < m; ++mi) {
        acc += lut[static_cast<std::size_t>(mi) * k + code[mi]];
    }
    return acc;
}

// Scores `count` codes, eight at a time (with 4-way and 1-way tails). Interleaving
// independent accumulators hides the latency of the serial add chain and prefetches
// upcoming code rows; each code still sums its m entries in order from 0.0f, so every
// result equals inner_product_from_lut() bit for bit.
template <typename CodeAt>
void inner_product_from_lut_many(std::span<const float> lut, std::uint32_t m, std::uint32_t k,
                                 CodeAt code_at, std::size_t count, float* out) noexcept {
    const float* table = lut.data();
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const std::uint8_t* c0 = code_at(i);
        const std::uint8_t* c1 = code_at(i + 1);
        const std::uint8_t* c2 = code_at(i + 2);
        const std::uint8_t* c3 = code_at(i + 3);
        const std::uint8_t* c4 = code_at(i + 4);
        const std::uint8_t* c5 = code_at(i + 5);
        const std::uint8_t* c6 = code_at(i + 6);
        const std::uint8_t* c7 = code_at(i + 7);
        if (i + 16 <= count) {
            prefetch_read(code_at(i + 8));
            prefetch_read(code_at(i + 12));
        }
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        for (std::uint32_t mi = 0; mi < m; ++mi) {
            const float* row = table + static_cast<std::size_t>(mi) * k;
            a0 += row[c0[mi]];
            a1 += row[c1[mi]];
            a2 += row[c2[mi]];
            a3 += row[c3[mi]];
            a4 += row[c4[mi]];
            a5 += row[c5[mi]];
            a6 += row[c6[mi]];
            a7 += row[c7[mi]];
        }
        out[i] = a0;
        out[i + 1] = a1;
        out[i + 2] = a2;
        out[i + 3] = a3;
        out[i + 4] = a4;
        out[i + 5] = a5;
        out[i + 6] = a6;
        out[i + 7] = a7;
    }
    for (; i + 4 <= count; i += 4) {
        const std::uint8_t* c0 = code_at(i);
        const std::uint8_t* c1 = code_at(i + 1);
        const std::uint8_t* c2 = code_at(i + 2);
        const std::uint8_t* c3 = code_at(i + 3);
        float a0 = 0.0f;
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        for (std::uint32_t mi = 0; mi < m; ++mi) {
            const float* row = table + static_cast<std::size_t>(mi) * k;
            a0 += row[c0[mi]];
            a1 += row[c1[mi]];
            a2 += row[c2[mi]];
            a3 += row[c3[mi]];
        }
        out[i] = a0;
        out[i + 1] = a1;
        out[i + 2] = a2;
        out[i + 3] = a3;
    }
    for (; i < count; ++i) {
        out[i] = inner_product_from_lut(lut, m, k, code_at(i));
    }
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
    const float* codebooks_data() const noexcept { return codebooks_.data(); }
    void import_codebooks(std::span<const float> codebooks, bool trained) {
        codebooks_.assign(codebooks.begin(), codebooks.end());
        trained_ = trained;
    }

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

        struct WorkerScratch {
            std::vector<float> sub;
            std::vector<std::uint32_t> assign;
            std::vector<float> new_centroids;
            std::vector<std::uint32_t> counts;
            std::vector<float> d2;
        };

        auto train_subspace = [&](std::uint32_t mi, WorkerScratch& ws) {
            // Slice subspace `mi` from each training vector.
            for (std::uint32_t i = 0; i < n_train; ++i) {
                std::memcpy(ws.sub.data() + static_cast<std::size_t>(i) * dsub_,
                            training + static_cast<std::size_t>(i) * cfg_.dim + mi * dsub_,
                            dsub_ * sizeof(float));
            }

            // k-means++ init, deterministic via splitmix64 stream seeded by
            // (cfg_.seed, mi). Picks the first centroid as training row 0
            // (deterministic; data ordering is the caller's responsibility).
            float* cb = codebooks_.data() + static_cast<std::size_t>(mi) * k * dsub_;
            std::memcpy(cb, ws.sub.data(), dsub_ * sizeof(float));

            std::fill(ws.d2.begin(), ws.d2.end(), std::numeric_limits<float>::infinity());
            std::uint64_t rng_state =
                splitmix64_mix(cfg_.seed ^ (static_cast<std::uint64_t>(mi) + 1));
            for (std::uint32_t ki = 1; ki < k; ++ki) {
                // Update d2 with the just-added centroid.
                const float* last = cb + static_cast<std::size_t>(ki - 1) * dsub_;
                double total = 0.0;
                std::uint32_t i = 0;
                for (; i + 4 <= n_train; i += 4) {
                    const float* batch = ws.sub.data() + static_cast<std::size_t>(i) * dsub_;
                    float distances[4];
                    l2_sq4(last, batch, batch + dsub_, batch + 2 * dsub_, batch + 3 * dsub_,
                           distances, dsub_);
                    for (std::uint32_t lane = 0; lane < 4; ++lane) {
                        if (distances[lane] < ws.d2[i + lane])
                            ws.d2[i + lane] = distances[lane];
                        total += ws.d2[i + lane];
                    }
                }
                for (; i < n_train; ++i) {
                    const float distance =
                        l2_sq(ws.sub.data() + static_cast<std::size_t>(i) * dsub_, last, dsub_);
                    if (distance < ws.d2[i])
                        ws.d2[i] = distance;
                    total += ws.d2[i];
                }
                rng_state = splitmix64_mix(rng_state);
                std::uint32_t pick = 0;
                if (total > 0.0) {
                    const double u =
                        static_cast<double>(rng_state >> 11) / static_cast<double>(1ULL << 53);
                    double target = u * total;
                    double cum = 0.0;
                    for (std::uint32_t j = 0; j < n_train; ++j) {
                        cum += ws.d2[j];
                        if (cum >= target) {
                            pick = j;
                            break;
                        }
                    }
                } else {
                    // All training points coincide with existing centroids —
                    // pick a deterministic fallback.
                    pick = static_cast<std::uint32_t>(rng_state % n_train);
                }
                std::memcpy(cb + static_cast<std::size_t>(ki) * dsub_,
                            ws.sub.data() + static_cast<std::size_t>(pick) * dsub_,
                            dsub_ * sizeof(float));
            }

            // Lloyd iterations.
            for (std::uint32_t it = 0; it < n_iters; ++it) {
                // Assignment step.
                std::uint32_t changes = 0;
                for (std::uint32_t i = 0; i < n_train; ++i) {
                    const float* x = ws.sub.data() + static_cast<std::size_t>(i) * dsub_;
                    const std::uint32_t best = nearest_centroid(x, cb, k, dsub_);
                    if (it == 0 || ws.assign[i] != best)
                        ++changes;
                    ws.assign[i] = best;
                }

                // Update step.
                std::fill(ws.new_centroids.begin(), ws.new_centroids.end(), 0.0f);
                std::fill(ws.counts.begin(), ws.counts.end(), 0u);
                for (std::uint32_t i = 0; i < n_train; ++i) {
                    const std::uint32_t ki = ws.assign[i];
                    const float* x = ws.sub.data() + static_cast<std::size_t>(i) * dsub_;
                    float* c = ws.new_centroids.data() + static_cast<std::size_t>(ki) * dsub_;
                    for (std::uint32_t d = 0; d < dsub_; ++d)
                        c[d] += x[d];
                    ++ws.counts[ki];
                }
                for (std::uint32_t ki = 0; ki < k; ++ki) {
                    if (ws.counts[ki] == 0) {
                        // Empty cluster: reseed from a deterministic pick.
                        rng_state = splitmix64_mix(rng_state);
                        const std::uint32_t pick = static_cast<std::uint32_t>(rng_state % n_train);
                        std::memcpy(cb + static_cast<std::size_t>(ki) * dsub_,
                                    ws.sub.data() + static_cast<std::size_t>(pick) * dsub_,
                                    dsub_ * sizeof(float));
                    } else {
                        const float inv = 1.0f / static_cast<float>(ws.counts[ki]);
                        float* c = ws.new_centroids.data() + static_cast<std::size_t>(ki) * dsub_;
                        float* dst = cb + static_cast<std::size_t>(ki) * dsub_;
                        for (std::uint32_t d = 0; d < dsub_; ++d)
                            dst[d] = c[d] * inv;
                    }
                }

                if (changes == 0)
                    break;
            }
        };

        auto make_scratch = [&]() {
            WorkerScratch ws;
            ws.sub.resize(static_cast<std::size_t>(n_train) * dsub_);
            ws.assign.resize(n_train);
            ws.new_centroids.resize(static_cast<std::size_t>(k) * dsub_);
            ws.counts.resize(k);
            ws.d2.resize(n_train);
            return ws;
        };

        const unsigned hw = std::thread::hardware_concurrency();
        const std::uint32_t n_threads =
            (cfg_.m >= 8 && n_train >= 1024 && hw > 1)
                ? std::min<std::uint32_t>(cfg_.m, static_cast<std::uint32_t>(hw))
                : 1u;

        if (n_threads <= 1u) {
            auto ws = make_scratch();
            for (std::uint32_t mi = 0; mi < cfg_.m; ++mi) {
                train_subspace(mi, ws);
            }
        } else {
            std::atomic<std::uint32_t> next_mi{0};
            auto worker_fn = [&]() {
                auto ws = make_scratch();
                while (true) {
                    const std::uint32_t mi = next_mi.fetch_add(1u, std::memory_order_relaxed);
                    if (mi >= cfg_.m)
                        break;
                    train_subspace(mi, ws);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(n_threads - 1);
            for (std::uint32_t t = 1; t < n_threads; ++t) {
                workers.emplace_back(worker_fn);
            }
            worker_fn();
            for (auto& w : workers) {
                w.join();
            }
        }

        trained_ = true;
    }

    void encode(const float* vec, std::uint8_t* code) const noexcept {
        const std::uint32_t dsub_ = dsub();
        for (std::uint32_t mi = 0; mi < cfg_.m; ++mi) {
            const float* x = vec + mi * dsub_;
            const float* cb = codebooks_.data() + static_cast<std::size_t>(mi) * cfg_.k * dsub_;
            code[mi] = static_cast<std::uint8_t>(nearest_centroid(x, cb, cfg_.k, dsub_));
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
std::span<const float> ProductQuantizer::codebooks() const noexcept {
    const std::size_t count = static_cast<std::size_t>(impl_->m()) * impl_->k() * impl_->dsub();
    return {impl_->codebooks_data(), count};
}
void ProductQuantizer::import_codebooks(std::span<const float> codebooks, bool trained) {
    const std::size_t expected = static_cast<std::size_t>(impl_->m()) * impl_->k() * impl_->dsub();
    if (codebooks.size() != expected) {
        throw std::invalid_argument("simeon::ProductQuantizer: invalid imported codebook size");
    }
    impl_->import_codebooks(codebooks, trained);
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
        return inner_product_from_lut(lut_ip_, m_, k_, code);
    }

    std::span<const float> lut_l2_sq() const noexcept { return {lut_l2_.data(), lut_l2_.size()}; }
    std::span<const float> lut_ip() const noexcept { return {lut_ip_.data(), lut_ip_.size()}; }

private:
    std::uint32_t m_;
    std::uint32_t k_;
    std::vector<float> lut_l2_; // m * k, squared L2 from query subspace to each centroid
    std::vector<float> lut_ip_; // m * k, inner product from query subspace to each centroid
};

PQQuery::PQQuery(const ProductQuantizer& pq, const float* query) {
    if (query == nullptr) {
        throw std::invalid_argument("simeon::PQQuery: query must not be null");
    }
    impl_ = std::make_unique<Impl>(pq, query);
}
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

class PQInnerProductQuery::Impl {
public:
    Impl(const ProductQuantizer& pq, const float* query) : m_(pq.m()), k_(pq.k()) {
        const std::uint32_t dsub = pq.dsub();
        lut_ip_.resize(static_cast<std::size_t>(m_) * k_);
        for (std::uint32_t mi = 0; mi < m_; ++mi) {
            const float* q = query + mi * dsub;
            for (std::uint32_t ki = 0; ki < k_; ++ki) {
                const float* c = pq.centroid(mi, ki);
                lut_ip_[static_cast<std::size_t>(mi) * k_ + ki] = dot(q, c, dsub);
            }
        }
        if (k_ <= 16) {
            build_fastscan_lut();
        }
    }

    float inner_product(const std::uint8_t* code) const noexcept {
        return inner_product_from_lut(lut_ip_, m_, k_, code);
    }

    void inner_product_batch(const std::uint8_t* codes, std::size_t count,
                             float* out) const noexcept {
        const std::size_t stride = m_;
        inner_product_from_lut_many(
            lut_ip_, m_, k_, [codes, stride](std::size_t i) { return codes + i * stride; }, count,
            out);
    }

    void inner_product_batch_fastscan(const std::uint8_t* codes, std::size_t count,
                                      float* out) const noexcept {
        if (k_ <= 16) {
            simd::pq_fastscan_lut16(lut_u8_.data(), codes, m_, count, fastscan_scale_,
                                    fastscan_offset_, out);
            return;
        }
        inner_product_batch(codes, count, out);
    }

    void inner_product_gather(const std::uint8_t* codes, const std::size_t* indices,
                              std::size_t count, float* out) const noexcept {
        const std::size_t stride = m_;
        inner_product_from_lut_many(
            lut_ip_, m_, k_,
            [codes, indices, stride](std::size_t i) { return codes + indices[i] * stride; }, count,
            out);
    }

    std::span<const float> lut_ip() const noexcept { return {lut_ip_.data(), lut_ip_.size()}; }

private:
    void build_fastscan_lut() {
        lut_u8_.assign(static_cast<std::size_t>(m_) * 16u, 0u);
        std::vector<float> mins(m_, 0.0f);
        float max_span = 0.0f;
        double offset_sum = 0.0;
        for (std::uint32_t mi = 0; mi < m_; ++mi) {
            const float* row = lut_ip_.data() + static_cast<std::size_t>(mi) * k_;
            float mn = row[0];
            float mx = row[0];
            for (std::uint32_t ki = 1; ki < k_; ++ki) {
                mn = std::min(mn, row[ki]);
                mx = std::max(mx, row[ki]);
            }
            mins[mi] = mn;
            offset_sum += static_cast<double>(mn);
            max_span = std::max(max_span, mx - mn);
        }
        fastscan_offset_ = static_cast<float>(offset_sum);
        if (max_span <= 0.0f) {
            fastscan_scale_ = 0.0f;
            return;
        }
        fastscan_scale_ = max_span / 255.0f;
        const float inv_scale = 255.0f / max_span;
        for (std::uint32_t mi = 0; mi < m_; ++mi) {
            const float* row = lut_ip_.data() + static_cast<std::size_t>(mi) * k_;
            std::uint8_t* dst = lut_u8_.data() + static_cast<std::size_t>(mi) * 16u;
            const float mn = mins[mi];
            for (std::uint32_t ki = 0; ki < k_; ++ki) {
                const float q = std::clamp((row[ki] - mn) * inv_scale, 0.0f, 255.0f);
                dst[ki] = static_cast<std::uint8_t>(std::lround(q));
            }
        }
    }

    std::uint32_t m_;
    std::uint32_t k_;
    std::vector<float> lut_ip_;
    std::vector<std::uint8_t> lut_u8_;
    float fastscan_scale_ = 0.0f;
    float fastscan_offset_ = 0.0f;
};

PQInnerProductQuery::PQInnerProductQuery(const ProductQuantizer& pq, const float* query) {
    if (query == nullptr) {
        throw std::invalid_argument("simeon::PQInnerProductQuery: query must not be null");
    }
    impl_ = std::make_unique<Impl>(pq, query);
}
PQInnerProductQuery::~PQInnerProductQuery() = default;
PQInnerProductQuery::PQInnerProductQuery(PQInnerProductQuery&&) noexcept = default;
PQInnerProductQuery& PQInnerProductQuery::operator=(PQInnerProductQuery&&) noexcept = default;

float PQInnerProductQuery::inner_product(const std::uint8_t* code) const noexcept {
    return impl_->inner_product(code);
}
void PQInnerProductQuery::inner_product_batch(const std::uint8_t* codes, std::size_t count,
                                              float* out) const noexcept {
    impl_->inner_product_batch(codes, count, out);
}
void PQInnerProductQuery::inner_product_batch_fastscan(const std::uint8_t* codes, std::size_t count,
                                                       float* out) const noexcept {
    impl_->inner_product_batch_fastscan(codes, count, out);
}
void PQInnerProductQuery::inner_product_gather(const std::uint8_t* codes,
                                               const std::size_t* indices, std::size_t count,
                                               float* out) const noexcept {
    impl_->inner_product_gather(codes, indices, count, out);
}
std::span<const float> PQInnerProductQuery::lut_ip() const noexcept {
    return impl_->lut_ip();
}

} // namespace simeon
