#include "simeon/projection.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "simeon/hasher.hpp"

#if defined(SIMEON_HAS_NEON)
#include <arm_neon.h>
#endif
#if defined(SIMEON_HAS_AVX2)
#include <immintrin.h>
#endif

namespace simeon {

namespace {

// Achlioptas (2003): entries in {-1, 0, +1} with probabilities {1/6, 2/3, 1/6}.
// Scaled by sqrt(3) so that JL distance preservation in expectation holds.
float achlioptas_entry(std::uint32_t row, std::uint32_t col, std::uint64_t seed) noexcept {
    const std::uint64_t key = (static_cast<std::uint64_t>(row) << 32) ^ col;
    const std::uint64_t h = splitmix64_mix(key ^ seed);
    // 0..2^64-1 mapped to {−1 (p=1/6), 0 (p=2/3), +1 (p=1/6)}.
    constexpr std::uint64_t LOW = 0x2AAAAAAAAAAAAAABULL;  // ~(2^64 / 6)
    constexpr std::uint64_t HIGH = 0xD555555555555555ULL; // ~(5 * 2^64 / 6)
    static const float scale = std::sqrt(3.0f);
    if (h < LOW)
        return -scale;
    if (h >= HIGH)
        return +scale;
    return 0.0f;
}

// Dense Gaussian via Box–Muller on two splitmix64 streams. Deterministic per (row, col, seed).
float gaussian_entry(std::uint32_t row, std::uint32_t col, std::uint64_t seed) noexcept {
    const std::uint64_t key = (static_cast<std::uint64_t>(row) << 32) ^ col;
    const std::uint64_t h0 = splitmix64_mix(key ^ seed);
    const std::uint64_t h1 = splitmix64_mix(h0 ^ 0x9E3779B97F4A7C15ULL);
    const float u0 = (static_cast<float>((h0 >> 11) | 1ULL) / static_cast<float>(1ULL << 53));
    const float u1 = (static_cast<float>((h1 >> 11) | 1ULL) / static_cast<float>(1ULL << 53));
    const float r = std::sqrt(-2.0f * std::log(u0));
    const float theta = 6.2831853071795864769f * u1;
    return r * std::cos(theta);
}

// In-place iterative Walsh-Hadamard transform. n must be a power of 2.
// Unnormalized: applying twice multiplies by n. Caller scales as needed.
void fwht_inplace(float* data, std::uint32_t n) noexcept {
    for (std::uint32_t h = 1; h < n; h <<= 1) {
        for (std::uint32_t i = 0; i < n; i += h << 1) {
            for (std::uint32_t j = i; j < i + h; ++j) {
                const float x = data[j];
                const float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }
}

std::uint32_t next_pow2(std::uint32_t n) noexcept {
    if (n <= 1)
        return 1;
    std::uint32_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

// Walsh-Hadamard matrix entry: H[i, j] = (-1)^popcount(i & j). Rows of H are
// orthogonal with squared norm = pad_n; H/sqrt(pad_n) is orthonormal.
int hadamard_sign(std::uint32_t i, std::uint32_t j) noexcept {
    return (__builtin_popcount(i & j) & 1) ? -1 : +1;
}

// Very sparse (Li 2006): entries scaled by sqrt(s), nonzero with probability 1/s.
float very_sparse_entry(std::uint32_t row, std::uint32_t col, std::uint64_t seed,
                        std::uint32_t sketch_dim) noexcept {
    if (sketch_dim == 0)
        return 0.0f;
    const float s = std::max(1.0f, std::sqrt(static_cast<float>(sketch_dim)));
    const std::uint64_t key = (static_cast<std::uint64_t>(row) << 32) ^ col;
    const std::uint64_t h = splitmix64_mix(key ^ seed ^ 0xABCDEF0123456789ULL);
    const float u = static_cast<float>(h >> 11) / static_cast<float>(1ULL << 53);
    const float p_nonzero = 1.0f / s;
    if (u < p_nonzero * 0.5f)
        return -std::sqrt(s);
    if (u < p_nonzero)
        return +std::sqrt(s);
    return 0.0f;
}

} // namespace

Projection::Projection(std::uint32_t sketch_dim, std::uint32_t output_dim, ProjectionMode mode,
                       std::uint64_t seed, float sparse_jl_eps)
    : sketch_dim_(sketch_dim), output_dim_(output_dim), mode_(mode), seed_(seed) {
    if (mode != ProjectionMode::None && output_dim == 0) {
        throw std::invalid_argument("simeon::Projection: output_dim must be > 0 when mode != None");
    }
    if (mode_ == ProjectionMode::SparseJL && !(sparse_jl_eps > 0.0f && sparse_jl_eps <= 1.0f)) {
        throw std::invalid_argument("simeon::Projection: sparse_jl_eps must be in (0, 1]");
    }
    if (mode_ == ProjectionMode::None)
        return;

    inv_scale_ = 1.0f / std::sqrt(static_cast<float>(output_dim_));
    achlioptas_scale_ = std::sqrt(3.0f) * inv_scale_;

    // SparseJL (Kane–Nelson 2010): per input column, pick s distinct output
    // rows. Assign ±1/sqrt(s) at each picked entry. With s = ceil(eps * k),
    // E[||Px||^2] = ||x||^2 (unbiased) and the JL-style distortion bound
    // tightens as eps shrinks at the cost of more per-row work.
    std::uint32_t s_jl = 0;
    float sparse_jl_value = 0.0f;
    if (mode_ == ProjectionMode::SparseJL) {
        const float raw = std::ceil(sparse_jl_eps * static_cast<float>(output_dim_));
        s_jl = std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(raw));
        s_jl = std::min(s_jl, output_dim_);
        sparse_jl_value = 1.0f / std::sqrt(static_cast<float>(s_jl));
    }

    // Still materialize the dense view so Projection::entry(row, col) works
    // for tests and callers that probe the matrix. The hot path in apply()
    // uses the specialized per-mode cache below.
    dense_.assign(static_cast<std::size_t>(output_dim_) * sketch_dim_, 0.0f);
    if (mode_ == ProjectionMode::AchlioptasSparse)
        achlioptas_.resize(output_dim_);
    if (mode_ == ProjectionMode::VerySparse)
        sparse_.resize(output_dim_);
    if (mode_ == ProjectionMode::SparseJL)
        sparse_.resize(output_dim_);

    if (mode_ == ProjectionMode::Fwht) {
        // SRHT-style FJLT: zero-pad sketch to next power of 2, multiply by
        // ±1 sign diagonal D, apply Walsh-Hadamard, then subsample
        // output_dim distinct rows. Scale absorbs the orthonormal H factor
        // (1/sqrt(pad_n)) and the subsample correction (sqrt(pad_n/k)) into
        // a single 1/sqrt(output_dim) per output entry.
        pad_n_ = next_pow2(sketch_dim_);
        fwht_scale_ = 1.0f / std::sqrt(static_cast<float>(output_dim_));

        signs_.resize(pad_n_);
        std::uint64_t rng = splitmix64_mix(seed_ ^ 0x9E3779B97F4A7C15ULL);
        for (std::uint32_t i = 0; i < pad_n_; ++i) {
            rng = splitmix64_mix(rng + i);
            signs_[i] = (rng & 1ULL) ? -1.0f : +1.0f;
        }

        // Sample output_dim_ distinct row indices from [0, pad_n_) by
        // partial Fisher-Yates over a scratch index array. Same pattern
        // as the SparseJL column-row picker above.
        if (output_dim_ > pad_n_) {
            throw std::invalid_argument(
                "simeon::Projection: Fwht requires output_dim <= next_pow2(sketch_dim)");
        }
        std::vector<std::uint32_t> idx(pad_n_);
        for (std::uint32_t i = 0; i < pad_n_; ++i)
            idx[i] = i;
        std::uint64_t srng = splitmix64_mix(seed_ ^ 0xBF58476D1CE4E5B9ULL);
        for (std::uint32_t i = 0; i < output_dim_; ++i) {
            srng = splitmix64_mix(srng + i);
            const std::uint32_t pick =
                i + static_cast<std::uint32_t>(srng % static_cast<std::uint64_t>(pad_n_ - i));
            std::swap(idx[i], idx[pick]);
        }
        sample_.assign(idx.begin(), idx.begin() + output_dim_);

        // Materialize dense_ for entry()/test parity. Closed form:
        //   P[r, c] = D[c] * (-1)^popcount(sample_[r] & c) / sqrt(output_dim_)
        // for c in [0, sketch_dim_); padded columns don't enter dense_
        // because the input sketch is zero in those columns.
        for (std::uint32_t r = 0; r < output_dim_; ++r) {
            const std::uint32_t s = sample_[r];
            float* row_ptr = dense_.data() + static_cast<std::size_t>(r) * sketch_dim_;
            for (std::uint32_t c = 0; c < sketch_dim_; ++c) {
                const float sign = static_cast<float>(hadamard_sign(s, c));
                row_ptr[c] = signs_[c] * sign * fwht_scale_;
            }
        }
        return;
    }

    if (mode_ == ProjectionMode::SparseJL) {
        // Per input column pick s distinct output rows via partial
        // Fisher-Yates over a row-index scratch. For each picked (col, row)
        // pair, append (col, ±1/sqrt(s)) to that row's sparse entry list.
        // The apply path then gathers per row exactly the same way as
        // VerySparse; per-row count varies but mean is s * sketch_dim /
        // output_dim, so total nonzeros ≈ s * sketch_dim.
        std::vector<std::uint32_t> idx(output_dim_);
        for (std::uint32_t col = 0; col < sketch_dim_; ++col) {
            for (std::uint32_t i = 0; i < output_dim_; ++i)
                idx[i] = i;
            const std::uint64_t col_seed =
                splitmix64_mix(seed_ ^ (static_cast<std::uint64_t>(col) * 0x9E3779B97F4A7C15ULL));
            std::uint64_t rng = col_seed;
            for (std::uint32_t i = 0; i < s_jl; ++i) {
                rng = splitmix64_mix(rng + i);
                const std::uint32_t pick =
                    i +
                    static_cast<std::uint32_t>(rng % static_cast<std::uint64_t>(output_dim_ - i));
                std::swap(idx[i], idx[pick]);
            }
            for (std::uint32_t i = 0; i < s_jl; ++i) {
                const std::uint32_t row = idx[i];
                const std::uint64_t sign_h = splitmix64_mix(
                    col_seed ^ (static_cast<std::uint64_t>(row) * 0xBF58476D1CE4E5B9ULL));
                const float w = (sign_h & 1ULL) ? -sparse_jl_value : +sparse_jl_value;
                sparse_[row].cols.push_back(col);
                sparse_[row].weights.push_back(w);
                dense_[static_cast<std::size_t>(row) * sketch_dim_ + col] = w;
            }
        }
        // Sort each row's gather by column index for sequential sketch
        // access during apply().
        for (std::uint32_t row = 0; row < output_dim_; ++row) {
            auto& cols = sparse_[row].cols;
            auto& ws = sparse_[row].weights;
            std::vector<std::pair<std::uint32_t, float>> tmp(cols.size());
            for (std::size_t i = 0; i < cols.size(); ++i)
                tmp[i] = {cols[i], ws[i]};
            std::sort(tmp.begin(), tmp.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::size_t i = 0; i < cols.size(); ++i) {
                cols[i] = tmp[i].first;
                ws[i] = tmp[i].second;
            }
        }
        return;
    }

    for (std::uint32_t row = 0; row < output_dim_; ++row) {
        auto* row_ptr = dense_.data() + static_cast<std::size_t>(row) * sketch_dim_;
        for (std::uint32_t col = 0; col < sketch_dim_; ++col) {
            float w = 0.0f;
            switch (mode_) {
                case ProjectionMode::AchlioptasSparse:
                    w = achlioptas_entry(row, col, seed_);
                    if (w > 0.0f)
                        achlioptas_[row].pos_cols.push_back(col);
                    else if (w < 0.0f)
                        achlioptas_[row].neg_cols.push_back(col);
                    break;
                case ProjectionMode::DenseGaussian:
                    w = gaussian_entry(row, col, seed_);
                    break;
                case ProjectionMode::VerySparse:
                    w = very_sparse_entry(row, col, seed_, sketch_dim_);
                    if (w != 0.0f) {
                        sparse_[row].cols.push_back(col);
                        sparse_[row].weights.push_back(w);
                    }
                    break;
                default:
                    w = 0.0f;
            }
            row_ptr[col] = w;
        }
    }
}

Projection::~Projection() = default;

float Projection::entry(std::uint32_t row, std::uint32_t col) const {
    if (mode_ == ProjectionMode::None)
        return (row == col) ? 1.0f : 0.0f;
    return dense_[static_cast<std::size_t>(row) * sketch_dim_ + col];
}

void Projection::apply(const std::int32_t* sketch, float* out) const {
    if (mode_ == ProjectionMode::None) {
        for (std::uint32_t i = 0; i < sketch_dim_; ++i) {
            out[i] = static_cast<float>(sketch[i]);
        }
        return;
    }

    // Achlioptas: weights are {-sqrt(3), 0, +sqrt(3)}. Gather per-row by
    // sign and compute (sum_pos - sum_neg) as plain int64 accumulation, then
    // scale once. No float multiply inside the inner loop.
    if (mode_ == ProjectionMode::AchlioptasSparse) {
        for (std::uint32_t row = 0; row < output_dim_; ++row) {
            const auto& ar = achlioptas_[row];
            std::int64_t pos = 0;
            std::int64_t neg = 0;
            const auto* pc = ar.pos_cols.data();
            const std::size_t np = ar.pos_cols.size();
            for (std::size_t i = 0; i < np; ++i)
                pos += sketch[pc[i]];
            const auto* nc = ar.neg_cols.data();
            const std::size_t nn = ar.neg_cols.size();
            for (std::size_t i = 0; i < nn; ++i)
                neg += sketch[nc[i]];
            out[row] = static_cast<float>(pos - neg) * achlioptas_scale_;
        }
        return;
    }

    if (mode_ == ProjectionMode::VerySparse) {
        for (std::uint32_t row = 0; row < output_dim_; ++row) {
            const auto& sr = sparse_[row];
            float acc = 0.0f;
            const auto* cols = sr.cols.data();
            const auto* ws = sr.weights.data();
            const std::size_t n = sr.cols.size();
            for (std::size_t i = 0; i < n; ++i) {
                acc += ws[i] * static_cast<float>(sketch[cols[i]]);
            }
            out[row] = acc * inv_scale_;
        }
        return;
    }

    if (mode_ == ProjectionMode::Fwht) {
        // Apply: zero-pad sketch into pad_n_ float buffer, multiply by D,
        // run in-place FWHT, gather sample_[] entries with fwht_scale_.
        // Total work: O(pad_n_ log pad_n_) for the FWHT + O(output_dim_)
        // for the gather — independent of output_dim_ in the inner loop.
        thread_local std::vector<float> buf;
        buf.assign(pad_n_, 0.0f);
        for (std::uint32_t i = 0; i < sketch_dim_; ++i) {
            buf[i] = static_cast<float>(sketch[i]) * signs_[i];
        }
        fwht_inplace(buf.data(), pad_n_);
        for (std::uint32_t r = 0; r < output_dim_; ++r) {
            out[r] = buf[sample_[r]] * fwht_scale_;
        }
        return;
    }

    if (mode_ == ProjectionMode::SparseJL) {
        // Same gather pattern as VerySparse but the per-row weights are
        // ±1/sqrt(s) and there is no global inv_scale: the scale is
        // already baked into the row weights so the JL distortion bound
        // is k-independent of output_dim. Apply step is therefore one
        // gather + s FMAs per row, total work O(output_dim * s).
        for (std::uint32_t row = 0; row < output_dim_; ++row) {
            const auto& sr = sparse_[row];
            float acc = 0.0f;
            const auto* cols = sr.cols.data();
            const auto* ws = sr.weights.data();
            const std::size_t n = sr.cols.size();
            for (std::size_t i = 0; i < n; ++i) {
                acc += ws[i] * static_cast<float>(sketch[cols[i]]);
            }
            out[row] = acc;
        }
        return;
    }

    // Dense Gaussian: convert sketch once, then process 4 output rows per
    // sketch_f sweep so each `sf[col]` load amortizes over 4 FMAs. This
    // reuses the input across rows and turns the per-doc work into a small
    // GEMM-like shape (1xK * Kx4) that saturates NEON's FMA throughput.
    thread_local std::vector<float> sketch_f;
    sketch_f.resize(sketch_dim_);
    for (std::uint32_t i = 0; i < sketch_dim_; ++i) {
        sketch_f[i] = static_cast<float>(sketch[i]);
    }
    const float* sf = sketch_f.data();
    const std::uint32_t sd = sketch_dim_;
    std::uint32_t row = 0;
#if defined(SIMEON_HAS_NEON)
    for (; row + 4 <= output_dim_; row += 4) {
        const float* r0p = dense_.data() + static_cast<std::size_t>(row + 0) * sd;
        const float* r1p = dense_.data() + static_cast<std::size_t>(row + 1) * sd;
        const float* r2p = dense_.data() + static_cast<std::size_t>(row + 2) * sd;
        const float* r3p = dense_.data() + static_cast<std::size_t>(row + 3) * sd;
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        std::uint32_t col = 0;
        for (; col + 8 <= sd; col += 8) {
            float32x4_t s0 = vld1q_f32(sf + col);
            float32x4_t s1 = vld1q_f32(sf + col + 4);
            a0 = vfmaq_f32(a0, vld1q_f32(r0p + col), s0);
            a0 = vfmaq_f32(a0, vld1q_f32(r0p + col + 4), s1);
            a1 = vfmaq_f32(a1, vld1q_f32(r1p + col), s0);
            a1 = vfmaq_f32(a1, vld1q_f32(r1p + col + 4), s1);
            a2 = vfmaq_f32(a2, vld1q_f32(r2p + col), s0);
            a2 = vfmaq_f32(a2, vld1q_f32(r2p + col + 4), s1);
            a3 = vfmaq_f32(a3, vld1q_f32(r3p + col), s0);
            a3 = vfmaq_f32(a3, vld1q_f32(r3p + col + 4), s1);
        }
        float s0 = vaddvq_f32(a0);
        float s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2);
        float s3 = vaddvq_f32(a3);
        for (; col < sd; ++col) {
            s0 += r0p[col] * sf[col];
            s1 += r1p[col] * sf[col];
            s2 += r2p[col] * sf[col];
            s3 += r3p[col] * sf[col];
        }
        out[row + 0] = s0 * inv_scale_;
        out[row + 1] = s1 * inv_scale_;
        out[row + 2] = s2 * inv_scale_;
        out[row + 3] = s3 * inv_scale_;
    }
#endif
    for (; row < output_dim_; ++row) {
        const float* row_ptr = dense_.data() + static_cast<std::size_t>(row) * sd;
        float acc = 0.0f;
        std::uint32_t col = 0;
#if defined(SIMEON_HAS_NEON)
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        for (; col + 8 <= sd; col += 8) {
            a0 = vfmaq_f32(a0, vld1q_f32(row_ptr + col), vld1q_f32(sf + col));
            a1 = vfmaq_f32(a1, vld1q_f32(row_ptr + col + 4), vld1q_f32(sf + col + 4));
        }
        acc = vaddvq_f32(vaddq_f32(a0, a1));
#elif defined(SIMEON_HAS_AVX2)
        __m256 a0 = _mm256_setzero_ps();
        for (; col + 8 <= sd; col += 8) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(row_ptr + col), _mm256_loadu_ps(sf + col), a0);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, a0);
        for (int k = 0; k < 8; ++k)
            acc += tmp[k];
#endif
        for (; col < sd; ++col)
            acc += row_ptr[col] * sf[col];
        out[row] = acc * inv_scale_;
    }
}

} // namespace simeon
