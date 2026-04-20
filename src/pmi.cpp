#include "simeon/pmi.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace simeon {

namespace {

// SplitMix64 PRNG, inlined so pmi.cpp stays self-contained from the SIMD
// and hasher TUs. Matches the seed advancement used elsewhere in simeon.
inline std::uint64_t splitmix64(std::uint64_t& state) noexcept {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Box-Muller standard normal. Consumes two PRNG draws per sample.
inline float gauss(std::uint64_t& state) noexcept {
    std::uint64_t a = splitmix64(state) | 1ULL;
    std::uint64_t b = splitmix64(state) | 1ULL;
    const float u0 = static_cast<float>(a >> 11) / static_cast<float>(1ULL << 53);
    const float u1 = static_cast<float>(b >> 11) / static_cast<float>(1ULL << 53);
    const float r = std::sqrt(-2.0f * std::log(u0));
    const float theta = 6.2831853071795864769f * u1;
    return r * std::cos(theta);
}

inline bool is_word_char(unsigned char c) noexcept {
    return std::isalnum(c) != 0 || c == '_';
}

// Walks `text` emitting lowercased word tokens via `cb(tok)`. Matches the
// tokenizer's `emit_word_tokens` word boundary rules.
template <typename Cb>
void for_each_word(std::string_view text, Cb&& cb) {
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        while (i < n && !is_word_char(static_cast<unsigned char>(text[i]))) ++i;
        const std::size_t start = i;
        while (i < n && is_word_char(static_cast<unsigned char>(text[i]))) ++i;
        if (start < i) {
            std::string tok(text.substr(start, i - start));
            for (auto& c : tok)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            cb(std::move(tok));
        }
    }
}

// CSR-like sparse symmetric matrix with one entry per (i, j) pair — both
// (i, j) and (j, i) are stored explicitly so matmul is a single row scan.
struct SparseSym {
    std::uint32_t n = 0;
    std::vector<std::uint32_t> row_ptr;  // size n+1
    std::vector<std::uint32_t> col_idx;  // size nnz
    std::vector<float> vals;             // size nnz
};

// y = M * x. x, y are n-by-cols column-major dense matrices. M is symmetric
// and both triangles are stored, so we walk rows once.
void spmm(const SparseSym& M, const float* x, std::uint32_t cols, float* y) {
    const std::uint32_t n = M.n;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint32_t beg = M.row_ptr[i];
        const std::uint32_t end = M.row_ptr[i + 1];
        for (std::uint32_t c = 0; c < cols; ++c) y[i + c * n] = 0.0f;
        for (std::uint32_t e = beg; e < end; ++e) {
            const std::uint32_t j = M.col_idx[e];
            const float v = M.vals[e];
            for (std::uint32_t c = 0; c < cols; ++c) {
                y[i + c * n] += v * x[j + c * n];
            }
        }
    }
}

// QR via Gram-Schmidt with one reorthogonalization pass for numerical
// stability. A is n x k column-major; overwritten with Q.
void qr_inplace(float* A, std::uint32_t n, std::uint32_t k) {
    for (std::uint32_t col = 0; col < k; ++col) {
        float* q = A + col * n;
        for (int pass = 0; pass < 2; ++pass) {
            for (std::uint32_t prev = 0; prev < col; ++prev) {
                const float* qp = A + prev * n;
                double dot = 0.0;
                for (std::uint32_t i = 0; i < n; ++i) dot += static_cast<double>(qp[i]) * q[i];
                const float d = static_cast<float>(dot);
                for (std::uint32_t i = 0; i < n; ++i) q[i] -= d * qp[i];
            }
        }
        double sq = 0.0;
        for (std::uint32_t i = 0; i < n; ++i) sq += static_cast<double>(q[i]) * q[i];
        const double nrm = std::sqrt(sq);
        if (nrm > 1e-12) {
            const float inv = static_cast<float>(1.0 / nrm);
            for (std::uint32_t i = 0; i < n; ++i) q[i] *= inv;
        } else {
            // Degenerate column — zero it. Rare on real corpora.
            for (std::uint32_t i = 0; i < n; ++i) q[i] = 0.0f;
        }
    }
}

// Jacobi eigendecomposition of a symmetric k x k matrix (column-major).
// On return, A is diagonal with eigenvalues, V holds eigenvectors in its
// columns. Unordered output.
void jacobi_eigh(std::vector<double>& A, std::vector<double>& V, std::uint32_t k) {
    V.assign(static_cast<std::size_t>(k) * k, 0.0);
    for (std::uint32_t i = 0; i < k; ++i) V[i + i * k] = 1.0;

    const std::uint32_t max_sweeps = 64;
    for (std::uint32_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (std::uint32_t p = 0; p < k; ++p) {
            for (std::uint32_t q = p + 1; q < k; ++q) {
                const double a = A[p + q * k];
                off += a * a;
            }
        }
        if (off < 1e-24) break;

        for (std::uint32_t p = 0; p < k - 1; ++p) {
            for (std::uint32_t q = p + 1; q < k; ++q) {
                const double apq = A[p + q * k];
                if (std::fabs(apq) < 1e-14) continue;
                const double app = A[p + p * k];
                const double aqq = A[q + q * k];
                const double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e18) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    t = (theta >= 0.0 ? 1.0 : -1.0) /
                        (std::fabs(theta) + std::sqrt(1.0 + theta * theta));
                }
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                A[p + p * k] = app - t * apq;
                A[q + q * k] = aqq + t * apq;
                A[p + q * k] = 0.0;
                A[q + p * k] = 0.0;

                for (std::uint32_t i = 0; i < k; ++i) {
                    if (i == p || i == q) continue;
                    const double aip = A[i + p * k];
                    const double aiq = A[i + q * k];
                    A[i + p * k] = c * aip - s * aiq;
                    A[p + i * k] = A[i + p * k];
                    A[i + q * k] = s * aip + c * aiq;
                    A[q + i * k] = A[i + q * k];
                }
                for (std::uint32_t i = 0; i < k; ++i) {
                    const double vip = V[i + p * k];
                    const double viq = V[i + q * k];
                    V[i + p * k] = c * vip - s * viq;
                    V[i + q * k] = s * vip + c * viq;
                }
            }
        }
    }
}

}  // namespace

PmiEmbeddings PmiEmbeddings::learn(std::span<const std::string_view> seed_corpus,
                                   const PmiConfig& cfg) {
    if (seed_corpus.empty()) {
        throw std::invalid_argument("PmiEmbeddings::learn: seed_corpus must not be empty");
    }
    if (cfg.target_rank == 0) {
        throw std::invalid_argument("PmiEmbeddings::learn: target_rank must be > 0");
    }
    if (cfg.window_size == 0) {
        throw std::invalid_argument("PmiEmbeddings::learn: window_size must be > 0");
    }
    if (cfg.max_vocab_size == 0) {
        throw std::invalid_argument("PmiEmbeddings::learn: max_vocab_size must be > 0");
    }

    // Pass 1 — tokenize each doc into lowercased words, count unigram
    // frequency, remember per-doc token sequences for pass 2.
    std::vector<std::vector<std::string>> docs_tokens;
    docs_tokens.reserve(seed_corpus.size());
    std::unordered_map<std::string, std::uint64_t> unigram;
    unigram.reserve(1024);
    for (auto text : seed_corpus) {
        std::vector<std::string> toks;
        for_each_word(text, [&](std::string&& t) {
            ++unigram[t];
            toks.push_back(std::move(t));
        });
        docs_tokens.push_back(std::move(toks));
    }

    // Apply min_token_count, then keep top max_vocab_size by frequency.
    std::vector<std::pair<std::string, std::uint64_t>> ranked;
    ranked.reserve(unigram.size());
    for (auto& kv : unigram) {
        if (kv.second >= cfg.min_token_count) {
            ranked.emplace_back(std::move(const_cast<std::string&>(kv.first)), kv.second);
        }
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) return a.second > b.second;
                  return a.first < b.first;  // break ties deterministically
              });
    if (ranked.size() > cfg.max_vocab_size) ranked.resize(cfg.max_vocab_size);
    if (ranked.empty()) {
        throw std::invalid_argument(
            "PmiEmbeddings::learn: no tokens met min_token_count threshold");
    }

    std::vector<std::string> vocab;
    std::unordered_map<std::string, std::uint32_t> index;
    vocab.reserve(ranked.size());
    index.reserve(ranked.size() * 2);
    for (std::uint32_t i = 0; i < ranked.size(); ++i) {
        vocab.push_back(ranked[i].first);
        index.emplace(ranked[i].first, i);
    }
    const std::uint32_t n = static_cast<std::uint32_t>(vocab.size());

    // Pass 2 — count symmetric co-occurrence within +/- window_size. Store
    // upper-triangular plus diagonal in a dense-hashed sparse map. For each
    // pair (i, j) we record c_ij once; the diagonal holds c_ii (same-token
    // co-occurrence within window).
    std::vector<std::uint64_t> per_tok(n, 0);  // marginal context counts
    std::uint64_t total_pairs = 0;
    std::unordered_map<std::uint64_t, std::uint64_t> pair_counts;
    pair_counts.reserve(static_cast<std::size_t>(n) * 8);

    const std::uint32_t w = cfg.window_size;
    for (const auto& toks : docs_tokens) {
        // Map each document token to its vocab id (or -1 if filtered out).
        std::vector<std::int32_t> ids;
        ids.reserve(toks.size());
        for (const auto& t : toks) {
            auto it = index.find(t);
            ids.push_back(it != index.end()
                              ? static_cast<std::int32_t>(it->second)
                              : -1);
        }
        const std::size_t m = ids.size();
        for (std::size_t i = 0; i < m; ++i) {
            if (ids[i] < 0) continue;
            const std::uint32_t ti = static_cast<std::uint32_t>(ids[i]);
            const std::size_t lo = (i > w) ? (i - w) : 0;
            const std::size_t hi = std::min(m, i + w + 1);
            for (std::size_t j = lo; j < hi; ++j) {
                if (j == i) continue;
                if (ids[j] < 0) continue;
                const std::uint32_t tj = static_cast<std::uint32_t>(ids[j]);
                per_tok[ti] += 1;
                total_pairs += 1;
                // Canonical (a, b) with a <= b so each pair is counted once
                // across both directions — the matrix is symmetric.
                const std::uint32_t a = std::min(ti, tj);
                const std::uint32_t b = std::max(ti, tj);
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
                pair_counts[key] += 1;
            }
        }
    }

    if (total_pairs == 0) {
        throw std::invalid_argument(
            "PmiEmbeddings::learn: no co-occurrence pairs in seed_corpus");
    }

    // Build SPPMI CSR. For each pair (a, b) compute
    //   pmi = log( (c_ab * D) / (c_a * c_b) ) - log(k)
    // where D = total_pairs counts both directions (since we incremented
    // per_tok[ti] for each neighbour visit). Clip to max(0, pmi).
    const double D = static_cast<double>(total_pairs);
    const double shift = static_cast<double>(cfg.shift_log_k);

    // Collect rows as (row, col, val) triples first, then compact to CSR.
    std::vector<std::vector<std::pair<std::uint32_t, float>>> rows_tmp(n);
    for (const auto& kv : pair_counts) {
        const std::uint64_t key = kv.first;
        const std::uint64_t cnt = kv.second;
        const std::uint32_t a = static_cast<std::uint32_t>(key >> 32);
        const std::uint32_t b = static_cast<std::uint32_t>(key & 0xFFFFFFFFULL);
        const double c_ab = 2.0 * static_cast<double>(cnt);  // a->b and b->a
        if (a == b) {
            // Diagonal: pair counted once per i (since we skipped i==j).
            // Still symmetric; only store once in the diagonal slot.
            const double c_a = static_cast<double>(per_tok[a]);
            if (c_a <= 0.0) continue;
            const double pmi = std::log((static_cast<double>(cnt) * D) / (c_a * c_a)) - shift;
            if (pmi > 0.0) rows_tmp[a].emplace_back(a, static_cast<float>(pmi));
            continue;
        }
        const double c_a = static_cast<double>(per_tok[a]);
        const double c_b = static_cast<double>(per_tok[b]);
        if (c_a <= 0.0 || c_b <= 0.0) continue;
        const double pmi = std::log((c_ab * D) / (c_a * c_b)) - shift;
        if (pmi <= 0.0) continue;
        const float v = static_cast<float>(pmi);
        rows_tmp[a].emplace_back(b, v);
        rows_tmp[b].emplace_back(a, v);
    }

    SparseSym M;
    M.n = n;
    M.row_ptr.resize(static_cast<std::size_t>(n) + 1, 0);
    std::uint32_t nnz = 0;
    for (std::uint32_t i = 0; i < n; ++i) nnz += static_cast<std::uint32_t>(rows_tmp[i].size());
    M.col_idx.resize(nnz);
    M.vals.resize(nnz);
    std::uint32_t pos = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        M.row_ptr[i] = pos;
        auto& r = rows_tmp[i];
        std::sort(r.begin(), r.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        for (const auto& e : r) {
            M.col_idx[pos] = e.first;
            M.vals[pos] = e.second;
            ++pos;
        }
        rows_tmp[i].clear();
        rows_tmp[i].shrink_to_fit();
    }
    M.row_ptr[n] = pos;

    // Randomized SVD / eigendecomposition. SPPMI is symmetric, so we can
    // run the symmetric range-finder followed by a small Jacobi eigh.
    // Clamp `r + oversample` to the vocab size — small seed corpora
    // legitimately produce vocabularies below the configured rank.
    std::uint32_t r = cfg.target_rank;
    if (r > n) r = n;
    std::uint32_t k = r + cfg.svd_oversample;
    if (k > n) k = n;

    // Draw Gaussian test matrix Ω (n x k) and compute Y = M Ω.
    std::uint64_t rng = cfg.svd_seed;
    std::vector<float> Omega(static_cast<std::size_t>(n) * k);
    for (auto& x : Omega) x = gauss(rng);
    std::vector<float> Y(static_cast<std::size_t>(n) * k, 0.0f);
    spmm(M, Omega.data(), k, Y.data());
    qr_inplace(Y.data(), n, k);

    // Power iterations: Y = M M Y with re-orthonormalization between each
    // multiply. M is symmetric so M^T M = M * M.
    std::vector<float> tmp(static_cast<std::size_t>(n) * k, 0.0f);
    for (std::uint32_t it = 0; it < cfg.svd_iters; ++it) {
        spmm(M, Y.data(), k, tmp.data());
        qr_inplace(tmp.data(), n, k);
        spmm(M, tmp.data(), k, Y.data());
        qr_inplace(Y.data(), n, k);
    }

    // Q = Y after final QR. Small matrix C = Q^T M Q is k x k symmetric;
    // eigendecompose.
    std::vector<float> MQ(static_cast<std::size_t>(n) * k, 0.0f);
    spmm(M, Y.data(), k, MQ.data());

    std::vector<double> C(static_cast<std::size_t>(k) * k, 0.0);
    for (std::uint32_t p = 0; p < k; ++p) {
        for (std::uint32_t q = 0; q < k; ++q) {
            double s = 0.0;
            const float* qp = Y.data() + static_cast<std::size_t>(p) * n;
            const float* mq = MQ.data() + static_cast<std::size_t>(q) * n;
            for (std::uint32_t i = 0; i < n; ++i) s += static_cast<double>(qp[i]) * mq[i];
            C[p + q * k] = s;
        }
    }
    // Symmetrize to absorb FP noise.
    for (std::uint32_t p = 0; p < k; ++p) {
        for (std::uint32_t q = p + 1; q < k; ++q) {
            const double avg = 0.5 * (C[p + q * k] + C[q + p * k]);
            C[p + q * k] = avg;
            C[q + p * k] = avg;
        }
    }
    std::vector<double> Vk;
    jacobi_eigh(C, Vk, k);

    // Pick the top r eigenvalues by |λ| (SPPMI is PSD-ish but not
    // guaranteed PSD after shifting). Row embeddings: U = Q Vk, then
    // scale column j by sqrt(max(λ_j, 0)) per Levy-Goldberg's symmetric
    // factorization.
    std::vector<std::pair<double, std::uint32_t>> order;
    order.reserve(k);
    for (std::uint32_t j = 0; j < k; ++j) order.emplace_back(C[j + j * k], j);
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (order.size() > r) order.resize(r);

    PmiEmbeddings out;
    out.dim_ = r;
    out.vocab_ = std::move(vocab);
    out.index_ = std::move(index);
    out.rows_.assign(static_cast<std::size_t>(n) * r, 0.0f);

    std::vector<float> scale(r);
    for (std::uint32_t j = 0; j < r; ++j) {
        const double lam = order[j].first;
        scale[j] = static_cast<float>(std::sqrt(lam > 0.0 ? lam : 0.0));
    }

    // U[:, j] = Q * Vk[:, order[j].second], then multiply by scale[j].
    for (std::uint32_t j = 0; j < r; ++j) {
        const std::uint32_t src = order[j].second;
        const float s = scale[j];
        for (std::uint32_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::uint32_t p = 0; p < k; ++p) {
                acc += static_cast<double>(Y[i + p * n]) * Vk[p + src * k];
            }
            out.rows_[static_cast<std::size_t>(i) * r + j] = static_cast<float>(acc) * s;
        }
    }

    return out;
}

const float* PmiEmbeddings::row(std::string_view tok) const noexcept {
    // Lowercase into a stack-allocated buffer up to 64 bytes; spill to the
    // heap for longer tokens. Keeps the hot path allocation-free for
    // typical word lengths.
    char stack_buf[64];
    std::string heap_buf;
    char* buf;
    const std::size_t n = tok.size();
    if (n <= sizeof(stack_buf)) {
        buf = stack_buf;
    } else {
        heap_buf.resize(n);
        buf = heap_buf.data();
    }
    for (std::size_t i = 0; i < n; ++i) {
        buf[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(tok[i])));
    }
    auto it = index_.find(std::string(buf, n));
    if (it == index_.end()) return nullptr;
    return rows_.data() + static_cast<std::size_t>(it->second) * dim_;
}

namespace {

constexpr char kMagic[8] = {'S', 'M', 'E', 'P', 'M', 'I', '0', '1'};

void write_u32(std::string& out, std::uint32_t v) {
    char bytes[4];
    bytes[0] = static_cast<char>(v & 0xFF);
    bytes[1] = static_cast<char>((v >> 8) & 0xFF);
    bytes[2] = static_cast<char>((v >> 16) & 0xFF);
    bytes[3] = static_cast<char>((v >> 24) & 0xFF);
    out.append(bytes, 4);
}

void write_f32(std::string& out, float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    write_u32(out, u);
}

std::uint32_t read_u32(const char* p) noexcept {
    return (static_cast<std::uint32_t>(static_cast<unsigned char>(p[0]))) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(p[1])) << 8) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(p[2])) << 16) |
           (static_cast<std::uint32_t>(static_cast<unsigned char>(p[3])) << 24);
}

float read_f32(const char* p) noexcept {
    std::uint32_t u = read_u32(p);
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

}  // namespace

std::string PmiEmbeddings::serialize() const {
    std::string out;
    out.reserve(16 + vocab_.size() * 16 + rows_.size() * 4);
    out.append(kMagic, sizeof(kMagic));
    write_u32(out, dim_);
    write_u32(out, static_cast<std::uint32_t>(vocab_.size()));
    for (const auto& tok : vocab_) {
        write_u32(out, static_cast<std::uint32_t>(tok.size()));
        out.append(tok);
    }
    for (float v : rows_) write_f32(out, v);
    return out;
}

PmiEmbeddings PmiEmbeddings::from_bytes(std::string_view bytes) {
    if (bytes.size() < 16) {
        throw std::invalid_argument("PmiEmbeddings::from_bytes: truncated header");
    }
    if (std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
        throw std::invalid_argument("PmiEmbeddings::from_bytes: bad magic");
    }
    const std::uint32_t dim = read_u32(bytes.data() + 8);
    const std::uint32_t vn = read_u32(bytes.data() + 12);
    std::size_t cur = 16;
    PmiEmbeddings out;
    out.dim_ = dim;
    out.vocab_.reserve(vn);
    out.index_.reserve(vn * 2);
    for (std::uint32_t i = 0; i < vn; ++i) {
        if (cur + 4 > bytes.size()) {
            throw std::invalid_argument("PmiEmbeddings::from_bytes: truncated vocab");
        }
        const std::uint32_t len = read_u32(bytes.data() + cur);
        cur += 4;
        if (cur + len > bytes.size()) {
            throw std::invalid_argument("PmiEmbeddings::from_bytes: truncated token");
        }
        std::string tok(bytes.data() + cur, len);
        cur += len;
        out.index_.emplace(tok, i);
        out.vocab_.push_back(std::move(tok));
    }
    const std::size_t need = static_cast<std::size_t>(vn) * dim * 4;
    if (cur + need != bytes.size()) {
        throw std::invalid_argument("PmiEmbeddings::from_bytes: row payload size mismatch");
    }
    out.rows_.resize(static_cast<std::size_t>(vn) * dim);
    for (std::size_t i = 0; i < out.rows_.size(); ++i) {
        out.rows_[i] = read_f32(bytes.data() + cur);
        cur += 4;
    }
    return out;
}

}  // namespace simeon
