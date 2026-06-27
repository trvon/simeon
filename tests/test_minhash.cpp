#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "simeon/minhash.hpp"
#include "simeon/tokenizer.hpp"

using simeon::MinHashConfig;
using simeon::MinHashEncoder;

namespace {

// Build a deterministic random ASCII string from a vocab pool, with the
// requested per-word overlap fraction against `base`.
std::string make_string_with_jaccard(const std::vector<std::string>& vocab,
                                     std::uint32_t total_words, std::uint32_t shared_words,
                                     std::uint32_t seed) {
    assert(shared_words <= total_words);
    assert(total_words <= vocab.size());
    std::mt19937 rng(seed);
    std::vector<std::uint32_t> idx(vocab.size());
    for (std::uint32_t i = 0; i < vocab.size(); ++i)
        idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    std::string out;
    for (std::uint32_t i = 0; i < total_words; ++i) {
        if (!out.empty())
            out += ' ';
        if (i < shared_words) {
            out += vocab[i]; // shared prefix slot of vocab
        } else {
            out += vocab[idx[i]];
        }
    }
    return out;
}

std::vector<std::string> make_vocab(std::uint32_t n) {
    std::vector<std::string> v;
    v.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        // Each "word" is six lower-case chars derived from i, ensuring
        // n-gram tokens (n=3..5) cover the whole word.
        std::string w(6, 'a');
        std::uint32_t x = i + 1;
        for (int p = 0; p < 6; ++p) {
            w[p] = static_cast<char>('a' + (x % 26));
            x = x * 1103515245u + 12345u;
        }
        v.push_back(std::move(w));
    }
    return v;
}

void test_determinism_same_input() {
    MinHashEncoder enc(MinHashConfig{});
    std::vector<std::uint32_t> a(256), b(256);
    enc.encode("the quick brown fox jumps over the lazy dog", a.data());
    enc.encode("the quick brown fox jumps over the lazy dog", b.data());
    assert(a == b);
}

void test_seed_sensitivity() {
    MinHashConfig c1{}, c2{};
    c2.hash_seed = c1.hash_seed ^ 0x1u;
    MinHashEncoder e1(c1), e2(c2);
    std::vector<std::uint32_t> a(256), b(256);
    e1.encode("hello world", a.data());
    e2.encode("hello world", b.data());
    // Different seed -> almost certainly different signatures.
    bool any_diff = false;
    for (std::uint32_t i = 0; i < 256; ++i) {
        if (a[i] != b[i]) {
            any_diff = true;
            break;
        }
    }
    assert(any_diff);
}

void test_jaccard_estimate_self() {
    MinHashEncoder enc(MinHashConfig{});
    std::vector<std::uint32_t> sig(256);
    enc.encode("identical text identical text identical text", sig.data());
    const float j = simeon::jaccard_estimate(sig.data(), sig.data(), 256);
    assert(std::fabs(j - 1.0f) < 1e-6f);
}

// Build the same char-ngram set the encoder uses, so we can compute the
// ground-truth Jaccard and compare it against the MinHash estimate. The
// encoder runs ngrams over the full string including spaces, so we replicate
// that here.
std::unordered_set<std::string> char_ngram_set(std::string_view s, std::uint32_t kmin,
                                               std::uint32_t kmax) {
    std::unordered_set<std::string> out;
    const std::size_t n = s.size();
    for (std::uint32_t k = kmin; k <= kmax; ++k) {
        if (n < k)
            continue;
        for (std::size_t i = 0; i + k <= n; ++i) {
            out.insert(std::string(s.substr(i, k)));
        }
    }
    return out;
}

float ground_truth_jaccard(std::string_view a, std::string_view b, std::uint32_t kmin,
                           std::uint32_t kmax) {
    auto sa = char_ngram_set(a, kmin, kmax);
    auto sb = char_ngram_set(b, kmin, kmax);
    if (sa.empty() && sb.empty())
        return 0.0f;
    std::uint32_t inter = 0;
    for (const auto& g : sa) {
        if (sb.count(g))
            ++inter;
    }
    const std::uint32_t uni = static_cast<std::uint32_t>(sa.size() + sb.size()) - inter;
    return static_cast<float>(inter) / static_cast<float>(uni);
}

// MinHash is an unbiased estimator: for k slots, std-error is ~sqrt(j(1-j)/k).
// Average across pairs at a known empirical Jaccard and assert the estimator
// mean is close to ground truth.
void test_jaccard_estimate_unbiased() {
    constexpr std::uint32_t k = 1024;
    constexpr std::uint32_t pairs = 30;
    auto vocab = make_vocab(40);

    MinHashConfig cfg;
    cfg.k = k;
    MinHashEncoder enc(cfg);

    double sum_est = 0.0;
    double sum_true = 0.0;
    std::vector<std::uint32_t> sa(k), sb(k);
    for (std::uint32_t i = 0; i < pairs; ++i) {
        auto ta = make_string_with_jaccard(vocab, 30, 15, 1000 + i);
        auto tb = make_string_with_jaccard(vocab, 30, 15, 9000 + i);
        enc.encode(ta, sa.data());
        enc.encode(tb, sb.data());
        sum_est += simeon::jaccard_estimate(sa.data(), sb.data(), k);
        sum_true += ground_truth_jaccard(ta, tb, cfg.ngram_min, cfg.ngram_max);
    }
    const double mean_est = sum_est / pairs;
    const double mean_true = sum_true / pairs;
    // Aggregated std-error scales by 1/sqrt(pairs * k); easily under 0.05.
    assert(std::fabs(mean_est - mean_true) < 0.05);
}

void test_densification_no_empty_slots() {
    // Tiny input - many bins will be empty after the per-token pass.
    MinHashConfig cfg;
    cfg.k = 256;
    MinHashEncoder enc(cfg);
    std::vector<std::uint32_t> sig(256);
    enc.encode("abc", sig.data()); // single 3-gram
    // Densification must fill every slot.
    constexpr std::uint32_t kEmpty = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t i = 0; i < 256; ++i) {
        assert(sig[i] != kEmpty);
    }
}

void test_densification_short_text_stable() {
    // Two encodes of the same short text must agree slot-for-slot, meaning
    // the densification fill-in path is deterministic.
    MinHashConfig cfg;
    cfg.k = 256;
    MinHashEncoder enc(cfg);
    std::vector<std::uint32_t> a(256), b(256);
    enc.encode("ab", a.data()); // text shorter than ngram_min -> all empty
    enc.encode("ab", b.data());
    assert(a == b);
}

void test_jaccard_disjoint_low_score() {
    // Two strings with no shared word substrings -> Jaccard near zero.
    constexpr std::uint32_t k = 512;
    MinHashConfig cfg;
    cfg.k = k;
    MinHashEncoder enc(cfg);
    std::vector<std::uint32_t> a(k), b(k);
    enc.encode("aaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbb", a.data());
    enc.encode("xxxxxxxxxxxxxxxxxxxx yyyyyyyyyyyyyyyyyyyy", b.data());
    const float j = simeon::jaccard_estimate(a.data(), b.data(), k);
    // Some accidental matches can happen via densification, but should be
    // small. 1/sqrt(k) ~ 0.044; tolerate up to a few times that.
    assert(j < 0.1f);
}

// Statistical regression guard on densification quality in the sparse regime
// (|tokens| << k, most bins densified). One-permutation MinHash cannot beat
// the information floor of ~|occupied bins| effective samples, so MSE sits a
// few-fold above the classical k-permutation bound J(1-J)/k here by design.
// Measured 2026-06: the shipped strided+mixed scheme lands at ~3.6x (4-word
// cell) / ~7.8x (2-word cell); a per-bin hashed-probe variant (textbook
// "optimal densification", arXiv:1703.04664) measured no better (9.2x on the
// 2-word cell), so the strided scheme stays. This test pins the 4-word cell
// at <6x to catch future regressions in donor selection.
void test_densification_variance_sparse() {
    constexpr std::uint32_t k = 512;
    MinHashConfig cfg;
    cfg.k = k;
    MinHashEncoder enc(cfg);

    struct SetSink final : simeon::NGramEmitter {
        std::unordered_set<std::string> toks;
        void on_token(std::string_view t, float) override { toks.emplace(t); }
    };
    const simeon::TokenizerConfig tcfg{cfg.ngram_min, cfg.ngram_max, true, false};

    auto vocab = make_vocab(64);

    double se_sum = 0.0;
    double bound_sum = 0.0;
    std::uint32_t n_pairs = 0;

    std::vector<std::uint32_t> sig_a(k);
    std::vector<std::uint32_t> sig_b(k);

    for (std::uint32_t trial = 0; trial < 400; ++trial) {
        const auto text_a = make_string_with_jaccard(vocab, 4, 2, 1000 + trial * 2);
        const auto text_b = make_string_with_jaccard(vocab, 4, 2, 1001 + trial * 2);

        SetSink sa;
        SetSink sb;
        simeon::tokenize(text_a, tcfg, sa);
        simeon::tokenize(text_b, tcfg, sb);
        if (sa.toks.empty() || sb.toks.empty())
            continue;

        std::uint32_t inter = 0;
        for (const auto& t : sa.toks) {
            if (sb.toks.count(t))
                ++inter;
        }
        const std::uint32_t uni =
            static_cast<std::uint32_t>(sa.toks.size() + sb.toks.size()) - inter;
        const double j_exact = static_cast<double>(inter) / static_cast<double>(uni);
        if (j_exact <= 0.02 || j_exact >= 0.98)
            continue;

        enc.encode(text_a, sig_a.data());
        enc.encode(text_b, sig_b.data());
        const double j_est = simeon::jaccard_estimate(sig_a.data(), sig_b.data(), k);

        const double err = j_est - j_exact;
        se_sum += err * err;
        bound_sum += j_exact * (1.0 - j_exact) / static_cast<double>(k);
        ++n_pairs;
    }

    assert(n_pairs > 300);
    const double mse = se_sum / n_pairs;
    const double classical = bound_sum / n_pairs;
    const double inflation = mse / classical;
    std::printf("  densification sparse-regime MSE inflation vs classical: %.2fx "
                "(mse=%.3e classical=%.3e, %u pairs)\n",
                inflation, mse, classical, n_pairs);
    assert(inflation < 6.0 && "densification MSE regressed past the measured "
                              "one-permutation floor for this cell");
}

} // namespace

int main() {
    test_determinism_same_input();
    test_seed_sensitivity();
    test_jaccard_estimate_self();
    test_jaccard_estimate_unbiased();
    test_densification_no_empty_slots();
    test_densification_short_text_stable();
    test_jaccard_disjoint_low_score();
    test_densification_variance_sparse();
    return 0;
}
