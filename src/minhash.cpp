#include "simeon/minhash.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "simeon/hasher.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

constexpr std::uint32_t kEmpty = std::numeric_limits<std::uint32_t>::max();

struct MinSink final : NGramEmitter {
    std::uint32_t* slots;
    std::uint32_t k;
    HashFamily family;
    std::uint64_t seed;

    void on_token(std::string_view tok, float) override {
        const std::uint64_t h = hash64(tok, seed, family);
        const std::uint32_t bin = static_cast<std::uint32_t>(h % k);
        // Use the high 32 bits as the within-bin value. Splitting like this
        // gives a uniform-in-bin value uncorrelated with the bin assignment,
        // which is the correctness condition for one-permutation MinHash.
        const std::uint32_t val = static_cast<std::uint32_t>(h >> 32);
        if (val < slots[bin])
            slots[bin] = val;
    }
};

// Densification: for every empty slot, look at slot (i + r * step) mod k for
// r = 1, 2, 3, ... until a non-empty slot is found. Combine its value with the
// attempt count `r` so that two signatures with the same neighbour land on
// the same final value iff they agree on both the neighbour and the attempt
// index. This is the optimal-densification scheme from §3 of Shrivastava
// 2017 (arXiv:1703.04664) — variance matches classical k-permutation MinHash.
//
// `step` is an odd seed-derived stride; coprime with k for any k when k is
// odd, and for even k still hits every bin in at most k iterations because
// the rotation-attempt index r breaks the cycle. To keep the linear-time
// guarantee we cap attempts at k and fall back to the seed itself, which
// can only happen on an all-empty signature (text shorter than n_min).
void densify(std::uint32_t* slots, std::uint32_t k, std::uint64_t seed) noexcept {
    // Stride: any odd value derived from the seed. Hashing keeps it spread
    // out so consecutive seeds don't produce near-identical strides.
    const std::uint32_t step = static_cast<std::uint32_t>(splitmix64_mix(seed) | 1ULL);

    for (std::uint32_t i = 0; i < k; ++i) {
        if (slots[i] != kEmpty)
            continue;
        std::uint32_t r = 1;
        for (; r <= k; ++r) {
            const std::uint32_t j = (i + r * step) % k;
            if (slots[j] != kEmpty) {
                // Mix the donor value with the attempt index so that
                // donors at different rotation distances are distinguishable.
                const std::uint64_t mixed =
                    splitmix64_mix((static_cast<std::uint64_t>(slots[j]) << 32) | r);
                slots[i] = static_cast<std::uint32_t>(mixed);
                break;
            }
        }
        if (r > k) {
            // Entire signature was empty (unusable input). Fill with the
            // mixed seed so two empty signatures collide deterministically.
            const std::uint32_t fill = static_cast<std::uint32_t>(splitmix64_mix(seed ^ i));
            slots[i] = fill;
        }
    }
}

} // namespace

MinHashEncoder::MinHashEncoder(MinHashConfig cfg) noexcept : cfg_(cfg) {}

void MinHashEncoder::encode(std::string_view text, std::uint32_t* out) const {
    if (cfg_.k == 0)
        throw std::runtime_error("MinHashEncoder: k must be > 0");
    std::fill_n(out, cfg_.k, kEmpty);

    MinSink sink{};
    sink.slots = out;
    sink.k = cfg_.k;
    sink.family = cfg_.hash;
    sink.seed = cfg_.hash_seed;

    const TokenizerConfig tcfg{cfg_.ngram_min, cfg_.ngram_max, true, false};
    tokenize(text, tcfg, sink);

    densify(out, cfg_.k, cfg_.hash_seed);
}

float jaccard_estimate(const std::uint32_t* a, const std::uint32_t* b, std::uint32_t k) noexcept {
    if (k == 0)
        return 0.0f;
    std::uint32_t matches = 0;
    for (std::uint32_t i = 0; i < k; ++i) {
        if (a[i] == b[i])
            ++matches;
    }
    return static_cast<float>(matches) / static_cast<float>(k);
}

} // namespace simeon
