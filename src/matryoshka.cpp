#include "simeon/matryoshka.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "simeon/simeon.hpp"

namespace simeon {

std::vector<float> compute_matryoshka_weights(
    const EncoderConfig& cfg, std::span<const std::string_view> seed_corpus) {
    if (seed_corpus.empty()) {
        throw std::invalid_argument(
            "simeon::compute_matryoshka_weights: seed_corpus must not be empty");
    }

    // Encode the seed corpus with a "raw" config: matryoshka off so we observe
    // the unbiased projected energy, and L2 normalize off so the variance we
    // compute reflects per-row signal magnitude (not just direction).
    EncoderConfig probe = cfg;
    probe.matryoshka = false;
    probe.matryoshka_weights.clear();
    probe.l2_normalize = false;

    Encoder enc(probe);
    const std::uint32_t d = enc.output_dim();

    std::vector<double> mean(d, 0.0);
    std::vector<double> sq(d, 0.0);
    std::vector<float> tmp(d, 0.0f);

    const double n = static_cast<double>(seed_corpus.size());
    for (auto text : seed_corpus) {
        enc.encode(text, tmp.data());
        for (std::uint32_t r = 0; r < d; ++r) {
            const double v = static_cast<double>(tmp[r]);
            mean[r] += v;
            sq[r] += v * v;
        }
    }

    std::vector<float> weights(d, 0.0f);
    double sum = 0.0;
    for (std::uint32_t r = 0; r < d; ++r) {
        const double mu = mean[r] / n;
        const double var = sq[r] / n - mu * mu;
        // Floor at 0 to absorb tiny negative noise from FP cancellation.
        const double w = std::sqrt(var > 0.0 ? var : 0.0);
        weights[r] = static_cast<float>(w);
        sum += w;
    }

    // L1-normalize so mean weight = 1. Keeps post-weight magnitude comparable
    // to the unweighted projection — which matters before encoder L2 norm.
    if (sum > 0.0) {
        const double scale = static_cast<double>(d) / sum;
        for (auto& w : weights) {
            w = static_cast<float>(static_cast<double>(w) * scale);
        }
    } else {
        // Degenerate: all-zero variance (e.g. empty texts). Fall back to
        // uniform weights so encoder still produces a sane vector.
        for (auto& w : weights) w = 1.0f;
    }

    return weights;
}

}  // namespace simeon
