#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "simeon/simeon.hpp"

namespace simeon {

// Compute data-aware matryoshka row weights from a seed corpus.
//
// Builds a temporary encoder from `cfg` (with matryoshka *off* and L2-norm
// *off* so we observe raw projected energy), encodes each seed-corpus doc,
// then returns a per-row weight equal to sqrt(per-row variance) across the
// corpus. Rows with more variance — i.e. more retrieval signal — get larger
// weights, biasing energy toward those rows when the resulting weight vector
// is plugged into `EncoderConfig::matryoshka_weights`.
//
// Weights are L1-normalized so their mean is 1.0; this keeps the post-weight
// vector roughly the same magnitude as the unweighted projection, which
// matters before the encoder's L2 normalization step.
//
// The returned vector has length `cfg.output_dim` (or `cfg.sketch_dim` if
// projection is None).
//
// `cfg` must satisfy `projection != None` — matryoshka weights only make
// sense over a fixed-dim projected space.
std::vector<float> compute_matryoshka_weights(const EncoderConfig& cfg,
                                              std::span<const std::string_view> seed_corpus);

}  // namespace simeon
