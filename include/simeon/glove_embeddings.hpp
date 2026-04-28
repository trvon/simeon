#pragma once

#include <cstdint>
#include <string>

#include "simeon/pmi.hpp"

namespace simeon {

// Load pre-trained GloVe or fastText text-format vectors into a PmiEmbeddings
// object so they can be plugged into EncoderConfig::pmi_rows unchanged.
//
// File format: one vector per line, `word f1 f2 ... fn\n`. An optional leading
// `vocab_size dim` header line (some distributions) is skipped automatically.
// Both GloVe (Stanford) and fastText text export (Facebook) use this layout.
//
// max_vocab: stop after loading this many words (0 = unlimited). GloVe files
// are sorted by corpus frequency (most frequent first); 200 000 covers the
// content-word head that simeon's PMI tokenizer sees.
//
// All rows are L2-normalized on load — matching PmiEmbeddings::learn() output.
PmiEmbeddings load_glove(const std::string& path, std::uint32_t max_vocab = 0);

} // namespace simeon
