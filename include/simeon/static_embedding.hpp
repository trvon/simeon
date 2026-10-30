#pragma once

// Static (lookup) text embeddings: WordPiece tokens -> mean of table rows -> L2 normalize.
//
// The table is a SMSTAT01 file (see tools/static_embedding/export_static_table.py), typically exported from a
// model2vec model with PCA and per-row int8. Tokenization follows HuggingFace's BertNormalizer,
// BertPreTokenizer and WordPiece. Limits: accent stripping and lowercasing are exact for Latin,
// Greek and Cyrillic (tables generated from Unicode data); other scripts pass through unchanged.

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace simeon {

class StaticEmbedding {
public:
    // Parses a SMSTAT01 table. Throws std::runtime_error on a malformed or truncated file.
    static StaticEmbedding load(const std::string& path);
    static StaticEmbedding from_bytes(std::span<const std::uint8_t> bytes);

    std::uint32_t dim() const noexcept { return dim_; }
    std::uint32_t vocab_size() const noexcept { return static_cast<std::uint32_t>(tokens_.size()); }
    // UINT32_MAX when the table has no unknown token.
    std::uint32_t unk_id() const noexcept { return unk_id_; }

    // WordPiece ids for `text`, unknown words included as unk_id().
    std::vector<std::uint32_t> tokenize(std::string_view text) const;

    // Mean of the rows of the known tokens of `text`, L2-normalized, written to `out` (dim()
    // floats). Returns false (and writes zeros) when no known token is found.
    bool encode(std::string_view text, std::span<float> out) const;
    std::vector<float> encode(std::string_view text) const;

private:
    std::string normalize(std::string_view text) const;
    void wordpiece(std::string_view word, std::vector<std::uint32_t>& out) const;

    std::uint32_t dim_ = 0;
    std::uint32_t unk_id_ = UINT32_MAX;
    std::uint32_t max_word_chars_ = 100;
    // Longest vocabulary piece in code points ("##" excluded); longer substrings cannot match.
    std::size_t max_piece_chars_ = 0;
    bool lowercase_ = true;
    bool strip_accents_ = true;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, std::uint32_t> ids_;
    std::vector<float> scales_;
    std::vector<std::int8_t> rows_;
};

} // namespace simeon
