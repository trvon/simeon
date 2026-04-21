#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace simeon {

// TextRank sentence ranker (Mihalcea & Tarau 2004). Splits text into
// sentences at ASCII sentence terminators and paragraph breaks, builds a
// graph whose edge weight is the token-overlap similarity between
// sentence pairs, and runs damped power iteration to convergence. Top-k
// sentences (by score) are returned, each as a view into the caller's
// input buffer.
//
// Used by yams as the fallback title-extractor for plain text with no
// structural title (see GLiNER-replacement plan). Deterministic and
// training-free; no external dependencies.

struct TextRankConfig {
    // Damping factor (probability of NOT teleporting). Mihalcea's paper
    // uses 0.85, matching the PageRank original.
    float damping = 0.85f;

    // Power iteration cap and convergence epsilon (l1 score delta).
    std::uint32_t max_iters = 30;
    float convergence_epsilon = 1e-4f;

    // Sentences shorter than this (in word tokens) are dropped before
    // ranking — typically fragments or noise lines in plain text.
    std::uint32_t min_sentence_tokens = 3;

    // Hard cap on sentence length. Longer sentences are truncated in
    // the token-set comparison; this bounds the O(nS²) graph build.
    std::uint32_t max_sentence_tokens = 60;

    // Hard cap on sentences considered. Beyond this, excess sentences
    // are dropped from the end of the input (graph size is O(cap²)).
    std::uint32_t max_sentences = 256;
};

struct RankedSentence {
    // View into the caller's input buffer — valid only as long as
    // that buffer is alive.
    std::string_view text;
    float score;
    std::uint32_t index; // 0-based sentence position in the source
};

class TextRank {
public:
    explicit TextRank(TextRankConfig cfg = {}) noexcept : cfg_(cfg) {}

    // Rank sentences in `text` and return the top_k by score.
    // `top_k == 0` is treated as "return all sentences ranked".
    // Non-throwing; returns empty vector on empty input or no valid
    // sentences after filtering.
    std::vector<RankedSentence> rank(std::string_view text, std::size_t top_k = 1) const;

    // Return just the top-1 sentence text, or empty view if none.
    // Convenience for the title-extraction use case.
    std::string_view top_sentence(std::string_view text) const;

    const TextRankConfig& config() const noexcept { return cfg_; }

private:
    TextRankConfig cfg_;
};

} // namespace simeon
