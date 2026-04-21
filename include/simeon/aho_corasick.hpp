#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simeon {

// Aho-Corasick dictionary matcher (Aho & Corasick 1975). Trie of patterns +
// failure links + output links; single-pass scan of input yields
// longest-match non-overlapping spans.
//
// Designed for the GLiNER-replacement path: dictionaries of size up to ~2M
// patterns (UMLS + Wikidata tech subset), single-threaded ingest-time
// throughput ≥50 MB/s on M-series. No SIMD required — AC is
// memory-bound on the goto transition table, not compute-bound.
//
// Deterministic: same input yields byte-identical AcMatch sequence.

struct AhoCorasickConfig {
    // ASCII-only case folding (A..Z → a..z) at build time and at match
    // time. Non-ASCII bytes pass through unchanged, which keeps the
    // implementation UTF-8 safe (ASCII is a subset of valid UTF-8).
    bool case_insensitive = true;

    // When true, a match span must begin and end on a word boundary —
    // i.e. the byte immediately before begin and immediately after end
    // must be end-of-string or non-alphanumeric ASCII. Prevents
    // substring hits like "cat" inside "catalog".
    bool whole_word = true;

    // Safety cap on pattern count; build() returns an error if exceeded.
    std::uint32_t max_patterns = 2'000'000;

    // When true (default), emit only the longest match anchored at each
    // position (classic non-overlapping longest-match). When false, emit
    // every pattern occurrence (overlaps allowed).
    bool longest_match = true;
};

struct AcMatch {
    std::uint32_t pattern_id; // index into the build() input
    std::uint32_t begin_utf8; // byte offset into input
    std::uint32_t end_utf8;   // byte offset just past the match
    std::uint16_t type_id;    // caller-provided class tag (e.g. UMLS semantic type)
};

struct AhoCorasickBuildError {
    std::string message;
};

class AhoCorasick {
public:
    explicit AhoCorasick(AhoCorasickConfig cfg = {}) noexcept;

    // Build the automaton. Patterns and type_ids must have equal length;
    // type_ids[i] is associated with pattern i. Empty patterns are
    // rejected. Duplicates are allowed (last type_id wins on overlap).
    // Returns error on empty input or cap exceeded; the instance is
    // reset to an empty state on error.
    [[nodiscard]] std::optional<AhoCorasickBuildError>
    build(std::span<const std::string_view> patterns, std::span<const std::uint16_t> type_ids);

    // Scan `text` and emit matches. Non-throwing, reentrant.
    std::vector<AcMatch> match(std::string_view text) const;

    // Accessors for bench / test introspection.
    std::size_t pattern_count() const noexcept { return pattern_count_; }
    std::size_t node_count() const noexcept { return next_.size() / 256; }

    const AhoCorasickConfig& config() const noexcept { return cfg_; }

    // Reset the automaton to the empty state. Not normally needed —
    // build() already resets before constructing. Exposed for tests.
    void clear() noexcept;

private:
    AhoCorasickConfig cfg_;

    // Dense goto table: next_[state * 256 + byte] = child state (0 = no
    // child). Initial state is 1; state 0 is the null sentinel so we
    // can use 0 to mean "no transition" without a separate bitmap.
    std::vector<std::uint32_t> next_;

    // fail_[state] = failure link. fail_[1] = 1 (root loops to itself).
    std::vector<std::uint32_t> fail_;

    // output_[state] = first pattern id reachable via output link from
    // this state, or UINT32_MAX if none. Chained via next_output_.
    std::vector<std::uint32_t> output_;
    std::vector<std::uint32_t> next_output_;

    // dict_suffix_[state] = nearest ancestor via failure links whose
    // output_ is set, or 0 (kNull) if no such ancestor. Lets match()
    // collect suffix outputs in O(matches) rather than O(depth).
    std::vector<std::uint32_t> dict_suffix_;

    // Per-pattern metadata.
    std::vector<std::uint16_t> pattern_type_;
    std::vector<std::uint32_t> pattern_length_;
    std::size_t pattern_count_ = 0;

    static std::uint8_t fold_(std::uint8_t c, bool case_insensitive) noexcept;
    static bool is_word_char_(std::uint8_t c) noexcept;
    void add_pattern_(std::string_view p, std::uint32_t pattern_id);
    void build_failure_links_();
};

} // namespace simeon
