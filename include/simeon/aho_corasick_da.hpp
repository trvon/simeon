#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simeon {

// PROTOTYPE: double-array trie backed Aho-Corasick.
// Lives on branch ac-da-trie-wip to compare match-side throughput against
// the dense 256-col goto table. NOT wired into the GLiNER-replacement
// path; do not use from production code.
//
// Representation:
//   goto(s, c) = da_base_[s] + c, valid only if da_check_[goto] == s.
// Theoretical advantages over dense:
//   - Memory ≈ O(edges) instead of O(nodes × 256).
//   - Lookup is one compute + two loads instead of one load, which may
//     or may not be faster depending on cache behavior.

struct AhoCorasickDaConfig {
    bool case_insensitive = true;
    bool whole_word = true;
    std::uint32_t max_patterns = 2'000'000;
    bool longest_match = true;
};

struct AcMatchDa {
    std::uint32_t pattern_id;
    std::uint32_t begin_utf8;
    std::uint32_t end_utf8;
    std::uint16_t type_id;
};

struct AhoCorasickDaBuildError {
    std::string message;
};

class AhoCorasickDa {
public:
    explicit AhoCorasickDa(AhoCorasickDaConfig cfg = {}) noexcept;

    [[nodiscard]] std::optional<AhoCorasickDaBuildError>
    build(std::span<const std::string_view> patterns, std::span<const std::uint16_t> type_ids);

    std::vector<AcMatchDa> match(std::string_view text) const;

    std::size_t pattern_count() const noexcept { return pattern_count_; }
    std::size_t da_slot_count() const noexcept { return da_check_.size(); }
    std::size_t node_count() const noexcept { return n_trie_nodes_; }

    const AhoCorasickDaConfig& config() const noexcept { return cfg_; }
    void clear() noexcept;

private:
    AhoCorasickDaConfig cfg_;

    // Intermediate trie (build-time only). Freed after DA construction.
    struct TrieNode {
        std::vector<std::pair<std::uint8_t, std::uint32_t>> children;
        std::uint32_t output = 0xFFFFFFFFu;
    };
    std::vector<TrieNode> trie_;
    std::size_t n_trie_nodes_ = 0;

    // Double-array arrays. Slot 0 is the null sentinel, slot 1 is root.
    std::vector<std::int32_t> da_base_;
    std::vector<std::uint32_t> da_check_;

    // Per-DA-slot metadata. Same semantics as the dense AC variant.
    std::vector<std::uint32_t> fail_;
    std::vector<std::uint32_t> output_;
    std::vector<std::uint32_t> next_output_;
    std::vector<std::uint32_t> dict_suffix_;

    std::vector<std::uint16_t> pattern_type_;
    std::vector<std::uint32_t> pattern_length_;
    std::size_t pattern_count_ = 0;

    static std::uint8_t fold_(std::uint8_t c, bool case_insensitive) noexcept;
    static bool is_word_char_(std::uint8_t c) noexcept;
    void add_pattern_to_trie_(std::string_view p, std::uint32_t pattern_id);
    void construct_da_();
    void build_failure_links_();

    // Hot-path lookup, inlined for the benefit of match().
    inline std::uint32_t goto_(std::uint32_t s, std::uint8_t c) const noexcept {
        const auto t = static_cast<std::size_t>(da_base_[s]) + c;
        if (t < da_check_.size() && da_check_[t] == s) {
            return static_cast<std::uint32_t>(t);
        }
        return 0;
    }
};

} // namespace simeon
