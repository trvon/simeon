#include "simeon/aho_corasick.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>

namespace simeon {

namespace {

constexpr std::uint32_t kRoot = 1;
constexpr std::uint32_t kNull = 0;
constexpr std::uint32_t kNoOutput = std::numeric_limits<std::uint32_t>::max();

inline std::uint32_t goto_(const std::vector<std::uint32_t>& next, std::uint32_t state,
                           std::uint8_t c) noexcept {
    return next[state * 256u + c];
}

inline void set_goto(std::vector<std::uint32_t>& next, std::uint32_t state, std::uint8_t c,
                     std::uint32_t child) noexcept {
    next[state * 256u + c] = child;
}

} // namespace

AhoCorasick::AhoCorasick(AhoCorasickConfig cfg) noexcept : cfg_(cfg) {
    clear();
}

std::uint8_t AhoCorasick::fold_(std::uint8_t c, bool case_insensitive) noexcept {
    if (case_insensitive && c >= 'A' && c <= 'Z') {
        return static_cast<std::uint8_t>(c - 'A' + 'a');
    }
    return c;
}

bool AhoCorasick::is_word_char_(std::uint8_t c) noexcept {
    // ASCII word-boundary definition. Non-ASCII bytes (UTF-8
    // continuation / multibyte lead) count as word chars so multibyte
    // letters don't spuriously break boundaries.
    if (c >= 0x80)
        return true;
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

void AhoCorasick::clear() noexcept {
    next_.assign(2u * 256u, kNull); // sentinel + root
    fail_.assign(2u, kRoot);
    output_.assign(2u, kNoOutput);
    next_output_.clear();
    dict_suffix_.assign(2u, kNull);
    pattern_type_.clear();
    pattern_length_.clear();
    pattern_count_ = 0;
}

void AhoCorasick::add_pattern_(std::string_view p, std::uint32_t pattern_id) {
    std::uint32_t state = kRoot;
    for (char raw : p) {
        const auto c = fold_(static_cast<std::uint8_t>(raw), cfg_.case_insensitive);
        std::uint32_t child = goto_(next_, state, c);
        if (child == kNull) {
            child = static_cast<std::uint32_t>(next_.size() / 256u);
            next_.resize(next_.size() + 256u, kNull);
            fail_.push_back(kRoot);
            output_.push_back(kNoOutput);
            dict_suffix_.push_back(kNull);
            set_goto(next_, state, c, child);
        }
        state = child;
    }
    // Link this pattern id onto the terminal state's output chain.
    next_output_.push_back(output_[state]);
    output_[state] = pattern_id;
}

void AhoCorasick::build_failure_links_() {
    // BFS from root; fail_[root] = root, fail_[depth-1 child] = root.
    std::queue<std::uint32_t> q;
    for (std::uint32_t c = 0; c < 256u; ++c) {
        const auto child = goto_(next_, kRoot, c);
        if (child != kNull) {
            fail_[child] = kRoot;
            q.push(child);
        }
    }
    while (!q.empty()) {
        const auto s = q.front();
        q.pop();
        for (std::uint32_t c = 0; c < 256u; ++c) {
            const auto child = goto_(next_, s, c);
            if (child == kNull)
                continue;
            // Walk failure chain from parent to find longest proper suffix.
            std::uint32_t f = fail_[s];
            while (f != kRoot && goto_(next_, f, c) == kNull) {
                f = fail_[f];
            }
            const auto f_child = goto_(next_, f, c);
            fail_[child] = (f_child != kNull && f_child != child) ? f_child : kRoot;
            // Shortcut to nearest ancestor with output.
            const auto fc = fail_[child];
            dict_suffix_[child] = (output_[fc] != kNoOutput) ? fc : dict_suffix_[fc];
            q.push(child);
        }
    }
}

std::optional<AhoCorasickBuildError> AhoCorasick::build(std::span<const std::string_view> patterns,
                                                        std::span<const std::uint16_t> type_ids) {
    clear();
    if (patterns.size() != type_ids.size()) {
        return AhoCorasickBuildError{"patterns and type_ids must have equal length"};
    }
    if (patterns.size() > cfg_.max_patterns) {
        return AhoCorasickBuildError{"patterns exceed max_patterns cap"};
    }
    for (std::size_t i = 0; i < patterns.size(); ++i) {
        if (patterns[i].empty()) {
            clear();
            return AhoCorasickBuildError{"empty pattern not allowed"};
        }
    }
    pattern_type_.assign(type_ids.begin(), type_ids.end());
    pattern_length_.resize(patterns.size());
    for (std::size_t i = 0; i < patterns.size(); ++i) {
        pattern_length_[i] = static_cast<std::uint32_t>(patterns[i].size());
        add_pattern_(patterns[i], static_cast<std::uint32_t>(i));
    }
    pattern_count_ = patterns.size();
    build_failure_links_();
    return std::nullopt;
}

std::vector<AcMatch> AhoCorasick::match(std::string_view text) const {
    std::vector<AcMatch> out;
    if (pattern_count_ == 0 || text.empty())
        return out;

    // Candidate buffer when longest_match is on: store all hits, then
    // reduce to non-overlapping longest-first. For longest_match off,
    // emit directly.
    std::vector<AcMatch> hits;
    hits.reserve(text.size() / 4);

    std::uint32_t state = kRoot;
    const auto n = text.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto c = fold_(static_cast<std::uint8_t>(text[i]), cfg_.case_insensitive);
        // Follow failure chain until we have a transition.
        while (state != kRoot && goto_(next_, state, c) == kNull) {
            state = fail_[state];
        }
        const auto next_state = goto_(next_, state, c);
        state = (next_state != kNull) ? next_state : kRoot;

        // Walk state's own output chain then jump to the next ancestor
        // with output via dict_suffix_ (skips non-output states).
        auto emit_from = [&](std::uint32_t s) {
            for (std::uint32_t o = output_[s]; o != kNoOutput; o = next_output_[o]) {
                const auto len = pattern_length_[o];
                const auto end = static_cast<std::uint32_t>(i + 1);
                if (len > end)
                    return;
                const auto begin = end - len;

                if (cfg_.whole_word) {
                    const bool left_ok =
                        (begin == 0) || !is_word_char_(static_cast<std::uint8_t>(text[begin - 1]));
                    const bool right_ok =
                        (end == n) || !is_word_char_(static_cast<std::uint8_t>(text[end]));
                    if (!left_ok || !right_ok)
                        continue;
                }

                hits.push_back(AcMatch{o, begin, end, pattern_type_[o]});
            }
        };
        if (output_[state] != kNoOutput)
            emit_from(state);
        for (std::uint32_t s = dict_suffix_[state]; s != kNull; s = dict_suffix_[s]) {
            emit_from(s);
        }
    }

    if (!cfg_.longest_match) {
        return hits;
    }

    // Longest-match non-overlapping reduction.
    // Sort by begin asc, then by length desc (longer wins at same start).
    std::sort(hits.begin(), hits.end(), [&](const AcMatch& a, const AcMatch& b) {
        if (a.begin_utf8 != b.begin_utf8)
            return a.begin_utf8 < b.begin_utf8;
        const auto la = a.end_utf8 - a.begin_utf8;
        const auto lb = b.end_utf8 - b.begin_utf8;
        return la > lb;
    });
    std::uint32_t cursor = 0;
    for (const auto& h : hits) {
        if (h.begin_utf8 < cursor)
            continue;
        out.push_back(h);
        cursor = h.end_utf8;
    }
    return out;
}

} // namespace simeon
