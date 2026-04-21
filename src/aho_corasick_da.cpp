#include "simeon/aho_corasick_da.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>

namespace simeon {

namespace {

constexpr std::uint32_t kRootDa = 1;
constexpr std::uint32_t kNullDa = 0;
constexpr std::uint32_t kNoOutputDa = std::numeric_limits<std::uint32_t>::max();

} // namespace

AhoCorasickDa::AhoCorasickDa(AhoCorasickDaConfig cfg) noexcept : cfg_(cfg) {
    clear();
}

std::uint8_t AhoCorasickDa::fold_(std::uint8_t c, bool case_insensitive) noexcept {
    if (case_insensitive && c >= 'A' && c <= 'Z') {
        return static_cast<std::uint8_t>(c - 'A' + 'a');
    }
    return c;
}

bool AhoCorasickDa::is_word_char_(std::uint8_t c) noexcept {
    if (c >= 0x80)
        return true;
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
}

void AhoCorasickDa::clear() noexcept {
    trie_.assign(1, TrieNode{}); // trie node 0 is the root
    n_trie_nodes_ = 1;
    da_base_.clear();
    da_check_.clear();
    fail_.clear();
    output_.clear();
    next_output_.clear();
    dict_suffix_.clear();
    pattern_type_.clear();
    pattern_length_.clear();
    pattern_count_ = 0;
}

void AhoCorasickDa::add_pattern_to_trie_(std::string_view p, std::uint32_t pattern_id) {
    std::uint32_t tid = 0; // trie root
    for (char raw : p) {
        const auto c = fold_(static_cast<std::uint8_t>(raw), cfg_.case_insensitive);
        // Scan by index — trie_.emplace_back() below may reallocate
        // trie_, which would invalidate any reference to a node.
        std::uint32_t found = 0xFFFFFFFFu;
        const auto& kids_ro = trie_[tid].children;
        for (const auto& e : kids_ro) {
            if (e.first == c) {
                found = e.second;
                break;
            }
        }
        if (found == 0xFFFFFFFFu) {
            const auto child = static_cast<std::uint32_t>(trie_.size());
            trie_.emplace_back();
            trie_[tid].children.emplace_back(c, child);
            tid = child;
        } else {
            tid = found;
        }
    }
    const auto prev = trie_[tid].output;
    next_output_.push_back(prev);
    trie_[tid].output = pattern_id;
}

void AhoCorasickDa::construct_da_() {
    n_trie_nodes_ = trie_.size();

    // Map from trie node id to DA slot id.
    std::vector<std::uint32_t> trie_to_da(trie_.size(), 0);
    trie_to_da[0] = kRootDa;

    // DA arrays. Slot 0 = null sentinel, slot 1 = root.
    auto grow = [&](std::size_t need) {
        if (need > da_check_.size()) {
            const auto new_sz = std::max(need, da_check_.size() * 2 + 1);
            da_base_.resize(new_sz, 0);
            da_check_.resize(new_sz, kNullDa);
        }
    };
    grow(2);

    // `used[i]` tracks whether slot i has been claimed (either as a node
    // itself or as a transition target). Claiming slot 0 and slot 1 up
    // front keeps them out of the free pool.
    std::vector<std::uint8_t> used(da_check_.size(), 0);
    used[0] = 1;
    used[1] = 1;

    // Start of linear scan for unused ranges. Advances monotonically as
    // the automaton grows; O(n²) worst case but fine for a prototype.
    std::size_t scan_start = 2;

    std::queue<std::uint32_t> bfs;
    bfs.push(0);
    while (!bfs.empty()) {
        const auto tid = bfs.front();
        bfs.pop();
        auto& kids = trie_[tid].children;
        if (kids.empty())
            continue;
        // Sort children by byte for deterministic layout (not required).
        std::sort(kids.begin(), kids.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        const auto da_parent = trie_to_da[tid];
        const std::uint8_t min_c = kids.front().first;

        // Find base b such that for every c in kids, slot (b + c) is free.
        // A slot beyond used.size() counts as free (allocated lazily below).
        std::int32_t b = static_cast<std::int32_t>(scan_start) - min_c;
        if (b < 1)
            b = 1;
        for (;; ++b) {
            bool ok = true;
            for (const auto& [c, _] : kids) {
                const std::size_t slot = static_cast<std::size_t>(b) + c;
                if (slot < used.size() && used[slot]) {
                    ok = false;
                    break;
                }
            }
            if (ok)
                break;
        }

        // Commit: grow DA arrays once, then mark slots.
        const std::size_t max_slot = static_cast<std::size_t>(b) + kids.back().first;
        grow(max_slot + 1);
        if (used.size() < max_slot + 1)
            used.resize(max_slot + 1, 0);

        da_base_[da_parent] = b;
        for (const auto& [c, child_tid] : kids) {
            const std::size_t slot = static_cast<std::size_t>(b) + c;
            da_check_[slot] = da_parent;
            used[slot] = 1;
            trie_to_da[child_tid] = static_cast<std::uint32_t>(slot);
            bfs.push(child_tid);
        }

        // Advance scan_start past consumed region (conservative).
        while (scan_start < used.size() && used[scan_start])
            ++scan_start;
    }

    // Shrink DA arrays to the last used slot + 1.
    std::size_t last_used = 0;
    for (std::size_t i = used.size(); i-- > 0;) {
        if (used[i]) {
            last_used = i;
            break;
        }
    }
    da_base_.resize(last_used + 1);
    da_check_.resize(last_used + 1);

    // Populate per-slot metadata.
    fail_.assign(da_check_.size(), kRootDa);
    output_.assign(da_check_.size(), kNoOutputDa);
    dict_suffix_.assign(da_check_.size(), kNullDa);
    for (std::size_t tid = 0; tid < trie_.size(); ++tid) {
        const auto da = trie_to_da[tid];
        if (trie_[tid].output != kNoOutputDa)
            output_[da] = trie_[tid].output;
    }

    // Free trie memory.
    std::vector<TrieNode>().swap(trie_);
}

void AhoCorasickDa::build_failure_links_() {
    std::queue<std::uint32_t> q;
    // Root's direct children: iterate over all 256 bytes and probe.
    // (For DA, we don't have an explicit child list; probe slots.)
    for (std::uint32_t c = 0; c < 256; ++c) {
        const auto child = goto_(kRootDa, static_cast<std::uint8_t>(c));
        if (child != kNullDa) {
            fail_[child] = kRootDa;
            q.push(child);
        }
    }
    while (!q.empty()) {
        const auto s = q.front();
        q.pop();
        for (std::uint32_t c = 0; c < 256; ++c) {
            const auto child = goto_(s, static_cast<std::uint8_t>(c));
            if (child == kNullDa)
                continue;
            std::uint32_t f = fail_[s];
            while (f != kRootDa && goto_(f, static_cast<std::uint8_t>(c)) == kNullDa) {
                f = fail_[f];
            }
            const auto f_child = goto_(f, static_cast<std::uint8_t>(c));
            fail_[child] = (f_child != kNullDa && f_child != child) ? f_child : kRootDa;
            const auto fc = fail_[child];
            dict_suffix_[child] = (output_[fc] != kNoOutputDa) ? fc : dict_suffix_[fc];
            q.push(child);
        }
    }
}

std::optional<AhoCorasickDaBuildError>
AhoCorasickDa::build(std::span<const std::string_view> patterns,
                     std::span<const std::uint16_t> type_ids) {
    clear();
    if (patterns.size() != type_ids.size())
        return AhoCorasickDaBuildError{"patterns and type_ids must have equal length"};
    if (patterns.size() > cfg_.max_patterns)
        return AhoCorasickDaBuildError{"patterns exceed max_patterns cap"};
    for (std::size_t i = 0; i < patterns.size(); ++i) {
        if (patterns[i].empty()) {
            clear();
            return AhoCorasickDaBuildError{"empty pattern not allowed"};
        }
    }

    pattern_type_.assign(type_ids.begin(), type_ids.end());
    pattern_length_.resize(patterns.size());
    for (std::size_t i = 0; i < patterns.size(); ++i) {
        pattern_length_[i] = static_cast<std::uint32_t>(patterns[i].size());
        add_pattern_to_trie_(patterns[i], static_cast<std::uint32_t>(i));
    }
    pattern_count_ = patterns.size();

    construct_da_();
    build_failure_links_();
    return std::nullopt;
}

std::vector<AcMatchDa> AhoCorasickDa::match(std::string_view text) const {
    std::vector<AcMatchDa> out;
    if (pattern_count_ == 0 || text.empty())
        return out;

    std::vector<AcMatchDa> hits;
    hits.reserve(text.size() / 4);

    std::uint32_t state = kRootDa;
    const auto n = text.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto c = fold_(static_cast<std::uint8_t>(text[i]), cfg_.case_insensitive);
        std::uint32_t next_state = goto_(state, c);
        while (state != kRootDa && next_state == kNullDa) {
            state = fail_[state];
            next_state = goto_(state, c);
        }
        state = (next_state != kNullDa) ? next_state : kRootDa;

        auto emit_from = [&](std::uint32_t s) {
            for (std::uint32_t o = output_[s]; o != kNoOutputDa; o = next_output_[o]) {
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
                hits.push_back(AcMatchDa{o, begin, end, pattern_type_[o]});
            }
        };
        if (output_[state] != kNoOutputDa)
            emit_from(state);
        for (std::uint32_t s = dict_suffix_[state]; s != kNullDa; s = dict_suffix_[s]) {
            emit_from(s);
        }
    }

    if (!cfg_.longest_match) {
        return hits;
    }

    std::sort(hits.begin(), hits.end(), [&](const AcMatchDa& a, const AcMatchDa& b) {
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
