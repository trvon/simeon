#include "simeon/tokenizer_bpe.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace simeon {

namespace {

constexpr std::uint64_t pair_key(std::uint32_t left, std::uint32_t right) noexcept {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

// Whitespace splitter shared between learner and apply path. Empty inputs
// produce zero words; consecutive whitespace is collapsed.
template <typename F>
void for_each_word(std::string_view text, F&& fn) {
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        const std::size_t start = i;
        while (i < n && !std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (start < i) fn(text.substr(start, i - start));
    }
}

void escape_byte(unsigned char c, std::string& out) {
    // Anything outside printable-ASCII (excluding space, tab, backslash) is
    // emitted as \xNN. Backslash itself is doubled. Tab and newline are
    // delimiters in the file format so they always get escaped.
    if (c == '\\') {
        out += "\\\\";
    } else if (c >= 0x21 && c <= 0x7E) {
        out += static_cast<char>(c);
    } else {
        static constexpr char hex[] = "0123456789abcdef";
        out += "\\x";
        out += hex[c >> 4];
        out += hex[c & 0x0F];
    }
}

std::string escape_bytes(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) escape_byte(c, out);
    return out;
}

// Returns false on parse error. Reads either '\\\\' (backslash), '\\xNN'
// (hex escape), or any other byte literally. Advances `i`.
bool parse_one_byte(std::string_view s, std::size_t& i, unsigned char& out) {
    if (i >= s.size()) return false;
    if (s[i] == '\\') {
        if (i + 1 >= s.size()) return false;
        const char c = s[i + 1];
        if (c == '\\') {
            out = '\\';
            i += 2;
            return true;
        }
        if (c == 'x') {
            if (i + 4 > s.size()) return false;
            unsigned int byte_val = 0;
            const auto* p = s.data() + i + 2;
            auto [ptr, ec] = std::from_chars(p, p + 2, byte_val, 16);
            if (ec != std::errc{} || ptr != p + 2) return false;
            out = static_cast<unsigned char>(byte_val);
            i += 4;
            return true;
        }
        return false;
    }
    out = static_cast<unsigned char>(s[i]);
    ++i;
    return true;
}

bool parse_token_field(std::string_view s, std::string& out) {
    out.clear();
    std::size_t i = 0;
    while (i < s.size()) {
        unsigned char b;
        if (!parse_one_byte(s, i, b)) return false;
        out += static_cast<char>(b);
    }
    return true;
}

}  // namespace

BpeMerges::BpeMerges() {
    // Initial vocab: every single byte is its own token. Token id == byte value.
    vocab_.reserve(256);
    for (int b = 0; b < 256; ++b) {
        vocab_.emplace_back(1, static_cast<char>(b));
    }
}

BpeMerges BpeMerges::learn(std::span<const std::string_view> seed_corpus,
                           std::uint32_t target_vocab_size) {
    BpeMerges m;
    if (target_vocab_size <= 256) return m;

    // Collect distinct word "shapes" with counts, encoded as id sequences.
    // Many corpora have heavy word-frequency skew, so deduplicating by word
    // string before BPE counting cuts work without changing the result.
    std::unordered_map<std::string, std::uint64_t> word_counts;
    for (auto text : seed_corpus) {
        for_each_word(text, [&](std::string_view w) { ++word_counts[std::string(w)]; });
    }

    struct WordSeq {
        std::vector<std::uint32_t> ids;
        std::uint64_t count;
    };
    std::vector<WordSeq> words;
    words.reserve(word_counts.size());
    for (auto& [w, c] : word_counts) {
        WordSeq ws;
        ws.ids.reserve(w.size());
        for (unsigned char b : w) ws.ids.push_back(b);
        ws.count = c;
        words.push_back(std::move(ws));
    }

    const std::uint32_t merges_to_add = target_vocab_size - 256;
    for (std::uint32_t step = 0; step < merges_to_add; ++step) {
        // Count adjacent pairs across all words, weighted by word count.
        std::unordered_map<std::uint64_t, std::uint64_t> pair_counts;
        for (const auto& ws : words) {
            const auto& ids = ws.ids;
            for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
                pair_counts[pair_key(ids[i], ids[i + 1])] += ws.count;
            }
        }
        if (pair_counts.empty()) break;

        // Find the most-frequent pair. Tie-break by pair_key for
        // determinism (same input -> same merges).
        std::uint64_t best_key = 0;
        std::uint64_t best_count = 0;
        for (const auto& [k, c] : pair_counts) {
            if (c > best_count || (c == best_count && k < best_key)) {
                best_count = c;
                best_key = k;
            }
        }
        if (best_count < 2) break;  // No pair worth merging.

        const std::uint32_t left = static_cast<std::uint32_t>(best_key >> 32);
        const std::uint32_t right = static_cast<std::uint32_t>(best_key & 0xFFFFFFFFULL);
        const std::uint32_t new_id = static_cast<std::uint32_t>(m.vocab_.size());
        m.vocab_.push_back(m.vocab_[left] + m.vocab_[right]);
        m.pair_to_id_.emplace(best_key, new_id);

        // In-place rewrite: replace every (left,right) adjacency with new_id.
        for (auto& ws : words) {
            auto& ids = ws.ids;
            if (ids.size() < 2) continue;
            std::size_t r = 0, w = 0;
            while (r < ids.size()) {
                if (r + 1 < ids.size() && ids[r] == left && ids[r + 1] == right) {
                    ids[w++] = new_id;
                    r += 2;
                } else {
                    ids[w++] = ids[r++];
                }
            }
            ids.resize(w);
        }
    }
    return m;
}

BpeMerges BpeMerges::learn_from_text(std::string_view text,
                                     std::uint32_t target_vocab_size) {
    std::array<std::string_view, 1> arr = {text};
    return learn(arr, target_vocab_size);
}

std::string BpeMerges::serialize() const {
    std::string out;
    // Emit only the merge entries. Initial 256 byte vocab is implicit.
    for (std::uint32_t id = 256; id < vocab_.size(); ++id) {
        // Recover (left, right) from the merge map. Linear scan is acceptable
        // here because serialize() is rare; preserves a single source of
        // truth (pair_to_id_) without storing parallel data.
        std::uint32_t left = 0, right = 0;
        bool found = false;
        for (const auto& [k, v] : pair_to_id_) {
            if (v == id) {
                left = static_cast<std::uint32_t>(k >> 32);
                right = static_cast<std::uint32_t>(k & 0xFFFFFFFFULL);
                found = true;
                break;
            }
        }
        if (!found) continue;
        out += escape_bytes(vocab_[left]);
        out += '\t';
        out += escape_bytes(vocab_[right]);
        out += '\n';
    }
    return out;
}

BpeMerges BpeMerges::from_text(std::string_view serialized) {
    BpeMerges m;
    // Vocab string -> id, used to look up the parents of each line.
    std::unordered_map<std::string, std::uint32_t> by_string;
    by_string.reserve(serialized.size() / 4);
    for (std::uint32_t i = 0; i < 256; ++i) by_string.emplace(m.vocab_[i], i);

    std::size_t i = 0;
    const std::size_t n = serialized.size();
    while (i < n) {
        const std::size_t line_start = i;
        while (i < n && serialized[i] != '\n') ++i;
        std::string_view line = serialized.substr(line_start, i - line_start);
        if (i < n) ++i;  // consume \n
        if (line.empty()) continue;

        const std::size_t tab = line.find('\t');
        if (tab == std::string_view::npos) continue;
        std::string left, right;
        if (!parse_token_field(line.substr(0, tab), left)) continue;
        if (!parse_token_field(line.substr(tab + 1), right)) continue;

        auto lit = by_string.find(left);
        auto rit = by_string.find(right);
        if (lit == by_string.end() || rit == by_string.end()) continue;
        const std::uint32_t lid = lit->second;
        const std::uint32_t rid = rit->second;
        const std::uint32_t new_id = static_cast<std::uint32_t>(m.vocab_.size());
        std::string merged = left + right;
        m.vocab_.push_back(merged);
        m.pair_to_id_.emplace(pair_key(lid, rid), new_id);
        by_string.emplace(std::move(merged), new_id);
    }
    return m;
}

void BpeMerges::apply(std::string_view word, NGramEmitter& sink) const {
    if (word.empty()) return;

    thread_local std::vector<std::uint32_t> ids;
    ids.assign(word.size(), 0);
    for (std::size_t i = 0; i < word.size(); ++i) {
        ids[i] = static_cast<unsigned char>(word[i]);
    }

    // Greedy lowest-rank: at each pass scan every adjacent pair, find the
    // applicable merge with the smallest target id (oldest merge first),
    // apply it everywhere it appears, and loop. O(L^2 * passes) per word
    // in the worst case; words are short so passes are bounded by ~L.
    while (ids.size() >= 2) {
        std::uint32_t best_target = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t best_key = 0;
        for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
            const std::uint64_t k = pair_key(ids[i], ids[i + 1]);
            const auto it = pair_to_id_.find(k);
            if (it != pair_to_id_.end() && it->second < best_target) {
                best_target = it->second;
                best_key = k;
            }
        }
        if (best_target == std::numeric_limits<std::uint32_t>::max()) break;

        const std::uint32_t left = static_cast<std::uint32_t>(best_key >> 32);
        const std::uint32_t right = static_cast<std::uint32_t>(best_key & 0xFFFFFFFFULL);
        std::size_t r = 0, w = 0;
        while (r < ids.size()) {
            if (r + 1 < ids.size() && ids[r] == left && ids[r + 1] == right) {
                ids[w++] = best_target;
                r += 2;
            } else {
                ids[w++] = ids[r++];
            }
        }
        ids.resize(w);
    }

    for (std::uint32_t id : ids) sink.on_token(vocab_[id], 1.0f);
}

}  // namespace simeon
