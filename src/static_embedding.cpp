#include "simeon/static_embedding.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace simeon {

namespace {

struct CodepointRange {
    std::uint32_t lo;
    std::uint32_t hi;
};

struct FoldEntry {
    std::uint32_t cp;
    std::uint8_t len;
    std::uint8_t utf8[7];
};

#include "static_embedding_unicode.inc"

template <std::size_t N> bool in_ranges(const CodepointRange (&table)[N], std::uint32_t cp) {
    const auto* end = table + N;
    const auto* it = std::upper_bound(
        table, end, cp, [](std::uint32_t v, const CodepointRange& r) { return v < r.lo; });
    return it != table && cp <= (it - 1)->hi;
}

template <std::size_t N> const FoldEntry* find_fold(const FoldEntry (&table)[N], std::uint32_t cp) {
    const auto* end = table + N;
    const auto* it = std::lower_bound(table, end, cp,
                                      [](const FoldEntry& e, std::uint32_t v) { return e.cp < v; });
    return it != end && it->cp == cp ? it : nullptr;
}

constexpr std::uint32_t kReplacement = 0xFFFD;

// Decodes one UTF-8 code point at s[i]; invalid sequences yield U+FFFD and consume one byte.
std::uint32_t next_codepoint(std::string_view s, std::size_t& i) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    auto cont = [&](std::size_t k) -> int {
        if (i + k >= s.size())
            return -1;
        const auto b = static_cast<unsigned char>(s[i + k]);
        return (b & 0xC0) == 0x80 ? (b & 0x3F) : -1;
    };
    if (b0 < 0x80) {
        ++i;
        return b0;
    }
    int n = 0;
    std::uint32_t cp = 0;
    if ((b0 & 0xE0) == 0xC0) {
        n = 1;
        cp = b0 & 0x1F;
    } else if ((b0 & 0xF0) == 0xE0) {
        n = 2;
        cp = b0 & 0x0F;
    } else if ((b0 & 0xF8) == 0xF0) {
        n = 3;
        cp = b0 & 0x07;
    } else {
        ++i;
        return kReplacement;
    }
    for (int k = 1; k <= n; ++k) {
        const int c = cont(static_cast<std::size_t>(k));
        if (c < 0) {
            ++i;
            return kReplacement;
        }
        cp = (cp << 6) | static_cast<std::uint32_t>(c);
    }
    i += static_cast<std::size_t>(n) + 1;
    return cp;
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

bool is_whitespace(std::uint32_t cp) {
    if (cp < 0x80)
        return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r';
    return cp < 0x10000 && in_ranges(kSpaceSeparator, cp);
}

bool is_control(std::uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r')
        return false;
    if (cp < 0x80)
        return cp < 0x20 || cp == 0x7F;
    return cp < 0x10000 && in_ranges(kControlOrFormat, cp);
}

// BERT's "Chinese character" blocks (CJK ideographs), padded with spaces by the normalizer.
bool is_cjk(std::uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool is_punctuation(std::uint32_t cp) {
    if (cp < 0x80)
        return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
               (cp >= 123 && cp <= 126);
    return cp < 0x10000 && in_ranges(kPunctuation, cp);
}

// Pre-tokenizer whitespace: the normalizer has already turned \t \n \r and Zs into ' '; U+2028
// and U+2029 are still Unicode whitespace.
bool is_split_space(std::uint32_t cp) {
    return is_whitespace(cp) || cp == 0x2028 || cp == 0x2029 || cp == 0x0085;
}

struct Reader {
    std::span<const std::uint8_t> bytes;
    std::size_t pos = 0;

    void need(std::size_t n) const {
        if (bytes.size() - pos < n)
            throw std::runtime_error("StaticEmbedding: truncated table");
    }
    std::uint32_t u32() {
        need(4);
        std::uint32_t v = static_cast<std::uint32_t>(bytes[pos]) |
                          (static_cast<std::uint32_t>(bytes[pos + 1]) << 8) |
                          (static_cast<std::uint32_t>(bytes[pos + 2]) << 16) |
                          (static_cast<std::uint32_t>(bytes[pos + 3]) << 24);
        pos += 4;
        return v;
    }
    std::uint16_t u16() {
        need(2);
        const auto v = static_cast<std::uint16_t>(bytes[pos] | (bytes[pos + 1] << 8));
        pos += 2;
        return v;
    }
};

} // namespace

StaticEmbedding StaticEmbedding::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("StaticEmbedding: cannot open " + path);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                    std::istreambuf_iterator<char>());
    return from_bytes(bytes);
}

StaticEmbedding StaticEmbedding::from_bytes(std::span<const std::uint8_t> bytes) {
    Reader r{bytes};
    r.need(8);
    if (std::memcmp(bytes.data(), "SMSTAT01", 8) != 0)
        throw std::runtime_error("StaticEmbedding: bad magic");
    r.pos = 8;
    const std::uint32_t version = r.u32();
    if (version != 1 && version != 2)
        throw std::runtime_error("StaticEmbedding: unsupported version");
    StaticEmbedding e;
    e.dim_ = r.u32();
    const std::uint32_t vocab = r.u32();
    const std::uint32_t flags = r.u32();
    e.unk_id_ = r.u32();
    e.max_word_chars_ = r.u32();
    if (e.dim_ == 0 || e.dim_ > 8192 || vocab == 0 || vocab > (1u << 24))
        throw std::runtime_error("StaticEmbedding: implausible dim or vocabulary size");
    if (e.unk_id_ != UINT32_MAX && e.unk_id_ >= vocab)
        throw std::runtime_error("StaticEmbedding: unk id out of range");
    e.lowercase_ = (flags & 1u) != 0;
    e.strip_accents_ = (flags & 2u) != 0;

    e.tokens_.reserve(vocab);
    e.ids_.reserve(vocab);
    for (std::uint32_t i = 0; i < vocab; ++i) {
        const std::uint16_t len = r.u16();
        r.need(len);
        std::string tok(reinterpret_cast<const char*>(bytes.data() + r.pos), len);
        r.pos += len;
        {
            std::string_view body(tok);
            if (body.starts_with("##"))
                body.remove_prefix(2);
            std::size_t cps = 0;
            for (std::size_t k = 0; k < body.size();) {
                next_codepoint(body, k);
                ++cps;
            }
            e.max_piece_chars_ = std::max(e.max_piece_chars_, cps);
        }
        e.ids_.emplace(tok, i);
        e.tokens_.push_back(std::move(tok));
    }
    if (version >= 2) {
        const std::uint32_t added = r.u32();
        if (added > vocab)
            throw std::runtime_error("StaticEmbedding: implausible added-token count");
        for (std::uint32_t k = 0; k < added; ++k) {
            const std::uint32_t id = r.u32();
            const std::uint32_t aflags = r.u32();
            if (id >= vocab)
                throw std::runtime_error("StaticEmbedding: added token id out of range");
            const std::string& content = e.tokens_[id];
            if (content.empty())
                throw std::runtime_error("StaticEmbedding: empty added token");
            AddedTokenSet& set = (aflags & 1u) != 0 ? e.addedNormalized_ : e.addedRaw_;
            set.ids.emplace(content, id);
            set.firstBytes.set(static_cast<unsigned char>(content.front()));
            set.maxBytes = std::max(set.maxBytes, content.size());
        }
    }
    r.need(static_cast<std::size_t>(vocab) * 4);
    e.scales_.resize(vocab);
    for (std::uint32_t i = 0; i < vocab; ++i) {
        const std::uint32_t bits = r.u32();
        float f;
        std::memcpy(&f, &bits, sizeof f);
        if (!std::isfinite(f))
            throw std::runtime_error("StaticEmbedding: non-finite row scale");
        e.scales_[i] = f;
    }
    const std::size_t row_bytes = static_cast<std::size_t>(vocab) * e.dim_;
    if (bytes.size() - r.pos != row_bytes)
        throw std::runtime_error("StaticEmbedding: table size does not match header");
    e.rows_.resize(row_bytes);
    std::memcpy(e.rows_.data(), bytes.data() + r.pos, row_bytes);
    return e;
}

std::string StaticEmbedding::normalize(std::string_view text) const {
    std::string out;
    out.reserve(text.size() + 8);
    std::size_t i = 0;
    while (i < text.size()) {
        const auto byte = static_cast<unsigned char>(text[i]);
        if (byte < 0x80) {
            // ASCII: no table lookups.
            ++i;
            if (byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r') {
                out.push_back(' ');
            } else if (byte < 0x20 || byte == 0x7F) {
                continue;
            } else if (lowercase_ && byte >= 'A' && byte <= 'Z') {
                out.push_back(static_cast<char>(byte + ('a' - 'A')));
            } else {
                out.push_back(static_cast<char>(byte));
            }
            continue;
        }
        const std::uint32_t cp = next_codepoint(text, i);
        if (cp == kReplacement || is_control(cp))
            continue;
        if (is_whitespace(cp)) {
            out.push_back(' ');
            continue;
        }
        if (is_cjk(cp)) {
            out.push_back(' ');
            append_utf8(out, cp);
            out.push_back(' ');
            continue;
        }
        if (strip_accents_ && cp < 0x10000 && in_ranges(kCombiningMark, cp))
            continue;
        const FoldEntry* fold = nullptr;
        if (strip_accents_ && lowercase_)
            fold = find_fold(kFoldStripLower, cp);
        else if (lowercase_)
            fold = find_fold(kFoldLower, cp);
        else if (strip_accents_)
            fold = find_fold(kFoldStrip, cp);
        if (fold) {
            out.append(reinterpret_cast<const char*>(fold->utf8), fold->len);
        } else if (lowercase_ && cp >= 'A' && cp <= 'Z') {
            out.push_back(static_cast<char>(cp + ('a' - 'A')));
        } else {
            append_utf8(out, cp);
        }
    }
    return out;
}

void StaticEmbedding::wordpiece(std::string_view word, std::vector<std::uint32_t>& out) const {
    // Code point boundaries of the word (buffers reused across calls on this thread).
    thread_local std::vector<std::size_t> starts;
    starts.clear();
    for (std::size_t i = 0; i < word.size();) {
        starts.push_back(i);
        next_codepoint(word, i);
    }
    const std::size_t n = starts.size();
    if (n == 0)
        return;
    if (n > max_word_chars_) {
        if (unk_id_ != UINT32_MAX)
            out.push_back(unk_id_);
        return;
    }
    starts.push_back(word.size());

    const std::size_t mark = out.size();
    thread_local std::string piece;
    std::size_t start = 0;
    while (start < n) {
        std::size_t end = std::min(n, start + max_piece_chars_);
        std::uint32_t found = UINT32_MAX;
        while (start < end) {
            piece.clear();
            if (start > 0)
                piece = "##";
            piece.append(word.substr(starts[start], starts[end] - starts[start]));
            if (const auto it = ids_.find(piece); it != ids_.end()) {
                found = it->second;
                break;
            }
            --end;
        }
        if (found == UINT32_MAX) {
            out.resize(mark);
            if (unk_id_ != UINT32_MAX)
                out.push_back(unk_id_);
            return;
        }
        out.push_back(found);
        start = end;
    }
}

template <class Fn>
void StaticEmbedding::splitAdded(std::string_view text, const AddedTokenSet& set,
                                 std::vector<std::uint32_t>& out, Fn&& onSegment) {
    if (set.empty()) {
        onSegment(text);
        return;
    }
    thread_local std::string probe;
    std::size_t segStart = 0;
    std::size_t i = 0;
    while (i < text.size()) {
        if (!set.firstBytes.test(static_cast<unsigned char>(text[i]))) {
            ++i;
            continue;
        }
        std::size_t matched = 0;
        std::uint32_t id = 0;
        for (std::size_t len = std::min(set.maxBytes, text.size() - i); len > 0; --len) {
            probe.assign(text.substr(i, len));
            if (const auto it = set.ids.find(probe); it != set.ids.end()) {
                matched = len;
                id = it->second;
                break;
            }
        }
        if (matched == 0) {
            ++i;
            continue;
        }
        if (i > segStart)
            onSegment(text.substr(segStart, i - segStart));
        out.push_back(id);
        i += matched;
        segStart = i;
    }
    if (segStart < text.size())
        onSegment(text.substr(segStart));
}

void StaticEmbedding::pretokenize(std::string_view norm, std::vector<std::uint32_t>& ids) const {
    std::size_t word_start = std::string::npos;
    auto flush = [&](std::size_t end) {
        if (word_start != std::string::npos) {
            wordpiece(norm.substr(word_start, end - word_start), ids);
            word_start = std::string::npos;
        }
    };
    std::size_t i = 0;
    while (i < norm.size()) {
        const std::size_t at = i;
        const std::uint32_t cp = next_codepoint(norm, i);
        if (is_split_space(cp)) {
            flush(at);
        } else if (is_punctuation(cp)) {
            flush(at);
            wordpiece(norm.substr(at, i - at), ids);
        } else if (word_start == std::string::npos) {
            word_start = at;
        }
    }
    flush(norm.size());
}

void StaticEmbedding::normalizedSegment(std::string_view raw,
                                        std::vector<std::uint32_t>& ids) const {
    const std::string norm = normalize(raw);
    splitAdded(norm, addedNormalized_, ids,
               [&](std::string_view segment) { pretokenize(segment, ids); });
}

std::vector<std::uint32_t> StaticEmbedding::tokenize(std::string_view text) const {
    std::vector<std::uint32_t> ids;
    splitAdded(text, addedRaw_, ids,
               [&](std::string_view segment) { normalizedSegment(segment, ids); });
    return ids;
}

bool StaticEmbedding::encode(std::string_view text, std::span<float> out) const {
    if (out.size() != dim_)
        throw std::invalid_argument("StaticEmbedding::encode: output size must equal dim()");
    std::fill(out.begin(), out.end(), 0.0f);
    // Accumulate straight into `out`: float * int8 rows vectorize; the mean's 1/n cancels in the
    // L2 normalization.
    float* acc = out.data();
    std::size_t n = 0;
    for (const std::uint32_t id : tokenize(text)) {
        if (id == unk_id_)
            continue;
        const float s = scales_[id];
        const std::int8_t* row = rows_.data() + static_cast<std::size_t>(id) * dim_;
        for (std::uint32_t d = 0; d < dim_; ++d)
            acc[d] += static_cast<float>(row[d]) * s;
        ++n;
    }
    if (n == 0)
        return false;
    double norm = 0.0;
    for (std::uint32_t d = 0; d < dim_; ++d)
        norm += static_cast<double>(acc[d]) * acc[d];
    if (norm <= 0.0) {
        std::fill(out.begin(), out.end(), 0.0f);
        return false;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(norm));
    for (std::uint32_t d = 0; d < dim_; ++d)
        acc[d] *= inv;
    return true;
}

std::vector<float> StaticEmbedding::encode(std::string_view text) const {
    std::vector<float> out(dim_);
    encode(text, out);
    return out;
}

} // namespace simeon
