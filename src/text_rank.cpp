#include "simeon/text_rank.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace simeon {

namespace {

bool is_sentence_terminator(char c) noexcept {
    return c == '.' || c == '!' || c == '?';
}

bool is_paragraph_break(std::string_view s, std::size_t i) noexcept {
    // "\n\n" or "\n\r\n" treated as a hard sentence break.
    return i + 1 < s.size() && s[i] == '\n' && s[i + 1] == '\n';
}

bool is_word_char(char c) noexcept {
    // Non-ASCII bytes (UTF-8 continuations / lead bytes) treated as
    // word chars so multibyte letters don't split tokens.
    const auto u = static_cast<std::uint8_t>(c);
    if (u >= 0x80)
        return true;
    return (u >= '0' && u <= '9') || (u >= 'A' && u <= 'Z') || (u >= 'a' && u <= 'z') || u == '_' ||
           u == '-' || u == '\'';
}

// Split `text` into sentences. Returns each sentence as a
// (begin, end) byte-offset pair into `text`. Leading/trailing whitespace
// is trimmed; empty sentences are skipped.
std::vector<std::pair<std::size_t, std::size_t>> split_sentences(std::string_view text,
                                                                 std::size_t cap) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    out.reserve(32);
    const auto n = text.size();
    std::size_t start = 0;
    std::size_t i = 0;
    auto push_span = [&](std::size_t b, std::size_t e) {
        // trim
        while (b < e && (static_cast<std::uint8_t>(text[b]) <= ' '))
            ++b;
        while (e > b && (static_cast<std::uint8_t>(text[e - 1]) <= ' '))
            --e;
        if (b < e)
            out.emplace_back(b, e);
    };
    while (i < n && out.size() < cap) {
        if (is_sentence_terminator(text[i])) {
            push_span(start, i);
            // Skip consecutive terminators ("...", "!!", etc.).
            while (i < n && is_sentence_terminator(text[i]))
                ++i;
            start = i;
            continue;
        }
        if (is_paragraph_break(text, i)) {
            push_span(start, i);
            i += 2;
            start = i;
            continue;
        }
        ++i;
    }
    if (out.size() < cap) {
        push_span(start, n);
    }
    return out;
}

// Extract the word-token set for a sentence, case-folded (ASCII only)
// and capped at max_tokens. Short tokens (< 2 chars) dropped as noise.
std::unordered_set<std::string> tokenize_sentence(std::string_view sentence,
                                                  std::uint32_t max_tokens) {
    std::unordered_set<std::string> out;
    std::string cur;
    cur.reserve(16);
    auto flush = [&] {
        if (cur.size() >= 2 && out.size() < max_tokens) {
            out.insert(cur);
        }
        cur.clear();
    };
    for (char c : sentence) {
        if (is_word_char(c)) {
            const auto u = static_cast<std::uint8_t>(c);
            if (u < 0x80 && u >= 'A' && u <= 'Z') {
                cur.push_back(static_cast<char>(u - 'A' + 'a'));
            } else {
                cur.push_back(c);
            }
        } else {
            flush();
        }
    }
    flush();
    return out;
}

std::size_t intersection_size(const std::unordered_set<std::string>& a,
                              const std::unordered_set<std::string>& b) noexcept {
    const auto& small = (a.size() < b.size()) ? a : b;
    const auto& large = (a.size() < b.size()) ? b : a;
    std::size_t n = 0;
    for (const auto& tok : small) {
        if (large.count(tok))
            ++n;
    }
    return n;
}

} // namespace

std::vector<RankedSentence> TextRank::rank(std::string_view text, std::size_t top_k) const {
    std::vector<RankedSentence> empty;
    if (text.empty())
        return empty;

    const auto spans = split_sentences(text, cfg_.max_sentences);
    if (spans.empty())
        return empty;

    // Tokenize each sentence once.
    std::vector<std::unordered_set<std::string>> token_sets;
    token_sets.reserve(spans.size());
    std::vector<std::uint32_t> kept_indices; // sentence positions we keep
    kept_indices.reserve(spans.size());
    for (std::size_t i = 0; i < spans.size(); ++i) {
        const auto sentence = text.substr(spans[i].first, spans[i].second - spans[i].first);
        auto toks = tokenize_sentence(sentence, cfg_.max_sentence_tokens);
        if (toks.size() < cfg_.min_sentence_tokens)
            continue;
        token_sets.push_back(std::move(toks));
        kept_indices.push_back(static_cast<std::uint32_t>(i));
    }
    const std::size_t n = token_sets.size();
    if (n == 0)
        return empty;
    if (n == 1) {
        // Only one sentence — return it with score 1.0.
        const auto idx = kept_indices[0];
        const auto [b, e] = spans[idx];
        RankedSentence rs{text.substr(b, e - b), 1.0f, idx};
        return {rs};
    }

    // Build adjacency: w[i][j] = |tok_i ∩ tok_j| / (log|tok_i| + log|tok_j|)
    // Mihalcea's normalization; denominator guards against single-token
    // sentences but we already enforce min_sentence_tokens >= 2.
    std::vector<std::vector<float>> w(n, std::vector<float>(n, 0.0f));
    std::vector<float> row_sum(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const auto li = std::log(static_cast<float>(token_sets[i].size()) + 1.0f);
        for (std::size_t j = i + 1; j < n; ++j) {
            const auto lj = std::log(static_cast<float>(token_sets[j].size()) + 1.0f);
            const auto inter = static_cast<float>(intersection_size(token_sets[i], token_sets[j]));
            const float denom = li + lj;
            const float wij = (denom > 0.0f) ? (inter / denom) : 0.0f;
            w[i][j] = wij;
            w[j][i] = wij;
            row_sum[i] += wij;
            row_sum[j] += wij;
        }
    }

    // Power iteration. PageRank-style: s[i] = (1-d) + d * Σ_j w[j][i] / row_sum[j] * s[j].
    std::vector<float> s(n, 1.0f);
    std::vector<float> ns(n, 0.0f);
    const float d = cfg_.damping;
    const float base = 1.0f - d;
    for (std::uint32_t iter = 0; iter < cfg_.max_iters; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            float acc = 0.0f;
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j)
                    continue;
                const float rs = row_sum[j];
                if (rs <= 0.0f)
                    continue;
                acc += w[j][i] / rs * s[j];
            }
            ns[i] = base + d * acc;
        }
        float delta = 0.0f;
        for (std::size_t i = 0; i < n; ++i) {
            delta += std::fabs(ns[i] - s[i]);
        }
        s.swap(ns);
        if (delta < cfg_.convergence_epsilon)
            break;
    }

    std::vector<RankedSentence> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto idx = kept_indices[i];
        const auto [b, e] = spans[idx];
        out.push_back(RankedSentence{text.substr(b, e - b), s[i], idx});
    }
    // Partial sort top_k by score descending (ties broken by sentence
    // index ascending — earlier sentences win, matching "lead-bias").
    auto cmp = [](const RankedSentence& a, const RankedSentence& b) {
        if (a.score != b.score)
            return a.score > b.score;
        return a.index < b.index;
    };
    const std::size_t k = (top_k == 0 || top_k > out.size()) ? out.size() : top_k;
    std::partial_sort(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(k), out.end(), cmp);
    out.resize(k);
    return out;
}

std::string_view TextRank::top_sentence(std::string_view text) const {
    const auto r = rank(text, 1);
    return r.empty() ? std::string_view{} : r[0].text;
}

} // namespace simeon
