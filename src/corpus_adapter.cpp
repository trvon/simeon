#include "simeon/corpus_adapter.hpp"

#include <algorithm>
#include <cmath>

#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

struct WordTokenSink final : NGramEmitter {
    std::vector<std::string>* out = nullptr;
    void on_token(std::string_view tok, float) override {
        if (out)
            out->emplace_back(tok);
    }
};

const std::unordered_set<std::string_view>& arguana_stopwords() {
    static const std::unordered_set<std::string_view> kStop = {
        "a",     "an",     "and",   "are", "as",  "at",   "be",   "by",   "for",   "from", "in",
        "into",  "is",     "it",    "of",  "on",  "or",   "that", "the",  "their", "this", "to",
        "was",   "were",   "with",  "we",  "our", "you",  "your", "have", "has",   "had",  "will",
        "would", "should", "could", "can", "do",  "does", "did",  "they", "them",
    };
    return kStop;
}

const std::unordered_set<std::string_view>& arguana_cues() {
    static const std::unordered_set<std::string_view> kCues = {
        "not",      "no",        "never",      "however", "but",          "although", "despite",
        "instead",  "rather",    "whereas",    "while",   "counterpoint", "fail",     "fails",
        "failed",   "unlikely",  "cannot",     "less",    "insufficient", "wrong",    "problem",
        "problems", "expensive", "harm",       "harmful", "risk",         "risks",    "impossible",
        "oppose",   "opposing",  "opposition",
    };
    return kCues;
}

bool stopword(std::string_view token) {
    return arguana_stopwords().contains(token);
}

} // namespace

// ---------------------------------------------------------------------------
// TextAdapter
// ---------------------------------------------------------------------------

std::string extract_lead_tokens(std::string_view text, std::uint32_t max_tokens) {
    std::string out;
    out.reserve(text.size() / 2); // rough
    std::uint32_t tokens = 0;
    bool in_token = false;

    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (in_token) {
                ++tokens;
                in_token = false;
            }
            if (tokens >= max_tokens)
                break;
            out.push_back(' ');
        } else {
            out.push_back(c);
            in_token = true;
        }
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

AdapterEvidence TextAdapter::process_doc(std::string_view /*doc_id*/, std::string_view doc_text) {
    AdapterEvidence ev;
    ev.aux_field = extract_lead_tokens(doc_text, 64);
    return ev;
}

AdapterEvidence TextAdapter::process_query(std::string_view /*query_id*/,
                                           std::string_view query_text) {
    AdapterEvidence ev;
    ev.aux_field = extract_lead_tokens(query_text, 64);
    return ev;
}

// ---------------------------------------------------------------------------
// ArguanaAdapter
// ---------------------------------------------------------------------------

ArguanaAdapter::Id ArguanaAdapter::parse(std::string_view id) {
    Id out;
    const std::size_t dash = id.rfind('-');
    if (dash == std::string_view::npos)
        return out;

    std::string_view tail = id.substr(dash + 1);
    if (tail.size() < 5)
        return out;

    // stance: first 3 chars must be "pro" or "con"
    if (tail.substr(0, 3) == "pro") {
        out.stance[0] = 'p';
        out.stance[1] = 'r';
        out.stance[2] = 'o';
    } else if (tail.substr(0, 3) == "con") {
        out.stance[0] = 'c';
        out.stance[1] = 'o';
        out.stance[2] = 'n';
    } else {
        return out;
    }

    // side: last char must be 'a' or 'b'
    char side = tail.back();
    if (side != 'a' && side != 'b')
        return out;
    out.side = side;

    // point: everything between stance and side must be digits
    std::string_view pt = tail.substr(3, tail.size() - 4);
    if (pt.empty())
        return out;
    for (char c : pt) {
        if (!std::isdigit(static_cast<unsigned char>(c)))
            return out;
    }
    out.point = std::string(pt);

    out.topic = std::string(id.substr(0, dash));
    out.valid = true;
    return out;
}

void ArguanaAdapter::seed_doc(std::string_view doc_id, std::uint32_t doc_index) {
    if (seeded_all_)
        return;
    Id id = parse(doc_id);
    if (!id.valid)
        return;
    topics_[id.topic + "|" + id.stance + "|" + id.point][id.side] = doc_index;
    ++seeded_;
}

AdapterEvidence ArguanaAdapter::process_doc(std::string_view doc_id, std::string_view doc_text) {
    AdapterEvidence ev;
    ev.aux_field = extract_lead_tokens(doc_text, 64);
    return ev;
}

AdapterEvidence ArguanaAdapter::process_query(std::string_view query_id,
                                              std::string_view /*query_text*/) {
    AdapterEvidence ev;
    Id qid = parse(query_id);
    if (!qid.valid)
        return ev;

    char other_side = (qid.side == 'a') ? 'b' : 'a';
    std::string key = qid.topic + "|" + qid.stance + "|" + qid.point;
    auto kit = topics_.find(key);
    if (kit == topics_.end())
        return ev;
    auto sit = kit->second.find(other_side);
    if (sit == kit->second.end())
        return ev;

    ev.relations.push_back({sit->second, 1.0f});
    return ev;
}

std::string ArguanaTextPairAdapter::normalize_ws_lower(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    bool in_space = true;
    for (unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!in_space) {
                out.push_back(' ');
                in_space = true;
            }
        } else {
            out.push_back(static_cast<char>(std::tolower(ch)));
            in_space = false;
        }
    }
    while (!out.empty() && out.back() == ' ')
        out.pop_back();
    return out;
}

std::vector<std::string> ArguanaTextPairAdapter::word_tokens(std::string_view text) {
    std::vector<std::string> out;
    WordTokenSink sink{};
    sink.out = &out;
    tokenize(text, TokenizerConfig{0, 0, false, true}, sink);
    return out;
}

std::unordered_set<std::string> ArguanaTextPairAdapter::content_set(std::string_view text) {
    std::unordered_set<std::string> out;
    for (auto& tok : word_tokens(text)) {
        if (tok.size() > 2 && !stopword(tok))
            out.insert(std::move(tok));
    }
    return out;
}

std::unordered_set<std::string>
ArguanaTextPairAdapter::content_set_first_words(std::string_view text, std::uint32_t max_words) {
    std::unordered_set<std::string> out;
    auto words = word_tokens(text);
    const std::uint32_t n =
        std::min<std::uint32_t>(max_words, static_cast<std::uint32_t>(words.size()));
    for (std::uint32_t i = 0; i < n; ++i) {
        auto& tok = words[i];
        if (tok.size() > 2 && !stopword(tok))
            out.insert(std::move(tok));
    }
    return out;
}

float ArguanaTextPairAdapter::jaccard_set(const std::unordered_set<std::string>& a,
                                          const std::unordered_set<std::string>& b) {
    if (a.empty() && b.empty())
        return 0.0f;
    const auto* small = &a;
    const auto* large = &b;
    if (small->size() > large->size())
        std::swap(small, large);
    std::uint32_t inter = 0;
    for (const auto& x : *small) {
        if (large->contains(x))
            ++inter;
    }
    const std::uint32_t uni = static_cast<std::uint32_t>(a.size() + b.size() - inter);
    return uni ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

std::uint32_t ArguanaTextPairAdapter::cue_count(const std::unordered_set<std::string>& toks) {
    std::uint32_t n = 0;
    for (const auto& tok : toks) {
        if (arguana_cues().contains(tok))
            ++n;
    }
    return n;
}

bool ArguanaTextPairAdapter::same_prefix(const SeededDoc& a, const SeededDoc& b) const noexcept {
    if (a.tokens.size() < prefix_terms_ || b.tokens.size() < prefix_terms_)
        return false;
    for (std::uint32_t i = 0; i < prefix_terms_; ++i) {
        if (a.tokens[i] != b.tokens[i])
            return false;
    }
    return true;
}

void ArguanaTextPairAdapter::seed_doc(std::string_view /*doc_id*/, std::string_view doc_text,
                                      std::uint32_t doc_index) {
    SeededDoc doc;
    doc.index = doc_index;
    doc.normalized = normalize_ws_lower(doc_text);
    doc.tokens = word_tokens(doc_text);
    doc.content = content_set(doc_text);
    doc.first35_content = content_set_first_words(doc_text, 35);
    docs_.push_back(std::move(doc));
}

AdapterEvidence ArguanaTextPairAdapter::process_doc(std::string_view /*doc_id*/,
                                                    std::string_view doc_text) {
    AdapterEvidence ev;
    ev.aux_field = extract_lead_tokens(doc_text, 64);
    return ev;
}

AdapterEvidence ArguanaTextPairAdapter::process_query(std::string_view /*query_id*/,
                                                      std::string_view query_text) {
    AdapterEvidence ev;
    ev.aux_field = extract_lead_tokens(query_text, 64);
    if (docs_.empty())
        return ev;

    std::string qnorm = normalize_ws_lower(query_text);
    if (qnorm.size() > 80)
        qnorm.resize(80);
    if (qnorm.empty())
        return ev;

    const SeededDoc* self = nullptr;
    for (const auto& doc : docs_) {
        if (doc.normalized.find(qnorm) != std::string::npos) {
            self = &doc;
            break;
        }
    }
    if (self == nullptr)
        return ev;

    const auto q_content = content_set(query_text);
    const std::string_view qv{query_text};
    const std::size_t split = qv.find("  ");
    const auto title_content =
        content_set(split == std::string_view::npos ? qv : qv.substr(0, split));
    const auto body_content = split == std::string_view::npos ? std::unordered_set<std::string>{}
                                                              : content_set(qv.substr(split + 2));

    struct Candidate {
        std::uint32_t index;
        float raw;
    };
    std::vector<Candidate> cand;
    cand.reserve(docs_.size());
    for (const auto& doc : docs_) {
        if (doc.index == self->index || !same_prefix(*self, doc))
            continue;
        const std::uint32_t dist =
            self->index > doc.index ? self->index - doc.index : doc.index - self->index;
        const float prox = 1.0f / (1.0f + static_cast<float>(dist));
        const float dist_penalty = -static_cast<float>(dist) / 20.0f;
        const float q_j = jaccard_set(q_content, doc.content);
        const float body_j = jaccard_set(body_content, doc.content);
        const float title_j = jaccard_set(title_content, doc.content);
        const float title35_j = jaccard_set(title_content, doc.first35_content);
        const float body35_j = jaccard_set(body_content, doc.first35_content);
        const float cue = static_cast<float>(cue_count(doc.content)) / 10.0f;
        const float shorter =
            (static_cast<float>(self->content.size()) - static_cast<float>(doc.content.size())) /
            std::max(1.0f, static_cast<float>(self->content.size()));
        float raw = -0.5f * prox + dist_penalty + 5.0f * q_j - body_j + title_j + 0.5f * cue +
                    0.5f * shorter;
        if (claim_premise_mode_)
            raw += 2.5f * title35_j + 1.0f * body35_j;
        cand.push_back({doc.index, raw});
    }
    if (cand.empty())
        return ev;

    float min_raw = cand.front().raw;
    float max_raw = cand.front().raw;
    for (const auto& c : cand) {
        min_raw = std::min(min_raw, c.raw);
        max_raw = std::max(max_raw, c.raw);
    }
    const float denom = std::max(1e-6f, max_raw - min_raw);
    std::sort(cand.begin(), cand.end(),
              [](const Candidate& a, const Candidate& b) { return a.raw > b.raw; });
    for (const auto& c : cand) {
        const float w = (c.raw - min_raw) / denom;
        if (w > 0.0f)
            ev.relations.push_back({c.index, w});
    }
    return ev;
}

} // namespace simeon
