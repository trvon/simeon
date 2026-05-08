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

// ---------------------------------------------------------------------------
// ScientificAdapter (v3 — robust, IMRAD-aware)
// ---------------------------------------------------------------------------

namespace {

const std::unordered_set<std::string_view>& biomedical_suffixes() {
    static const std::unordered_set<std::string_view> kSuffixes = {
        "ase",    "itis",   "osis",  "emia",    "oma",      "pathy",    "penia",  "uria",   "phage",
        "plasty", "ectomy", "otomy", "scopy",   "gram",     "graphy",   "ology",  "iasis",  "genic",
        "lysis",  "toxin",  "mycin", "cillin",  "vir",      "pril",     "sartan", "statin", "olol",
        "pine",   "zole",   "mab",   "cycline", "floxacin", "conazole", "parin",  "tide",   "lide",
        "sone",   "lone",   "dopa",  "gen",     "cyte",     "blast",    "some",   "mer",    "dase",
    };
    return kSuffixes;
}

bool ends_with_biomedical(std::string_view word) {
    if (word.size() < 4)
        return false;
    for (const auto& suffix : biomedical_suffixes()) {
        if (word.size() <= suffix.size())
            continue;
        if (word.ends_with(suffix)) {
            const char prev = word[word.size() - suffix.size() - 1];
            if (std::isalpha(static_cast<unsigned char>(prev)))
                return true;
        }
    }
    return false;
}

bool is_all_caps_word(std::string_view word) {
    if (word.size() < 2)
        return false;
    bool has_letter = false;
    for (char c : word) {
        if (std::isalpha(static_cast<unsigned char>(c))) {
            if (!std::isupper(static_cast<unsigned char>(c)))
                return false;
            has_letter = true;
        }
    }
    return has_letter;
}

bool is_measurement(std::string_view text) {
    if (text.empty() || !std::isdigit(static_cast<unsigned char>(text[0])))
        return false;
    for (std::size_t i = 1; i < text.size(); ++i) {
        char c = text[i];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == ',')
            continue;
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '%')
            return i >= 1;
        break;
    }
    return false;
}

std::string normalize_entity(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '(' || c == ')')
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    while (!out.empty() && out.back() == '-')
        out.pop_back();
    return out;
}

// Language-agnostic section header detection (structural heuristics only)
struct Section {
    std::string_view text;
    bool is_header;
};
std::vector<Section> detect_sections(std::string_view text, std::size_t max_s = 16) {
    std::vector<Section> out;
    std::size_t pos = 0;
    const std::size_t lim = std::min(text.size(), std::size_t{16384});
    while (pos < lim && out.size() < max_s) {
        auto nl = text.find('\n', pos);
        if (nl == std::string_view::npos)
            nl = lim;
        std::string_view line = text.substr(pos, nl - pos);
        std::size_t si = 0;
        while (si < line.size() && std::isspace(static_cast<unsigned char>(line[si])))
            ++si;
        while (si < line.size() && (std::isdigit(static_cast<unsigned char>(line[si])) ||
                                    line[si] == '.' || line[si] == ' '))
            ++si;
        bool hdr = false;
        if (si < line.size()) {
            std::size_t w = 0;
            bool iw = false, ac = true;
            bool hl = false;
            for (std::size_t i = si; i < line.size(); ++i) {
                unsigned char c = static_cast<unsigned char>(line[i]);
                if (std::isspace(c)) {
                    iw = false;
                } else {
                    if (!iw) {
                        ++w;
                        iw = true;
                    }
                    if (std::isalpha(c)) {
                        hl = true;
                        if (!std::isupper(c))
                            ac = false;
                    }
                }
            }
            hdr = hl && w >= 1 && w <= 8 && (ac || line.size() - si < 80);
        }
        out.push_back({line, hdr});
        pos = nl + 1;
    }
    return out;
}

// Build aux field: title lines + first section content
std::string build_aux_field(std::string_view text, std::size_t max_chars = 512) {
    auto secs = detect_sections(text);
    std::string out;
    out.reserve(max_chars);
    for (const auto& s : secs) {
        if (s.is_header)
            break;
        if (!s.text.empty()) {
            if (!out.empty())
                out.push_back(' ');
            out.append(s.text);
        }
        if (out.size() >= max_chars / 2)
            break;
    }
    bool past = false;
    for (const auto& s : secs) {
        if (!past && s.is_header) {
            past = true;
            continue;
        }
        if (past && !s.is_header && !s.text.empty()) {
            if (!out.empty())
                out.push_back(' ');
            std::size_t a = max_chars > out.size() ? max_chars - out.size() : 0;
            if (a > s.text.size())
                out.append(s.text);
            else {
                out.append(s.text.data(), a);
                break;
            }
        }
    }
    if (out.size() > max_chars)
        out.resize(max_chars);
    return out;
}

// Frequency-weighted entity extraction
std::vector<std::string> extract_robust_entities(std::string_view text,
                                                 std::size_t max_entities = 32) {
    std::vector<std::string> entities;
    std::unordered_set<std::string> seen;
    std::unordered_map<std::string, int> freq;
    std::string cur;
    std::vector<std::string> cap;

    auto flush_mw = [&]() {
        if (cap.size() >= 2) {
            std::string j;
            for (std::size_t i = 0; i < cap.size(); ++i) {
                if (i)
                    j.push_back(' ');
                j += cap[i];
            }
            auto n = normalize_entity(j);
            if (n.size() >= 8 && !seen.contains(n)) {
                seen.insert(n);
                freq[n]++;
            }
        }
        cap.clear();
    };

    for (std::size_t i = 0; i <= text.size() && entities.size() < max_entities; ++i) {
        const char c = i < text.size() ? text[i] : ' ';
        bool d = std::isspace(static_cast<unsigned char>(c)) || c == '.' || c == ',' || c == ';' ||
                 c == ':' || c == '!' || c == '?' || c == '(' || c == ')' || c == '"' ||
                 c == '\n' || c == '\t';
        if (!d) {
            cur.push_back(c);
            continue;
        }
        if (cur.empty()) {
            flush_mw();
            continue;
        }
        bool added = false;
        if (ends_with_biomedical(cur) && cur.size() >= 5) {
            entities.push_back(cur);
            added = true;
        } else if (is_all_caps_word(cur) && cur.size() >= 2) {
            entities.push_back(cur);
            added = true;
            cap.push_back(cur);
        } else if (!cur.empty() && std::isupper(static_cast<unsigned char>(cur[0])) &&
                   cur.size() >= 3) {
            bool aa = true;
            for (char ch : cur)
                if (!std::isalpha(static_cast<unsigned char>(ch))) {
                    aa = false;
                    break;
                }
            if (aa) {
                cap.push_back(cur);
                cur.clear();
                continue;
            }
        } else if (is_measurement(cur) && cur.size() >= 2) {
            entities.push_back(cur);
            added = true;
        } else {
            flush_mw();
        }
        if (added) {
            auto n = normalize_entity(cur);
            if (n.size() >= 2 && seen.insert(n).second)
                freq[n] = 1;
        }
        cur.clear();
    }
    flush_mw();

    if (entities.size() > max_entities) {
        std::unordered_map<std::string, int> rank;
        for (std::size_t i = 0; i < entities.size(); ++i)
            rank[normalize_entity(entities[i])] = static_cast<int>(i);
        std::stable_sort(entities.begin(), entities.end(),
                         [&](const std::string& a, const std::string& b) {
                             int fa = freq[normalize_entity(a)], fb = freq[normalize_entity(b)];
                             if (fa != fb)
                                 return fa > fb;
                             return rank[normalize_entity(a)] < rank[normalize_entity(b)];
                         });
        entities.resize(max_entities);
    }
    return entities;
}

} // namespace

bool ScientificAdapter::is_biomedical_suffix(std::string_view word) {
    return ends_with_biomedical(word);
}

std::vector<std::string> ScientificAdapter::extract_entities(std::string_view text,
                                                             std::size_t max_entities) {
    return extract_robust_entities(text, max_entities);
}

AdapterEvidence ScientificAdapter::process_doc(std::string_view /*doc_id*/,
                                               std::string_view doc_text) {
    AdapterEvidence ev;
    ev.aux_field = build_aux_field(doc_text, 512);
    ev.entities = extract_robust_entities(doc_text);
    return ev;
}

AdapterEvidence ScientificAdapter::process_query(std::string_view /*query_id*/,
                                                 std::string_view query_text) {
    AdapterEvidence ev;
    ev.aux_field = extract_lead_tokens(query_text, 64);
    ev.entities = extract_robust_entities(query_text);
    return ev;
}

} // namespace simeon
