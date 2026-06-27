#include "simeon/corpus_adapter.hpp"

#include <algorithm>
#include <cmath>

#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {} // namespace

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
// ---------------------------------------------------------------------------

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
