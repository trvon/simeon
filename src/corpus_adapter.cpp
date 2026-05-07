#include "simeon/corpus_adapter.hpp"

namespace simeon {

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

} // namespace simeon
