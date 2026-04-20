#include "simeon/concept_mining.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "simeon/hasher.hpp"
#include "simeon/tokenizer.hpp"

namespace simeon {

namespace {

struct WordHashSink final : NGramEmitter {
    std::vector<std::uint64_t>* out;
    HashFamily family;
    std::uint64_t seed;
    void on_token(std::string_view tok, float) override {
        out->push_back(hash64(tok, seed, family));
    }
};

constexpr TokenizerConfig word_only_cfg() noexcept {
    return TokenizerConfig{0, 0, false, true};
}

// Per-(concept, doc) Atire BM25 contribution.
inline float score_atire(float tf, float dl, float idf, float k1, float b, float avg_dl) noexcept {
    if (avg_dl <= 0.0f)
        return 0.0f;
    const float denom = tf + k1 * (1.0f - b + b * dl / avg_dl);
    return idf * tf * (k1 + 1.0f) / denom;
}

} // namespace

std::uint64_t ConceptIndex::hash_bigram(std::uint64_t a, std::uint64_t b) noexcept {
    // Same mix Bm25Index::hash_bigram uses, but ordered (a, b) — concepts
    // are position-sensitive multi-word terms, unlike SDM unordered bigrams.
    return splitmix64_mix(a ^ splitmix64_mix(b + 0x9E3779B97F4A7C15ULL));
}

const ConceptEntry* ConceptIndex::find(std::uint64_t concept_hash) const noexcept {
    const auto it = concepts_.find(concept_hash);
    return it == concepts_.end() ? nullptr : &it->second;
}

void ConceptIndex::score(std::string_view query, std::span<float> out_scores) const {
    if (concepts_.empty() || out_scores.size() != doc_count_)
        return;

    std::vector<std::uint64_t> qwords;
    WordHashSink sink{};
    sink.out = &qwords;
    sink.family = hash_family_;
    sink.seed = hash_seed_;
    const auto tcfg = word_only_cfg();
    tokenize(query, tcfg, sink);

    if (qwords.size() < 2)
        return;

    // Collect distinct query bigrams that match a mined concept.
    std::unordered_map<std::uint64_t, float> query_concept_pmi;
    query_concept_pmi.reserve(qwords.size());
    for (std::size_t i = 1; i < qwords.size(); ++i) {
        const std::uint64_t bh = hash_bigram(qwords[i - 1], qwords[i]);
        auto it = concepts_.find(bh);
        if (it != concepts_.end())
            query_concept_pmi[bh] = it->second.pmi;
    }

    // For each matched concept, walk its postings and add PMI-weighted
    // BM25 to the corresponding doc's score entry.
    for (const auto& [bh, pmi] : query_concept_pmi) {
        const auto it = concepts_.find(bh);
        if (it == concepts_.end())
            continue;
        const ConceptEntry& e = it->second;
        for (const auto& [did, tf] : e.docs) {
            if (did >= out_scores.size())
                continue;
            const float dl = static_cast<float>(bigram_doc_lengths_[did]);
            out_scores[did] +=
                pmi * score_atire(static_cast<float>(tf), dl, e.idf, k1_, b_, avg_bigram_dl_);
        }
    }
}

ConceptIndex mine_concepts(const Bm25Index& idx, std::span<const std::string_view> docs,
                           const ConceptConfig& cfg) {
    ConceptIndex out;
    out.k1_ = cfg.k1;
    out.b_ = cfg.b;
    out.hash_family_ = idx.config().hash;
    out.hash_seed_ = idx.config().hash_seed;
    out.doc_count_ = static_cast<std::uint32_t>(docs.size());
    out.bigram_doc_lengths_.assign(docs.size(), 0u);

    if (docs.empty())
        return out;

    // Scratch buffers reused per doc.
    std::vector<std::uint64_t> words;
    std::unordered_map<std::uint64_t, std::uint32_t> doc_bgm_tf;

    // Raw accumulator: concept_hash -> (total_tf, postings).
    struct RawEntry {
        std::uint64_t total_tf = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> docs;
        std::uint64_t a_hash = 0; // first unigram component (for PMI)
        std::uint64_t b_hash = 0; // second unigram component (for PMI)
    };
    std::unordered_map<std::uint64_t, RawEntry> raw;

    // Pass 1: tokenize each doc, accumulate bigram postings and doc lengths.
    const auto tcfg = word_only_cfg();
    std::uint64_t total_bigrams_across_corpus = 0;
    for (std::uint32_t did = 0; did < docs.size(); ++did) {
        words.clear();
        WordHashSink sink{};
        sink.out = &words;
        sink.family = out.hash_family_;
        sink.seed = out.hash_seed_;
        tokenize(docs[did], tcfg, sink);

        if (words.size() < 2)
            continue;

        const std::uint32_t doc_bgm_count = static_cast<std::uint32_t>(words.size() - 1);
        out.bigram_doc_lengths_[did] = doc_bgm_count;
        total_bigrams_across_corpus += doc_bgm_count;

        // Pair each bigram hash with its first-seen (a, b) components in
        // this doc, then roll into per-doc tf.
        doc_bgm_tf.clear();
        for (std::size_t i = 1; i < words.size(); ++i) {
            const std::uint64_t a = words[i - 1];
            const std::uint64_t b = words[i];
            const std::uint64_t bh = ConceptIndex::hash_bigram(a, b);
            const std::uint32_t was = doc_bgm_tf[bh];
            doc_bgm_tf[bh] = was + 1u;
            if (was == 0u) {
                auto& e = raw[bh];
                if (e.a_hash == 0 && e.b_hash == 0) {
                    e.a_hash = a;
                    e.b_hash = b;
                }
            }
        }
        for (const auto& [bh, tf] : doc_bgm_tf) {
            auto& e = raw[bh];
            e.total_tf += tf;
            e.docs.emplace_back(did, tf);
        }
    }

    const float avg_bigram_dl =
        out.doc_count_ == 0 ? 0.0f
                            : static_cast<float>(static_cast<double>(total_bigrams_across_corpus) /
                                                 static_cast<double>(out.doc_count_));
    out.avg_bigram_dl_ = avg_bigram_dl;

    // Total unigram tokens across corpus — denominator for unigram P(t).
    // Bm25Index caches this as total_tokens().
    const double total_tokens_d = static_cast<double>(idx.total_tokens());
    if (total_tokens_d <= 0.0)
        return out;

    // Pass 2: PMI filter. PMI(a, b) = log( P(a,b) / (P(a) * P(b)) )
    //   P(a,b) = bigram.total_tf / total_bigrams_across_corpus
    //   P(a)   = idx.total_tf(a) / idx.total_tokens()
    //   P(b)   = idx.total_tf(b) / idx.total_tokens()
    // Skip concepts with total_tf < min_ttf, PMI < pmi_floor, or
    // unknown component stats.
    const double total_bgm_d = static_cast<double>(total_bigrams_across_corpus);
    if (total_bgm_d <= 0.0)
        return out;

    std::vector<std::pair<std::uint64_t, float>> accepted; // (hash, PMI)
    accepted.reserve(raw.size() / 4);

    for (const auto& [bh, e] : raw) {
        if (e.total_tf < cfg.min_ttf)
            continue;
        const std::uint64_t tf_a = idx.total_tf_by_hash(e.a_hash);
        const std::uint64_t tf_b = idx.total_tf_by_hash(e.b_hash);
        if (tf_a == 0 || tf_b == 0)
            continue;
        const double p_ab = static_cast<double>(e.total_tf) / total_bgm_d;
        const double p_a = static_cast<double>(tf_a) / total_tokens_d;
        const double p_b = static_cast<double>(tf_b) / total_tokens_d;
        const double pmi = std::log(p_ab / (p_a * p_b));
        if (!std::isfinite(pmi))
            continue;
        if (pmi < static_cast<double>(cfg.pmi_floor))
            continue;
        accepted.emplace_back(bh, static_cast<float>(pmi));
    }

    // Enforce max_concepts cap by keeping highest-PMI concepts.
    if (accepted.size() > cfg.max_concepts) {
        std::nth_element(
            accepted.begin(), accepted.begin() + static_cast<std::ptrdiff_t>(cfg.max_concepts),
            accepted.end(), [](const auto& x, const auto& y) { return x.second > y.second; });
        accepted.resize(cfg.max_concepts);
    }

    // Build concept index from accepted set. idf uses the BM25 document
    // frequency formula applied to doc_count_ and posting size.
    const float nf = static_cast<float>(out.doc_count_);
    out.concepts_.reserve(accepted.size());
    for (const auto& [bh, pmi] : accepted) {
        const auto rit = raw.find(bh);
        if (rit == raw.end())
            continue;
        ConceptEntry entry;
        entry.pmi = pmi;
        entry.total_tf = rit->second.total_tf;
        entry.docs = rit->second.docs;
        std::sort(entry.docs.begin(), entry.docs.end(),
                  [](const auto& x, const auto& y) { return x.first < y.first; });
        const float df = static_cast<float>(entry.docs.size());
        entry.idf = std::log((nf - df + 0.5f) / (df + 0.5f) + 1.0f);
        out.concepts_.emplace(bh, std::move(entry));
    }

    return out;
}

void score_bm25_with_concepts(const Bm25Index& idx, const ConceptIndex& concepts,
                              std::string_view query, float concept_weight,
                              std::span<float> out_scores) {
    std::fill(out_scores.begin(), out_scores.end(), 0.0f);
    idx.score(query, out_scores);
    if (concept_weight == 0.0f || concepts.size() == 0)
        return;
    std::vector<float> cscores(out_scores.size(), 0.0f);
    concepts.score(query, std::span<float>{cscores});
    for (std::size_t i = 0; i < out_scores.size(); ++i)
        out_scores[i] += concept_weight * cscores[i];
}

} // namespace simeon
