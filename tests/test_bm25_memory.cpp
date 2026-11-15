// Memory contract and structural invariants for Bm25Index.
//
// Covers:
//   - FlatHashMapU64::reserve grows geometrically under incremental calls
//   - reserve_docs() stays small for large expected corpora
//   - memory_bytes() grows linearly with the corpus, never shrinks while
//     adding, and matches a full recount (validate())
//   - max_memory_bytes refuses docs without corrupting the index
//   - every variant keeps validate() clean, scores finite, and ranks the
//     same regardless of insertion order

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/flat_hash_map_u64.hpp"

using simeon::Bm25Config;
using simeon::Bm25Index;
using simeon::Bm25MemoryBudgetExceeded;
using simeon::Bm25Variant;

namespace {

#define CHECK_VALID(idx)                                                                           \
    do {                                                                                           \
        const std::string err_ = (idx).validate();                                                 \
        if (!err_.empty()) {                                                                       \
            std::fprintf(stderr, "%s:%d validate(): %s\n", __FILE__, __LINE__, err_.c_str());      \
            assert(false && "Bm25Index::validate failed");                                         \
        }                                                                                          \
    } while (0)

// Deterministic corpus with a skewed vocabulary, so some terms are common,
// most are rare, and n-gram tables see real collisions.
struct Rng {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

std::string make_word(Rng& r, std::size_t vocab) {
    // Square of a uniform draw skews toward low ids (common words).
    const double u = static_cast<double>(r.next() % 1'000'000) / 1'000'000.0;
    const std::size_t id = static_cast<std::size_t>(u * u * static_cast<double>(vocab));
    std::string w = "w";
    std::size_t x = id;
    do {
        w.push_back(static_cast<char>('a' + x % 26));
        x /= 26;
    } while (x);
    return w;
}

std::vector<std::string> make_corpus(std::size_t docs, std::size_t vocab, std::uint64_t seed,
                                     std::size_t min_len = 5, std::size_t max_len = 60) {
    Rng r{seed | 1};
    std::vector<std::string> out;
    out.reserve(docs);
    for (std::size_t d = 0; d < docs; ++d) {
        const std::size_t len = min_len + r.next() % (max_len - min_len + 1);
        std::string text;
        for (std::size_t i = 0; i < len; ++i) {
            if (i)
                text.push_back(' ');
            text += make_word(r, vocab);
        }
        out.push_back(std::move(text));
    }
    return out;
}

const Bm25Variant kAllVariants[] = {
    Bm25Variant::Atire,    Bm25Variant::BM25Plus,
    Bm25Variant::BM25L,    Bm25Variant::DLH13,
    Bm25Variant::PL2,      Bm25Variant::DPH,
    Bm25Variant::Dcm,      Bm25Variant::SubwordAwareBackoff,
    Bm25Variant::AtireLTD, Bm25Variant::Layered,
    Bm25Variant::LayeredW,
};

Bm25Config make_cfg(Bm25Variant v, bool bigrams) {
    Bm25Config cfg;
    cfg.variant = v;
    cfg.build_word_bigrams = bigrams || v == Bm25Variant::Layered || v == Bm25Variant::LayeredW;
    if (v == Bm25Variant::SubwordAwareBackoff)
        cfg.subword_gamma = 5.0f;
    return cfg;
}

void test_flat_map_reserve_is_geometric() {
    simeon::FlatHashMapU64<std::uint32_t> m;
    std::size_t changes = 0;
    std::size_t last = m.memory_bytes();
    for (std::uint64_t k = 0; k < 20'000; ++k) {
        // Same pattern Bm25Index::add_doc uses: reserve size + new terms.
        m.reserve(m.size() + 1);
        m[k * 0x9E3779B97F4A7C15ULL] = 1;
        if (m.memory_bytes() != last) {
            ++changes;
            last = m.memory_bytes();
        }
    }
    // Exact reservation reallocated on every insert (~20k changes).
    assert(changes < 64);
}

void test_shrink_to_fit_never_grows() {
    simeon::FlatHashMapU64<std::uint32_t> m;
    m.reserve(3); // below the 16-slot default
    m[1] = 1;
    m[2] = 2;
    const std::size_t before = m.memory_bytes();
    m.shrink_to_fit();
    assert(m.memory_bytes() <= before);
    assert(m.find(1) != m.end() && m.find(2) != m.end());

    // Found by fuzz_bm25: a tiny aux field made finalize() grow the index.
    Bm25Config cfg = make_cfg(Bm25Variant::SubwordAwareBackoff, false);
    Bm25Index idx(cfg);
    idx.reserve_docs(6);
    for (int d = 0; d < 6; ++d)
        idx.add_doc("the quick brown fox", d == 2 ? "lead" : "");
    const std::size_t pre = idx.memory_bytes();
    idx.finalize();
    assert(idx.memory_bytes() <= pre);
    CHECK_VALID(idx);
}

void test_reserve_docs_is_bounded() {
    for (bool bigrams : {false, true}) {
        Bm25Index idx(make_cfg(Bm25Variant::SubwordAwareBackoff, bigrams));
        idx.reserve_docs(90'000);
        // Before the fix this reserved ~99M n-gram entries per table (GBs).
        assert(idx.memory_bytes() < (std::size_t{256} << 20));
        CHECK_VALID(idx);
    }
    Bm25Config cfg = make_cfg(Bm25Variant::SubwordAwareBackoff, false);
    cfg.max_memory_bytes = std::size_t{8} << 20;
    Bm25Index capped(cfg);
    capped.reserve_docs(1'000'000);
    assert(capped.memory_bytes() <= cfg.max_memory_bytes / 2);

    // Tiny budgets: up-front space stays within half the budget, and a
    // budget too small for any table reserves nothing.
    for (std::size_t tiny : {std::size_t{1}, std::size_t{64}, std::size_t{1024}}) {
        Bm25Config tcfg = make_cfg(Bm25Variant::SubwordAwareBackoff, true);
        tcfg.max_memory_bytes = tiny;
        Bm25Index t(tcfg);
        t.reserve_docs(1'000);
        assert(t.memory_bytes() <= tiny / 2);
        if (tiny == 1)
            assert(t.memory_bytes() == 0);
        // Docs are admitted until the cap is reached, then refused.
        std::size_t added = 0;
        while (t.try_add_doc("alpha beta gamma delta"))
            ++added;
        assert(added >= 1);
        assert(t.memory_bytes() >= tiny);
        CHECK_VALID(t);
    }
}

void test_accounting_tracks_every_add() {
    const auto corpus = make_corpus(300, 2'000, 7);
    for (auto v : kAllVariants) {
        for (bool aux : {false, true}) {
            Bm25Index idx(make_cfg(v, false));
            std::size_t prev = idx.memory_bytes();
            for (std::size_t d = 0; d < corpus.size(); ++d) {
                // Aux field switches on part way through, as it can in yams.
                const std::string_view aux_text =
                    (aux && d >= 50)
                        ? std::string_view(corpus[(d * 7) % corpus.size()]).substr(0, 20)
                        : std::string_view{};
                idx.add_doc(corpus[d], aux_text);
                const std::size_t now = idx.memory_bytes();
                assert(now >= prev);
                prev = now;
                if (d % 37 == 0)
                    CHECK_VALID(idx);
            }
            CHECK_VALID(idx);
            const auto bd = idx.memory_breakdown();
            assert(bd.total() == idx.memory_bytes());
            assert(bd.word > 0);
            assert((bd.ngram > 0) == (v == Bm25Variant::SubwordAwareBackoff));
            assert((bd.aux_word > 0) == aux);
            idx.finalize();
            CHECK_VALID(idx);
            // Finalize shrinks lists; it must not grow the index.
            assert(idx.memory_bytes() <= prev);
        }
    }
}

void test_memory_grows_linearly() {
    for (auto v : {Bm25Variant::Atire, Bm25Variant::SubwordAwareBackoff}) {
        std::size_t bytes[2] = {0, 0};
        const std::size_t sizes[2] = {2'000, 4'000};
        for (int i = 0; i < 2; ++i) {
            const auto corpus = make_corpus(sizes[i], 20'000, 11);
            Bm25Index idx(make_cfg(v, false));
            idx.reserve_docs(corpus.size());
            for (const auto& doc : corpus)
                idx.add_doc(doc);
            bytes[i] = idx.memory_bytes();
        }
        // Doubling the corpus should at most roughly double memory; capacity
        // doubling can add slack, quadratic growth would be ~4x.
        const double ratio = static_cast<double>(bytes[1]) / static_cast<double>(bytes[0]);
        if (ratio > 2.6)
            std::fprintf(stderr, "variant %d: memory ratio %.2f (%zu -> %zu)\n",
                         static_cast<int>(v), ratio, bytes[0], bytes[1]);
        assert(ratio > 1.2 && ratio <= 2.6);
    }
}

void test_budget_refuses_without_corruption() {
    const auto corpus = make_corpus(2'000, 5'000, 23);
    for (auto v : {Bm25Variant::Atire, Bm25Variant::SubwordAwareBackoff, Bm25Variant::Layered}) {
        Bm25Config cfg = make_cfg(v, false);
        cfg.max_memory_bytes = std::size_t{1} << 20;
        Bm25Index idx(cfg);
        idx.reserve_docs(corpus.size());
        std::size_t added = 0;
        for (const auto& doc : corpus) {
            if (!idx.try_add_doc(doc))
                break;
            ++added;
        }
        assert(added > 0 && added < corpus.size());
        assert(idx.doc_count() == added);
        // Over budget by at most one doc's growth, which can include one map doubling.
        assert(idx.memory_bytes() >= cfg.max_memory_bytes);
        assert(idx.memory_bytes() < 2 * cfg.max_memory_bytes);
        CHECK_VALID(idx);

        // Refusal is sticky and leaves the index untouched.
        const std::size_t before = idx.memory_bytes();
        assert(!idx.try_add_doc("one more doc"));
        bool threw = false;
        try {
            idx.add_doc("one more doc");
        } catch (const Bm25MemoryBudgetExceeded&) {
            threw = true;
        }
        assert(threw);
        assert(idx.doc_count() == added);
        assert(idx.memory_bytes() == before);

        // A partial index is still usable.
        idx.finalize();
        CHECK_VALID(idx);
        std::vector<float> s(idx.doc_count());
        idx.score("wa wb wc", s);
        for (float x : s)
            assert(std::isfinite(x));
    }
}

void test_variants_finite_and_order_independent() {
    const auto corpus = make_corpus(120, 800, 31, 1, 40);
    std::vector<std::size_t> perm(corpus.size());
    std::iota(perm.begin(), perm.end(), 0);
    Rng r{99};
    for (std::size_t i = perm.size(); i > 1; --i)
        std::swap(perm[i - 1], perm[r.next() % i]);

    const char* queries[] = {"wa wb",           "wc",          "",      "!!! ???",
                             "zzzzzzzz unseen", "wa wa wa wd", "WA Wb", "w"};
    for (auto v : kAllVariants) {
        for (bool bigrams : {false, true}) {
            const Bm25Config cfg = make_cfg(v, bigrams);
            Bm25Index a(cfg);
            Bm25Index b(cfg);
            for (const auto& doc : corpus)
                a.add_doc(doc);
            for (std::size_t i = 0; i < corpus.size(); ++i)
                b.add_doc(corpus[perm[i]]);
            a.finalize();
            b.finalize();
            CHECK_VALID(a);
            CHECK_VALID(b);

            std::vector<float> sa(corpus.size());
            std::vector<float> sb(corpus.size());
            for (const char* q : queries) {
                a.score(q, sa);
                b.score(q, sb);
                for (std::size_t i = 0; i < corpus.size(); ++i) {
                    assert(std::isfinite(sb[i]));
                    // Doc perm[i] in `a` is doc i in `b`.
                    const float x = sa[perm[i]];
                    const float y = sb[i];
                    if (std::abs(x - y) > 1e-5f * std::max(1.0f, std::abs(x))) {
                        std::fprintf(stderr, "variant %d bigrams %d query '%s': %g vs %g\n",
                                     static_cast<int>(v), bigrams, q, x, y);
                        assert(false && "score depends on insertion order");
                    }
                }
            }
        }
    }
}

void test_finalize_idempotent_and_sealed() {
    const auto corpus = make_corpus(50, 300, 5);
    Bm25Index idx(make_cfg(Bm25Variant::SubwordAwareBackoff, true));
    for (const auto& doc : corpus)
        idx.add_doc(doc);
    idx.finalize();
    std::vector<float> s1(idx.doc_count());
    std::vector<float> s2(idx.doc_count());
    idx.score("wa wb wc", s1);
    const std::size_t bytes = idx.memory_bytes();
    idx.finalize();
    idx.score("wa wb wc", s2);
    assert(s1 == s2);
    assert(idx.memory_bytes() == bytes);
    CHECK_VALID(idx);

    bool threw = false;
    try {
        (void)idx.try_add_doc("late doc");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(idx.doc_count() == corpus.size());
}

void test_forward_index_is_counted() {
    const auto corpus = make_corpus(80, 400, 13);
    Bm25Index idx;
    for (const auto& doc : corpus)
        idx.add_doc(doc);
    idx.finalize();
    assert(idx.memory_breakdown().forward == 0);
    const std::uint32_t top[] = {0, 1, 2};
    const float w[] = {1.0f, 0.5f, 0.25f};
    std::vector<std::pair<std::uint64_t, float>> terms;
    idx.build_relevance_model(top, w, terms);
    assert(!terms.empty());
    assert(idx.memory_breakdown().forward > 0);
}

} // namespace

int main() {
    test_flat_map_reserve_is_geometric();
    test_shrink_to_fit_never_grows();
    test_reserve_docs_is_bounded();
    test_accounting_tracks_every_add();
    test_memory_grows_linearly();
    test_budget_refuses_without_corruption();
    test_variants_finite_and_order_independent();
    test_finalize_idempotent_and_sealed();
    test_forward_index_is_counted();
    std::puts("test_bm25_memory: ok");
    return 0;
}
