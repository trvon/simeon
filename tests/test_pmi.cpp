#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/pmi.hpp"
#include "simeon/simeon.hpp"

using simeon::Encoder;
using simeon::EncoderConfig;
using simeon::NGramMode;
using simeon::PmiConfig;
using simeon::PmiEmbeddings;
using simeon::ProjectionMode;

namespace {

std::vector<std::string_view> view_of(const std::vector<std::string>& v) {
    std::vector<std::string_view> out;
    out.reserve(v.size());
    for (const auto& s : v)
        out.emplace_back(s);
    return out;
}

double cosine(const float* a, const float* b, std::uint32_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0.0 || nb <= 0.0)
        return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

std::vector<std::string> build_synonym_corpus() {
    // Each of cat/feline vs dog/canine co-occurs with its own verb and
    // adverb (purr/soft vs bark/loud). Repeat enough times for min_count
    // and SPPMI to have real signal.
    std::vector<std::string> docs;
    const std::vector<std::string> templates = {
        "the cat purrs softly in the room", "the feline purrs softly in the room",
        "a cat purred softly last night",   "a feline purred softly last night",
        "the dog barks loudly in the yard", "the canine barks loudly in the yard",
        "a dog barked loudly this morning", "a canine barked loudly this morning",
    };
    for (int rep = 0; rep < 40; ++rep) {
        for (const auto& t : templates)
            docs.push_back(t);
    }
    return docs;
}

PmiConfig default_pmi_cfg() {
    PmiConfig cfg;
    cfg.window_size = 5;
    cfg.target_rank = 16;
    cfg.min_token_count = 5;
    cfg.max_vocab_size = 200;
    cfg.svd_iters = 4;
    cfg.svd_oversample = 8;
    cfg.svd_seed = 0xABCD1234ULL;
    return cfg;
}

void test_learn_empty_throws() {
    std::vector<std::string_view> empty;
    bool threw = false;
    try {
        (void)PmiEmbeddings::learn(std::span<const std::string_view>(empty), default_pmi_cfg());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_learn_deterministic() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto a = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());
    auto b = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());
    assert(a.dim() == b.dim());
    assert(a.vocab_size() == b.vocab_size());
    assert(a.serialize() == b.serialize());
}

void test_serialize_round_trip() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto a = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());
    const std::string bytes = a.serialize();
    auto b = PmiEmbeddings::from_bytes(bytes);
    assert(a.dim() == b.dim());
    assert(a.vocab_size() == b.vocab_size());
    assert(a.serialize() == b.serialize());

    // Row-level byte-identical check on a known in-vocab token.
    const float* ra = a.row("cat");
    const float* rb = b.row("cat");
    assert(ra != nullptr && rb != nullptr);
    for (std::uint32_t i = 0; i < a.dim(); ++i)
        assert(ra[i] == rb[i]);
}

void test_synonym_cosine_ordering() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());

    const float* cat = emb.row("cat");
    const float* feline = emb.row("feline");
    const float* dog = emb.row("dog");
    const float* canine = emb.row("canine");
    assert(cat && feline && dog && canine);

    const double cos_cat_feline = cosine(cat, feline, emb.dim());
    const double cos_cat_dog = cosine(cat, dog, emb.dim());
    const double cos_dog_canine = cosine(dog, canine, emb.dim());
    const double cos_dog_feline = cosine(dog, feline, emb.dim());

    // Each pair should cluster with its synonym.
    assert(cos_cat_feline > cos_cat_dog);
    assert(cos_dog_canine > cos_dog_feline);
}

void test_oov_row_is_nullptr() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());
    assert(emb.row("quokka") == nullptr);
    // Case-insensitive lookup: "Cat" should find the "cat" row.
    const float* r1 = emb.row("cat");
    const float* r2 = emb.row("Cat");
    assert(r1 != nullptr && r2 != nullptr);
    for (std::uint32_t i = 0; i < emb.dim(); ++i)
        assert(r1[i] == r2[i]);
}

void test_from_bytes_rejects_bad_magic() {
    std::string bogus(32, 'X');
    bool threw = false;
    try {
        (void)PmiEmbeddings::from_bytes(bogus);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_from_bytes_rejects_truncation() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());
    const std::string full = emb.serialize();
    std::string trunc = full.substr(0, full.size() / 2);
    bool threw = false;
    try {
        (void)PmiEmbeddings::from_bytes(trunc);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_encoder_pmi_smoke() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());

    EncoderConfig cfg;
    cfg.projection = ProjectionMode::None;
    cfg.sketch_dim = 1;         // unused on PMI path; validator ignores.
    cfg.output_dim = emb.dim(); // informational; PMI path overrides.
    cfg.ngram_mode = NGramMode::WordOnly;
    cfg.l2_normalize = true;
    cfg.pmi_rows = &emb;

    Encoder e(cfg);
    assert(e.output_dim() == emb.dim());
    assert(e.config().output_dim == emb.dim());

    std::vector<float> va(e.output_dim(), 0.0f);
    std::vector<float> vb(e.output_dim(), 0.0f);
    e.encode("the cat purrs softly", va.data());
    e.encode("the feline purrs softly", vb.data());

    // L2-normalized output.
    double mass = 0.0;
    for (float x : va)
        mass += static_cast<double>(x) * x;
    assert(std::fabs(mass - 1.0) < 1e-4);

    // cat/feline embeddings are close, so the doc-level cosine is high.
    const double cs = cosine(va.data(), vb.data(), e.output_dim());
    assert(cs > 0.8);
}

void test_encoder_pmi_implicit_output_dim_normalizes_config() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());

    EncoderConfig cfg;
    cfg.projection = ProjectionMode::None;
    cfg.sketch_dim = 1;
    cfg.output_dim = 0;
    cfg.ngram_mode = NGramMode::WordOnly;
    cfg.l2_normalize = true;
    cfg.pmi_rows = &emb;

    Encoder e(cfg);
    assert(e.output_dim() == emb.dim());
    assert(e.config().output_dim == emb.dim());
}

void test_encoder_pmi_all_oov_zero_vector() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());

    EncoderConfig cfg;
    cfg.projection = ProjectionMode::None;
    cfg.sketch_dim = 1;
    cfg.output_dim = emb.dim();
    cfg.ngram_mode = NGramMode::WordOnly;
    cfg.l2_normalize = true;
    cfg.pmi_rows = &emb;

    Encoder e(cfg);
    std::vector<float> v(e.output_dim(), 1.0f);
    e.encode("quokka wombat zebra", v.data()); // all OOV
    for (float x : v)
        assert(x == 0.0f);
}

void test_encoder_pmi_projection_conflict_throws() {
    auto docs = build_synonym_corpus();
    auto seed = view_of(docs);
    auto emb = PmiEmbeddings::learn(std::span<const std::string_view>(seed), default_pmi_cfg());

    EncoderConfig cfg;
    cfg.projection = ProjectionMode::AchlioptasSparse; // conflict
    cfg.sketch_dim = 256;
    cfg.output_dim = emb.dim();
    cfg.pmi_rows = &emb;

    bool threw = false;
    try {
        Encoder e(cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void test_encoder_without_pmi_still_works() {
    // Backward-compat: with pmi_rows == nullptr, Encoder behaves as before.
    EncoderConfig cfg;
    cfg.ngram_mode = NGramMode::CharOnly;
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.sketch_dim = 1024;
    cfg.output_dim = 64;
    cfg.projection = ProjectionMode::AchlioptasSparse;
    cfg.projection_seed = 42;

    Encoder e(cfg);
    std::vector<float> v(e.output_dim(), 0.0f);
    e.encode("hello world", v.data());
    double mass = 0.0;
    for (float x : v)
        mass += static_cast<double>(x) * x;
    assert(std::fabs(mass - 1.0) < 1e-4);
}

} // namespace

int main() {
    test_learn_empty_throws();
    test_learn_deterministic();
    test_serialize_round_trip();
    test_synonym_cosine_ordering();
    test_oov_row_is_nullptr();
    test_from_bytes_rejects_bad_magic();
    test_from_bytes_rejects_truncation();
    test_encoder_pmi_smoke();
    test_encoder_pmi_implicit_output_dim_normalizes_config();
    test_encoder_pmi_all_oov_zero_vector();
    test_encoder_pmi_projection_conflict_throws();
    test_encoder_without_pmi_still_works();
    return 0;
}
