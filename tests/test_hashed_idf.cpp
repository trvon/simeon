#include <cassert>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/hashed_idf.hpp"
#include "simeon/hasher.hpp"
#include "simeon/simeon.hpp"

namespace {

void write_u32(std::string& bytes, std::size_t offset, std::uint32_t value) {
    assert(offset + 4 <= bytes.size());
    for (unsigned shift = 0; shift < 32; shift += 8)
        bytes[offset + shift / 8] = static_cast<char>((value >> shift) & 0xFF);
}

simeon::EncoderConfig word_config() {
    auto config = simeon::compact_retrieval_config();
    config.ngram_mode = simeon::NGramMode::WordOnly;
    config.ngram_min = 1;
    config.ngram_max = 1;
    config.sketch_dim = 65'536;
    config.output_dim = 0;
    config.projection = simeon::ProjectionMode::None;
    config.l2_normalize = false;
    return config;
}

void test_document_frequency_weights_are_deterministic_and_discriminative() {
    const std::vector<std::string_view> corpus = {"common rare", "common other"};
    const auto config = word_config();
    const auto first = simeon::HashedIdf::learn(corpus, config, {.hash_dim = 65'536});
    const auto second = simeon::HashedIdf::learn(corpus, config, {.hash_dim = 65'536});

    const auto common = simeon::hash64("common", config.hash_seed, config.hash);
    const auto rare = simeon::hash64("rare", config.hash_seed, config.hash);
    assert(common % first.hash_dim() != rare % first.hash_dim());
    assert(first.weight_q10(common) < first.weight_q10(rare));
    assert(first.document_count() == corpus.size());
    assert(first.fingerprint() == second.fingerprint());
    assert(first.serialize() == second.serialize());
}

void test_serialization_round_trip_and_encoder_integration() {
    const std::vector<std::string_view> corpus = {"common rare", "common other"};
    auto base = word_config();
    const auto learned = simeon::HashedIdf::learn(corpus, base, {.hash_dim = 65'536});
    const auto restored = simeon::HashedIdf::from_bytes(learned.serialize());
    assert(restored.fingerprint() == learned.fingerprint());
    assert(restored.storage_bytes() == learned.storage_bytes());

    std::string truncated = learned.serialize();
    truncated.pop_back();
    try {
        (void)simeon::HashedIdf::from_bytes(truncated);
        assert(false && "truncated artifact must fail");
    } catch (const std::invalid_argument&) {
    }
    std::string bad_magic = learned.serialize();
    bad_magic[0] = 'X';
    try {
        (void)simeon::HashedIdf::from_bytes(bad_magic);
        assert(false && "bad artifact magic must fail");
    } catch (const std::invalid_argument&) {
    }

    base.hashed_idf = &restored;
    simeon::Encoder encoder(base);
    std::vector<float> output(encoder.output_dim(), 0.0f);
    encoder.encode("common rare", output.data());

    for (std::string_view token : {std::string_view("common"), std::string_view("rare")}) {
        const auto hash = simeon::hash64(token, base.hash_seed, base.hash);
        const auto bucket = static_cast<std::uint32_t>(hash) % base.sketch_dim;
        const float sign = ((hash >> 63) & 1ULL) ? -1.0f : 1.0f;
        assert(output[bucket] == sign * static_cast<float>(restored.weight_q10(hash)));
    }
}

void test_artifact_rejects_incompatible_encoder_identity() {
    const std::vector<std::string_view> corpus = {"alpha beta", "beta gamma"};
    auto config = word_config();
    const auto idf = simeon::HashedIdf::learn(corpus, config, {.hash_dim = 1024});
    config.hashed_idf = &idf;
    ++config.hash_seed;
    try {
        simeon::Encoder encoder(config);
        assert(false && "mismatched IDF artifact must be rejected");
    } catch (const std::invalid_argument&) {
    }
}

void test_serialization_rejects_wide_invalid_enum_values() {
    const std::vector<std::string_view> corpus = {"alpha beta", "beta gamma"};
    const auto config = word_config();
    const auto idf = simeon::HashedIdf::learn(corpus, config, {.hash_dim = 1024});
    for (const std::size_t offset :
         {std::size_t{32}, std::size_t{36}, std::size_t{40}, std::size_t{44}, std::size_t{48}}) {
        std::string malformed = idf.serialize();
        write_u32(malformed, offset, 256);
        try {
            (void)simeon::HashedIdf::from_bytes(malformed);
            assert(false && "wide invalid artifact enum must fail");
        } catch (const std::invalid_argument&) {
        }
    }
}

void test_word_only_scope_leaves_character_features_at_unit_weight() {
    const std::vector<std::string_view> corpus = {"common rare", "common other"};
    auto config = simeon::compact_retrieval_config();
    config.output_dim = 0;
    config.projection = simeon::ProjectionMode::None;
    const auto idf = simeon::HashedIdf::learn(
        corpus, config, {.hash_dim = 65'536, .scope = simeon::HashedIdfScope::Word});
    const auto common = simeon::hash64("common", config.hash_seed, config.hash);
    const auto trigram = simeon::hash64("com", config.hash_seed, config.hash);
    assert(idf.weight_q10(common, 0.5f) < 1024);
    assert(idf.weight_q10(trigram, 1.0f) == 1024);
    assert(idf.scope() == simeon::HashedIdfScope::Word);
}

void test_compact_corpus_adaptive_preset_attaches_compatible_artifact() {
    const std::vector<std::string_view> corpus = {"alpha beta", "beta gamma"};
    const auto base = simeon::compact_retrieval_config();
    const auto idf = simeon::HashedIdf::learn(corpus, base, {.hash_dim = 65'536});
    const auto config = simeon::compact_hashed_idf_retrieval_config(idf);
    assert(config.hashed_idf == &idf);
    assert(config.output_dim == 384);
    simeon::Encoder encoder(config);
    assert(encoder.output_dim() == 384);
}

void test_quality_corpus_adaptive_preset_retains_768_coordinates() {
    const std::vector<std::string_view> corpus = {"alpha beta", "beta gamma"};
    const auto base = simeon::compact_retrieval_config();
    const auto idf = simeon::HashedIdf::learn(corpus, base, {.hash_dim = 65'536});
    const auto config = simeon::quality_hashed_idf_retrieval_config(idf);
    assert(config.hashed_idf == &idf);
    assert(config.sketch_dim == 8192);
    assert(config.output_dim == 768);
    simeon::Encoder encoder(config);
    assert(encoder.output_dim() == 768);
}

void test_invalid_build_inputs_fail() {
    const std::vector<std::string_view> empty;
    const auto config = word_config();
    try {
        (void)simeon::HashedIdf::learn(empty, config);
        assert(false && "empty corpus must fail");
    } catch (const std::invalid_argument&) {
    }
    try {
        const std::vector<std::string_view> corpus = {"alpha"};
        (void)simeon::HashedIdf::learn(corpus, config, {.hash_dim = 0});
        assert(false && "zero hash dimension must fail");
    } catch (const std::invalid_argument&) {
    }
    auto invalid_identity = config;
    invalid_identity.text_normalization = static_cast<simeon::TextNormalization>(255);
    try {
        const std::vector<std::string_view> corpus = {"alpha"};
        (void)simeon::HashedIdf::learn(corpus, invalid_identity);
        assert(false && "invalid encoder identity must fail");
    } catch (const std::invalid_argument&) {
    }
}

} // namespace

int main() {
    test_document_frequency_weights_are_deterministic_and_discriminative();
    test_serialization_round_trip_and_encoder_integration();
    test_artifact_rejects_incompatible_encoder_identity();
    test_serialization_rejects_wide_invalid_enum_values();
    test_word_only_scope_leaves_character_features_at_unit_weight();
    test_compact_corpus_adaptive_preset_attaches_compatible_artifact();
    test_quality_corpus_adaptive_preset_retains_768_coordinates();
    test_invalid_build_inputs_fail();
    return 0;
}
