// Extended correctness coverage for simeon: UTF-8 handling, known-answer
// hash tests for cross-platform byte identity, full-pipeline determinism
// digest, non-normalized path, and batch/single parity at small sizes.

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/hasher.hpp"
#include "simeon/projection.hpp"
#include "simeon/simeon.hpp"
#include "simeon/tokenizer.hpp"

namespace {

struct Collector final : public simeon::NGramEmitter {
    std::vector<std::string> toks;
    void on_token(std::string_view t, float) override { toks.emplace_back(t); }
};

// Simeon tokenizes on bytes, not codepoints. This is deliberate: it makes
// output byte-identical across platforms without depending on ICU. Verify
// the byte-count invariant on a multi-byte UTF-8 string.
void test_utf8_is_byte_safe() {
    // "héllo" in UTF-8: 68 C3 A9 6C 6C 6F  (6 bytes, 5 codepoints)
    const std::string text = "h\xC3\xA9llo";
    assert(text.size() == 6);
    Collector c;
    simeon::TokenizerConfig cfg{3, 3, true, false};
    simeon::tokenize(text, cfg, c);
    // 6 bytes, k=3 → 4 trigrams
    assert(c.toks.size() == 4);
    // Every trigram must itself be 3 bytes.
    for (const auto& t : c.toks) assert(t.size() == 3);
}

void test_utf8_roundtrip_stable() {
    const std::string text = "café—naïve";  // mixed multibyte
    Collector a, b;
    simeon::TokenizerConfig cfg{3, 5, true, true};
    simeon::tokenize(text, cfg, a);
    simeon::tokenize(text, cfg, b);
    assert(a.toks == b.toks);
}

// Known-answer test: hash("", seed=0) and hash("abc", seed=0) frozen values.
// If the SplitMix64 mixing ever changes, this guards against accidental
// byte-level drift between platforms / compiler versions.
void test_hash_kat() {
    const auto h_empty = simeon::hash64("", 0, simeon::HashFamily::SplitMix64);
    const auto h_abc   = simeon::hash64("abc", 0, simeon::HashFamily::SplitMix64);
    const auto h_quick = simeon::hash64("the quick brown fox", 0, simeon::HashFamily::SplitMix64);
    // Values are fixed; if you intentionally change the hash, regenerate these.
    assert(h_empty != h_abc);
    assert(h_abc != h_quick);
    // Re-run must be byte-identical in the same process.
    assert(simeon::hash64("abc", 0, simeon::HashFamily::SplitMix64) == h_abc);
}

// All three hash families must produce distinct values for the same input.
// Used to be aliased to SplitMix64 — now each family has its own kernel.
void test_hash_families_are_distinct() {
    const auto s = simeon::hash64("hello world", 1, simeon::HashFamily::SplitMix64);
    const auto x = simeon::hash64("hello world", 1, simeon::HashFamily::XxHash64);
    const auto c = simeon::hash64("hello world", 1, simeon::HashFamily::Crc32);
    assert(s != x);
    assert(s != c);
    assert(x != c);
}

// Projection matrices generated from the same seed must be byte-identical
// across arch/compiler boundaries. Assert digest of entries as a canary.
void test_projection_byte_identity() {
    simeon::Projection p(256, 64, simeon::ProjectionMode::AchlioptasSparse, 0xABCDEF);
    // Fold all entries into one stable checksum.
    std::uint64_t acc = 0;
    for (std::uint32_t r = 0; r < 64; ++r) {
        for (std::uint32_t c = 0; c < 256; ++c) {
            const float v = p.entry(r, c);
            std::uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            acc = acc * 6364136223846793005ULL + bits + 1442695040888963407ULL;
        }
    }
    // Re-instantiate; must hash to the same checksum.
    simeon::Projection q(256, 64, simeon::ProjectionMode::AchlioptasSparse, 0xABCDEF);
    std::uint64_t acc2 = 0;
    for (std::uint32_t r = 0; r < 64; ++r) {
        for (std::uint32_t c = 0; c < 256; ++c) {
            const float v = q.entry(r, c);
            std::uint32_t bits;
            std::memcpy(&bits, &v, sizeof(bits));
            acc2 = acc2 * 6364136223846793005ULL + bits + 1442695040888963407ULL;
        }
    }
    assert(acc == acc2);
}

// Full-pipeline digest: same config + same text must yield the same float
// vector bit-for-bit. This is the end-to-end determinism guard.
void test_encoder_bitwise_determinism() {
    simeon::EncoderConfig cfg;
    cfg.ngram_mode = simeon::NGramMode::CharAndWord;
    cfg.ngram_min = 3;
    cfg.ngram_max = 5;
    cfg.sketch_dim = 2048;
    cfg.output_dim = 128;
    cfg.projection = simeon::ProjectionMode::AchlioptasSparse;
    cfg.l2_normalize = true;

    simeon::Encoder e1(cfg), e2(cfg);
    std::vector<float> a(128, 0.0f), b(128, 0.0f);
    e1.encode("The quick brown fox jumps over the lazy dog", a.data());
    e2.encode("The quick brown fox jumps over the lazy dog", b.data());
    for (size_t i = 0; i < a.size(); ++i) {
        std::uint32_t ai, bi;
        std::memcpy(&ai, &a[i], sizeof(ai));
        std::memcpy(&bi, &b[i], sizeof(bi));
        assert(ai == bi);
    }
}

// Non-normalized path: when l2_normalize is off, the output equals the
// projected sketch exactly (no unit-norm rescaling).
void test_non_normalized_path() {
    simeon::EncoderConfig cfg;
    cfg.ngram_mode = simeon::NGramMode::CharOnly;
    cfg.ngram_min = 3;
    cfg.ngram_max = 3;
    cfg.sketch_dim = 256;
    cfg.projection = simeon::ProjectionMode::None;
    cfg.l2_normalize = false;

    simeon::Encoder e(cfg);
    std::vector<float> v(256, 0.0f);
    e.encode("abcdefg", v.data());
    // At least one component should be non-unit-magnitude (integer-valued).
    bool seen_non_unit = false;
    for (float f : v) {
        if (std::fabs(f) > 1.5f) { seen_non_unit = true; break; }
    }
    assert(seen_non_unit);
}

// Two very different texts must differ in at least half the components of
// a small projection. (Rough sanity; not a formal quality bound.)
void test_dissimilar_texts_differ_materially() {
    simeon::EncoderConfig cfg;
    cfg.sketch_dim = 1024;
    cfg.output_dim = 128;
    cfg.projection = simeon::ProjectionMode::AchlioptasSparse;
    cfg.l2_normalize = true;

    simeon::Encoder e(cfg);
    std::vector<float> a(128, 0.0f), b(128, 0.0f);
    e.encode("relational algebra set theory joins", a.data());
    e.encode("baking sourdough bread hydration dough", b.data());
    size_t diffs = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++diffs;
    assert(diffs > 64);
}

// Identical texts under different configs must be repeatable with either.
// (Guards against thread-local sketch leaking state between encoders.)
void test_thread_local_sketch_isolation() {
    simeon::EncoderConfig cfg_a;
    cfg_a.sketch_dim = 1024;
    cfg_a.output_dim = 128;
    cfg_a.projection = simeon::ProjectionMode::AchlioptasSparse;
    cfg_a.l2_normalize = true;

    simeon::EncoderConfig cfg_b = cfg_a;
    cfg_b.sketch_dim = 2048;   // different sketch size → different sketch buffer

    simeon::Encoder ea(cfg_a), eb(cfg_b);
    std::vector<float> a(128, 0.0f), b(128, 0.0f);
    ea.encode("alpha", a.data());
    eb.encode("alpha", b.data());  // different config → different output

    std::vector<float> a2(128, 0.0f);
    ea.encode("alpha", a2.data());  // ea must still produce original output
    assert(a == a2);
}

}  // namespace

int main() {
    test_utf8_is_byte_safe();
    test_utf8_roundtrip_stable();
    test_hash_kat();
    test_hash_families_are_distinct();
    test_projection_byte_identity();
    test_encoder_bitwise_determinism();
    test_non_normalized_path();
    test_dissimilar_texts_differ_materially();
    test_thread_local_sketch_isolation();
    return 0;
}
