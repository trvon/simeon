// libFuzzer harness for simeon::tokenize. Best run under ASAN+UBSAN.
//
//   meson setup buildfuzz -Denable_fuzz=true -Db_sanitize=address,undefined \
//     -Db_lundef=false --buildtype=debug
//   meson compile -C buildfuzz
//   ./buildfuzz/tests/fuzz/fuzz_tokenize -max_total_time=60

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "simeon/tokenizer.hpp"

namespace {

struct NullSink final : public simeon::NGramEmitter {
    std::size_t count = 0;
    void on_token(std::string_view, float) override { ++count; }
};

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size == 0) return 0;
    // Use the first byte as a knob over the (k_min, k_max, mode) space so the
    // fuzzer explores all tokenize() branches, not just the default config.
    const std::uint8_t knob = data[0];
    const std::uint32_t k_min = 1u + (knob & 0x07u);          // 1..8
    const std::uint32_t k_max = k_min + ((knob >> 3) & 0x07u);  // k_min..k_min+7
    const bool emit_char = (knob & 0x40u) != 0;
    const bool emit_word = (knob & 0x80u) != 0 || !emit_char;

    simeon::TokenizerConfig cfg{
        .ngram_min = k_min,
        .ngram_max = k_max,
        .emit_char = emit_char,
        .emit_word = emit_word,
    };
    NullSink sink;
    const std::string_view text(reinterpret_cast<const char*>(data + 1), size - 1);
    simeon::tokenize(text, cfg, sink);
    return 0;
}
