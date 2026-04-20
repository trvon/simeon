// libFuzzer harness for simeon::Encoder::encode. Best run under ASAN+UBSAN.
//
//   meson setup buildfuzz -Denable_fuzz=true -Db_sanitize=address,undefined \
//     -Db_lundef=false --buildtype=debug
//   meson compile -C buildfuzz
//   ./buildfuzz/tests/fuzz/fuzz_encode -max_total_time=60

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "simeon/simeon.hpp"

namespace {

simeon::Encoder& shared_encoder() {
    static simeon::Encoder enc([] {
        simeon::EncoderConfig cfg;
        cfg.ngram_mode = simeon::NGramMode::CharAndWord;
        cfg.ngram_min = 3;
        cfg.ngram_max = 5;
        cfg.sketch_dim = 1024;
        cfg.output_dim = 128;
        cfg.projection = simeon::ProjectionMode::AchlioptasSparse;
        cfg.l2_normalize = true;
        return cfg;
    }());
    return enc;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    auto& enc = shared_encoder();
    static thread_local std::vector<float> out(enc.output_dim());
    const std::string_view text(reinterpret_cast<const char*>(data), size);
    enc.encode(text, out.data());
    return 0;
}
