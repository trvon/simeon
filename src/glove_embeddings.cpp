#include "simeon/glove_embeddings.hpp"

#ifdef SIMEON_ENABLE_RESEARCH

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace simeon {

PmiEmbeddings load_glove(const std::string& path, std::uint32_t max_vocab) {
    std::ifstream in(path);
    if (!in.is_open())
        throw std::runtime_error("load_glove: cannot open " + path);

    std::vector<std::string> vocab;
    std::vector<float> rows;
    std::uint32_t dim = 0;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty())
            continue;

        // Split off the word token (everything before the first space).
        const std::size_t sep = line.find(' ');
        if (sep == std::string::npos)
            continue;

        // Some distributions prepend a bare "vocab_size dim" header line.
        // Detect by checking that the first token is entirely digits.
        if (vocab.empty()) {
            bool all_digits = true;
            for (std::size_t i = 0; i < sep; ++i)
                if (line[i] < '0' || line[i] > '9') {
                    all_digits = false;
                    break;
                }
            if (all_digits)
                continue;
        }

        // Parse float values after the word.
        std::vector<float> vec;
        vec.reserve(dim > 0 ? dim : 300u);
        const char* p = line.data() + sep + 1;
        const char* end = line.data() + line.size();
        while (p < end) {
            char* next;
            const float v = std::strtof(p, &next);
            if (next == p)
                break;
            vec.push_back(v);
            p = next;
        }
        if (vec.empty())
            continue;

        // First valid word establishes the dimension; mismatched lines are skipped.
        if (dim == 0) {
            dim = static_cast<std::uint32_t>(vec.size());
        } else if (static_cast<std::uint32_t>(vec.size()) != dim) {
            continue;
        }

        // L2-normalize each row on load to match PmiEmbeddings::learn() output.
        float norm_sq = 0.0f;
        for (float v : vec)
            norm_sq += v * v;
        if (norm_sq > 0.0f) {
            const float inv = 1.0f / std::sqrt(norm_sq);
            for (float& v : vec)
                v *= inv;
        }

        vocab.emplace_back(line.data(), sep);
        rows.insert(rows.end(), vec.begin(), vec.end());

        if (max_vocab > 0 && static_cast<std::uint32_t>(vocab.size()) >= max_vocab)
            break;
    }

    if (vocab.empty())
        throw std::runtime_error("load_glove: no vectors parsed from " + path);

    return PmiEmbeddings::from_external(dim, std::move(vocab), std::move(rows));
}

} // namespace simeon

#endif // SIMEON_ENABLE_RESEARCH
