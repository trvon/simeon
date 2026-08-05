#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include "simeon/concept_mining.hpp"

namespace simeon::detail {

// Build-local identity helper for regression tests and benchmarks. This header
// is deliberately excluded from the installed public headers.
struct ConceptIndexFingerprint {
    static std::uint64_t compute(const ConceptIndex& index) {
        constexpr std::uint64_t offset = 1469598103934665603ULL;
        constexpr std::uint64_t prime = 1099511628211ULL;
        std::uint64_t digest = offset;
        const auto append = [&](std::uint64_t value) {
            for (unsigned shift = 0; shift < 64; shift += 8) {
                digest ^= (value >> shift) & 0xffu;
                digest *= prime;
            }
        };

        append(index.doc_count_);
        append(std::bit_cast<std::uint32_t>(index.avg_bigram_dl_));
        append(std::bit_cast<std::uint32_t>(index.k1_));
        append(std::bit_cast<std::uint32_t>(index.b_));
        append(static_cast<std::uint64_t>(index.hash_family_));
        append(index.hash_seed_);
        append(index.bigram_doc_lengths_.size());
        for (const auto length : index.bigram_doc_lengths_)
            append(length);

        std::vector<std::uint64_t> hashes;
        hashes.reserve(index.concepts_.size());
        for (const auto& [hash, entry] : index.concepts_) {
            (void)entry;
            hashes.push_back(hash);
        }
        std::sort(hashes.begin(), hashes.end());
        append(hashes.size());
        for (const auto hash : hashes) {
            const auto& entry = index.concepts_.at(hash);
            append(hash);
            append(std::bit_cast<std::uint32_t>(entry.pmi));
            append(std::bit_cast<std::uint32_t>(entry.idf));
            append(entry.total_tf);
            append(entry.a_hash);
            append(entry.b_hash);
            append(entry.docs.size());
            for (const auto& [doc, frequency] : entry.docs) {
                append(doc);
                append(frequency);
            }
        }
        return digest;
    }
};

} // namespace simeon::detail
