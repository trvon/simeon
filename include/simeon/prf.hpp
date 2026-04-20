#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "simeon/bm25.hpp"

namespace simeon {

// RM3 pseudo-relevance feedback (Lavrenko & Croft 2001, "Relevance-Based
// Language Models", SIGIR). Runs a first-pass BM25 retrieval, treats the
// top-K docs as pseudo-relevant, estimates an RM1 relevance model
// p(w | R) = Σ_{d ∈ top_k} (tf(w, d) / |d|) * p(d | Q), keeps the top-N
// highest-weighted terms, and re-scores every doc under the blended query
// θ' = (1 - α) * θ_Q + α * θ_R.
//
// Training-free: all inputs are corpus statistics; no labeled data, no
// gradient descent. Deterministic given (idx, query, config).
//
// For SubwordAwareBackoff indexes, the first-pass retrieval uses the full
// SAB path (exact + char-n-gram backoff on OOV terms); rescoring the
// expanded query uses the exact word-posting path only — see
// Bm25Index::score_weighted_hashes. Non-SAB variants use the same scoring
// dispatch for both passes.
struct PrfConfig {
    // Number of feedback docs treated as pseudo-relevant. Canonical RM3
    // setting is 10 (Lavrenko & Croft 2001, consistent with TREC RM3 runs).
    std::uint32_t k = 10;
    // Number of expansion terms kept from the relevance model after ranking
    // by p(w | R). TREC-era default is 20; lower settings reduce topic drift
    // on short queries.
    std::uint32_t n_terms = 20;
    // Mixing weight between original query and expansion: θ' = (1-α)θ_Q + αθ_R.
    // α=0 recovers BM25; α=1 retrieves on the expansion alone. Canonical
    // setting is 0.5.
    float alpha = 0.5f;
};

// Score all docs under RM3 expansion of `query`. Out-scores size must equal
// idx.doc_count(). Throws if idx is not finalized.
void score_with_prf(const Bm25Index& idx, std::string_view query,
                    std::span<float> out_scores, const PrfConfig& cfg = {});

} // namespace simeon
