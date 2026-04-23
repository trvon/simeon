#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "simeon/flat_hash_map_u64.hpp"
#include "simeon/simeon.hpp"

namespace simeon {

// BM25 scoring variant. Atire is the conventional Robertson-style BM25 with
// IDF = log((N-df+0.5)/(df+0.5)+1); the others are well-studied refinements.
enum class Bm25Variant : std::uint8_t {
    // Robertson BM25 (Atire-style IDF). Default; matches pre-variant behavior.
    Atire,
    // Lv & Zhai 2011 (CIKM, "Lower-Bounding Term Frequency Normalization").
    // Adds a δ floor to per-term contribution to fix long-doc lower-bound
    // violation: idf*(tf*(k1+1)/(tf + k1*(1-b+b*dl/avg)) + δ).
    BM25Plus,
    // Lv & Zhai 2011 (SIGIR, "When Documents Are Very Long, BM25 Fails!").
    // Replaces normalized-tf with c' = tf / (1 - b + b*dl/avg), then
    // (k1+1)*(c'+δ)/(k1+c'+δ). Different functional form than BM25+.
    BM25L,
    // Amati DFR DLH13 (Terrier reference). Parameter-free divergence-from-
    // randomness scorer; uses corpus-wide term frequency, no k1/b/δ.
    DLH13,
    // Amati & van Rijsbergen 2002 DFR PL2 (Poisson with Laplace after-effect
    // and L2 length normalization, Terrier reference). Score uses
    // tfn = tf * log2(1 + c * avg_dl / dl), then the Stirling-Poisson form
    // (1/(tfn+1)) * (tfn*log2(tfn/λ) + (λ-tfn)*log2(e) + 0.5*log2(2π·tfn))
    // with λ = ttf / N. Free parameter c defaults to 1.0 per Terrier.
    PL2,
    // Amati 2007 DFR DPH (hypergeometric, Terrier reference). Parameter-free:
    // norm = (1 - tf/dl)² / (tf + 1) gates the Stirling log-argument
    // tf * avg_dl / dl * N / ttf. No free parameter.
    DPH,
    // Dirichlet-smoothed LM / DCM (Madsen-Kauchak-Elkan 2005, ICML, "Modeling
    // Word Burstiness"; related to Zhai-Lafferty 2001 Dirichlet smoothing).
    // Per-term contribution log(1 + tf * total_tokens / (α_sum * ttf)), where
    // α_sum defaults to avg_dl. Captures burstiness via the Polya-urn-style
    // log-tf growth without needing per-term α_t tuning.
    Dcm,
    // Subword-aware BM25+ backoff (this codebase, training-free): for each
    // query term t, blend exact BM25+ with a score computed from the
    // character n-grams of t against a parallel n-gram inverted index of
    // the corpus. Per-term mix is α = df(t)/(df(t)+γ) when df(t)>0, 0 when
    // df(t)=0; γ=0 recovers the strict OOV-fallback variant. Roughly
    // doubles index size and finalize time.
    SubwordAwareBackoff,
    // Atire BM25 with Fang-Zhai 2005 axiomatic Length Term-Discrimination
    // (LTD) correction. Replaces BM25's linear length normalization
    //     L(dl) = 1 - b + b · (dl/avg_dl)
    // with a sublinear power form
    //     L(dl) = 1 - b + b · (dl/avg_dl)^α
    // where α ∈ (0, 1] is `Bm25Config::ltd_alpha`. α=1 recovers Atire
    // byte-identically; α<1 reduces the long-doc penalty (BM25's b
    // overpenalizes long docs that legitimately satisfy LTD). Default
    // α=0.5 per Fang-Zhai's recommended midpoint.
    AtireLTD,
};

struct Bm25Config {
    float k1 = 1.2f;
    float b = 0.75f;
    HashFamily hash = HashFamily::SplitMix64;
    std::uint64_t hash_seed = 0xB252B252B252B252ULL;

    Bm25Variant variant = Bm25Variant::Atire;
    // δ floor for BM25Plus / BM25L. Lv & Zhai recommend 1.0 as default.
    float delta = 1.0f;
    // Smoothing factor for SubwordAwareBackoff: per-term mix between exact
    // and n-gram score is α = df / (df + subword_gamma). Set to 0 for the
    // strict "n-grams only when df=0" backoff.
    float subword_gamma = 0.0f;
    // Char n-gram range used by SubwordAwareBackoff for the secondary index.
    // Ignored by other variants.
    std::uint32_t ngram_min = 3;
    std::uint32_t ngram_max = 5;
    // Dirichlet concentration for Dcm variant. 0 means "derive from corpus"
    // (uses avg_dl at finalize() time, matching Zhai-Lafferty μ=avg_dl).
    // Ignored by other variants.
    float dcm_alpha_sum = 0.0f;
    // Free parameter c for PL2. Terrier default is 1.0. Ignored by other
    // variants.
    float pl2_c = 1.0f;
    // Length-norm exponent α ∈ (0, 1] for AtireLTD. α=1 recovers Atire
    // byte-identically; α=0.5 is Fang-Zhai 2005's recommended midpoint
    // for LTD compliance on long-doc corpora. Ignored by other variants.
    float ltd_alpha = 0.5f;
    // Step 1l: opt-in parallel word-bigram postings for SDM (Metzler & Croft
    // 2005). When true, finalize() builds ordered and unordered bigram
    // indexes so score_sdm() can run. Cost is ~30% extra index size and
    // ~30% longer finalize() (Metzler's published figure). Default false
    // preserves byte-identical behavior for callers that never use SDM.
    bool build_word_bigrams = false;
    // Window size for the unordered-bigram index: pair (q_i, q_j) is counted
    // when both appear within this many word positions of each other in the
    // doc. Metzler's default is 8. Fixed at index-build time; score_sdm()
    // reads it from the index, not from SdmConfig.
    std::uint32_t bigram_unordered_window = 8u;
};

// Sequential Dependence Model (Metzler & Croft 2005, SIGIR). Adds ordered
// and windowed-unordered bigram scoring on top of BM25. Full score is
//   score = λ_u · Σ BM25(unigram) + λ_o · Σ BM25(ordered bigram)
//                                 + λ_uw · Σ BM25(unordered bigram)
// with Metzler's published defaults (0.85, 0.10, 0.05) that work without
// per-corpus tuning. Training-free: all signals are corpus statistics.
struct SdmConfig {
    float lambda_unigram = 0.85f;
    float lambda_ordered = 0.10f;
    float lambda_unordered = 0.05f;
};

// Weighted SDM (Bendersky & Croft 2010, "Learning Concept Importance Using
// a Weighted Dependence Model"). Replaces fixed λ_o / λ_uw with per-bigram
// weights derived from the bigram's discriminative power, scaled so the
// query-mean equals the corresponding fixed λ — the recipe is byte-
// identical to fixed SDM on a single-bigram query but rewards rare /
// discriminative bigrams over common ones on multi-term queries.
//
// Per Bendersky-Croft §4: full WSDM combines linear-regressed
// per-feature weights over {tf_collection, df, length-norm, plus external
// resources}; the training-free reduction kept here uses bigram IDF as the
// single feature, which §6.4 reports captures most of the gain on TREC
// corpora when the regression weights are unavailable.
//
// Per-bigram weight (BigramIdfNorm strategy):
//   λ_b = λ_base * (idf_b / mean_query_bigram_idf)^β
// where β controls how aggressively rare bigrams are upweighted (β=0
// recovers fixed SDM; β=1 is canonical Bendersky-Croft IDF reweighting).
struct WeightedSdmConfig {
    // Same defaults as fixed SDM so WSDM(β=0) = SDM(0.85, 0.10, 0.05).
    float lambda_unigram = 0.85f;
    float lambda_ordered = 0.10f;
    float lambda_unordered = 0.05f;
    // Bendersky-Croft canonical IDF-normalization exponent. β=0 disables
    // weighting (recovers fixed-λ SDM); β=1 is the published recipe.
    float beta = 1.0f;
};

// Streaming BM25 index over word tokens. Tokenization reuses the simeon
// tokenizer with `emit_word=true, emit_char=false` so term boundaries match
// other simeon components when the same text is processed by both.
//
// Add docs in id order via `add_doc`, then call `finalize()` once before
// scoring. `score()` writes one BM25 score per doc into out_scores.
class Bm25Index {
public:
    explicit Bm25Index(Bm25Config cfg = {}) noexcept;

    void add_doc(std::string_view text);
    // Optional second field for BM25F-style scoring. This is an opt-in utility:
    // the default router and shipped production rows remain body-only unless the
    // caller explicitly uses score_bm25f().
    void add_doc(std::string_view text, std::string_view aux_text);
    void finalize();
    void score(std::string_view query, std::span<float> out_scores) const;
    // Opt-in multi-field scoring path. `weight_aux = 0` recovers body-only
    // BM25. Useful for explicit auxiliary fields, but not part of the default
    // routed retrieval path.
    void score_bm25f(std::string_view query, std::span<float> out_scores, float weight_body = 1.0f,
                     float weight_aux = 0.0f) const;
    void score_bm25f(std::string_view body_query, std::string_view aux_query,
                     std::span<float> out_scores, float weight_body = 1.0f,
                     float weight_aux = 0.0f) const;

    std::uint32_t doc_count() const noexcept {
        return static_cast<std::uint32_t>(doc_lengths_.size());
    }
    const Bm25Config& config() const noexcept { return cfg_; }

    // Per-term lookup for pre-retrieval predictors (QueryRouter). Tokenizes
    // `term` exactly like score() does (word-only) and returns the document
    // frequency / IDF; both are 0 when the term is OOV. Cheap: one tokenize
    // + one hash + one map lookup. Requires finalize() to have been called.
    std::uint32_t df(std::string_view term) const noexcept;
    float idf(std::string_view term) const noexcept;
    // Corpus-wide term frequency (Σ_d tf(term, d)). 0 for OOV terms. Used by
    // Step 1k Pass E predictors (SCQ, simplified clarity) which need
    // collection statistics beyond df/idf.
    std::uint64_t total_tf(std::string_view term) const noexcept;
    // Direct hash lookup for total_tf. Returns 0 when hash is unknown.
    // Used by concept mining (Step 1n) which operates on raw word hashes
    // to compute PMI without re-materializing strings. Hash must be
    // produced by this index's hash_term() or the result is meaningless.
    std::uint64_t total_tf_by_hash(std::uint64_t term_hash) const noexcept;
    // Total token count across all docs = Σ_d doc_length(d). Cheap; cached
    // at finalize(). Used by simplified clarity to normalize collection LM.
    std::uint64_t total_tokens() const noexcept { return total_tokens_; }

    // Hash `term` using the same scheme score() / df() / idf() use. Returns
    // the raw uint64 hash (not the IDF). Cheap: one tokenize of the single
    // term plus one hash call; exposes no corpus state. Used by PRF to build
    // weighted-term queries without re-tokenizing through the full pipeline.
    std::uint64_t hash_term(std::string_view term) const noexcept;

    // Per-doc length in word tokens (the same dl used in BM25 scoring).
    // Returns 0 for out-of-range doc_id.
    std::uint32_t doc_length(std::uint32_t doc_id) const noexcept {
        return doc_id < doc_lengths_.size() ? doc_lengths_[doc_id] : 0u;
    }

    // Build an RM1-style relevance model from the top-K feedback docs.
    // For each term w present in any feedback doc, writes
    //   weight(w) = Σ_{d ∈ top_k_docs} (tf(w, d) / dl(d)) * doc_weights[d_idx]
    // into `out_terms` as (term_hash, weight). Caller normalizes / trims.
    // top_k_docs and doc_weights must have the same size; doc_weights are
    // typically normalized first-pass BM25 scores.
    void build_relevance_model(std::span<const std::uint32_t> top_k_docs,
                               std::span<const float> doc_weights,
                               std::vector<std::pair<std::uint64_t, float>>& out_terms) const;

    // Score each doc by Σ_w weight(w) * per-term-contribution(w, d), using
    // the same Bm25Variant-dispatched scoring inner loop score() uses.
    // `weighted_terms` is (term_hash, weight) — hashes must be produced via
    // this index's hash_term() or df()/idf() will mis-match. Unlike score(),
    // this path never invokes the SubwordAwareBackoff n-gram fallback (no
    // per-term strings available); for SAB indexes the expansion is scored
    // on the exact word-posting path only.
    void score_weighted_hashes(std::span<const std::pair<std::uint64_t, float>> weighted_terms,
                               std::span<float> out_scores) const;

    // SDM (Metzler & Croft 2005): three-leg blend of unigram BM25 (via the
    // configured Bm25Variant), ordered-bigram BM25, and windowed-unordered-
    // bigram BM25. Requires cfg_.build_word_bigrams == true at construction;
    // if the index was built without bigrams, the bigram legs contribute
    // zero and score_sdm() degenerates to λ_u · score(). Throws if the
    // index is not finalized. Bigram legs always use Atire-style BM25
    // (k1/b from cfg_); the unigram leg uses the configured variant.
    //
    // This is an opt-in enrichment path. The benchmark fixtures showed it can
    // help some corpora modestly, but it is not part of the default routed
    // retrieval recipe.
    void score_sdm(std::string_view query, std::span<float> out_scores,
                   const SdmConfig& cfg = {}) const;

    // Weighted SDM (Bendersky & Croft 2010): same three-leg structure as
    // score_sdm() but per-bigram λ scales with the bigram's IDF, normalized
    // so the query-mean weight equals the fixed-λ baseline. Reduces to
    // score_sdm() when cfg.beta == 0. Same finalize() / build_word_bigrams
    // requirements as score_sdm(). Experimental / opt-in: the current
    // benchmark evidence did not justify routing this by default.
    void score_wsdm(std::string_view query, std::span<float> out_scores,
                    const WeightedSdmConfig& cfg = {}) const;

    // Combine two 64-bit term hashes into a stable 64-bit bigram hash using
    // splitmix64_mix. Ordered: pass (a, b) as-is. Unordered: canonicalize
    // via (min(a,b), max(a,b)) so (a,b) and (b,a) collide. Public so tests
    // and composing callers can verify determinism.
    std::uint64_t hash_bigram(std::uint64_t a, std::uint64_t b) const noexcept;

private:
    // After finalize(), each posting list carries the term's IDF inline, so
    // score() can iterate one map lookup per query term instead of two
    // (postings + idf). total_tf is the corpus-wide term frequency used by
    // DLH13. The (doc_id, tf) layout is unchanged.
    struct TermPostings {
        float idf = 0.0f;
        std::uint64_t total_tf = 0;
        std::vector<std::pair<std::uint32_t, std::uint32_t>> docs;
    };

    Bm25Config cfg_;
    std::vector<std::uint32_t> doc_lengths_;
    FlatHashMapU64<TermPostings> postings_;
    float avg_dl_ = 0.0f;
    // Cached at finalize(): Σ doc_lengths_ (used by Dcm, computed once).
    std::uint64_t total_tokens_ = 0;
    // Cached at finalize() when variant == Dcm: α_sum override or avg_dl.
    float alpha_sum_ = 0.0f;
    bool finalized_ = false;

    // Optional auxiliary text field for BM25F-style linear field fusion.
    // Indexed in parallel with the body so doc ids stay aligned; callers that
    // only use add_doc(text) get an all-empty aux field.
    std::vector<std::uint32_t> aux_doc_lengths_;
    FlatHashMapU64<TermPostings> aux_postings_;
    float aux_avg_dl_ = 0.0f;
    std::uint64_t aux_total_tokens_ = 0;
    float aux_alpha_sum_ = 0.0f;

    // SubwordAwareBackoff secondary index: char n-gram postings over the
    // same docs. Only populated when cfg_.variant == SubwordAwareBackoff.
    std::vector<std::uint32_t> ngram_doc_lengths_;
    FlatHashMapU64<TermPostings> ngram_postings_;
    float ngram_avg_dl_ = 0.0f;
    std::vector<std::uint32_t> aux_ngram_doc_lengths_;
    FlatHashMapU64<TermPostings> aux_ngram_postings_;
    float aux_ngram_avg_dl_ = 0.0f;

    // Step 1l word-bigram secondary indexes. Only populated when
    // cfg_.build_word_bigrams == true. Scored by Atire BM25 using the word
    // doc_lengths_ (same avg_dl_) for length normalization; bigram tf values
    // are intrinsically smaller than unigram tf so the BM25 saturation
    // behaves as intended.
    FlatHashMapU64<TermPostings> ordered_bigram_postings_;
    FlatHashMapU64<TermPostings> unordered_bigram_postings_;
};

} // namespace simeon
