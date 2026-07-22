#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "simeon/hashed_idf.hpp"
#include "simeon/simeon.hpp"

namespace simeon::experiment {

// The manifest deliberately knows nothing about individual experiment kinds.
// Each benchmark resolves the parameters for the kinds it owns, which keeps
// the manifest/result contract reusable across encoders, lexical legs, fusion,
// and future retrieval strategies.
struct VariantSpec {
    std::string name;
    std::string kind;
    std::map<std::string, std::string> parameters;
};

struct Manifest {
    std::uint32_t schema_version = 0;
    std::string name;
    std::map<std::string, std::string> metadata;
    std::vector<VariantSpec> variants;
};

Manifest parse_manifest(std::istream& input);
Manifest load_manifest(const std::string& path);

// Semantic fingerprints are stable across whitespace, comments, quoting, and
// parameter ordering. The hash is a provenance identifier, not a security
// primitive.
std::string manifest_fingerprint(const Manifest& manifest);

// When a manifest declares selection/holdout metadata, enforce it. An
// exploration manifest cannot read the holdout; a frozen holdout manifest must
// contain one selected variant and identify the exploration manifest it came
// from via selection_manifest.
void validate_run_policy(const Manifest& manifest, std::string_view split);

// Artifact-free manifests may contain only fixed encoders. Corpus-adaptive
// manifests may additionally contain hashed-IDF encoders and retrieval recipes
// whose lexical or semantic artifacts consume corpus statistics; evaluation-
// only state is never accepted by these training-free runners.
void validate_training_regime(const Manifest& manifest);

struct Qrel {
    std::uint32_t query = 0;
    std::uint32_t document = 0;
    std::uint32_t relevance = 0;
};

struct Fixture {
    std::string name;
    std::string split;
    std::vector<std::string> query_ids;
    std::vector<std::string> query_texts;
    std::vector<std::string> doc_ids;
    std::vector<std::string> doc_texts;
    std::vector<Qrel> qrels;
};

// Loads corpus.tsv plus queries[_<split>].tsv and qrels[_<split>].tsv.
// "test" uses the unsuffixed query/qrel files for compatibility with the
// existing research fixtures.
Fixture load_fixture(const std::string& directory, const std::string& split);
std::string fixture_fingerprint(const Fixture& fixture);

using Ranking = std::vector<std::pair<float, std::uint32_t>>;

struct Metrics {
    double ndcg_at_10 = 0.0;
    double precision_at_10 = 0.0;
    double recall_at_10 = 0.0;
    double recall_at_100 = 0.0;
    double mrr_at_10 = 0.0;
    std::size_t evaluated_queries = 0;
};

// metric_profile=trec-v1:
//   nDCG@10 uses qrel values as gains, P@10 has a fixed denominator of ten,
//   recall divides by all positive qrels, and equal scores tie by doc index.
Metrics evaluate_rankings(const std::vector<Ranking>& rankings, const Fixture& fixture);

// Resolve all defaults and reject unknown keys so a typo cannot silently
// create a different experiment than the manifest claims.
EncoderConfig resolve_encoder_config(const VariantSpec& variant);

struct HashedIdfEncoderConfig {
    EncoderConfig encoder;
    HashedIdfConfig idf;
};

HashedIdfEncoderConfig resolve_hashed_idf_encoder_config(const VariantSpec& variant);

struct WsdmIdfFusionConfig {
    HashedIdfEncoderConfig idf_encoder;
    float idf_weight = 0.0f;
    bool idf_candidates = false;
    std::uint32_t pool_per_leg = 100;
};

// Frozen WSDM(SAB)/WSDM(Atire) score fusion over the production six-leg
// candidate union, optionally extended and reranked by a hashed-IDF encoder.
// Only the three fusion-specific fields above vary; the IDF encoder remains a
// fully resolved, provenance-bearing configuration.
WsdmIdfFusionConfig resolve_wsdm_idf_fusion_config(const VariantSpec& variant);

enum class CoordinateTransform : std::uint8_t {
    None,
    Center,
    Standardize,
};

enum class CoordinateRoutingPolicy : std::uint8_t {
    Fixed,
    Blend,
    Selective,
    SelectiveEnergy,
};

struct CoordinateCalibrationConfig {
    HashedIdfEncoderConfig idf_encoder;
    std::uint32_t retrieval_dim = 0;
    CoordinateTransform transform = CoordinateTransform::None;
    CoordinateRoutingPolicy routing_policy = CoordinateRoutingPolicy::Fixed;
    // Full-width score contribution for Blend and for Selective's rejected-
    // chart fallback. Zero is the prefix, one is the maximum-width chart.
    float full_weight = 1.0f;
    float min_chart_overlap = 0.0f;
    float max_chart_distortion = 2.0f;
    // Maximum absolute deviation of observed query prefix energy from the
    // uniform FWHT expectation retrieval_dim/output_dim.
    float max_energy_deviation = 1.0f;
    // For Standardize, mix each coordinate variance toward the global mean
    // before inverse-standard-deviation scaling. Zero is full diagonal
    // standardization; one is centered-only after cosine normalization.
    float variance_shrinkage = 0.0f;
    // Lower bound relative to the mean coordinate variance.
    float min_variance_ratio = 1.0e-4f;
};

// Corpus-coordinate calibration over one maximum-width hashed-IDF projection.
// Variants may select nested prefix widths and transforms, while the lexical,
// IDF, projection, and maximum-width identity remains fixed.
CoordinateCalibrationConfig resolve_coordinate_calibration_config(const VariantSpec& variant);

enum class FeatureFamilyNormalization : std::uint8_t {
    // Normalize each chart independently; fusion is a convex combination of
    // the two cosine scores.
    Independent,
    // Concatenate weighted raw chart prefixes and normalize once. This
    // preserves query/document-specific family energy.
    Joint,
    // As Joint, after scaling each family by its corpus document RMS. This
    // preserves within-family energy variation while making the fixed weight
    // portable across corpora with different character/word count ratios.
    JointRms,
};

enum class FeatureFamilyPolicy : std::uint8_t {
    // Historical atlas behavior: the family chart replaces the combined
    // encoder. Retained so earlier manifests preserve their meaning.
    FamilyOnly,
    // Safe combined maximum-width chart without a residual.
    BaseOnly,
    // Add the family chart to every query with a fixed convex weight.
    ResidualBlend,
    // Add the residual only when every label-free query/chart observation is
    // inside its declared interval; rejected queries keep the base score.
    Selective,
};

enum class FeatureResidualScoreNormalization : std::uint8_t {
    RawCosine,
    QueryZScore,
    RankRrf,
};

struct FeatureFamilyAtlasConfig {
    // A combined character+word identity is resolved from the manifest, then
    // split into independent character-only and word-only encoders/artifacts
    // by the atlas workspace. Keeping one base identity makes accidental
    // tokenizer, hash, or projection drift between charts impossible.
    HashedIdfEncoderConfig idf_encoder;
    std::uint32_t char_dim = 0;
    std::uint32_t word_dim = 0;
    std::uint32_t storage_budget_dim = 0;
    FeatureFamilyNormalization normalization = FeatureFamilyNormalization::Independent;
    // Score of the deployable block embedding whose character and word
    // blocks are scaled by sqrt(char_weight) and sqrt(1-char_weight).
    float char_weight = 0.5f;
    FeatureFamilyPolicy policy = FeatureFamilyPolicy::FamilyOnly;
    FeatureResidualScoreNormalization residual_score_normalization =
        FeatureResidualScoreNormalization::RawCosine;
    // Family-chart contribution when it augments the combined base.
    float residual_weight = 0.0f;
    float residual_rrf_k = 60.0f;
    // Admission intervals for Selective. Word energy is measured from the
    // query after corpus-RMS family calibration; overlap is between the two
    // family top-K charts. Defaults admit every observed query.
    float min_word_energy = 0.0f;
    float max_word_energy = 1.0f;
    float min_family_overlap = 0.0f;
    float max_family_overlap = 1.0f;
    float min_base_family_overlap = 0.0f;
    float max_base_family_overlap = 1.0f;
};

// Corpus-adaptive, label-free feature-chart atlas. Family-only variants use
// exactly storage_budget_dim floats; residual policies additionally retain the
// combined output_dim chart and report the full sum. Relevance labels are
// consumed only by the common evaluator.
FeatureFamilyAtlasConfig resolve_feature_family_atlas_config(const VariantSpec& variant);
std::string encoder_config_json(const EncoderConfig& config);

struct EncoderRetrievalRun {
    std::vector<Ranking> rankings;
    std::uint32_t output_dim = 0;
    double query_encode_us = 0.0;
    double document_encode_us = 0.0;
    double score_us = 0.0;
    double artifact_build_us = 0.0;
    std::uint64_t artifact_bytes = 0;
    std::uint64_t peak_working_bytes = 0;
};

// Exact dense retrieval with bounded document-embedding and ranking memory.
// Changing document_block_size changes memory use only. Returned rows are
// sorted best-first and contain the exact top retrieval_depth documents under
// the evaluator's deterministic score/doc-index ordering.
EncoderRetrievalRun run_encoder_retrieval(const EncoderConfig& config, const Fixture& fixture,
                                          std::size_t document_block_size,
                                          std::size_t retrieval_depth = 100);

struct ResultContext {
    std::string experiment;
    std::string manifest_fingerprint;
    std::string variant;
    std::string kind = "encoder";
    std::string fixture;
    std::string fixture_fingerprint;
    std::string split;
    std::string code_revision = "unrecorded";
    std::string metric_profile = "trec-v1";
    std::map<std::string, std::string> metadata;
    std::size_t document_block_size = 0;
    std::size_t retrieval_depth = 100;
};

std::string encoder_result_json(const ResultContext& context, const EncoderConfig& config,
                                const EncoderRetrievalRun& run, const Metrics& metrics,
                                double evaluation_us, std::size_t query_count,
                                std::size_t document_count, std::size_t qrel_count);

std::string json_escape(std::string_view value);

} // namespace simeon::experiment
