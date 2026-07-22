#include "experiment_support.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "simeon/simd.hpp"

namespace simeon::experiment {
namespace {

std::string trim(std::string_view value) {
    constexpr std::string_view whitespace = " \t\r\n";
    const auto first = value.find_first_not_of(whitespace);
    if (first == std::string_view::npos)
        return {};
    const auto last = value.find_last_not_of(whitespace);
    return std::string(value.substr(first, last - first + 1));
}

std::string strip_comment(std::string_view line) {
    char quote = '\0';
    bool escaped = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (quote == '"' && c == '\\') {
            escaped = true;
            continue;
        }
        if (quote != '\0') {
            if (c == quote)
                quote = '\0';
            continue;
        }
        if (c == '"' || c == '\'') {
            quote = c;
        } else if (c == '#') {
            return std::string(line.substr(0, i));
        }
    }
    return std::string(line);
}

[[noreturn]] void manifest_error(std::size_t line, const std::string& message) {
    throw std::runtime_error("manifest line " + std::to_string(line) + ": " + message);
}

std::string parse_value(std::string_view raw, std::size_t line) {
    std::string value = trim(raw);
    if (value.empty())
        manifest_error(line, "value must not be empty");
    if (value.front() != '"' && value.front() != '\'')
        return value;
    const char quote = value.front();
    if (value.size() < 2 || value.back() != quote)
        manifest_error(line, "unterminated quoted value");
    std::string out;
    out.reserve(value.size() - 2);
    for (std::size_t i = 1; i + 1 < value.size(); ++i) {
        const char c = value[i];
        if (quote == '"' && c == '\\') {
            if (i + 2 >= value.size())
                manifest_error(line, "unterminated escape sequence");
            const char escaped = value[++i];
            switch (escaped) {
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case '\\':
                case '"':
                    out.push_back(escaped);
                    break;
                default:
                    manifest_error(line, std::string("unsupported escape \\") + escaped);
            }
        } else {
            out.push_back(c);
        }
    }
    return out;
}

std::uint64_t parse_u64(std::string_view value, const std::string& key) {
    int base = 10;
    if (value.size() > 2 && value[0] == '0' && (value[1] == 'x' || value[1] == 'X')) {
        value.remove_prefix(2);
        base = 16;
    }
    std::uint64_t out = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), out, base);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::runtime_error("invalid unsigned integer for '" + key +
                                 "': " + std::string(value));
    return out;
}

std::uint32_t parse_u32(std::string_view value, const std::string& key) {
    const auto parsed = parse_u64(value, key);
    if (parsed > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("unsigned integer out of range for '" + key + "'");
    return static_cast<std::uint32_t>(parsed);
}

float parse_float(std::string_view value, const std::string& key) {
    std::istringstream input{std::string(value)};
    input.imbue(std::locale::classic());
    float parsed = 0.0f;
    if (!(input >> parsed) || !std::isfinite(parsed))
        throw std::runtime_error("invalid finite float for '" + key + "': " + std::string(value));
    input >> std::ws;
    if (!input.eof())
        throw std::runtime_error("invalid finite float for '" + key + "': " + std::string(value));
    return parsed;
}

bool parse_bool(std::string_view value, const std::string& key) {
    if (value == "true" || value == "1" || value == "yes" || value == "on")
        return true;
    if (value == "false" || value == "0" || value == "no" || value == "off")
        return false;
    throw std::runtime_error("invalid boolean for '" + key + "': " + std::string(value));
}

class Fingerprint {
public:
    void add(std::string_view value) noexcept {
        add_u64(value.size());
        for (const unsigned char byte : value) {
            value_ ^= byte;
            value_ *= kPrime;
        }
    }

    void add_u64(std::uint64_t value) noexcept {
        for (int shift = 0; shift < 64; shift += 8) {
            value_ ^= static_cast<std::uint8_t>(value >> shift);
            value_ *= kPrime;
        }
    }

    std::string finish() const {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << value_;
        return out.str();
    }

private:
    static constexpr std::uint64_t kOffset = 14695981039346656037ULL;
    static constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t value_ = kOffset;
};

std::vector<std::pair<std::string, std::string>> read_tsv2(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open fixture file " + path.string());
    std::vector<std::pair<std::string, std::string>> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty())
            continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos || tab == 0)
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     ": expected id<TAB>text");
        rows.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    return rows;
}

struct QrelRow {
    std::string query_id;
    std::string doc_id;
    std::uint32_t relevance = 0;
};

std::vector<QrelRow> read_qrels(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open fixture file " + path.string());
    std::vector<QrelRow> rows;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty())
            continue;
        const auto first = line.find('\t');
        const auto second = first == std::string::npos ? first : line.find('\t', first + 1);
        if (first == std::string::npos || second == std::string::npos || first == 0 ||
            second == first + 1)
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) +
                                     ": expected query_id<TAB>doc_id<TAB>relevance");
        const std::string_view rel{line.data() + second + 1, line.size() - second - 1};
        rows.push_back({line.substr(0, first), line.substr(first + 1, second - first - 1),
                        parse_u32(rel, "relevance")});
    }
    return rows;
}

template <typename Names>
std::unordered_map<std::string, std::uint32_t> make_id_map(const Names& names,
                                                           std::string_view kind) {
    if (names.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error(std::string(kind) + " ids exceed uint32 index capacity");
    std::unordered_map<std::string, std::uint32_t> result;
    result.reserve(names.size());
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (!result.emplace(names[i], static_cast<std::uint32_t>(i)).second)
            throw std::runtime_error("duplicate " + std::string(kind) + " id: " + names[i]);
    }
    return result;
}

const char* ngram_mode_name(NGramMode mode) {
    switch (mode) {
        case NGramMode::CharOnly:
            return "char";
        case NGramMode::WordOnly:
            return "word";
        case NGramMode::CharAndWord:
            return "char_word";
        case NGramMode::Subword:
            return "subword";
        case NGramMode::CharSubword:
            return "char_subword";
    }
    return "unknown";
}

const char* hash_family_name(HashFamily family) {
    switch (family) {
        case HashFamily::SplitMix64:
            return "splitmix64";
        case HashFamily::XxHash64:
            return "xxhash64";
        case HashFamily::Crc32:
            return "crc32";
        case HashFamily::MixedTabulation:
            return "mixed_tabulation";
    }
    return "unknown";
}

const char* text_normalization_name(TextNormalization normalization) {
    switch (normalization) {
        case TextNormalization::None:
            return "none";
        case TextNormalization::AsciiLower:
            return "ascii_lower";
    }
    return "unknown";
}

const char* char_ngram_scope_name(CharNGramScope scope) {
    switch (scope) {
        case CharNGramScope::Text:
            return "text";
        case CharNGramScope::WordBounded:
            return "word_bounded";
    }
    return "unknown";
}

const char* sketch_weighting_name(SketchWeighting weighting) {
    switch (weighting) {
        case SketchWeighting::Raw:
            return "raw";
        case SketchWeighting::SignedSqrt:
            return "signed_sqrt";
    }
    return "unknown";
}

const char* feature_weighting_name(FeatureWeighting weighting) {
    switch (weighting) {
        case FeatureWeighting::Raw:
            return "raw";
        case FeatureWeighting::SqrtTf:
            return "sqrt_tf";
    }
    return "unknown";
}

const char* projection_mode_name(ProjectionMode mode) {
    switch (mode) {
        case ProjectionMode::None:
            return "none";
        case ProjectionMode::AchlioptasSparse:
            return "achlioptas";
        case ProjectionMode::DenseGaussian:
            return "dense_gaussian";
        case ProjectionMode::VerySparse:
            return "very_sparse";
        case ProjectionMode::SparseJL:
            return "sparse_jl";
        case ProjectionMode::Fwht:
            return "fwht";
    }
    return "unknown";
}

bool ranking_better(const Ranking::value_type& left, const Ranking::value_type& right) {
    if (left.first != right.first)
        return left.first > right.first;
    return left.second < right.second;
}

void retain_top(Ranking& heap, Ranking::value_type candidate, std::size_t depth) {
    if (heap.size() < depth) {
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), ranking_better);
        return;
    }
    if (!ranking_better(candidate, heap.front()))
        return;
    std::pop_heap(heap.begin(), heap.end(), ranking_better);
    heap.back() = candidate;
    std::push_heap(heap.begin(), heap.end(), ranking_better);
}

} // namespace

Manifest parse_manifest(std::istream& input) {
    Manifest manifest;
    std::unordered_set<std::string> top_keys;
    std::unordered_set<std::string> variant_names;
    VariantSpec* current = nullptr;
    std::string line;
    std::size_t line_number = 0;

    while (std::getline(input, line)) {
        ++line_number;
        const std::string clean = trim(strip_comment(line));
        if (clean.empty())
            continue;

        if (clean.front() == '[') {
            if (clean.back() != ']')
                manifest_error(line_number, "unterminated section header");
            constexpr std::string_view prefix = "variant.";
            const std::string section = trim(std::string_view(clean).substr(1, clean.size() - 2));
            if (!section.starts_with(prefix) || section.size() == prefix.size())
                manifest_error(line_number, "expected [variant.<name>] section");
            const std::string name = section.substr(prefix.size());
            if (!variant_names.insert(name).second)
                manifest_error(line_number, "duplicate variant '" + name + "'");
            manifest.variants.push_back(VariantSpec{.name = name, .kind = {}, .parameters = {}});
            current = &manifest.variants.back();
            continue;
        }

        const auto equals = clean.find('=');
        if (equals == std::string::npos)
            manifest_error(line_number, "expected key=value");
        const std::string key = trim(std::string_view(clean).substr(0, equals));
        if (key.empty())
            manifest_error(line_number, "key must not be empty");
        const std::string value =
            parse_value(std::string_view(clean).substr(equals + 1), line_number);

        if (current != nullptr) {
            if (key == "kind") {
                if (!current->kind.empty())
                    manifest_error(line_number, "duplicate key 'kind'");
                current->kind = value;
            } else {
                if (!current->parameters.emplace(key, value).second)
                    manifest_error(line_number, "duplicate key '" + key + "'");
            }
            continue;
        }

        if (!top_keys.insert(key).second)
            manifest_error(line_number, "duplicate key '" + key + "'");
        if (key == "schema") {
            manifest.schema_version = parse_u32(value, key);
        } else if (key == "name") {
            manifest.name = value;
        } else {
            manifest.metadata.emplace(key, value);
        }
    }

    if (manifest.schema_version != 1)
        throw std::runtime_error("manifest schema must be 1");
    if (manifest.name.empty())
        throw std::runtime_error("manifest name is required");
    if (manifest.variants.empty())
        throw std::runtime_error("manifest must define at least one variant");
    for (const auto& variant : manifest.variants) {
        if (variant.kind.empty())
            throw std::runtime_error("variant '" + variant.name + "' is missing kind");
    }
    return manifest;
}

Manifest load_manifest(const std::string& path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open experiment manifest " + path);
    return parse_manifest(input);
}

std::string manifest_fingerprint(const Manifest& manifest) {
    Fingerprint fingerprint;
    fingerprint.add("simeon.experiment.manifest.v1");
    fingerprint.add_u64(manifest.schema_version);
    fingerprint.add(manifest.name);
    for (const auto& [key, value] : manifest.metadata) {
        fingerprint.add(key);
        fingerprint.add(value);
    }
    for (const auto& variant : manifest.variants) {
        fingerprint.add(variant.name);
        fingerprint.add(variant.kind);
        for (const auto& [key, value] : variant.parameters) {
            fingerprint.add(key);
            fingerprint.add(value);
        }
    }
    return fingerprint.finish();
}

void validate_run_policy(const Manifest& manifest, std::string_view split) {
    const auto phase = manifest.metadata.find("phase");
    const auto selection = manifest.metadata.find("selection_split");
    const auto holdout = manifest.metadata.find("holdout_split");
    const bool has_policy = phase != manifest.metadata.end() ||
                            selection != manifest.metadata.end() ||
                            holdout != manifest.metadata.end();
    if (!has_policy)
        return;
    if (phase == manifest.metadata.end() || selection == manifest.metadata.end() ||
        holdout == manifest.metadata.end())
        throw std::runtime_error(
            "run policy requires phase, selection_split, and holdout_split together");
    if (selection->second == holdout->second)
        throw std::runtime_error("selection_split and holdout_split must differ");
    if (phase->second != "exploration" && phase->second != "frozen")
        throw std::runtime_error("phase must be 'exploration' or 'frozen'");
    if (split != selection->second && split != holdout->second)
        throw std::runtime_error("split '" + std::string(split) +
                                 "' is not declared by the manifest run policy");
    if (split != holdout->second)
        return;
    if (phase->second != "frozen")
        throw std::runtime_error("exploration manifest cannot run on its holdout split");
    if (manifest.variants.size() != 1)
        throw std::runtime_error(
            "frozen holdout manifest must contain exactly one selected variant");
    const auto lineage = manifest.metadata.find("selection_manifest");
    if (lineage == manifest.metadata.end() || lineage->second.empty())
        throw std::runtime_error("frozen holdout manifest requires selection_manifest lineage");
}

void validate_training_regime(const Manifest& manifest) {
    const auto found = manifest.metadata.find("training_regime");
    if (found == manifest.metadata.end())
        throw std::runtime_error("embedding experiment requires training_regime metadata");
    if (found->second != "artifact-free" && found->second != "corpus-adaptive")
        throw std::runtime_error("embedding experiment cannot run training_regime '" +
                                 found->second + "'");
    for (const auto& variant : manifest.variants) {
        const bool fixed = variant.kind == "encoder";
        const bool adaptive = variant.kind == "hashed_idf_encoder";
        const bool adaptive_fusion = variant.kind == "wsdm_idf_fusion";
        const bool adaptive_coordinates = variant.kind == "coordinate_calibrated_idf";
        const bool adaptive_family_atlas = variant.kind == "feature_family_atlas_idf";
        const bool adaptive_reranker = variant.kind == "fragment_geometry_reranker";
        if (!fixed && !(found->second == "corpus-adaptive" &&
                        (adaptive || adaptive_fusion || adaptive_coordinates ||
                         adaptive_family_atlas || adaptive_reranker)))
            throw std::runtime_error("training_regime '" + found->second +
                                     "' cannot run variant kind '" + variant.kind + "'");
    }
}

Fixture load_fixture(const std::string& directory, const std::string& split) {
    namespace fs = std::filesystem;
    if (split.empty())
        throw std::runtime_error("fixture split must not be empty");
    const bool valid_split = std::all_of(split.begin(), split.end(), [](unsigned char byte) {
        return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
               (byte >= '0' && byte <= '9') || byte == '_' || byte == '-';
    });
    if (!valid_split)
        throw std::runtime_error("fixture split contains an unsupported character");
    const fs::path root{directory};
    const std::string suffix = split == "test" ? "" : "_" + split;

    Fixture fixture;
    fixture.name = root.filename().string();
    fixture.split = split;
    for (auto& [id, text] : read_tsv2(root / "corpus.tsv")) {
        fixture.doc_ids.push_back(std::move(id));
        fixture.doc_texts.push_back(std::move(text));
    }
    for (auto& [id, text] : read_tsv2(root / ("queries" + suffix + ".tsv"))) {
        fixture.query_ids.push_back(std::move(id));
        fixture.query_texts.push_back(std::move(text));
    }
    if (fixture.doc_ids.empty() || fixture.query_ids.empty())
        throw std::runtime_error("fixture corpus and query files must not be empty");

    const auto query_ids = make_id_map(fixture.query_ids, "query");
    const auto doc_ids = make_id_map(fixture.doc_ids, "document");
    std::unordered_set<std::uint64_t> seen_qrels;
    for (const auto& row : read_qrels(root / ("qrels" + suffix + ".tsv"))) {
        if (row.relevance == 0)
            continue;
        const auto query = query_ids.find(row.query_id);
        const auto document = doc_ids.find(row.doc_id);
        if (query == query_ids.end())
            throw std::runtime_error("qrel references unknown query id: " + row.query_id);
        if (document == doc_ids.end())
            throw std::runtime_error("qrel references unknown document id: " + row.doc_id);
        const std::uint64_t key =
            (static_cast<std::uint64_t>(query->second) << 32) | document->second;
        if (!seen_qrels.insert(key).second)
            throw std::runtime_error("duplicate positive qrel for query '" + row.query_id +
                                     "' and document '" + row.doc_id + "'");
        fixture.qrels.push_back({query->second, document->second, row.relevance});
    }
    if (fixture.qrels.empty())
        throw std::runtime_error("fixture must contain at least one positive qrel");
    return fixture;
}

std::string fixture_fingerprint(const Fixture& fixture) {
    Fingerprint fingerprint;
    fingerprint.add("simeon.experiment.fixture.v1");
    fingerprint.add(fixture.split);
    fingerprint.add_u64(fixture.query_ids.size());
    for (std::size_t i = 0; i < fixture.query_ids.size(); ++i) {
        fingerprint.add(fixture.query_ids[i]);
        fingerprint.add(fixture.query_texts.at(i));
    }
    fingerprint.add_u64(fixture.doc_ids.size());
    for (std::size_t i = 0; i < fixture.doc_ids.size(); ++i) {
        fingerprint.add(fixture.doc_ids[i]);
        fingerprint.add(fixture.doc_texts.at(i));
    }
    auto qrels = fixture.qrels;
    std::sort(qrels.begin(), qrels.end(), [](const Qrel& a, const Qrel& b) {
        if (a.query != b.query)
            return a.query < b.query;
        if (a.document != b.document)
            return a.document < b.document;
        return a.relevance < b.relevance;
    });
    fingerprint.add_u64(qrels.size());
    for (const auto& qrel : qrels) {
        fingerprint.add_u64(qrel.query);
        fingerprint.add_u64(qrel.document);
        fingerprint.add_u64(qrel.relevance);
    }
    return fingerprint.finish();
}

Metrics evaluate_rankings(const std::vector<Ranking>& rankings, const Fixture& fixture) {
    if (rankings.size() != fixture.query_ids.size())
        throw std::runtime_error("ranking count must match fixture query count");

    std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> relevance(
        fixture.query_ids.size());
    for (const auto& qrel : fixture.qrels) {
        if (qrel.query >= relevance.size() || qrel.document >= fixture.doc_ids.size())
            throw std::runtime_error("qrel index is outside the fixture");
        auto& grade = relevance[qrel.query][qrel.document];
        grade = std::max(grade, qrel.relevance);
    }

    Metrics result;
    for (std::size_t query = 0; query < rankings.size(); ++query) {
        const auto& qrels = relevance[query];
        if (qrels.empty())
            continue;
        ++result.evaluated_queries;

        Ranking sorted = rankings[query];
        std::unordered_set<std::uint32_t> ranked_documents;
        ranked_documents.reserve(sorted.size());
        for (const auto& [score, document] : sorted) {
            if (!std::isfinite(score))
                throw std::runtime_error("ranking contains a non-finite score");
            if (document >= fixture.doc_ids.size())
                throw std::runtime_error("ranking contains a document index outside the fixture");
            if (!ranked_documents.insert(document).second)
                throw std::runtime_error("ranking contains a duplicate document index");
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        double dcg = 0.0;
        std::size_t hits_at_10 = 0;
        std::size_t hits_at_100 = 0;
        double reciprocal_rank = 0.0;
        for (std::size_t rank = 0; rank < sorted.size() && rank < 100; ++rank) {
            const auto found = qrels.find(sorted[rank].second);
            const std::uint32_t grade = found == qrels.end() ? 0 : found->second;
            if (rank < 10 && grade > 0) {
                dcg += static_cast<double>(grade) / std::log2(static_cast<double>(rank) + 2.0);
                ++hits_at_10;
                if (reciprocal_rank == 0.0)
                    reciprocal_rank = 1.0 / (static_cast<double>(rank) + 1.0);
            }
            if (grade > 0)
                ++hits_at_100;
        }

        std::vector<std::uint32_t> ideal;
        ideal.reserve(qrels.size());
        for (const auto& [_, grade] : qrels)
            ideal.push_back(grade);
        std::sort(ideal.begin(), ideal.end(), std::greater<>());
        double idcg = 0.0;
        for (std::size_t rank = 0; rank < ideal.size() && rank < 10; ++rank) {
            idcg += static_cast<double>(ideal[rank]) / std::log2(static_cast<double>(rank) + 2.0);
        }

        result.ndcg_at_10 += idcg == 0.0 ? 0.0 : dcg / idcg;
        result.precision_at_10 += static_cast<double>(hits_at_10) / 10.0;
        result.recall_at_10 += static_cast<double>(hits_at_10) / qrels.size();
        result.recall_at_100 += static_cast<double>(hits_at_100) / qrels.size();
        result.mrr_at_10 += reciprocal_rank;
    }

    if (result.evaluated_queries != 0) {
        const double count = static_cast<double>(result.evaluated_queries);
        result.ndcg_at_10 /= count;
        result.precision_at_10 /= count;
        result.recall_at_10 /= count;
        result.recall_at_100 /= count;
        result.mrr_at_10 /= count;
    }
    return result;
}

EncoderConfig resolve_encoder_config(const VariantSpec& variant) {
    if (variant.kind != "encoder")
        throw std::runtime_error("variant '" + variant.name + "' has kind '" + variant.kind +
                                 "'; expected 'encoder'");

    static const std::unordered_set<std::string> known = {
        "ngram_mode",
        "ngram_min",
        "ngram_max",
        "char_ngram_scope",
        "hash",
        "hash_seed",
        "sketch_dim",
        "output_dim",
        "projection",
        "projection_seed",
        "text_normalization",
        "feature_weighting",
        "sketch_weighting",
        "l2_normalize",
        "matryoshka",
        "matryoshka_decay",
        "sparse_jl_eps",
    };
    for (const auto& [key, _] : variant.parameters) {
        if (!known.contains(key))
            throw std::runtime_error("unknown encoder parameter '" + key + "' in variant '" +
                                     variant.name + "'");
    }

    EncoderConfig config;
    const auto value = [&](std::string_view key) -> const std::string* {
        const auto found = variant.parameters.find(std::string(key));
        return found == variant.parameters.end() ? nullptr : &found->second;
    };
    if (const auto* v = value("ngram_mode")) {
        if (*v == "char")
            config.ngram_mode = NGramMode::CharOnly;
        else if (*v == "word")
            config.ngram_mode = NGramMode::WordOnly;
        else if (*v == "char_word")
            config.ngram_mode = NGramMode::CharAndWord;
        else
            throw std::runtime_error("unsupported ngram_mode '" + *v + "' in variant '" +
                                     variant.name + "'");
    }
    if (const auto* v = value("ngram_min"))
        config.ngram_min = parse_u32(*v, "ngram_min");
    if (const auto* v = value("ngram_max"))
        config.ngram_max = parse_u32(*v, "ngram_max");
    if (const auto* v = value("char_ngram_scope")) {
        if (*v == "text")
            config.char_ngram_scope = CharNGramScope::Text;
        else if (*v == "word_bounded")
            config.char_ngram_scope = CharNGramScope::WordBounded;
        else
            throw std::runtime_error("unsupported char_ngram_scope '" + *v + "' in variant '" +
                                     variant.name + "'");
    }
    if (const auto* v = value("hash")) {
        if (*v == "splitmix64")
            config.hash = HashFamily::SplitMix64;
        else if (*v == "xxhash64")
            config.hash = HashFamily::XxHash64;
        else if (*v == "crc32")
            config.hash = HashFamily::Crc32;
        else if (*v == "mixed_tabulation")
            config.hash = HashFamily::MixedTabulation;
        else
            throw std::runtime_error("unsupported hash '" + *v + "' in variant '" + variant.name +
                                     "'");
    }
    if (const auto* v = value("hash_seed"))
        config.hash_seed = parse_u64(*v, "hash_seed");
    if (const auto* v = value("sketch_dim"))
        config.sketch_dim = parse_u32(*v, "sketch_dim");
    if (const auto* v = value("output_dim"))
        config.output_dim = parse_u32(*v, "output_dim");
    if (const auto* v = value("projection")) {
        if (*v == "none")
            config.projection = ProjectionMode::None;
        else if (*v == "achlioptas")
            config.projection = ProjectionMode::AchlioptasSparse;
        else if (*v == "dense_gaussian")
            config.projection = ProjectionMode::DenseGaussian;
        else if (*v == "very_sparse")
            config.projection = ProjectionMode::VerySparse;
        else if (*v == "sparse_jl")
            config.projection = ProjectionMode::SparseJL;
        else if (*v == "fwht")
            config.projection = ProjectionMode::Fwht;
        else
            throw std::runtime_error("unsupported projection '" + *v + "' in variant '" +
                                     variant.name + "'");
    }
    if (const auto* v = value("projection_seed"))
        config.projection_seed = parse_u64(*v, "projection_seed");
    if (const auto* v = value("text_normalization")) {
        if (*v == "none")
            config.text_normalization = TextNormalization::None;
        else if (*v == "ascii_lower")
            config.text_normalization = TextNormalization::AsciiLower;
        else
            throw std::runtime_error("unsupported text_normalization '" + *v + "' in variant '" +
                                     variant.name + "'");
    }
    if (const auto* v = value("feature_weighting")) {
        if (*v == "raw")
            config.feature_weighting = FeatureWeighting::Raw;
        else if (*v == "sqrt_tf")
            config.feature_weighting = FeatureWeighting::SqrtTf;
        else
            throw std::runtime_error("unsupported feature_weighting '" + *v + "' in variant '" +
                                     variant.name + "'");
    }
    if (const auto* v = value("sketch_weighting")) {
        if (*v == "raw")
            config.sketch_weighting = SketchWeighting::Raw;
        else if (*v == "signed_sqrt")
            config.sketch_weighting = SketchWeighting::SignedSqrt;
        else
            throw std::runtime_error("unsupported sketch_weighting '" + *v + "' in variant '" +
                                     variant.name + "'");
    }
    if (const auto* v = value("l2_normalize"))
        config.l2_normalize = parse_bool(*v, "l2_normalize");
    if (const auto* v = value("matryoshka"))
        config.matryoshka = parse_bool(*v, "matryoshka");
    if (const auto* v = value("matryoshka_decay"))
        config.matryoshka_decay = parse_float(*v, "matryoshka_decay");
    if (const auto* v = value("sparse_jl_eps"))
        config.sparse_jl_eps = parse_float(*v, "sparse_jl_eps");
    return config;
}

HashedIdfEncoderConfig resolve_hashed_idf_encoder_config(const VariantSpec& variant) {
    if (variant.kind != "hashed_idf_encoder")
        throw std::runtime_error("variant '" + variant.name + "' has kind '" + variant.kind +
                                 "'; expected 'hashed_idf_encoder'");
    VariantSpec encoder_variant = variant;
    encoder_variant.kind = "encoder";
    HashedIdfEncoderConfig result;
    if (const auto found = encoder_variant.parameters.find("idf_hash_dim");
        found != encoder_variant.parameters.end()) {
        result.idf.hash_dim = parse_u32(found->second, "idf_hash_dim");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("idf_scope");
        found != encoder_variant.parameters.end()) {
        if (found->second == "all")
            result.idf.scope = HashedIdfScope::All;
        else if (found->second == "char")
            result.idf.scope = HashedIdfScope::Character;
        else if (found->second == "word")
            result.idf.scope = HashedIdfScope::Word;
        else
            throw std::runtime_error("unsupported idf_scope '" + found->second + "' in variant '" +
                                     variant.name + "'");
        encoder_variant.parameters.erase(found);
    }
    result.encoder = resolve_encoder_config(encoder_variant);
    return result;
}

WsdmIdfFusionConfig resolve_wsdm_idf_fusion_config(const VariantSpec& variant) {
    if (variant.kind != "wsdm_idf_fusion")
        throw std::runtime_error("variant '" + variant.name + "' has kind '" + variant.kind +
                                 "'; expected 'wsdm_idf_fusion'");

    VariantSpec encoder_variant = variant;
    encoder_variant.kind = "hashed_idf_encoder";
    WsdmIdfFusionConfig result;
    if (const auto found = encoder_variant.parameters.find("idf_weight");
        found != encoder_variant.parameters.end()) {
        result.idf_weight = parse_float(found->second, "idf_weight");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("idf_candidates");
        found != encoder_variant.parameters.end()) {
        result.idf_candidates = parse_bool(found->second, "idf_candidates");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("pool_per_leg");
        found != encoder_variant.parameters.end()) {
        result.pool_per_leg = parse_u32(found->second, "pool_per_leg");
        encoder_variant.parameters.erase(found);
    }
    if (!(result.idf_weight >= 0.0f && result.idf_weight <= 1.0f))
        throw std::runtime_error("idf_weight must be within [0, 1] in variant '" + variant.name +
                                 "'");
    if (result.pool_per_leg == 0)
        throw std::runtime_error("pool_per_leg must be greater than zero in variant '" +
                                 variant.name + "'");
    result.idf_encoder = resolve_hashed_idf_encoder_config(encoder_variant);
    return result;
}

CoordinateCalibrationConfig resolve_coordinate_calibration_config(const VariantSpec& variant) {
    if (variant.kind != "coordinate_calibrated_idf")
        throw std::runtime_error("variant '" + variant.name + "' has kind '" + variant.kind +
                                 "'; expected 'coordinate_calibrated_idf'");

    VariantSpec encoder_variant = variant;
    encoder_variant.kind = "hashed_idf_encoder";
    CoordinateCalibrationConfig result;
    if (const auto found = encoder_variant.parameters.find("retrieval_dim");
        found != encoder_variant.parameters.end()) {
        result.retrieval_dim = parse_u32(found->second, "retrieval_dim");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("coordinate_transform");
        found != encoder_variant.parameters.end()) {
        if (found->second == "none")
            result.transform = CoordinateTransform::None;
        else if (found->second == "center")
            result.transform = CoordinateTransform::Center;
        else if (found->second == "standardize")
            result.transform = CoordinateTransform::Standardize;
        else
            throw std::runtime_error("unsupported coordinate_transform '" + found->second +
                                     "' in variant '" + variant.name + "'");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("coordinate_policy");
        found != encoder_variant.parameters.end()) {
        if (found->second == "fixed")
            result.routing_policy = CoordinateRoutingPolicy::Fixed;
        else if (found->second == "blend")
            result.routing_policy = CoordinateRoutingPolicy::Blend;
        else if (found->second == "selective")
            result.routing_policy = CoordinateRoutingPolicy::Selective;
        else if (found->second == "selective_energy")
            result.routing_policy = CoordinateRoutingPolicy::SelectiveEnergy;
        else
            throw std::runtime_error("unsupported coordinate_policy '" + found->second +
                                     "' in variant '" + variant.name + "'");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("full_weight");
        found != encoder_variant.parameters.end()) {
        result.full_weight = parse_float(found->second, "full_weight");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("min_chart_overlap");
        found != encoder_variant.parameters.end()) {
        result.min_chart_overlap = parse_float(found->second, "min_chart_overlap");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("max_chart_distortion");
        found != encoder_variant.parameters.end()) {
        result.max_chart_distortion = parse_float(found->second, "max_chart_distortion");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("max_energy_deviation");
        found != encoder_variant.parameters.end()) {
        result.max_energy_deviation = parse_float(found->second, "max_energy_deviation");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("variance_shrinkage");
        found != encoder_variant.parameters.end()) {
        result.variance_shrinkage = parse_float(found->second, "variance_shrinkage");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("min_variance_ratio");
        found != encoder_variant.parameters.end()) {
        result.min_variance_ratio = parse_float(found->second, "min_variance_ratio");
        encoder_variant.parameters.erase(found);
    }
    result.idf_encoder = resolve_hashed_idf_encoder_config(encoder_variant);
    const auto maximum_dim = result.idf_encoder.encoder.output_dim;
    if (result.retrieval_dim == 0 || result.retrieval_dim > maximum_dim)
        throw std::runtime_error("retrieval_dim must be within [1, output_dim] in variant '" +
                                 variant.name + "'");
    if (result.idf_encoder.encoder.projection != ProjectionMode::Fwht)
        throw std::runtime_error("coordinate_calibrated_idf requires projection=fwht in variant '" +
                                 variant.name + "'");
    if (!result.idf_encoder.encoder.l2_normalize)
        throw std::runtime_error(
            "coordinate_calibrated_idf requires l2_normalize=true in variant '" + variant.name +
            "'");
    if (!(result.variance_shrinkage >= 0.0f && result.variance_shrinkage <= 1.0f))
        throw std::runtime_error("variance_shrinkage must be within [0, 1] in variant '" +
                                 variant.name + "'");
    if (!(result.full_weight >= 0.0f && result.full_weight <= 1.0f))
        throw std::runtime_error("full_weight must be within [0, 1] in variant '" + variant.name +
                                 "'");
    if (!(result.min_chart_overlap >= 0.0f && result.min_chart_overlap <= 1.0f))
        throw std::runtime_error("min_chart_overlap must be within [0, 1] in variant '" +
                                 variant.name + "'");
    if (!(result.max_chart_distortion >= 0.0f && result.max_chart_distortion <= 2.0f))
        throw std::runtime_error("max_chart_distortion must be within [0, 2] in variant '" +
                                 variant.name + "'");
    if (!(result.max_energy_deviation >= 0.0f && result.max_energy_deviation <= 1.0f))
        throw std::runtime_error("max_energy_deviation must be within [0, 1] in variant '" +
                                 variant.name + "'");
    if (!(result.min_variance_ratio > 0.0f && result.min_variance_ratio <= 1.0f))
        throw std::runtime_error("min_variance_ratio must be within (0, 1] in variant '" +
                                 variant.name + "'");
    return result;
}

FeatureFamilyAtlasConfig resolve_feature_family_atlas_config(const VariantSpec& variant) {
    if (variant.kind != "feature_family_atlas_idf")
        throw std::runtime_error("variant '" + variant.name + "' has kind '" + variant.kind +
                                 "'; expected 'feature_family_atlas_idf'");

    VariantSpec encoder_variant = variant;
    encoder_variant.kind = "hashed_idf_encoder";
    FeatureFamilyAtlasConfig result;
    if (const auto found = encoder_variant.parameters.find("char_dim");
        found != encoder_variant.parameters.end()) {
        result.char_dim = parse_u32(found->second, "char_dim");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("word_dim");
        found != encoder_variant.parameters.end()) {
        result.word_dim = parse_u32(found->second, "word_dim");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("storage_budget_dim");
        found != encoder_variant.parameters.end()) {
        result.storage_budget_dim = parse_u32(found->second, "storage_budget_dim");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("family_normalization");
        found != encoder_variant.parameters.end()) {
        if (found->second == "independent")
            result.normalization = FeatureFamilyNormalization::Independent;
        else if (found->second == "joint")
            result.normalization = FeatureFamilyNormalization::Joint;
        else if (found->second == "joint_rms")
            result.normalization = FeatureFamilyNormalization::JointRms;
        else
            throw std::runtime_error("unsupported family_normalization '" + found->second +
                                     "' in variant '" + variant.name + "'");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("char_weight");
        found != encoder_variant.parameters.end()) {
        result.char_weight = parse_float(found->second, "char_weight");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("family_policy");
        found != encoder_variant.parameters.end()) {
        if (found->second == "family_only")
            result.policy = FeatureFamilyPolicy::FamilyOnly;
        else if (found->second == "base_only")
            result.policy = FeatureFamilyPolicy::BaseOnly;
        else if (found->second == "residual_blend")
            result.policy = FeatureFamilyPolicy::ResidualBlend;
        else if (found->second == "selective")
            result.policy = FeatureFamilyPolicy::Selective;
        else
            throw std::runtime_error("unsupported family_policy '" + found->second +
                                     "' in variant '" + variant.name + "'");
        encoder_variant.parameters.erase(found);
    }
    if (const auto found = encoder_variant.parameters.find("residual_score_normalization");
        found != encoder_variant.parameters.end()) {
        if (found->second == "raw_cosine")
            result.residual_score_normalization = FeatureResidualScoreNormalization::RawCosine;
        else if (found->second == "query_zscore")
            result.residual_score_normalization = FeatureResidualScoreNormalization::QueryZScore;
        else if (found->second == "rank_rrf")
            result.residual_score_normalization = FeatureResidualScoreNormalization::RankRrf;
        else
            throw std::runtime_error("unsupported residual_score_normalization '" + found->second +
                                     "' in variant '" + variant.name + "'");
        encoder_variant.parameters.erase(found);
    }
    const auto parse_family_float = [&](const char* key, float& destination) {
        if (const auto found = encoder_variant.parameters.find(key);
            found != encoder_variant.parameters.end()) {
            destination = parse_float(found->second, key);
            encoder_variant.parameters.erase(found);
        }
    };
    parse_family_float("residual_weight", result.residual_weight);
    parse_family_float("residual_rrf_k", result.residual_rrf_k);
    parse_family_float("min_word_energy", result.min_word_energy);
    parse_family_float("max_word_energy", result.max_word_energy);
    parse_family_float("min_family_overlap", result.min_family_overlap);
    parse_family_float("max_family_overlap", result.max_family_overlap);
    parse_family_float("min_base_family_overlap", result.min_base_family_overlap);
    parse_family_float("max_base_family_overlap", result.max_base_family_overlap);
    result.idf_encoder = resolve_hashed_idf_encoder_config(encoder_variant);

    const auto maximum_dim = result.idf_encoder.encoder.output_dim;
    if (result.idf_encoder.encoder.ngram_mode != NGramMode::CharAndWord)
        throw std::runtime_error(
            "feature_family_atlas_idf requires ngram_mode=char_word in variant '" + variant.name +
            "'");
    if (result.idf_encoder.encoder.projection != ProjectionMode::Fwht)
        throw std::runtime_error("feature_family_atlas_idf requires projection=fwht in variant '" +
                                 variant.name + "'");
    if (!result.idf_encoder.encoder.l2_normalize)
        throw std::runtime_error(
            "feature_family_atlas_idf requires l2_normalize=true in variant '" + variant.name +
            "'");
    if (result.idf_encoder.idf.scope != HashedIdfScope::All)
        throw std::runtime_error("feature_family_atlas_idf requires idf_scope=all in variant '" +
                                 variant.name + "'");
    if (result.char_dim == 0 || result.char_dim > maximum_dim)
        throw std::runtime_error("char_dim must be within [1, output_dim] in variant '" +
                                 variant.name + "'");
    if (result.word_dim == 0 || result.word_dim > maximum_dim)
        throw std::runtime_error("word_dim must be within [1, output_dim] in variant '" +
                                 variant.name + "'");
    const std::uint64_t total_dim = static_cast<std::uint64_t>(result.char_dim) + result.word_dim;
    if (result.storage_budget_dim == 0 || total_dim != result.storage_budget_dim)
        throw std::runtime_error("char_dim + word_dim must equal storage_budget_dim in variant '" +
                                 variant.name + "'");
    if (!(result.char_weight >= 0.0f && result.char_weight <= 1.0f))
        throw std::runtime_error("char_weight must be within [0, 1] in variant '" + variant.name +
                                 "'");
    if (!(result.residual_weight >= 0.0f && result.residual_weight <= 1.0f))
        throw std::runtime_error("residual_weight must be within [0, 1] in variant '" +
                                 variant.name + "'");
    if (!(result.residual_rrf_k > 0.0f))
        throw std::runtime_error("residual_rrf_k must be greater than zero in variant '" +
                                 variant.name + "'");
    const auto validate_interval = [&](float minimum, float maximum, const char* name) {
        if (!(minimum >= 0.0f && maximum <= 1.0f && minimum <= maximum))
            throw std::runtime_error(std::string(name) +
                                     " interval must be ordered within [0, 1] in variant '" +
                                     variant.name + "'");
    };
    validate_interval(result.min_word_energy, result.max_word_energy, "word-energy");
    validate_interval(result.min_family_overlap, result.max_family_overlap, "family-overlap");
    validate_interval(result.min_base_family_overlap, result.max_base_family_overlap,
                      "base-family-overlap");
    if ((result.policy == FeatureFamilyPolicy::ResidualBlend ||
         result.policy == FeatureFamilyPolicy::Selective) &&
        result.residual_weight <= 0.0f)
        throw std::runtime_error("residual policies require residual_weight > 0 in variant '" +
                                 variant.name + "'");
    return result;
}

std::string encoder_config_json(const EncoderConfig& config) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"ngram_mode\":\"" << ngram_mode_name(config.ngram_mode) << "\""
        << ",\"ngram_min\":" << config.ngram_min << ",\"ngram_max\":" << config.ngram_max
        << ",\"char_ngram_scope\":\"" << char_ngram_scope_name(config.char_ngram_scope) << "\""
        << ",\"hash\":\"" << hash_family_name(config.hash) << "\""
        << ",\"hash_seed\":" << config.hash_seed << ",\"sketch_dim\":" << config.sketch_dim
        << ",\"output_dim\":" << config.output_dim << ",\"projection\":\""
        << projection_mode_name(config.projection) << "\""
        << ",\"projection_seed\":" << config.projection_seed << ",\"text_normalization\":\""
        << text_normalization_name(config.text_normalization) << "\""
        << ",\"feature_weighting\":\"" << feature_weighting_name(config.feature_weighting) << "\""
        << ",\"sketch_weighting\":\"" << sketch_weighting_name(config.sketch_weighting) << "\""
        << ",\"l2_normalize\":" << (config.l2_normalize ? "true" : "false")
        << ",\"matryoshka\":" << (config.matryoshka ? "true" : "false")
        << ",\"matryoshka_decay\":" << config.matryoshka_decay
        << ",\"sparse_jl_eps\":" << config.sparse_jl_eps << ",\"hashed_idf\":";
    if (config.hashed_idf == nullptr) {
        out << "null";
    } else {
        out << "{\"fingerprint\":\"" << json_escape(config.hashed_idf->fingerprint())
            << "\",\"hash_dim\":" << config.hashed_idf->hash_dim() << ",\"scope\":\""
            << (config.hashed_idf->scope() == HashedIdfScope::All         ? "all"
                : config.hashed_idf->scope() == HashedIdfScope::Character ? "char"
                                                                          : "word")
            << "\""
            << ",\"document_count\":" << config.hashed_idf->document_count()
            << ",\"storage_bytes\":" << config.hashed_idf->storage_bytes() << '}';
    }
    out << '}';
    return out.str();
}

EncoderRetrievalRun run_encoder_retrieval(const EncoderConfig& config, const Fixture& fixture,
                                          std::size_t document_block_size,
                                          std::size_t retrieval_depth) {
    using Clock = std::chrono::steady_clock;
    if (document_block_size == 0)
        throw std::runtime_error("document block size must be greater than zero");
    if (retrieval_depth == 0)
        throw std::runtime_error("retrieval depth must be greater than zero");
    if (fixture.query_ids.size() != fixture.query_texts.size() ||
        fixture.doc_ids.size() != fixture.doc_texts.size())
        throw std::runtime_error("fixture ids and texts have different row counts");
    if (fixture.query_ids.empty() || fixture.doc_ids.empty())
        throw std::runtime_error("encoder retrieval requires at least one query and document");
    if (fixture.query_ids.size() > std::numeric_limits<std::uint32_t>::max() ||
        fixture.doc_ids.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("encoder retrieval fixture exceeds uint32 index capacity");

    Encoder encoder(config);
    EncoderRetrievalRun run;
    run.output_dim = encoder.output_dim();
    const std::size_t query_count = fixture.query_texts.size();
    const std::size_t document_count = fixture.doc_texts.size();
    const std::size_t block_capacity = std::min(document_block_size, document_count);
    const std::size_t retained_depth = std::min(retrieval_depth, document_count);

    std::vector<std::string_view> query_views;
    query_views.reserve(query_count);
    for (const auto& text : fixture.query_texts)
        query_views.emplace_back(text);
    std::vector<float> query_vectors(query_count * run.output_dim);
    const auto query_start = Clock::now();
    encoder.encode_batch(query_views, query_vectors.data());
    run.query_encode_us =
        std::chrono::duration<double, std::micro>(Clock::now() - query_start).count();

    run.rankings.resize(query_count);
    for (auto& ranking : run.rankings)
        ranking.reserve(retained_depth);

    std::vector<std::string_view> document_views(block_capacity);
    std::vector<float> document_vectors(block_capacity * run.output_dim);
    for (std::size_t block_start = 0; block_start < document_count; block_start += block_capacity) {
        const std::size_t block_size = std::min(block_capacity, document_count - block_start);
        for (std::size_t i = 0; i < block_size; ++i)
            document_views[i] = fixture.doc_texts[block_start + i];

        const auto encode_start = Clock::now();
        encoder.encode_batch(std::span<const std::string_view>(document_views.data(), block_size),
                             document_vectors.data());
        run.document_encode_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - encode_start).count();

        const auto score_start = Clock::now();
        for (std::size_t query = 0; query < query_count; ++query) {
            const float* query_vector = query_vectors.data() + query * run.output_dim;
            std::size_t local = 0;
            for (; local + 4 <= block_size; local += 4) {
                float scores[4];
                const float* base = document_vectors.data() + local * run.output_dim;
                simd::dot4(query_vector, base, base + run.output_dim, base + 2 * run.output_dim,
                           base + 3 * run.output_dim, scores, run.output_dim);
                for (std::size_t lane = 0; lane < 4; ++lane) {
                    retain_top(
                        run.rankings[query],
                        {scores[lane], static_cast<std::uint32_t>(block_start + local + lane)},
                        retained_depth);
                }
            }
            for (; local < block_size; ++local) {
                const float score = simd::dot(
                    query_vector, document_vectors.data() + local * run.output_dim, run.output_dim);
                retain_top(run.rankings[query],
                           {score, static_cast<std::uint32_t>(block_start + local)},
                           retained_depth);
            }
        }
        run.score_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - score_start).count();
    }

    for (auto& ranking : run.rankings)
        std::sort(ranking.begin(), ranking.end(), ranking_better);

    const std::uint64_t vector_bytes =
        sizeof(float) * (query_count * static_cast<std::uint64_t>(run.output_dim) +
                         block_capacity * static_cast<std::uint64_t>(run.output_dim));
    const std::uint64_t ranking_bytes =
        query_count * static_cast<std::uint64_t>(retained_depth) * sizeof(Ranking::value_type);
    run.peak_working_bytes = vector_bytes + ranking_bytes;
    return run;
}

std::string encoder_result_json(const ResultContext& context, const EncoderConfig& config,
                                const EncoderRetrievalRun& run, const Metrics& metrics,
                                double evaluation_us, std::size_t query_count,
                                std::size_t document_count, std::size_t qrel_count) {
    const double query_us_mean =
        query_count == 0 ? 0.0
                         : (run.query_encode_us + run.score_us) / static_cast<double>(query_count);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << "{\"schema\":\"simeon.experiment.result.v1\""
        << ",\"experiment\":\"" << json_escape(context.experiment) << "\""
        << ",\"variant\":\"" << json_escape(context.variant) << "\""
        << ",\"kind\":\"" << json_escape(context.kind) << "\""
        << ",\"metric_profile\":\"" << json_escape(context.metric_profile) << "\""
        << ",\"provenance\":{\"manifest\":\"" << json_escape(context.manifest_fingerprint)
        << "\",\"fixture\":\"" << json_escape(context.fixture_fingerprint)
        << "\",\"fixture_name\":\"" << json_escape(context.fixture) << "\",\"split\":\""
        << json_escape(context.split) << "\",\"code_revision\":\""
        << json_escape(context.code_revision) << "\",\"simd_tier\":\""
        << json_escape(simd_tier_name(active_simd_tier())) << "\"}"
        << ",\"metadata\":{";
    bool first_metadata = true;
    for (const auto& [key, value] : context.metadata) {
        if (!first_metadata)
            out << ',';
        first_metadata = false;
        out << '"' << json_escape(key) << "\":\"" << json_escape(value) << '"';
    }
    out << "},\"config\":" << encoder_config_json(config)
        << ",\"execution\":{\"document_block_size\":" << context.document_block_size
        << ",\"retrieval_depth\":" << context.retrieval_depth
        << ",\"effective_output_dim\":" << run.output_dim << "}"
        << ",\"counts\":{\"queries\":" << query_count << ",\"documents\":" << document_count
        << ",\"qrels\":" << qrel_count << ",\"evaluated_queries\":" << metrics.evaluated_queries
        << "}"
        << ",\"metrics\":{\"ndcg_at_10\":" << metrics.ndcg_at_10
        << ",\"precision_at_10\":" << metrics.precision_at_10
        << ",\"recall_at_10\":" << metrics.recall_at_10
        << ",\"recall_at_100\":" << metrics.recall_at_100 << ",\"mrr_at_10\":" << metrics.mrr_at_10
        << "}"
        << ",\"timing_us\":{\"artifact_build_total\":" << run.artifact_build_us
        << ",\"query_encode_total\":" << run.query_encode_us
        << ",\"document_encode_total\":" << run.document_encode_us
        << ",\"score_total\":" << run.score_us << ",\"evaluation_total\":" << evaluation_us
        << ",\"query_score_mean\":" << query_us_mean << "}"
        << ",\"resources\":{\"artifact_bytes\":" << run.artifact_bytes
        << ",\"estimated_peak_working_bytes\":" << run.peak_working_bytes + run.artifact_bytes
        << "}}";
    return out.str();
}

std::string json_escape(std::string_view value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0');
    for (const unsigned char c : value) {
        switch (c) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (c < 0x20)
                    out << "\\u" << std::setw(4) << static_cast<unsigned>(c);
                else
                    out << static_cast<char>(c);
        }
    }
    return out.str();
}

} // namespace simeon::experiment
