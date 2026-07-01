// Slim recipe-level accuracy bench over a reference fixture.
//
// Loads a fixture directory (corpus.tsv / queries[_dev].tsv / qrels[_dev].tsv /
// reference[_dev].bin), builds the six production score legs, and evaluates the
// promoted retrieval recipes end-to-end (nDCG@10 et al.) through the *shipped*
// primitives: convex_fuse_z (fusion.hpp), explicit-feedback score_with_prf
// (prf.hpp), and score_fragment_geometry (fragment_geometry.hpp).
//
// Row semantics reproduce the retired research driver's fusion grid: per-leg
// top-K union pool, pool-restricted z-normalization (geometry -inf floored at
// the pool minimum), fixed convex weights. Expected frozen-test anchors are in
// docs/benchmarks.md (fusion pass table).
//
// Optional: SIMEON_WORKBENCH_OUT=<path> dumps one JSONL line per query with the
// union pool, per-pool relevance grades, and each leg's raw pool scores for
// out-of-tree rerank iteration (scripts/rerank_workbench.py).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/fusion.hpp"
#include "simeon/pmi.hpp"
#include "simeon/prf.hpp"
#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Fixture {
    std::vector<std::string> query_ids;
    std::vector<std::string> query_texts;
    std::vector<std::string> doc_ids;
    std::vector<std::string> doc_texts;

    struct Qrel {
        std::uint32_t q;
        std::uint32_t d;
        std::uint32_t rel;
    };
    std::vector<Qrel> qrels;

    std::uint32_t ref_dim = 0;
    std::string ref_model;
    std::vector<float> ref_query_embs;
    std::vector<float> ref_doc_embs;
};

std::vector<std::pair<std::string, std::string>> read_tsv2(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::pair<std::string, std::string>> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos)
            throw std::runtime_error("malformed TSV (missing tab) in " + path);
        out.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    return out;
}

std::vector<std::array<std::string, 3>> read_tsv3(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    std::vector<std::array<std::string, 3>> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        const auto t1 = line.find('\t');
        const auto t2 = line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos)
            throw std::runtime_error("malformed qrels (need 3 cols) in " + path);
        out.push_back({line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1), line.substr(t2 + 1)});
    }
    return out;
}

template <typename T> T read_le(std::istream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in)
        throw std::runtime_error("short read");
    return v;
}

void load_reference_bin(const std::string& path, Fixture& fx) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open " + path);
    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "SIMEONFX", 8) != 0)
        throw std::runtime_error("bad magic in " + path);
    if (read_le<std::uint32_t>(in) != 1u)
        throw std::runtime_error("unsupported reference.bin version");
    fx.ref_dim = read_le<std::uint32_t>(in);
    const std::uint32_t nq = read_le<std::uint32_t>(in);
    const std::uint32_t nd = read_le<std::uint32_t>(in);
    const std::uint32_t mn_len = read_le<std::uint32_t>(in);
    fx.ref_model.resize(mn_len);
    in.read(fx.ref_model.data(), mn_len);
    if (nq != fx.query_ids.size() || nd != fx.doc_ids.size())
        throw std::runtime_error("reference.bin row counts disagree with TSVs");
    if (fx.ref_dim == 0)
        throw std::runtime_error("reference.bin dim must be > 0");
    fx.ref_query_embs.resize(static_cast<std::size_t>(nq) * fx.ref_dim);
    fx.ref_doc_embs.resize(static_cast<std::size_t>(nd) * fx.ref_dim);
    in.read(reinterpret_cast<char*>(fx.ref_query_embs.data()),
            static_cast<std::streamsize>(fx.ref_query_embs.size() * sizeof(float)));
    in.read(reinterpret_cast<char*>(fx.ref_doc_embs.data()),
            static_cast<std::streamsize>(fx.ref_doc_embs.size() * sizeof(float)));
    if (!in)
        throw std::runtime_error("short read on reference.bin payload");
}

Fixture load_fixture(const std::string& dir, const std::string& split) {
    namespace fs = std::filesystem;
    Fixture fx;
    const std::string suffix = (split == "dev") ? "_dev" : "";

    auto corpus = read_tsv2((fs::path(dir) / "corpus.tsv").string());
    fx.doc_ids.reserve(corpus.size());
    fx.doc_texts.reserve(corpus.size());
    for (auto& [id, text] : corpus) {
        fx.doc_ids.push_back(std::move(id));
        fx.doc_texts.push_back(std::move(text));
    }

    auto queries = read_tsv2((fs::path(dir) / ("queries" + suffix + ".tsv")).string());
    fx.query_ids.reserve(queries.size());
    fx.query_texts.reserve(queries.size());
    for (auto& [id, text] : queries) {
        fx.query_ids.push_back(std::move(id));
        fx.query_texts.push_back(std::move(text));
    }

    std::unordered_map<std::string, std::uint32_t> qmap, dmap;
    qmap.reserve(fx.query_ids.size());
    dmap.reserve(fx.doc_ids.size());
    for (std::uint32_t i = 0; i < fx.query_ids.size(); ++i)
        qmap[fx.query_ids[i]] = i;
    for (std::uint32_t i = 0; i < fx.doc_ids.size(); ++i)
        dmap[fx.doc_ids[i]] = i;

    for (const auto& [qid, did, relstr] :
         read_tsv3((fs::path(dir) / ("qrels" + suffix + ".tsv")).string())) {
        const auto qit = qmap.find(qid);
        const auto dit = dmap.find(did);
        if (qit == qmap.end() || dit == dmap.end())
            continue;
        const int rel = std::atoi(relstr.c_str());
        if (rel <= 0)
            continue;
        fx.qrels.push_back({qit->second, dit->second, static_cast<std::uint32_t>(rel)});
    }

    load_reference_bin((fs::path(dir) / ("reference" + suffix + ".bin")).string(), fx);
    return fx;
}

struct Metrics {
    double ndcg_at_10;
    double precision_at_10;
    double recall_at_10;
    double recall_at_100;
    double mrr_at_10;
    std::size_t evaluated_queries;
};

// Per-query: rankings[qi][i] is (score, doc_idx); higher score = better rank.
// Semantics match the retired driver's score_rankings_full (tie-break by doc
// index, grade-weighted DCG, per-query IDCG over the full qrel multiset).
Metrics score_rankings(const std::vector<std::vector<std::pair<float, std::uint32_t>>>& rankings,
                       const Fixture& fx) {
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    double ndcg10_sum = 0.0, p10_sum = 0.0, r10_sum = 0.0, r100_sum = 0.0, mrr10_sum = 0.0;
    std::size_t n_eval = 0;

    for (std::uint32_t qi = 0; qi < rankings.size(); ++qi) {
        auto it = rel.find(qi);
        if (it == rel.end() || it->second.empty())
            continue;
        ++n_eval;
        const auto& qrel = it->second;

        auto sorted = rankings[qi];
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        double dcg = 0.0;
        std::size_t hits10 = 0, hits100 = 0;
        double first_rel_rank = 0.0;
        for (std::size_t r = 0; r < sorted.size(); ++r) {
            const auto rit = qrel.find(sorted[r].second);
            const std::uint32_t g = rit != qrel.end() ? rit->second : 0;
            if (r < 10 && g > 0) {
                dcg += static_cast<double>(g) / std::log2(static_cast<double>(r) + 2.0);
                ++hits10;
                if (first_rel_rank == 0.0)
                    first_rel_rank = static_cast<double>(r) + 1.0;
            }
            if (r < 100 && g > 0)
                ++hits100;
        }

        std::vector<std::uint32_t> rels;
        rels.reserve(qrel.size());
        for (const auto& [_, g] : qrel)
            rels.push_back(g);
        std::sort(rels.begin(), rels.end(), std::greater<>());
        double idcg = 0.0;
        for (std::size_t r = 0; r < rels.size() && r < 10; ++r)
            idcg += static_cast<double>(rels[r]) / std::log2(static_cast<double>(r) + 2.0);

        ndcg10_sum += idcg > 0.0 ? dcg / idcg : 0.0;
        p10_sum += static_cast<double>(hits10) / 10.0;
        r10_sum += static_cast<double>(hits10) /
                   static_cast<double>(std::min<std::size_t>(10, qrel.size()));
        r100_sum += static_cast<double>(hits100) /
                    static_cast<double>(std::min<std::size_t>(100, qrel.size()));
        mrr10_sum += first_rel_rank > 0.0 ? 1.0 / first_rel_rank : 0.0;
    }

    const double n = static_cast<double>(n_eval == 0 ? 1 : n_eval);
    return {ndcg10_sum / n, p10_sum / n, r10_sum / n, r100_sum / n, mrr10_sum / n, n_eval};
}

simeon::Bm25Index build_bm25(const Fixture& fx, simeon::Bm25Config cfg) {
    simeon::Bm25Index idx(cfg);
    idx.reserve_docs(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        idx.add_doc(d);
    idx.finalize();
    return idx;
}

void print_row(const char* name, const char* split, const Metrics& m, double query_us_mean) {
    std::printf("{\"row\":\"%s\",\"split\":\"%s\",\"ndcg_at_10\":%.4f,\"precision_at_10\":%.4f,"
                "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,\"mrr_at_10\":%.4f,"
                "\"evaluated_queries\":%zu,\"query_us_mean\":%.1f}\n",
                name, split, m.ndcg_at_10, m.precision_at_10, m.recall_at_10, m.recall_at_100,
                m.mrr_at_10, m.evaluated_queries, query_us_mean);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: recipe_accuracy_bench <fixture_dir> [split=test|dev] "
                             "[pool_per_leg=100] [mode=default|spectral]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const std::string split = argc > 2 ? argv[2] : "test";
    const std::uint32_t pool_per_leg =
        argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 100u;
    // spectral mode swaps the csls/fusion geometry variants for a
    // temper_spectrum(alpha) sweep with per-query whitening off.
    const bool spectral_mode = argc > 4 && std::string_view(argv[4]) == "spectral";

    const Fixture fx = load_fixture(dir, split);
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const std::uint32_t nq = static_cast<std::uint32_t>(fx.query_ids.size());
    std::fprintf(stderr, "[recipe] %s split=%s docs=%u queries=%u qrels=%zu ref=%s dim=%u\n",
                 dir.c_str(), split.c_str(), nd, nq, fx.qrels.size(), fx.ref_model.c_str(),
                 fx.ref_dim);

    // ---- Score legs (mirrors the retired fusion grid's production set).
    simeon::Bm25Config cfg_atire;
    cfg_atire.build_word_bigrams = true;
    const auto atire = build_bm25(fx, cfg_atire);
    simeon::Bm25Config cfg_sab;
    cfg_sab.variant = simeon::Bm25Variant::SubwordAwareBackoff;
    cfg_sab.delta = 1.0f;
    cfg_sab.subword_gamma = 5.0f;
    cfg_sab.build_word_bigrams = true;
    const auto sab = build_bm25(fx, cfg_sab);
    simeon::Bm25Config cfg_plus;
    cfg_plus.variant = simeon::Bm25Variant::BM25Plus;
    const auto plus = build_bm25(fx, cfg_plus);
    simeon::Bm25Config cfg_l;
    cfg_l.variant = simeon::Bm25Variant::BM25L;
    const auto bl = build_bm25(fx, cfg_l);
    simeon::Bm25Config cfg_dlh;
    cfg_dlh.variant = simeon::Bm25Variant::DLH13;
    const auto dlh = build_bm25(fx, cfg_dlh);
    const std::array<const simeon::Bm25Index*, 5> rrf5_set{&atire, &plus, &bl, &dlh, &sab};
    const simeon::WeightedSdmConfig wsdm_cfg; // beta = 1.0, published recipe

    // Fragment-geometry leg: PMI rank-128 encoder, richcov fragments, pure
    // geometry (alpha = 0), PHSS LargestGapApprox — the fusion grid's recipe.
    std::vector<std::string_view> seed_views;
    seed_views.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        seed_views.emplace_back(d);
    simeon::PmiConfig pcfg;
    pcfg.target_rank = 128;
    pcfg.min_token_count = 5;
    pcfg.max_vocab_size = 20'000;
    const auto pmi =
        simeon::PmiEmbeddings::learn(std::span<const std::string_view>(seed_views), pcfg);
    simeon::EncoderConfig ecfg;
    ecfg.ngram_mode = simeon::NGramMode::WordOnly;
    ecfg.ngram_min = 1;
    ecfg.ngram_max = 1;
    ecfg.sketch_dim = 0;
    ecfg.output_dim = pmi.dim();
    ecfg.projection = simeon::ProjectionMode::None;
    ecfg.l2_normalize = true;
    ecfg.pmi_rows = &pmi;
    const simeon::Encoder enc(ecfg);

    // Spectral-mode variants: tempered copies of the PMI rows, whiten off at
    // scoring time. alpha=0 isolates whiten-off vs the whiten-on geom_pure row.
    static constexpr float kTemperAlphas[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    constexpr std::size_t kNumTemper = std::size(kTemperAlphas);
    std::vector<simeon::PmiEmbeddings> temper_pmi;
    std::vector<simeon::Encoder> temper_enc;
    std::vector<std::vector<std::vector<simeon::SemanticFragment>>> temper_frags;
    if (spectral_mode) {
        temper_pmi.reserve(kNumTemper);
        temper_enc.reserve(kNumTemper);
        temper_frags.assign(kNumTemper, std::vector<std::vector<simeon::SemanticFragment>>(nd));
        for (std::size_t v = 0; v < kNumTemper; ++v) {
            temper_pmi.push_back(pmi);
            temper_pmi[v].temper_spectrum(kTemperAlphas[v]);
        }
        for (std::size_t v = 0; v < kNumTemper; ++v) {
            simeon::EncoderConfig vecfg = ecfg;
            vecfg.pmi_rows = &temper_pmi[v];
            temper_enc.emplace_back(vecfg);
        }
    }

    std::vector<std::vector<simeon::SemanticFragment>> frags(nd);
    for (std::size_t i = 0; i < nd; ++i) {
        const auto prep = simeon::prepare_doc(fx.doc_texts[i], atire, 6, 8, 0.0f);
        frags[i] = simeon::build_doc_semantic_fragments_rich_covered_from_prep(enc, fx.doc_texts[i],
                                                                               prep, 0.60f, 0.80f);
        for (std::size_t v = 0; v < temper_enc.size(); ++v) {
            temper_frags[v][i] = simeon::build_doc_semantic_fragments_rich_covered_from_prep(
                temper_enc[v], fx.doc_texts[i], prep, 0.60f, 0.80f);
        }
    }
    simeon::compress_fragments_to_bf16(frags, enc.output_dim());
    for (std::size_t v = 0; v < temper_enc.size(); ++v)
        simeon::compress_fragments_to_bf16(temper_frags[v], temper_enc[v].output_dim());

    // Query-independent per-doc fragment-topology features: over each doc's
    // own fragments (first 8, richcov ranking order), the largest adjacent
    // gap in the sorted pairwise-cosine list (0-D persistence, the PHSS
    // statistic applied per doc) and the mean pairwise cosine (coherence).
    std::vector<float> doc_topo_gap(nd, 0.0f), doc_topo_coh(nd, 0.0f);
    {
        const std::uint32_t fdim = enc.output_dim();
        std::vector<float> fv;
        std::vector<float> sims;
        for (std::size_t i = 0; i < nd; ++i) {
            const auto& df = frags[i];
            const std::size_t k = std::min<std::size_t>(df.size(), 8);
            if (k < 2)
                continue;
            fv.resize(k * fdim);
            for (std::size_t f = 0; f < k; ++f)
                simeon::read_frag_vec(df[f], fdim, fv.data() + f * fdim);
            sims.clear();
            for (std::size_t a = 0; a + 1 < k; ++a)
                for (std::size_t b = a + 1; b < k; ++b)
                    sims.push_back(
                        simeon::simd::dot(fv.data() + a * fdim, fv.data() + b * fdim, fdim));
            float sum = 0.0f;
            for (float s : sims)
                sum += s;
            doc_topo_coh[i] = sum / static_cast<float>(sims.size());
            std::sort(sims.begin(), sims.end());
            float gap = 0.0f;
            for (std::size_t s = 0; s + 1 < sims.size(); ++s)
                gap = std::max(gap, sims[s + 1] - sims[s]);
            doc_topo_gap[i] = gap;
        }
    }
    simeon::FragmentGeometryConfig gcfg;
    gcfg.pool_size = 100;
    gcfg.alpha = 0.0f;
    gcfg.top_fragments_per_doc = 8;
    gcfg.attention_scale = 8.0f;
    gcfg.knn = 8;
    gcfg.steps = 2;
    gcfg.use_phss = true;
    gcfg.phss_config.criterion = simeon::PhssConfig::Criterion::LargestGapApprox;
    std::fprintf(stderr, "[recipe] legs ready (pmi dim=%u)\n", enc.output_dim());

    enum Leg : std::uint8_t { L_ATIRE = 0, L_WSDM_AT, L_SAB, L_WSDM_SAB, L_GEOM, L_RRF5, N_LEGS };
    static constexpr const char* kLegName[N_LEGS] = {"atire",   "wsdmat", "sab",
                                                     "wsdmsab", "geom",   "rrf5"};

    // ---- Rows. Full-corpus baselines + pool-restricted fusion recipes.
    enum RowId : std::uint8_t {
        R_BM25_ATIRE = 0,
        R_BM25_SAB,
        R_WSDM_AT,
        R_WSDM_SAB,
        R_GEOM,
        R_GEOM_CSLS8_B10, // hubness-corrected geometry variants (research)
        R_GEOM_CSLS8_B05,
        R_GEOM_CSLS16_B10,
        R_CC,    // promoted: 0.6·z(wsdm_sab) + 0.4·z(wsdm_at)
        R_CCPRF, // promoted ⊕ 0.3·z(prf_fused)
        R_CCG10, // promoted ⊕ g·z(geom_csls_k8_b1.0), g = 0.10/0.20/0.30
        R_CCG20,
        R_CCG30,
        R_GEOM_T000, // spectral mode: temper_spectrum(alpha), whiten off
        R_GEOM_T025,
        R_GEOM_T050,
        R_GEOM_T075,
        R_GEOM_T100,
        R_CC_MINMAX,    // promoted weights, observed min-max normalization per leg
        R_CCTOPO_GAP10, // promoted ⊕ w·z(doc fragment-topology feature)
        R_CCTOPO_GAP20,
        R_CCTOPO_COH10,
        R_CCTOPO_COH20,

        R_MINILM,
        R_POOL_ORACLE,
        N_ROWS
    };
    static constexpr const char* kRowName[N_ROWS] = {"bm25_atire",
                                                     "bm25_sab",
                                                     "wsdm_at",
                                                     "wsdm_sab",
                                                     "geom_pure",
                                                     "geom_csls_k8_b1.0",
                                                     "geom_csls_k8_b0.5",
                                                     "geom_csls_k16_b1.0",
                                                     "fusion_cc_wsdmsab0.60_wsdmat0.40",
                                                     "fusion_ccprf_wsdmsab0.60_wsdmat0.40_pf0.30",
                                                     "fusion_ccgeomcsls_g0.10",
                                                     "fusion_ccgeomcsls_g0.20",
                                                     "fusion_ccgeomcsls_g0.30",
                                                     "geom_temper_a0.00_nowhiten",
                                                     "geom_temper_a0.25_nowhiten",
                                                     "geom_temper_a0.50_nowhiten",
                                                     "geom_temper_a0.75_nowhiten",
                                                     "geom_temper_a1.00_nowhiten",
                                                     "fusion_ccminmax_wsdmsab0.60_wsdmat0.40",
                                                     "fusion_cctopo_gap_w0.10",
                                                     "fusion_cctopo_gap_w0.20",
                                                     "fusion_cctopo_coh_w0.10",
                                                     "fusion_cctopo_coh_w0.20",
                                                     "reference_dense",
                                                     "pool_oracle"};
    static_assert(R_GEOM_T100 - R_GEOM_T000 + 1 == kNumTemper);

    struct CslsVariant {
        RowId row;
        std::uint32_t k;
        float beta;
    };
    static constexpr CslsVariant kCslsVariants[] = {
        {R_GEOM_CSLS8_B10, 8, 1.0f}, {R_GEOM_CSLS8_B05, 8, 0.5f}, {R_GEOM_CSLS16_B10, 16, 1.0f}};

    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    FILE* wb_fp = nullptr;
    if (const char* wb_path = std::getenv("SIMEON_WORKBENCH_OUT")) {
        wb_fp = std::fopen(wb_path, "w");
        if (wb_fp == nullptr)
            std::fprintf(stderr, "[recipe] cannot open workbench path %s\n", wb_path);
    }

    std::vector<std::vector<std::vector<std::pair<float, std::uint32_t>>>> row_rankings(
        N_ROWS, std::vector<std::vector<std::pair<float, std::uint32_t>>>(nq));
    std::array<std::vector<float>, N_LEGS> leg_scores;
    for (auto& v : leg_scores)
        v.assign(nd, 0.0f);
    std::array<std::vector<std::pair<std::uint32_t, float>>, N_LEGS> leg_topk;

    const auto t_q0 = Clock::now();
    for (std::uint32_t qi = 0; qi < nq; ++qi) {
        const auto& query = fx.query_texts[qi];
        atire.score(query, leg_scores[L_ATIRE]);
        atire.score_wsdm(query, leg_scores[L_WSDM_AT], wsdm_cfg);
        sab.score(query, leg_scores[L_SAB]);
        sab.score_wsdm(query, leg_scores[L_WSDM_SAB], wsdm_cfg);
        leg_scores[L_GEOM] = simeon::score_fragment_geometry(query, atire, enc, frags, gcfg);
        std::fill(leg_scores[L_RRF5].begin(), leg_scores[L_RRF5].end(), 0.0f);
        simeon::score_bm25_variants_rrf(std::span<const simeon::Bm25Index* const>(rrf5_set), query,
                                        leg_scores[L_RRF5]);

        // Union pool over per-leg top-K.
        std::vector<std::uint32_t> pool;
        {
            std::unordered_set<std::uint32_t> seen;
            for (int leg = 0; leg < N_LEGS; ++leg) {
                leg_topk[leg] = simeon::top_k(leg_scores[leg], pool_per_leg);
                for (const auto& [did, _s] : leg_topk[leg])
                    if (seen.insert(did).second)
                        pool.push_back(did);
            }
        }
        const std::size_t np = pool.size();

        // Raw pool-restricted legs; geometry's -inf (outside its internal BM25
        // pool) floored at the leg's pool minimum finite score.
        std::array<std::vector<float>, N_LEGS> pooled;
        for (int leg = 0; leg < N_LEGS; ++leg) {
            auto& pl = pooled[leg];
            pl.resize(np);
            for (std::size_t p = 0; p < np; ++p)
                pl[p] = leg_scores[leg][pool[p]];
            if (leg == L_GEOM) {
                float mn = std::numeric_limits<float>::infinity();
                for (float v : pl)
                    if (std::isfinite(v))
                        mn = std::min(mn, v);
                if (!std::isfinite(mn))
                    mn = 0.0f;
                for (float& v : pl)
                    if (!std::isfinite(v))
                        v = mn;
            }
        }

        // Promoted convex fusion via the shipped primitive.
        std::vector<float> cc(np, 0.0f);
        {
            const std::array<std::span<const float>, 2> legs{pooled[L_WSDM_SAB], pooled[L_WSDM_AT]};
            const std::array<float, 2> w{0.6f, 0.4f};
            simeon::convex_fuse_z(std::span<const std::span<const float>>(legs),
                                  std::span<const float>(w), cc);
        }

        // prf_fused: RM3 anchored on the promoted fusion's top-10 with
        // softmax feedback weights (explicit-feedback score_with_prf).
        std::vector<float> prf_fused_raw(np, 0.0f);
        {
            std::vector<std::size_t> ord(np);
            std::iota(ord.begin(), ord.end(), std::size_t{0});
            const std::size_t kfb = std::min<std::size_t>(10, np);
            std::partial_sort(ord.begin(), ord.begin() + kfb, ord.end(),
                              [&](std::size_t a, std::size_t b) {
                                  if (cc[a] != cc[b])
                                      return cc[a] > cc[b];
                                  return pool[a] < pool[b];
                              });
            std::vector<std::uint32_t> fb_ids(kfb);
            std::vector<float> fb_w(kfb);
            const float fmax = kfb > 0 ? cc[ord[0]] : 0.0f;
            for (std::size_t i = 0; i < kfb; ++i) {
                fb_ids[i] = pool[ord[i]];
                fb_w[i] = std::exp(cc[ord[i]] - fmax);
            }
            std::vector<float> full(nd, 0.0f);
            simeon::score_with_prf(atire, query, fb_ids, fb_w, full, simeon::PrfConfig{});
            for (std::size_t p = 0; p < np; ++p)
                prf_fused_raw[p] = full[pool[p]];
        }

        // ccprf: 0.3·z(prf_fused) + 0.7·(0.6·z(wsdm_sab) + 0.4·z(wsdm_at)),
        // expressed as one convex_fuse_z over three raw legs.
        std::vector<float> ccprf(np, 0.0f);
        {
            const std::array<std::span<const float>, 3> legs{prf_fused_raw, pooled[L_WSDM_SAB],
                                                             pooled[L_WSDM_AT]};
            const std::array<float, 3> w{0.30f, 0.42f, 0.28f};
            simeon::convex_fuse_z(std::span<const std::span<const float>>(legs),
                                  std::span<const float>(w), ccprf);
        }

        if (wb_fp != nullptr) {
            const auto rit = rel.find(qi);
            std::fprintf(wb_fp, "{\"qid\":\"%s\",\"pool\":[", fx.query_ids[qi].c_str());
            for (std::size_t p = 0; p < np; ++p)
                std::fprintf(wb_fp, "%s%u", p ? "," : "", pool[p]);
            std::fprintf(wb_fp, "],\"rel\":[");
            for (std::size_t p = 0; p < np; ++p) {
                std::uint32_t g = 0;
                if (rit != rel.end()) {
                    const auto qit = rit->second.find(pool[p]);
                    if (qit != rit->second.end())
                        g = qit->second;
                }
                std::fprintf(wb_fp, "%s%u", p ? "," : "", g);
            }
            std::fprintf(wb_fp, "],\"all_rels\":[");
            if (rit != rel.end()) {
                bool first = true;
                for (const auto& [_, g] : rit->second) {
                    std::fprintf(wb_fp, "%s%u", first ? "" : ",", g);
                    first = false;
                }
            }
            std::fprintf(wb_fp, "],\"legs\":{");
            for (int leg = 0; leg < N_LEGS; ++leg) {
                std::fprintf(wb_fp, "%s\"%s\":[", leg ? "," : "", kLegName[leg]);
                for (std::size_t p = 0; p < np; ++p) {
                    const float v = leg_scores[leg][pool[p]];
                    std::fprintf(wb_fp, "%s%.6g", p ? "," : "",
                                 std::isfinite(v) ? static_cast<double>(v) : -1e30);
                }
                std::fprintf(wb_fp, "]");
            }
            std::fprintf(wb_fp, "}}\n");
        }

        // ---- Row rankings.
        const auto full_corpus_row = [&](RowId row, const std::vector<float>& s) {
            auto& out = row_rankings[row][qi];
            out.reserve(nd);
            for (std::uint32_t d = 0; d < nd; ++d)
                if (std::isfinite(s[d]))
                    out.emplace_back(s[d], d);
        };
        full_corpus_row(R_BM25_ATIRE, leg_scores[L_ATIRE]);
        full_corpus_row(R_BM25_SAB, leg_scores[L_SAB]);
        full_corpus_row(R_WSDM_AT, leg_scores[L_WSDM_AT]);
        full_corpus_row(R_WSDM_SAB, leg_scores[L_WSDM_SAB]);

        const auto pool_row = [&](RowId row, const std::vector<float>& s) {
            auto& out = row_rankings[row][qi];
            out.reserve(np);
            for (std::size_t p = 0; p < np; ++p)
                out.emplace_back(s[p], pool[p]);
        };
        pool_row(R_GEOM, pooled[L_GEOM]);

        // Pool-restrict a full-corpus geometry score vector, flooring -inf
        // (outside the leg's internal BM25 pool) at the pool minimum.
        const auto pool_restrict_geom = [&](const std::vector<float>& vs) {
            std::vector<float> vp(np);
            float mn = std::numeric_limits<float>::infinity();
            for (std::size_t p = 0; p < np; ++p) {
                vp[p] = vs[pool[p]];
                if (std::isfinite(vp[p]))
                    mn = std::min(mn, vp[p]);
            }
            if (!std::isfinite(mn))
                mn = 0.0f;
            for (float& x : vp)
                if (!std::isfinite(x))
                    x = mn;
            return vp;
        };

        // Hubness-corrected geometry variants: same recipe, csls knobs on.
        // Pool membership stays fixed (variants are rerank rows, not pool
        // contributors); -inf handling matches the geom leg.
        if (!spectral_mode) {
            std::vector<float> geom_csls_k8; // kept for the fusion rows below
            for (const auto& v : kCslsVariants) {
                simeon::FragmentGeometryConfig vcfg = gcfg;
                vcfg.csls_k = v.k;
                vcfg.csls_beta = v.beta;
                auto vp = pool_restrict_geom(
                    simeon::score_fragment_geometry(query, atire, enc, frags, vcfg));
                pool_row(v.row, vp);
                if (v.row == R_GEOM_CSLS8_B10)
                    geom_csls_k8 = std::move(vp);
            }

            // Promoted WSDM pair ⊕ hubness-corrected geometry leg.
            const std::array<RowId, 3> ccg_rows{R_CCG10, R_CCG20, R_CCG30};
            const std::array<float, 3> gws{0.10f, 0.20f, 0.30f};
            for (std::size_t gi = 0; gi < gws.size(); ++gi) {
                const float g = gws[gi];
                std::vector<float> fused(np, 0.0f);
                const std::array<std::span<const float>, 3> legs{geom_csls_k8, pooled[L_WSDM_SAB],
                                                                 pooled[L_WSDM_AT]};
                const std::array<float, 3> w{g, 0.6f * (1.0f - g), 0.4f * (1.0f - g)};
                simeon::convex_fuse_z(std::span<const std::span<const float>>(legs),
                                      std::span<const float>(w), fused);
                pool_row(ccg_rows[gi], fused);
            }
        } else {
            // Spectral sweep: tempered PMI coordinates, whiten off.
            simeon::FragmentGeometryConfig nw_cfg = gcfg;
            nw_cfg.whiten = false;
            for (std::size_t v = 0; v < kNumTemper; ++v) {
                const auto vp = pool_restrict_geom(simeon::score_fragment_geometry(
                    query, atire, temper_enc[v], temper_frags[v], nw_cfg));
                pool_row(static_cast<RowId>(R_GEOM_T000 + v), vp);
            }
        }

        pool_row(R_CC, cc);
        pool_row(R_CCPRF, ccprf);

        // Min-max ablation of the promoted fusion (Bruch-Gai 2022 compare
        // normalization schemes; z was adopted without testing min-max).
        {
            std::vector<float> mm(np, 0.0f);
            const std::array<const std::vector<float>*, 2> legs{&pooled[L_WSDM_SAB],
                                                                &pooled[L_WSDM_AT]};
            const std::array<float, 2> w{0.6f, 0.4f};
            for (std::size_t l = 0; l < legs.size(); ++l) {
                float mn = std::numeric_limits<float>::infinity();
                float mx = -std::numeric_limits<float>::infinity();
                for (float x : *legs[l]) {
                    mn = std::min(mn, x);
                    mx = std::max(mx, x);
                }
                const float span = mx - mn;
                if (span <= 0.0f)
                    continue;
                for (std::size_t p = 0; p < np; ++p)
                    mm[p] += w[l] * (((*legs[l])[p] - mn) / span);
            }
            pool_row(R_CC_MINMAX, mm);
        }

        // Promoted fusion ⊕ per-doc fragment-topology feature legs.
        {
            std::vector<float> gap_pool(np), coh_pool(np);
            for (std::size_t p = 0; p < np; ++p) {
                gap_pool[p] = doc_topo_gap[pool[p]];
                coh_pool[p] = doc_topo_coh[pool[p]];
            }
            const struct {
                RowId row;
                const std::vector<float>* feat;
                float w;
            } topo_rows[] = {{R_CCTOPO_GAP10, &gap_pool, 0.10f},
                             {R_CCTOPO_GAP20, &gap_pool, 0.20f},
                             {R_CCTOPO_COH10, &coh_pool, 0.10f},
                             {R_CCTOPO_COH20, &coh_pool, 0.20f}};
            for (const auto& t : topo_rows) {
                std::vector<float> fused(np, 0.0f);
                const std::array<std::span<const float>, 3> legs{*t.feat, pooled[L_WSDM_SAB],
                                                                 pooled[L_WSDM_AT]};
                const std::array<float, 3> w{t.w, 0.6f * (1.0f - t.w), 0.4f * (1.0f - t.w)};
                simeon::convex_fuse_z(std::span<const std::span<const float>>(legs),
                                      std::span<const float>(w), fused);
                pool_row(t.row, fused);
            }
        }

        {
            auto& out = row_rankings[R_MINILM][qi];
            out.reserve(nd);
            const float* qv = fx.ref_query_embs.data() + static_cast<std::size_t>(qi) * fx.ref_dim;
            for (std::uint32_t d = 0; d < nd; ++d) {
                const float* dv = fx.ref_doc_embs.data() + static_cast<std::size_t>(d) * fx.ref_dim;
                out.emplace_back(simeon::simd::dot(qv, dv, fx.ref_dim), d);
            }
        }

        {
            auto& out = row_rankings[R_POOL_ORACLE][qi];
            out.reserve(np);
            const auto rit = rel.find(qi);
            for (std::size_t p = 0; p < np; ++p) {
                float g = 0.0f;
                if (rit != rel.end()) {
                    const auto qit = rit->second.find(pool[p]);
                    if (qit != rit->second.end())
                        g = static_cast<float>(qit->second);
                }
                out.emplace_back(g, pool[p]);
            }
        }
    }
    const double query_us_mean =
        std::chrono::duration<double, std::micro>(Clock::now() - t_q0).count() /
        static_cast<double>(nq == 0 ? 1 : nq);

    if (wb_fp != nullptr)
        std::fclose(wb_fp);

    for (int r = 0; r < N_ROWS; ++r) {
        // Rows not populated in this mode (csls/fusion vs spectral) stay empty.
        const bool active = std::any_of(row_rankings[r].begin(), row_rankings[r].end(),
                                        [](const auto& ranking) { return !ranking.empty(); });
        if (!active)
            continue;
        const Metrics m = score_rankings(row_rankings[r], fx);
        print_row(kRowName[r], split.c_str(), m, query_us_mean);
    }
    return 0;
}
