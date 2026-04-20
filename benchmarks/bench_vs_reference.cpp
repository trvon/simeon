// Reference embedding comparison bench.
//
// Loads a frozen IR fixture (corpus + queries + qrels + pre-computed reference
// embeddings) and evaluates simeon configurations against the reference model
// on the same workload. Emits one JSONL record per configuration.
//
// Fixture format documented in docs/reference_fixture.md.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/fusion.hpp"
#include "simeon/matryoshka.hpp"
#include "simeon/minhash.hpp"
#include "simeon/pmi.hpp"
#include "simeon/pq.hpp"
#include "simeon/prf.hpp"
#include "simeon/query_router.hpp"
#include "simeon/simd.hpp"
#include "simeon/simeon.hpp"

namespace {

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

// Optional per-query routing telemetry. When non-null, router-grid and
// oracle-router rows append one JSONL line per (config, query) describing the
// chosen recipe, query features, and per-query metrics. Lets post-hoc analysis
// run without re-executing the bench.
FILE* g_router_per_query_fp = nullptr;

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
        if (tab == std::string::npos) {
            throw std::runtime_error("malformed TSV (missing tab) in " + path);
        }
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
        if (t1 == std::string::npos || t2 == std::string::npos) {
            throw std::runtime_error("malformed qrels (need 3 cols) in " + path);
        }
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
    if (std::memcmp(magic, "SIMEONFX", 8) != 0) {
        throw std::runtime_error("bad magic in " + path);
    }
    const std::uint32_t version = read_le<std::uint32_t>(in);
    if (version != 1u) {
        throw std::runtime_error("unsupported reference.bin version");
    }
    fx.ref_dim = read_le<std::uint32_t>(in);
    const std::uint32_t nq = read_le<std::uint32_t>(in);
    const std::uint32_t nd = read_le<std::uint32_t>(in);
    const std::uint32_t mn_len = read_le<std::uint32_t>(in);
    fx.ref_model.resize(mn_len);
    in.read(fx.ref_model.data(), mn_len);

    if (nq != fx.query_ids.size() || nd != fx.doc_ids.size()) {
        throw std::runtime_error("reference.bin row counts disagree with TSVs");
    }
    if (fx.ref_dim == 0) {
        throw std::runtime_error("reference.bin dim must be > 0");
    }
    fx.ref_query_embs.resize(static_cast<std::size_t>(nq) * fx.ref_dim);
    fx.ref_doc_embs.resize(static_cast<std::size_t>(nd) * fx.ref_dim);
    in.read(reinterpret_cast<char*>(fx.ref_query_embs.data()),
            fx.ref_query_embs.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(fx.ref_doc_embs.data()),
            fx.ref_doc_embs.size() * sizeof(float));
    if (!in)
        throw std::runtime_error("short read on reference.bin payload");
}

Fixture load_fixture(const std::string& dir, const std::string& split = "test") {
    namespace fs = std::filesystem;
    Fixture fx;

    const std::string suffix = (split == "dev") ? "_dev" : "";
    const std::string queries_name = "queries" + suffix + ".tsv";
    const std::string qrels_name = "qrels" + suffix + ".tsv";
    const std::string ref_name = "reference" + suffix + ".bin";

    auto corpus = read_tsv2((fs::path(dir) / "corpus.tsv").string());
    fx.doc_ids.reserve(corpus.size());
    fx.doc_texts.reserve(corpus.size());
    for (auto& [id, text] : corpus) {
        fx.doc_ids.push_back(std::move(id));
        fx.doc_texts.push_back(std::move(text));
    }

    auto queries = read_tsv2((fs::path(dir) / queries_name).string());
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

    auto qrels = read_tsv3((fs::path(dir) / qrels_name).string());
    fx.qrels.reserve(qrels.size());
    for (const auto& [qid, did, relstr] : qrels) {
        const auto qit = qmap.find(qid);
        const auto dit = dmap.find(did);
        if (qit == qmap.end() || dit == dmap.end())
            continue;
        const int rel = std::atoi(relstr.c_str());
        if (rel <= 0)
            continue;
        fx.qrels.push_back({qit->second, dit->second, static_cast<std::uint32_t>(rel)});
    }

    load_reference_bin((fs::path(dir) / ref_name).string(), fx);
    return fx;
}

// Routes through the SIMD-dispatched dot kernel so the bench's per-query
// scoring measures the same code path retrieval users hit at runtime.
inline float dot(const float* a, const float* b, std::size_t n) {
    return simeon::simd::dot(a, b, static_cast<std::uint32_t>(n));
}

struct Metrics {
    double ndcg_at_10;
    double recall_at_10;
    double recall_at_100;
    double mrr_at_10;
    std::size_t evaluated_queries; // queries with ≥1 relevant doc
};

// Per-query: scored[i] is (score, doc_idx). Higher score = better rank.
Metrics score_rankings(const std::vector<std::vector<std::pair<float, std::uint32_t>>>& rankings,
                       const Fixture& fx) {
    // Bucket qrels by query.
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel;
    for (const auto& q : fx.qrels)
        rel[q.q][q.d] = q.rel;

    double ndcg10_sum = 0.0, r10_sum = 0.0, r100_sum = 0.0, mrr10_sum = 0.0;
    std::size_t n_eval = 0;

    for (std::uint32_t qi = 0; qi < rankings.size(); ++qi) {
        auto it = rel.find(qi);
        if (it == rel.end() || it->second.empty())
            continue;
        ++n_eval;
        const auto& qrel = it->second;

        // Sort by score descending; tie-break by doc index for determinism.
        auto sorted = rankings[qi];
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first)
                return a.first > b.first;
            return a.second < b.second;
        });

        // DCG@10
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

        // IDCG@10
        std::vector<std::uint32_t> rels;
        rels.reserve(qrel.size());
        for (const auto& [_, g] : qrel)
            rels.push_back(g);
        std::sort(rels.begin(), rels.end(), std::greater<>());
        double idcg = 0.0;
        for (std::size_t r = 0; r < rels.size() && r < 10; ++r) {
            idcg += static_cast<double>(rels[r]) / std::log2(static_cast<double>(r) + 2.0);
        }
        ndcg10_sum += idcg > 0.0 ? dcg / idcg : 0.0;

        const double denom10 = static_cast<double>(std::min<std::size_t>(10, qrel.size()));
        const double denom100 = static_cast<double>(std::min<std::size_t>(100, qrel.size()));
        r10_sum += static_cast<double>(hits10) / denom10;
        r100_sum += static_cast<double>(hits100) / denom100;
        mrr10_sum += first_rel_rank > 0.0 ? 1.0 / first_rel_rank : 0.0;
    }

    const double n = static_cast<double>(n_eval == 0 ? 1 : n_eval);
    return {ndcg10_sum / n, r10_sum / n, r100_sum / n, mrr10_sum / n, n_eval};
}

// Optional per-config timings. Zero means "not measured" — emit() omits the field.
struct Timing {
    double doc_encode_us = 0.0;  // total simeon doc-encode time (sum over corpus)
    double index_build_us = 0.0; // BM25/PQ build/finalize/train time
    double query_us = 0.0; // total per-query work (encode + score + rerank), summed across queries
};

void emit(const char* config_name, const Fixture& fx, const Metrics& m,
          std::uint32_t code_bytes_per_doc, const Timing& t = {}) {
    const double nd = static_cast<double>(fx.doc_ids.size());
    const double nq = static_cast<double>(fx.query_ids.size());
    const double encode_us_per_doc = t.doc_encode_us > 0 ? t.doc_encode_us / nd : 0.0;
    const double query_us_per_q = t.query_us > 0 ? t.query_us / nq : 0.0;
    const double encode_dps = encode_us_per_doc > 0 ? 1.0e6 / encode_us_per_doc : 0.0;
    const double query_qps = query_us_per_q > 0 ? 1.0e6 / query_us_per_q : 0.0;

    std::printf("{\"bench\":\"vs_reference\",\"config\":\"%s\",\"model\":\"%s\","
                "\"queries\":%zu,\"docs\":%zu,\"evaluated_queries\":%zu,"
                "\"code_bytes_per_doc\":%u,"
                "\"ndcg_at_10\":%.4f,\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                "\"mrr_at_10\":%.4f,"
                "\"encode_us_per_doc\":%.3f,\"query_us_per_q\":%.3f,"
                "\"encode_throughput_dps\":%.1f,\"query_throughput_qps\":%.1f,"
                "\"index_build_us\":%.1f,"
                "\"simd_tier\":\"%s\"}\n",
                config_name, fx.ref_model.c_str(), fx.query_ids.size(), fx.doc_ids.size(),
                m.evaluated_queries, code_bytes_per_doc, m.ndcg_at_10, m.recall_at_10,
                m.recall_at_100, m.mrr_at_10, encode_us_per_doc, query_us_per_q, encode_dps,
                query_qps, t.index_build_us, simeon::simd_tier_name(simeon::active_simd_tier()));
    std::fflush(stdout);
}

using Clock = std::chrono::steady_clock;
inline double elapsed_us(Clock::time_point t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

void run_simeon(const char* name, simeon::EncoderConfig cfg, const Fixture& fx) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    std::vector<float> dembs(static_cast<std::size_t>(fx.doc_ids.size()) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < fx.doc_ids.size(); ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        enc.encode(fx.query_texts[qi], qembs.data() + qi * dim);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(dot(qembs.data() + qi * dim, dembs.data() + di * dim, dim),
                                      di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
}

void run_simeon_pq(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                   std::uint32_t pq_m) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    simeon::PQConfig pcfg{.dim = dim, .m = pq_m, .k = 256, .seed = 0xBEEF1234ULL};
    simeon::ProductQuantizer pq(pcfg);
    t0 = Clock::now();
    pq.train(dembs.data(), nd, 15);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(nd) * pq_m, 0);
    pq.encode_batch(dembs.data(), nd, codes.data());
    t.index_build_us = elapsed_us(t0);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::PQQuery pq_q(pq, qembs.data() + qi * dim);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(pq_q.inner_product(codes.data() + di * pq_m), di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), pq_m, t);
}

void run_reference(const Fixture& fx) {
    Timing t;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(dot(fx.ref_query_embs.data() + qi * fx.ref_dim,
                                          fx.ref_doc_embs.data() + di * fx.ref_dim, fx.ref_dim),
                                      di);
        }
    }
    t.query_us = elapsed_us(t0);
    // Note: doc_encode_us is intentionally 0 here — reference-model encoding is
    // measured out of process so the benchmark can compare against a realistic
    // CPU throughput number without adding Python/ML dependencies to simeon.
    emit("reference", fx, score_rankings(rankings, fx),
         static_cast<std::uint32_t>(fx.ref_dim * sizeof(float)), t);
}

// Build a BM25 index over fx.doc_texts. Returns the index and the build time;
// caller reuses both across multiple bench rows so we only pay indexing once.
struct Bm25WithTiming {
    simeon::Bm25Index idx;
    double build_us = 0.0;
};

Bm25WithTiming build_bm25(const Fixture& fx, simeon::Bm25Config cfg = {}) {
    Bm25WithTiming out{simeon::Bm25Index(cfg), 0.0};
    auto t0 = Clock::now();
    for (const auto& d : fx.doc_texts)
        out.idx.add_doc(d);
    out.idx.finalize();
    out.build_us = elapsed_us(t0);
    return out;
}

void run_bm25(const char* name, const Fixture& fx, const simeon::Bm25Index& idx, double build_us) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], s);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    // BM25 has no per-doc fixed footprint (variable inverted-index size); emit 0.
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

// BM25 with RM3 pseudo-relevance feedback expansion. Same BM25 variant the
// passed-in idx was built with (Atire / BM25+ / SAB etc.); PRF only changes
// the query, not the scorer. See include/simeon/prf.hpp for algorithm.
void run_bm25_prf(const char* name, const Fixture& fx, const simeon::Bm25Index& idx,
                  double build_us, const simeon::PrfConfig& pc) {
    Timing t;
    t.index_build_us = build_us;
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> s(nd, 0.0f);
    auto t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::score_with_prf(idx, fx.query_texts[qi], s, pc);
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(s[di], di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), 0, t);
}

void run_simeon_bm25_rrf(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                         const simeon::Bm25Index& idx) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);
    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::pair<std::uint32_t, float>> r_simeon(nd), r_bm25(nd);
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (std::uint32_t di = 0; di < nd; ++di) {
            r_simeon[di] = {di, dot(qembs.data() + qi * dim, dembs.data() + di * dim, dim)};
        }
        idx.score(fx.query_texts[qi], bm25_scores);
        for (std::uint32_t di = 0; di < nd; ++di)
            r_bm25[di] = {di, bm25_scores[di]};

        std::array<simeon::Ranking, 2> ins = {simeon::Ranking(r_simeon), simeon::Ranking(r_bm25)};
        auto fused = simeon::rrf_fuse(ins, 60.0f);

        rankings[qi].reserve(fused.size());
        for (const auto& [did, sc] : fused) {
            rankings[qi].emplace_back(sc, did);
        }
    }
    t.query_us = elapsed_us(t0);
    // Footprint is the simeon vectors (BM25 index sits in a different storage tier
    // and is variable-size; flagged in docs).
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
}

// Cascade: BM25 top-K pool → re-rank within the pool by one of three modes.
// Documents outside the pool are assigned -inf so they never appear in top-N
// during metric scoring. The cascade is the only ensemble pattern that can
// improve over BM25-alone on lexical-heavy corpora; see plan Step 1.
enum class RerankMode {
    SimeonCosine, // pure simeon cosine within pool
    LinearAlpha,  // alpha * bm25_z + (1-alpha) * simeon_z, both z-scored within pool
    PoolRrf,      // RRF restricted to the pool (avoids the global-RRF dilution)
};

void run_bm25_pool_simeon_rerank(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                                 const simeon::Bm25Index& idx, std::uint32_t pool_size,
                                 RerankMode mode, float alpha = 0.5f) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);
    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        idx.score(fx.query_texts[qi], bm25_scores);
        auto pool = simeon::top_k(bm25_scores, pool_size);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(neg_inf, di);
        }

        if (mode == RerankMode::SimeonCosine) {
            for (const auto& [did, _] : pool) {
                const float s = dot(qembs.data() + qi * dim, dembs.data() + did * dim, dim);
                rankings[qi][did].first = s;
            }
        } else if (mode == RerankMode::LinearAlpha) {
            // Z-score both signals within the pool so alpha is a meaningful weight.
            std::vector<float> bm(pool.size()), si(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) {
                bm[i] = pool[i].second;
                si[i] = dot(qembs.data() + qi * dim, dembs.data() + pool[i].first * dim, dim);
            }
            auto zscore = [](std::vector<float>& v) {
                if (v.empty())
                    return;
                double mean = 0.0;
                for (float x : v)
                    mean += x;
                mean /= v.size();
                double var = 0.0;
                for (float x : v)
                    var += (x - mean) * (x - mean);
                const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
                for (float& x : v)
                    x = static_cast<float>((x - mean) / sd);
            };
            zscore(bm);
            zscore(si);
            for (std::size_t i = 0; i < pool.size(); ++i) {
                rankings[qi][pool[i].first].first = alpha * bm[i] + (1.0f - alpha) * si[i];
            }
        } else { // PoolRrf
            std::vector<std::pair<std::uint32_t, float>> r_bm, r_si;
            r_bm.reserve(pool.size());
            r_si.reserve(pool.size());
            for (const auto& [did, score] : pool)
                r_bm.emplace_back(did, score);
            for (const auto& [did, _] : pool) {
                const float s = dot(qembs.data() + qi * dim, dembs.data() + did * dim, dim);
                r_si.emplace_back(did, s);
            }
            std::array<simeon::Ranking, 2> ins = {simeon::Ranking(r_bm), simeon::Ranking(r_si)};
            auto fused = simeon::rrf_fuse(ins, 60.0f);
            for (const auto& [did, sc] : fused) {
                rankings[qi][did].first = sc;
            }
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
}

// Cascade: rank top-K by PQ codes (cheap), then re-rank those candidates by
// full simeon vectors. Index footprint is just the PQ codes; the full vectors
// are assumed to live in a slower storage tier (or be re-encoded on demand
// from the doc text).
void run_pq_first_then_full(const char* name, simeon::EncoderConfig cfg, const Fixture& fx,
                            std::uint32_t pq_m, std::uint32_t candidate_k = 100) {
    Timing t;
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);
    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        enc.encode(fx.query_texts[i], qembs.data() + i * dim);
    }

    simeon::PQConfig pcfg{.dim = dim, .m = pq_m, .k = 256, .seed = 0xBEEF1234ULL};
    simeon::ProductQuantizer pq(pcfg);
    t0 = Clock::now();
    pq.train(dembs.data(), nd, 15);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(nd) * pq_m, 0);
    pq.encode_batch(dembs.data(), nd, codes.data());
    t.index_build_us = elapsed_us(t0);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> pq_scores(nd, 0.0f);
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        simeon::PQQuery pq_q(pq, qembs.data() + qi * dim);
        for (std::uint32_t di = 0; di < nd; ++di) {
            pq_scores[di] = pq_q.inner_product(codes.data() + di * pq_m);
        }
        auto cands = simeon::top_k(pq_scores, candidate_k);

        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(-std::numeric_limits<float>::infinity(), di);
        }
        for (const auto& [did, _] : cands) {
            const float s = dot(qembs.data() + qi * dim, dembs.data() + did * dim, dim);
            rankings[qi][did].first = s;
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), pq_m, t);
}

// MinHash retrieval: encode all docs once, score every doc per query by
// jaccard_estimate. Deterministic given (cfg, seed). Code-bytes-per-doc is
// k * sizeof(uint32) — directly comparable to simeon dense rows.
void run_minhash(const char* name, simeon::MinHashConfig cfg, const Fixture& fx) {
    Timing t;
    simeon::MinHashEncoder enc(cfg);
    const std::uint32_t k = enc.k();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<std::uint32_t> dsigs(static_cast<std::size_t>(nd) * k, 0);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dsigs.data() + i * k);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<std::uint32_t> qsig(k, 0);
    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        enc.encode(fx.query_texts[qi], qsig.data());
        rankings[qi].reserve(nd);
        for (std::uint32_t di = 0; di < nd; ++di) {
            rankings[qi].emplace_back(
                simeon::jaccard_estimate(qsig.data(), dsigs.data() + di * k, k), di);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx),
         static_cast<std::uint32_t>(k * sizeof(std::uint32_t)), t);
}

// Three-way RRF: BM25 ⊕ simeon ⊕ MinHash. The MinHash leg covers
// near-duplicate / boilerplate cases that cosine-space simeon misses.
void run_simeon_bm25_minhash_rrf(const char* name, simeon::EncoderConfig sc,
                                 simeon::MinHashConfig mc, const Fixture& fx,
                                 const simeon::Bm25Index& bm25_idx) {
    Timing t;
    simeon::Encoder senc(sc);
    simeon::MinHashEncoder menc(mc);
    const std::uint32_t dim = senc.output_dim();
    const std::uint32_t k = menc.k();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    std::vector<std::uint32_t> dsigs(static_cast<std::size_t>(nd) * k, 0);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        senc.encode(fx.doc_texts[i], dembs.data() + i * dim);
        menc.encode(fx.doc_texts[i], dsigs.data() + i * k);
    }
    t.doc_encode_us = elapsed_us(t0);

    std::vector<float> qembs(static_cast<std::size_t>(fx.query_ids.size()) * dim, 0.0f);
    std::vector<std::uint32_t> qsigs(static_cast<std::size_t>(fx.query_ids.size()) * k, 0);
    for (std::size_t i = 0; i < fx.query_ids.size(); ++i) {
        senc.encode(fx.query_texts[i], qembs.data() + i * dim);
        menc.encode(fx.query_texts[i], qsigs.data() + i * k);
    }

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> bm25_scores(nd, 0.0f);
    std::vector<std::pair<std::uint32_t, float>> r_si(nd), r_bm(nd), r_mh(nd);
    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        for (std::uint32_t di = 0; di < nd; ++di) {
            r_si[di] = {di, dot(qembs.data() + qi * dim, dembs.data() + di * dim, dim)};
        }
        bm25_idx.score(fx.query_texts[qi], bm25_scores);
        for (std::uint32_t di = 0; di < nd; ++di)
            r_bm[di] = {di, bm25_scores[di]};
        for (std::uint32_t di = 0; di < nd; ++di) {
            r_mh[di] = {di,
                        simeon::jaccard_estimate(qsigs.data() + qi * k, dsigs.data() + di * k, k)};
        }
        std::array<simeon::Ranking, 3> ins = {simeon::Ranking(r_si), simeon::Ranking(r_bm),
                                              simeon::Ranking(r_mh)};
        auto fused = simeon::rrf_fuse(ins, 60.0f);
        rankings[qi].reserve(fused.size());
        for (const auto& [did, sc_] : fused) {
            rankings[qi].emplace_back(sc_, did);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx),
         static_cast<std::uint32_t>(dim * sizeof(float) + k * sizeof(std::uint32_t)), t);
}

// Router-driven cascade. Pre-builds Atire + SAB-smooth indexes and the
// simeon encoder; per query, the QueryRouter picks one of three recipes
// based on cheap pre-retrieval predictors (see docs/router_design.md). The
// emitted row reflects the *aggregated* metric across whichever recipes the
// router selected. Index-build cost is the sum of the two BM25 indexes plus
// simeon doc-encoding; query cost is the actual per-query dispatched work.
struct RouterCounts {
    std::uint32_t atire = 0;
    std::uint32_t sab = 0;
    std::uint32_t cascade = 0;
};

void run_router_cascade(const char* name, simeon::EncoderConfig sc, const Fixture& fx,
                        const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                        double bm25_build_us, std::uint32_t pool_size = 500,
                        float cascade_alpha = 0.75f, simeon::RouterConfig rc = {},
                        const simeon::PrfConfig* prf = nullptr) {
    Timing t;
    t.index_build_us = bm25_build_us;

    simeon::Encoder enc(sc);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    // Router uses the Atire index for IDF lookups (any finalized index works;
    // both share the corpus so df values are identical).
    simeon::QueryRouter router(atire_idx, rc);

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scores(nd, 0.0f);
    std::vector<float> q_emb(dim, 0.0f);
    RouterCounts counts;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto recipe = router.choose(fx.query_texts[qi]);
        rankings[qi].reserve(nd);
        if (recipe == simeon::Recipe::Bm25Atire) {
            ++counts.atire;
            if (prf)
                simeon::score_with_prf(atire_idx, fx.query_texts[qi], scores, *prf);
            else
                atire_idx.score(fx.query_texts[qi], scores);
            for (std::uint32_t di = 0; di < nd; ++di) {
                rankings[qi].emplace_back(scores[di], di);
            }
        } else if (recipe == simeon::Recipe::Bm25SabSmooth) {
            ++counts.sab;
            if (prf)
                simeon::score_with_prf(sab_idx, fx.query_texts[qi], scores, *prf);
            else
                sab_idx.score(fx.query_texts[qi], scores);
            for (std::uint32_t di = 0; di < nd; ++di) {
                rankings[qi].emplace_back(scores[di], di);
            }
        } else { // CascadeLinearAlpha — SAB pool, simeon rerank, z-score combine
            ++counts.cascade;
            sab_idx.score(fx.query_texts[qi], scores);
            auto pool = simeon::top_k(scores, pool_size);
            for (std::uint32_t di = 0; di < nd; ++di) {
                rankings[qi].emplace_back(neg_inf, di);
            }
            enc.encode(fx.query_texts[qi], q_emb.data());
            std::vector<float> bm(pool.size()), si(pool.size());
            for (std::size_t i = 0; i < pool.size(); ++i) {
                bm[i] = pool[i].second;
                si[i] = dot(q_emb.data(), dembs.data() + pool[i].first * dim, dim);
            }
            auto zscore = [](std::vector<float>& v) {
                if (v.empty())
                    return;
                double mean = 0.0;
                for (float x : v)
                    mean += x;
                mean /= v.size();
                double var = 0.0;
                for (float x : v)
                    var += (x - mean) * (x - mean);
                const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
                for (float& x : v)
                    x = static_cast<float>((x - mean) / sd);
            };
            zscore(bm);
            zscore(si);
            for (std::size_t i = 0; i < pool.size(); ++i) {
                rankings[qi][pool[i].first].first =
                    cascade_alpha * bm[i] + (1.0f - cascade_alpha) * si[i];
            }
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
    std::fprintf(stderr, "[router] %s: atire=%u sab=%u cascade=%u (of %u queries)\n", name,
                 counts.atire, counts.sab, counts.cascade,
                 static_cast<std::uint32_t>(fx.query_ids.size()));
}

// Per-query metric computation for the per-query telemetry log and the
// oracle router. Mirrors the aggregate logic in score_rankings(): sort by
// score (tie-break by doc index), compute nDCG@10 / R@10 / R@100 / MRR@10
// using the saturating-recall convention.
struct PerQueryMetric {
    double ndcg_at_10 = 0.0;
    double recall_at_10 = 0.0;
    double recall_at_100 = 0.0;
    double mrr_at_10 = 0.0;
    bool has_relevant = false; // false → caller should skip from aggregates
};

PerQueryMetric compute_per_query(const std::vector<std::pair<float, std::uint32_t>>& ranking,
                                 const std::unordered_map<std::uint32_t, std::uint32_t>& qrel) {
    PerQueryMetric out;
    if (qrel.empty())
        return out;
    out.has_relevant = true;

    auto sorted = ranking;
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
    for (std::size_t r = 0; r < rels.size() && r < 10; ++r) {
        idcg += static_cast<double>(rels[r]) / std::log2(static_cast<double>(r) + 2.0);
    }
    out.ndcg_at_10 = idcg > 0.0 ? dcg / idcg : 0.0;
    const double denom10 = static_cast<double>(std::min<std::size_t>(10, qrel.size()));
    const double denom100 = static_cast<double>(std::min<std::size_t>(100, qrel.size()));
    out.recall_at_10 = static_cast<double>(hits10) / denom10;
    out.recall_at_100 = static_cast<double>(hits100) / denom100;
    out.mrr_at_10 = first_rel_rank > 0.0 ? 1.0 / first_rel_rank : 0.0;
    return out;
}

// Per-spec config for run_router_grid. Each spec emits one labeled JSONL row.
struct RouterSweepSpec {
    std::string tag; // suffix appended to the row name
    simeon::RouterConfig rc;
    std::uint32_t pool_size = 500;
    float cascade_alpha = 0.75f;
    // Step 1g.1: when true, populate post-retrieval-lite predictors via
    // QueryRouter::features_with_pool() so Atire pool/decay AND-gates can
    // see real signal. Adds ~one extra BM25 score() per query for the SAB
    // pool used in pool_overlap_jaccard. Off by default to keep Pass A/B/C
    // behavior byte-identical.
    bool use_post_retrieval = false;
    std::uint32_t post_retrieval_k = 50;
};

// Score one query under a Recipe. Fills `out_ranking` with size-nd
// (score, doc_idx) pairs. SimeonCascade also requires the encoded corpus +
// pre-encoded query embedding.
void score_query_for_recipe(simeon::Recipe recipe, const Fixture& fx, std::uint32_t qi,
                            const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                            const std::vector<float>* dembs, std::uint32_t dim,
                            const float* q_emb_or_null, std::uint32_t pool_size,
                            float cascade_alpha, std::vector<float>& scratch_scores,
                            std::vector<std::pair<float, std::uint32_t>>& out_ranking) {
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    const float neg_inf = -std::numeric_limits<float>::infinity();
    out_ranking.clear();
    out_ranking.reserve(nd);

    if (recipe == simeon::Recipe::Bm25Atire) {
        atire_idx.score(fx.query_texts[qi], scratch_scores);
        for (std::uint32_t di = 0; di < nd; ++di) {
            out_ranking.emplace_back(scratch_scores[di], di);
        }
    } else if (recipe == simeon::Recipe::Bm25SabSmooth) {
        sab_idx.score(fx.query_texts[qi], scratch_scores);
        for (std::uint32_t di = 0; di < nd; ++di) {
            out_ranking.emplace_back(scratch_scores[di], di);
        }
    } else { // CascadeLinearAlpha — SAB pool, simeon rerank, z-scored combine
        sab_idx.score(fx.query_texts[qi], scratch_scores);
        auto pool = simeon::top_k(scratch_scores, pool_size);
        for (std::uint32_t di = 0; di < nd; ++di) {
            out_ranking.emplace_back(neg_inf, di);
        }
        std::vector<float> bm(pool.size()), si(pool.size());
        for (std::size_t i = 0; i < pool.size(); ++i) {
            bm[i] = pool[i].second;
            si[i] = dot(q_emb_or_null, dembs->data() + pool[i].first * dim, dim);
        }
        auto zscore = [](std::vector<float>& v) {
            if (v.empty())
                return;
            double mean = 0.0;
            for (float x : v)
                mean += x;
            mean /= v.size();
            double var = 0.0;
            for (float x : v)
                var += (x - mean) * (x - mean);
            const float sd = static_cast<float>(std::sqrt(var / v.size()) + 1e-12);
            for (float& x : v)
                x = static_cast<float>((x - mean) / sd);
        };
        zscore(bm);
        zscore(si);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            out_ranking[pool[i].first].first =
                cascade_alpha * bm[i] + (1.0f - cascade_alpha) * si[i];
        }
    }
}

// Sweep helper: encode the corpus once, then run each RouterSweepSpec as its
// own bench row. Cheaper than calling run_router_cascade() N times because the
// expensive simeon doc-encode happens exactly once. Per-query telemetry is
// written to g_router_per_query_fp when set.
void run_router_grid(const std::string& prefix, simeon::EncoderConfig sc, const Fixture& fx,
                     const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                     double bm25_build_us, std::span<const RouterSweepSpec> specs) {
    simeon::Encoder enc(sc);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    const double doc_encode_us = elapsed_us(t0);

    // Bucket qrels by query for telemetry. (Aggregate scoring uses the same
    // bucketed view inside score_rankings.)
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel_bucket;
    for (const auto& q : fx.qrels)
        rel_bucket[q.q][q.d] = q.rel;
    static const std::unordered_map<std::uint32_t, std::uint32_t> empty_qrel;

    for (const auto& spec : specs) {
        const std::string name = prefix + "_" + spec.tag;

        Timing t;
        t.doc_encode_us = doc_encode_us;
        t.index_build_us = bm25_build_us;

        simeon::QueryRouter router(atire_idx, spec.rc);
        std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
        std::vector<float> scratch(nd, 0.0f);
        std::vector<float> q_emb(dim, 0.0f);
        std::array<std::uint32_t, 3> route_counts{0, 0, 0};

        // Pool span for Step 1g.1 post-retrieval-lite predictors. Convention
        // matches docs/router_design.md: pools[0]=Atire, pools[1]=SAB so that
        // pool_overlap_jaccard measures the routing-relevant disagreement.
        const std::array<const simeon::Bm25Index*, 2> pool_span{&atire_idx, &sab_idx};

        auto t1 = Clock::now();
        for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
            const auto features =
                spec.use_post_retrieval
                    ? router.features_with_pool(fx.query_texts[qi],
                                                std::span<const simeon::Bm25Index* const>(
                                                    pool_span.data(), pool_span.size()),
                                                spec.post_retrieval_k)
                    : router.features(fx.query_texts[qi]);
            const auto recipe = router.choose(features);
            ++route_counts[static_cast<std::size_t>(recipe)];
            const float* q_ptr = nullptr;
            if (recipe == simeon::Recipe::CascadeLinearAlpha) {
                enc.encode(fx.query_texts[qi], q_emb.data());
                q_ptr = q_emb.data();
            }
            score_query_for_recipe(recipe, fx, qi, atire_idx, sab_idx, &dembs, dim, q_ptr,
                                   spec.pool_size, spec.cascade_alpha, scratch, rankings[qi]);

            if (g_router_per_query_fp) {
                const auto rit = rel_bucket.find(qi);
                const auto& qrel = rit != rel_bucket.end() ? rit->second : empty_qrel;
                const auto pq = compute_per_query(rankings[qi], qrel);
                std::fprintf(g_router_per_query_fp,
                             "{\"config\":\"%s\",\"query_id\":\"%s\",\"recipe\":\"%s\","
                             "\"oov_rate\":%.4f,\"avg_idf\":%.4f,\"max_idf\":%.4f,"
                             "\"min_idf\":%.4f,\"idf_stddev\":%.4f,"
                             "\"n_terms\":%u,\"avg_term_chars\":%.2f,"
                             "\"score_decay_rate\":%.4f,\"score_normalized_var\":%.4f,"
                             "\"top_k_score_entropy\":%.4f,\"pool_overlap_jaccard\":%.4f,"
                             "\"has_relevant\":%s,\"ndcg_at_10\":%.4f,"
                             "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                             "\"mrr_at_10\":%.4f}\n",
                             name.c_str(), fx.query_ids[qi].c_str(), simeon::recipe_name(recipe),
                             features.oov_rate, features.avg_idf, features.max_idf,
                             features.min_idf, features.idf_stddev, features.n_terms,
                             features.avg_term_chars, features.score_decay_rate,
                             features.score_normalized_var, features.top_k_score_entropy,
                             features.pool_overlap_jaccard, pq.has_relevant ? "true" : "false",
                             pq.ndcg_at_10, pq.recall_at_10, pq.recall_at_100, pq.mrr_at_10);
            }
        }
        t.query_us = elapsed_us(t1);
        emit(name.c_str(), fx, score_rankings(rankings, fx),
             static_cast<std::uint32_t>(dim * sizeof(float)), t);
        std::fprintf(stderr, "[router-grid] %s: atire=%u sab=%u cascade=%u (of %u queries)\n",
                     name.c_str(),
                     route_counts[static_cast<std::size_t>(simeon::Recipe::Bm25Atire)],
                     route_counts[static_cast<std::size_t>(simeon::Recipe::Bm25SabSmooth)],
                     route_counts[static_cast<std::size_t>(simeon::Recipe::CascadeLinearAlpha)],
                     static_cast<std::uint32_t>(fx.query_ids.size()));
    }
    if (g_router_per_query_fp)
        std::fflush(g_router_per_query_fp);
}

// Oracle router: per query, score under all three recipes and use the one with
// the best per-query nDCG@10 (tie-break: best R@100, then SAB > Atire > Cascade
// for stability). The aggregate is the upper bound on what any pre-retrieval
// router that picks among these three recipes can achieve at the given
// (pool_size, cascade_alpha) setting. Per-query telemetry logs the oracle
// choice for downstream regret analysis.
void run_oracle_router(const char* name, simeon::EncoderConfig sc, const Fixture& fx,
                       const simeon::Bm25Index& atire_idx, const simeon::Bm25Index& sab_idx,
                       double bm25_build_us, std::uint32_t pool_size = 500,
                       float cascade_alpha = 0.75f) {
    Timing t;
    t.index_build_us = bm25_build_us;

    simeon::Encoder enc(sc);
    const std::uint32_t dim = enc.output_dim();
    const std::uint32_t nd = static_cast<std::uint32_t>(fx.doc_ids.size());
    std::vector<float> dembs(static_cast<std::size_t>(nd) * dim, 0.0f);
    auto t0 = Clock::now();
    for (std::size_t i = 0; i < nd; ++i) {
        enc.encode(fx.doc_texts[i], dembs.data() + i * dim);
    }
    t.doc_encode_us = elapsed_us(t0);

    simeon::QueryRouter router(atire_idx, simeon::RouterConfig{}); // for features only

    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>> rel_bucket;
    for (const auto& q : fx.qrels)
        rel_bucket[q.q][q.d] = q.rel;
    static const std::unordered_map<std::uint32_t, std::uint32_t> empty_qrel;

    std::vector<std::vector<std::pair<float, std::uint32_t>>> rankings(fx.query_ids.size());
    std::vector<float> scratch(nd, 0.0f);
    std::vector<float> q_emb(dim, 0.0f);
    std::array<std::uint32_t, 3> oracle_counts{0, 0, 0};

    // Pool span: oracle telemetry includes Step 1g.1 post-retrieval predictors
    // so per-query rows have the full feature vector (the oracle itself does
    // not consult them — its choice is the per-query argmax over recipes).
    const std::array<const simeon::Bm25Index*, 2> pool_span{&atire_idx, &sab_idx};

    t0 = Clock::now();
    for (std::uint32_t qi = 0; qi < fx.query_ids.size(); ++qi) {
        const auto features = router.features_with_pool(
            fx.query_texts[qi],
            std::span<const simeon::Bm25Index* const>(pool_span.data(), pool_span.size()), 50);
        // Cascade always needs a query embedding; precompute once.
        enc.encode(fx.query_texts[qi], q_emb.data());

        std::vector<std::pair<float, std::uint32_t>> r_atire, r_sab, r_cascade;
        score_query_for_recipe(simeon::Recipe::Bm25Atire, fx, qi, atire_idx, sab_idx, &dembs, dim,
                               q_emb.data(), pool_size, cascade_alpha, scratch, r_atire);
        score_query_for_recipe(simeon::Recipe::Bm25SabSmooth, fx, qi, atire_idx, sab_idx, &dembs,
                               dim, q_emb.data(), pool_size, cascade_alpha, scratch, r_sab);
        score_query_for_recipe(simeon::Recipe::CascadeLinearAlpha, fx, qi, atire_idx, sab_idx,
                               &dembs, dim, q_emb.data(), pool_size, cascade_alpha, scratch,
                               r_cascade);

        const auto rit = rel_bucket.find(qi);
        const auto& qrel = rit != rel_bucket.end() ? rit->second : empty_qrel;
        const auto m_atire = compute_per_query(r_atire, qrel);
        const auto m_sab = compute_per_query(r_sab, qrel);
        const auto m_cascade = compute_per_query(r_cascade, qrel);

        // Pick best by nDCG@10; tie-break by R@100, then by stable preference
        // SAB > Atire > Cascade (matches the default-fallback recipe order).
        struct Candidate {
            simeon::Recipe r;
            const PerQueryMetric* m;
            std::vector<std::pair<float, std::uint32_t>>* ranking;
            int tie_pref; // higher = preferred on metric ties
        };
        std::array<Candidate, 3> cands = {{
            {simeon::Recipe::Bm25SabSmooth, &m_sab, &r_sab, 2},
            {simeon::Recipe::Bm25Atire, &m_atire, &r_atire, 1},
            {simeon::Recipe::CascadeLinearAlpha, &m_cascade, &r_cascade, 0},
        }};
        const Candidate* best = &cands[0];
        for (const auto& c : cands) {
            if (c.m->ndcg_at_10 > best->m->ndcg_at_10 ||
                (c.m->ndcg_at_10 == best->m->ndcg_at_10 &&
                 c.m->recall_at_100 > best->m->recall_at_100) ||
                (c.m->ndcg_at_10 == best->m->ndcg_at_10 &&
                 c.m->recall_at_100 == best->m->recall_at_100 && c.tie_pref > best->tie_pref)) {
                best = &c;
            }
        }
        rankings[qi] = std::move(*best->ranking);
        ++oracle_counts[static_cast<std::size_t>(best->r)];

        if (g_router_per_query_fp) {
            std::fprintf(g_router_per_query_fp,
                         "{\"config\":\"%s\",\"query_id\":\"%s\",\"recipe\":\"%s\","
                         "\"oov_rate\":%.4f,\"avg_idf\":%.4f,\"max_idf\":%.4f,"
                         "\"min_idf\":%.4f,\"idf_stddev\":%.4f,"
                         "\"n_terms\":%u,\"avg_term_chars\":%.2f,"
                         "\"score_decay_rate\":%.4f,\"score_normalized_var\":%.4f,"
                         "\"top_k_score_entropy\":%.4f,\"pool_overlap_jaccard\":%.4f,"
                         "\"has_relevant\":%s,\"ndcg_at_10\":%.4f,"
                         "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,"
                         "\"mrr_at_10\":%.4f,"
                         "\"ndcg_atire\":%.4f,\"ndcg_sab\":%.4f,\"ndcg_cascade\":%.4f}\n",
                         name, fx.query_ids[qi].c_str(), simeon::recipe_name(best->r),
                         features.oov_rate, features.avg_idf, features.max_idf, features.min_idf,
                         features.idf_stddev, features.n_terms, features.avg_term_chars,
                         features.score_decay_rate, features.score_normalized_var,
                         features.top_k_score_entropy, features.pool_overlap_jaccard,
                         best->m->has_relevant ? "true" : "false", best->m->ndcg_at_10,
                         best->m->recall_at_10, best->m->recall_at_100, best->m->mrr_at_10,
                         m_atire.ndcg_at_10, m_sab.ndcg_at_10, m_cascade.ndcg_at_10);
        }
    }
    t.query_us = elapsed_us(t0);
    emit(name, fx, score_rankings(rankings, fx), static_cast<std::uint32_t>(dim * sizeof(float)),
         t);
    std::fprintf(stderr, "[router-oracle] %s: atire=%u sab=%u cascade=%u (of %u queries)\n", name,
                 oracle_counts[static_cast<std::size_t>(simeon::Recipe::Bm25Atire)],
                 oracle_counts[static_cast<std::size_t>(simeon::Recipe::Bm25SabSmooth)],
                 oracle_counts[static_cast<std::size_t>(simeon::Recipe::CascadeLinearAlpha)],
                 static_cast<std::uint32_t>(fx.query_ids.size()));
    if (g_router_per_query_fp)
        std::fflush(g_router_per_query_fp);
}

} // namespace

int main(int argc, char** argv) {
    const char* fixture_dir = nullptr;
    std::string queries_from = "test"; // "test" | "dev"
    const char* router_per_query_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--queries-from" && i + 1 < argc) {
            queries_from = argv[++i];
            if (queries_from != "test" && queries_from != "dev") {
                std::fprintf(stderr, "--queries-from must be test|dev\n");
                return 2;
            }
        } else if (a == "--router-per-query" && i + 1 < argc) {
            router_per_query_path = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                         "usage: %s [flags] <fixture_dir>\n"
                         "  --queries-from {test,dev}    pick split (default test)\n"
                         "  --router-per-query <path>    write per-query router telemetry JSONL\n"
                         "  fixture_dir expects corpus.tsv, queries[_dev].tsv,\n"
                         "  qrels[_dev].tsv, reference[_dev].bin\n"
                         "  see docs/reference_fixture.md\n",
                         argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            return 2;
        } else if (!fixture_dir) {
            fixture_dir = argv[i];
        } else {
            std::fprintf(stderr, "extra positional argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (!fixture_dir) {
        std::fprintf(stderr, "missing fixture_dir (try --help)\n");
        return 2;
    }
    if (router_per_query_path) {
        g_router_per_query_fp = std::fopen(router_per_query_path, "w");
        if (!g_router_per_query_fp) {
            std::fprintf(stderr, "cannot open --router-per-query path %s\n", router_per_query_path);
            return 1;
        }
    }
    Fixture fx;
    try {
        fx = load_fixture(fixture_dir, queries_from);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fixture load failed: %s\n", e.what());
        if (g_router_per_query_fp)
            std::fclose(g_router_per_query_fp);
        return 1;
    }
    std::fprintf(stderr, "[bench] fixture=%s split=%s nq=%zu nd=%zu\n", fixture_dir,
                 queries_from.c_str(), fx.query_ids.size(), fx.doc_ids.size());

    using PM = simeon::ProjectionMode;
    using NM = simeon::NGramMode;

    auto base = [](std::uint32_t out, PM p, NM nm = NM::CharAndWord, std::uint32_t lo = 3,
                   std::uint32_t hi = 5) {
        simeon::EncoderConfig c;
        c.ngram_mode = nm;
        c.ngram_min = lo;
        c.ngram_max = hi;
        c.sketch_dim = 4096;
        c.output_dim = out;
        c.projection = p;
        c.l2_normalize = true;
        return c;
    };

    run_reference(fx);

    run_simeon("achlioptas_4096_384", base(384, PM::AchlioptasSparse), fx);
    run_simeon("achlioptas_4096_768", base(768, PM::AchlioptasSparse), fx);
    run_simeon("very_sparse_4096_384", base(384, PM::VerySparse), fx);
    run_simeon("gaussian_4096_384", base(384, PM::DenseGaussian), fx);

    auto matryoshka = base(384, PM::AchlioptasSparse);
    matryoshka.matryoshka = true;
    run_simeon("achlioptas_matryoshka_4096_384", matryoshka, fx);

    run_simeon_pq("achlioptas_4096_384_pq8", base(384, PM::AchlioptasSparse), fx, 8);
    run_simeon_pq("achlioptas_4096_384_pq16", base(384, PM::AchlioptasSparse), fx, 16);
    run_simeon_pq("achlioptas_4096_384_pq32", base(384, PM::AchlioptasSparse), fx, 32);

    auto bm25 = build_bm25(fx);
    run_bm25("bm25_only", fx, bm25.idx, bm25.build_us);
    run_simeon_bm25_rrf("bm25_rrf_simeon_4096_384", base(384, PM::AchlioptasSparse), fx, bm25.idx);
    run_pq_first_then_full("pq16_first_stage_then_full_k100", base(384, PM::AchlioptasSparse), fx,
                           16, 100);
    run_pq_first_then_full("pq16_first_stage_then_full_k500", base(384, PM::AchlioptasSparse), fx,
                           16, 500);

    // Cascade: BM25 top-K pool → simeon rerank within pool. Three rerank modes
    // and two simeon widths to bound headroom (see plan Step 1).
    auto cfg384 = base(384, PM::AchlioptasSparse);
    auto cfg768 = base(768, PM::AchlioptasSparse);
    using RM = RerankMode;

    run_bm25_pool_simeon_rerank("bm25_pool100_simeon_cos_4096_384", cfg384, fx, bm25.idx, 100,
                                RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_pool500_simeon_cos_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_pool1000_simeon_cos_4096_384", cfg384, fx, bm25.idx, 1000,
                                RM::SimeonCosine);

    run_bm25_pool_simeon_rerank("bm25_pool500_linear_alpha050_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::LinearAlpha, 0.5f);
    run_bm25_pool_simeon_rerank("bm25_pool500_linear_alpha075_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::LinearAlpha, 0.75f);
    run_bm25_pool_simeon_rerank("bm25_pool500_pool_rrf_4096_384", cfg384, fx, bm25.idx, 500,
                                RM::PoolRrf);

    // Higher-quality simeon variant — bounds cascade headroom on this corpus.
    run_bm25_pool_simeon_rerank("bm25_pool500_simeon_cos_4096_768", cfg768, fx, bm25.idx, 500,
                                RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_pool500_linear_alpha075_4096_768", cfg768, fx, bm25.idx, 500,
                                RM::LinearAlpha, 0.75f);

    // Step 1c — BM25 formulation ablation. Standalone variant rows + cascade
    // headline using the strongest pool source (SubwordAwareBackoff).
    auto bm25_variant = [&](const char* name, simeon::Bm25Variant v, float delta = 1.0f,
                            float subword_gamma = 0.0f) {
        simeon::Bm25Config cfg;
        cfg.variant = v;
        cfg.delta = delta;
        cfg.subword_gamma = subword_gamma;
        auto idx = build_bm25(fx, cfg);
        run_bm25(name, fx, idx.idx, idx.build_us);
        return idx;
    };
    bm25_variant("bm25_atire", simeon::Bm25Variant::Atire);
    bm25_variant("bm25_plus", simeon::Bm25Variant::BM25Plus);
    bm25_variant("bm25_l", simeon::Bm25Variant::BM25L);
    bm25_variant("bm25_dlh13", simeon::Bm25Variant::DLH13);
    bm25_variant("bm25_pl2", simeon::Bm25Variant::PL2);
    bm25_variant("bm25_dph", simeon::Bm25Variant::DPH);
    bm25_variant("bm25_dcm", simeon::Bm25Variant::Dcm);
    auto sab_strict =
        bm25_variant("bm25_sab_strict", simeon::Bm25Variant::SubwordAwareBackoff, 1.0f, 0.0f);
    auto sab_smooth = bm25_variant("bm25_sab_smooth_gamma5",
                                   simeon::Bm25Variant::SubwordAwareBackoff, 1.0f, 5.0f);
    // Headline cascade: SAB pool (strongest morphological recall) → simeon
    // cosine rerank inside the pool. This is the row that should beat
    // bm25_only on lexical+morphological corpora like scifact.
    run_bm25_pool_simeon_rerank("bm25_sab_pool500_simeon_cos_4096_768", cfg768, fx, sab_smooth.idx,
                                500, RM::SimeonCosine);
    run_bm25_pool_simeon_rerank("bm25_sab_strict_pool500_simeon_cos_4096_768", cfg768, fx,
                                sab_strict.idx, 500, RM::SimeonCosine);

    // Step 1k A1 — RM3 pseudo-relevance feedback (Lavrenko & Croft 2001).
    // Canonical TREC settings (K=10, N=20, α=0.5) applied on top of the two
    // strongest BM25 pool sources. Attacks the FiQA-style short+paraphrase
    // query failure mode; expected no-op to modest-lift on scifact's short
    // abstracts.
    simeon::PrfConfig prf_canonical;
    prf_canonical.k = 10;
    prf_canonical.n_terms = 20;
    prf_canonical.alpha = 0.5f;
    run_bm25_prf("bm25_atire_rm3_k10_a0.5", fx, bm25.idx, bm25.build_us, prf_canonical);
    run_bm25_prf("bm25_sab_smooth_rm3_k10_a0.5", fx, sab_smooth.idx, sab_smooth.build_us,
                 prf_canonical);

    // Step 1g.2 — training-free PMI / co-occurrence embeddings. Rows tagged
    // `_incorpus` use the evaluation corpus itself as the seed corpus; that is
    // a known leakage configuration and serves as a sanity ceiling only — the
    // headline number must come from a held-out fold or external seed corpus
    // (see docs/pmi_projection.md). In-corpus rows land first because they are
    // reproducible from the shipped fixture alone.
    std::vector<std::string_view> pmi_seed_view;
    pmi_seed_view.reserve(fx.doc_texts.size());
    for (const auto& d : fx.doc_texts)
        pmi_seed_view.emplace_back(d);
    auto make_pmi = [&](std::uint32_t rank) {
        simeon::PmiConfig pc;
        pc.target_rank = rank;
        pc.min_token_count = 5;
        pc.max_vocab_size = 50'000;
        pc.svd_iters = 4;
        pc.svd_oversample = 10;
        return simeon::PmiEmbeddings::learn(std::span<const std::string_view>(pmi_seed_view), pc);
    };
    auto pmi256 = make_pmi(256);
    auto pmi512 = make_pmi(512);

    auto pmi_cfg = [&](const simeon::PmiEmbeddings& e) {
        simeon::EncoderConfig c;
        c.ngram_mode = NM::WordOnly;
        c.ngram_min = 1;
        c.ngram_max = 1;
        c.sketch_dim = 0;
        c.output_dim = e.dim();
        c.projection = PM::None;
        c.l2_normalize = true;
        c.pmi_rows = &e;
        return c;
    };
    auto pmi256_cfg = pmi_cfg(pmi256);
    auto pmi512_cfg = pmi_cfg(pmi512);

    run_simeon("simeon_pmi256_incorpus", pmi256_cfg, fx);
    run_simeon("simeon_pmi512_incorpus", pmi512_cfg, fx);
    run_simeon_bm25_rrf("simeon_pmi256_rrf_bm25_incorpus", pmi256_cfg, fx, bm25.idx);
    run_bm25_pool_simeon_rerank("bm25_sab_pool500_simeon_pmi256_cos_rerank_incorpus", pmi256_cfg,
                                fx, sab_smooth.idx, 500, RM::SimeonCosine);

    // Step 4 — wider sketch sweep. Establishes max-quality reference points.
    auto wide = [&](std::uint32_t sketch, std::uint32_t out, PM p) {
        auto c = base(out, p);
        c.sketch_dim = sketch;
        return c;
    };
    run_simeon("achlioptas_8192_512", wide(8192, 512, PM::AchlioptasSparse), fx);
    run_simeon("achlioptas_8192_1024", wide(8192, 1024, PM::AchlioptasSparse), fx);
    run_simeon("achlioptas_16384_1024", wide(16384, 1024, PM::AchlioptasSparse), fx);

    // Step 5 — Mixed Tabulation hash + parameterized Sparse-JL.
    auto mixed_tab_cfg = base(384, PM::AchlioptasSparse);
    mixed_tab_cfg.hash = simeon::HashFamily::MixedTabulation;
    run_simeon("achlioptas_4096_384_mixed_tab", mixed_tab_cfg, fx);

    auto sparse_jl_cfg = base(384, PM::SparseJL);
    sparse_jl_cfg.sparse_jl_eps = 0.10f;
    run_simeon("sparse_jl_4096_384_eps0.10", sparse_jl_cfg, fx);
    sparse_jl_cfg.sparse_jl_eps = 0.05f;
    run_simeon("sparse_jl_4096_384_eps0.05", sparse_jl_cfg, fx);

    // Step 6 — data-aware matryoshka weights. Uses the corpus itself as the
    // seed for the variance estimate (no held-out fold available in this
    // fixture); documented in docs/benchmarks.md so the row's evaluation
    // context is explicit.
    {
        auto probe_cfg = base(384, PM::AchlioptasSparse);
        std::vector<std::string_view> seed_views;
        seed_views.reserve(fx.doc_texts.size());
        for (const auto& d : fx.doc_texts)
            seed_views.emplace_back(d);
        auto weights = simeon::compute_matryoshka_weights(probe_cfg, seed_views);
        auto data_aware_cfg = probe_cfg;
        data_aware_cfg.matryoshka = true;
        data_aware_cfg.matryoshka_weights = std::move(weights);
        run_simeon("achlioptas_4096_384_matryoshka_data_aware", data_aware_cfg, fx);
    }

    // Step 7 — FWHT (subsampled randomized Hadamard transform).
    run_simeon("fwht_4096_384", base(384, PM::Fwht), fx);
    run_simeon("fwht_8192_1024", wide(8192, 1024, PM::Fwht), fx);

    // Step 2 — Densified MinHash retrieval + three-way fusion.
    simeon::MinHashConfig mh256;
    mh256.k = 256;
    simeon::MinHashConfig mh512;
    mh512.k = 512;
    run_minhash("minhash_256", mh256, fx);
    run_minhash("minhash_512", mh512, fx);
    run_simeon_bm25_minhash_rrf("simeon_4096_384_rrf_bm25_rrf_minhash_256", cfg384, mh256, fx,
                                bm25.idx);

    // Router-driven cascade. Picks per-query among Bm25Atire / Bm25SabSmooth /
    // CascadeLinearAlpha using cheap pre-retrieval predictors. The Atire
    // index from `bm25` and SAB-smooth from `sab_smooth` are reused — no
    // additional index-build cost. Two router presets:
    //   - default thresholds (oov>0, idf>6, nterms>=4 + idf<=5)
    //   - aggressive cascade (lower nterms threshold, higher max_idf)
    {
        run_router_cascade("router_default_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                           bm25.build_us + sab_smooth.build_us);

        simeon::RouterConfig raggro;
        raggro.cascade_min_terms = 2;
        raggro.cascade_max_idf = 8.0f;
        run_router_cascade("router_cascade_aggressive_4096_768", cfg768, fx, bm25.idx,
                           sab_smooth.idx, bm25.build_us + sab_smooth.build_us, 500, 0.75f, raggro);

        simeon::RouterConfig rsab_only;
        rsab_only.cascade_min_terms = 9999;  // disable cascade route
        rsab_only.high_idf_threshold = 9999; // disable Atire route
        run_router_cascade("router_sab_only_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                           bm25.build_us + sab_smooth.build_us, 500, 0.75f, rsab_only);

        // Step 1g.2 — default-threshold router with the PMI256 encoder used
        // inside the cascade route (replaces the Achlioptas reranker).
        run_router_cascade("router_default_with_pmi256_cascade_incorpus", pmi256_cfg, fx, bm25.idx,
                           sab_smooth.idx, bm25.build_us + sab_smooth.build_us);

        // Step 1k A1 — router integration with RM3 expansion on the Atire and
        // SAB routes (cascade route's first-pass left un-expanded to keep its
        // pool membership comparable to router_default_4096_768).
        simeon::PrfConfig prf_router;
        prf_router.k = 10;
        prf_router.n_terms = 20;
        prf_router.alpha = 0.5f;
        run_router_cascade("router_default_with_rm3_k10_a0.5", cfg768, fx, bm25.idx, sab_smooth.idx,
                           bm25.build_us + sab_smooth.build_us, 500, 0.75f, simeon::RouterConfig{},
                           &prf_router);
    }

    // Step 1e — router ablation expansion. Two sweeps share an encoded corpus
    // and the existing Atire/SAB indices via run_router_grid:
    //   pass A: vary one threshold knob at a time around the default config
    //           (small Cartesian: 36 specs).
    //   pass B: fix thresholds to default, sweep pool_size × cascade_alpha
    //           (9 specs).
    // Plus an oracle row that picks the per-query best recipe (upper bound).
    {
        std::vector<RouterSweepSpec> specs;
        const float oovs[] = {0.0f, 0.25f};
        const float idfs[] = {3.0f, 5.0f, 9999.0f};
        const std::uint32_t nts[] = {2u, 4u, 6u};
        const float maxidfs[] = {5.0f, 7.0f};
        char tagbuf[128];
        for (float oov : oovs)
            for (float idf : idfs)
                for (std::uint32_t nt : nts)
                    for (float mi : maxidfs) {
                        simeon::RouterConfig rc;
                        rc.oov_threshold = oov;
                        rc.high_idf_threshold = idf;
                        rc.cascade_min_terms = nt;
                        rc.cascade_max_idf = mi;
                        std::snprintf(tagbuf, sizeof(tagbuf), "passA_oov%.2f_idf%.0f_nt%u_mi%.0f",
                                      oov, idf, nt, mi);
                        specs.push_back({tagbuf, rc, 500u, 0.75f});
                    }
        const std::uint32_t pools[] = {250u, 500u, 1000u};
        const float alphas[] = {0.5f, 0.75f, 0.85f};
        for (std::uint32_t p : pools)
            for (float a : alphas) {
                simeon::RouterConfig rc; // defaults
                std::snprintf(tagbuf, sizeof(tagbuf), "passB_pool%u_a%.2f", p, a);
                specs.push_back({tagbuf, rc, p, a});
            }
        // Pass C — Step 1f predictor enrichment. Hold the Pass A winners
        // (oov=0, idf=3, cascade nt=4 / max_idf=5, pool=500, alpha=0.75) and
        // sweep the new atire-route AND-gates: atire_min_terms (n_terms floor)
        // and atire_min_idf_floor (min_idf floor). The (0, 0) cell duplicates
        // the Pass A winner row and acts as a regression check.
        const std::uint32_t atire_nts[] = {0u, 6u, 10u, 14u};
        const float atire_mi_floors[] = {0.0f, 1.5f, 3.0f};
        for (std::uint32_t ant : atire_nts)
            for (float amif : atire_mi_floors) {
                simeon::RouterConfig rc;
                rc.oov_threshold = 0.0f;
                rc.high_idf_threshold = 3.0f;
                rc.cascade_min_terms = 4u;
                rc.cascade_max_idf = 5.0f;
                rc.atire_min_terms = ant;
                rc.atire_min_idf_floor = amif;
                std::snprintf(tagbuf, sizeof(tagbuf), "passC_ant%u_amif%.1f", ant, amif);
                specs.push_back({tagbuf, rc, 500u, 0.75f, false, 50u});
            }
        // Pass D — Step 1g.1 post-retrieval-lite predictors. Hold the Pass A
        // winners and Step 1f Pass C defaults; sweep the new Atire AND-gates
        // atire_max_pool_jaccard (route only when Atire-vs-SAB top-K pools
        // disagree enough) and atire_min_score_decay (route only when the
        // BM25 top is sharply peaked). use_post_retrieval=true triggers
        // QueryRouter::features_with_pool() in run_router_grid.
        const float atire_jacs[] = {0.5f, 0.7f, 0.9f, 1.0f};
        const float atire_decays[] = {0.0f, 0.3f, 0.6f};
        for (float jac : atire_jacs)
            for (float dec : atire_decays) {
                simeon::RouterConfig rc;
                rc.oov_threshold = 0.0f;
                rc.high_idf_threshold = 3.0f;
                rc.cascade_min_terms = 4u;
                rc.cascade_max_idf = 5.0f;
                rc.atire_max_pool_jaccard = jac;
                rc.atire_min_score_decay = dec;
                std::snprintf(tagbuf, sizeof(tagbuf), "passD_jac%.1f_dec%.1f", jac, dec);
                specs.push_back({tagbuf, rc, 500u, 0.75f, true, 50u});
            }
        // Pass E — Step 1k B2. Sum-SCQ (Zhao 2008) and simplified clarity
        // (Cronen-Townsend 2002) AND-gates on the Atire route. Both are
        // pre-retrieval (no pool needed), so use_post_retrieval=false.
        const float atire_scq_floors[] = {0.0f, 5.0f, 10.0f, 20.0f};
        const float atire_clarity_ceils[] = {
            std::numeric_limits<float>::infinity(), 3.0f, 5.0f, 8.0f};
        for (float scq : atire_scq_floors)
            for (float clar : atire_clarity_ceils) {
                simeon::RouterConfig rc;
                rc.oov_threshold = 0.0f;
                rc.high_idf_threshold = 3.0f;
                rc.cascade_min_terms = 4u;
                rc.cascade_max_idf = 5.0f;
                rc.atire_min_scq = scq;
                rc.atire_max_clarity = clar;
                const double clar_d = std::isinf(clar) ? 99.0 : clar;
                std::snprintf(tagbuf, sizeof(tagbuf), "passE_scq%.0f_clar%.1f", scq,
                              clar_d);
                specs.push_back({tagbuf, rc, 500u, 0.75f, false, 50u});
            }

        run_router_grid("router_grid_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                        bm25.build_us + sab_smooth.build_us,
                        std::span<const RouterSweepSpec>(specs));

        run_oracle_router("router_oracle_4096_768", cfg768, fx, bm25.idx, sab_smooth.idx,
                          bm25.build_us + sab_smooth.build_us, 500u, 0.75f);
    }

    if (g_router_per_query_fp)
        std::fclose(g_router_per_query_fp);
    return 0;
}
