// simeon accuracy microbench — synthetic clustered retrieval.
//
// Builds K topical clusters, each with a distinct vocabulary pool. Each
// document is a sequence of words drawn from the cluster's pool, with a
// small cross-cluster leakage rate for noise. The query is the cluster's
// seed sentence. Same-cluster docs are relevant.
//
// This is a lower-bound sanity test: if simeon preserves lexical
// structure through the projection at all, intra-cluster cosine should
// exceed inter-cluster cosine, and Recall@10 should approach 1.0.
// Failure here means the projection is destroying signal. Passing here
// does NOT prove semantic quality — that requires a real corpus.
//
// Emits one JSONL record per (config × corpus) cell.
//   {
//     "bench": "accuracy",
//     "projection": "achlioptas",
//     "sketch_dim": 4096, "output_dim": 384,
//     "ngram_min": 3, "ngram_max": 5,
//     "clusters": 8, "docs_per_cluster": 50, "leakage": 0.05,
//     "recall_at_10": 0.98, "recall_at_100": 1.00,
//     "mrr_at_10": 0.92,
//     "intra_cos_mean": 0.61, "inter_cos_mean": 0.12,
//     "separation": 0.49,
//     "simd_tier": "neon"
//   }

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/pq.hpp"
#include "simeon/simeon.hpp"

namespace {

struct Cluster {
    std::string seed;           // cluster description used as the query
    std::vector<std::string> vocab;  // word pool
};

// Eight synthetic topical clusters with distinct vocabularies. The seed
// is the retrieval query; same-cluster docs are the relevant set.
std::vector<Cluster> build_clusters() {
    return {
        {"linux kernel scheduler cpu task queue preempt",
         {"kernel","scheduler","task","cpu","preempt","runqueue","cfs","priority",
          "deadline","tick","nice","cgroup","process","context","switch","affinity",
          "migration","load","balance","yield","idle","wakeup","core","numa","sched",
          "timer","realtime","isolcpu","latency","throughput"}},
        {"compiler code generation optimization llvm gcc ir",
         {"compiler","optimizer","pass","ssa","llvm","gcc","clang","ir","codegen",
          "register","allocator","inline","loop","vector","unroll","constant","folding",
          "dce","phi","block","dominator","liveness","spill","instruction","backend",
          "frontend","attribute","pragma","opt","pipeline"}},
        {"tls handshake certificate key exchange cipher suite",
         {"tls","ssl","handshake","certificate","cipher","suite","key","exchange",
          "x509","ca","session","resume","ticket","pki","rsa","ecdsa","ecdhe","hkdf",
          "hmac","sha","aes","gcm","chacha","record","alert","extension","sni","alpn",
          "hello","finished"}},
        {"dna sequencing genome alignment read fastq bwa",
         {"dna","rna","genome","sequencing","read","alignment","bwa","samtools",
          "fastq","bam","sam","variant","snp","indel","reference","assembly","contig",
          "scaffold","mapping","quality","phred","kmer","bloom","index","chromosome",
          "gene","exon","intron","annotation","fasta"}},
        {"distributed consensus raft paxos leader election log",
         {"raft","paxos","consensus","leader","follower","election","term","log",
          "entry","commit","quorum","heartbeat","append","vote","snapshot","replication",
          "partition","split","brain","linearizable","serializable","lease","epoch",
          "proposer","acceptor","learner","majority","quorum","sync","durable"}},
        {"neural network transformer attention embedding token batch",
         {"transformer","attention","embedding","token","encoder","decoder","layer",
          "softmax","residual","norm","gelu","relu","gradient","backprop","optimizer",
          "adam","loss","batch","epoch","checkpoint","finetune","pretrain","logit",
          "head","dim","vocab","bpe","sentencepiece","query","key"}},
        {"postgres query planner index btree hash join sort",
         {"postgres","planner","query","optimizer","index","btree","hash","join",
          "sort","seqscan","indexscan","nestloop","merge","bitmap","tuple","heap",
          "wal","vacuum","analyze","statistics","explain","cost","row","estimate",
          "partition","inherit","catalog","mvcc","snapshot","xid"}},
        {"rust borrow checker lifetime ownership trait generic",
         {"rust","borrow","checker","lifetime","ownership","move","copy","clone",
          "trait","generic","impl","struct","enum","match","pattern","closure","async",
          "await","future","pin","send","sync","unsafe","deref","drop","cargo","crate",
          "mod","pub","let"}},
    };
}

struct Doc {
    std::string text;
    std::uint32_t cluster_id;
};

std::string make_doc(std::mt19937& rng, const Cluster& c,
                     const std::vector<std::string>& global_pool,
                     double leakage, std::size_t words) {
    std::uniform_int_distribution<std::size_t> local(0, c.vocab.size() - 1);
    std::uniform_int_distribution<std::size_t> global_idx(0, global_pool.size() - 1);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::string out;
    out.reserve(words * 8);
    for (std::size_t i = 0; i < words; ++i) {
        if (coin(rng) < leakage) {
            out.append(global_pool[global_idx(rng)]);
        } else {
            out.append(c.vocab[local(rng)]);
        }
        out.push_back(' ');
    }
    return out;
}

struct Corpus {
    std::vector<Doc> docs;
    std::vector<Cluster> clusters;
};

Corpus make_corpus(std::mt19937& rng, std::size_t docs_per_cluster,
                   double leakage, std::size_t words_per_doc) {
    Corpus c;
    c.clusters = build_clusters();
    // Build a global vocabulary pool as the union of all clusters for leakage.
    std::vector<std::string> global;
    for (const auto& cl : c.clusters) {
        global.insert(global.end(), cl.vocab.begin(), cl.vocab.end());
    }
    c.docs.reserve(c.clusters.size() * docs_per_cluster);
    for (std::uint32_t k = 0; k < c.clusters.size(); ++k) {
        for (std::size_t i = 0; i < docs_per_cluster; ++i) {
            c.docs.push_back(Doc{make_doc(rng, c.clusters[k], global, leakage, words_per_doc), k});
        }
    }
    return c;
}

// Cosine similarity assuming both vectors are already unit-norm (simeon
// outputs are L2-normalized, and we normalize queries the same way).
double cosine(const float* a, const float* b, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += static_cast<double>(a[i]) * b[i];
    return s;
}

struct Metrics {
    double recall_at_10;
    double recall_at_100;
    double mrr_at_10;
    double intra_cos_mean;
    double inter_cos_mean;
};

Metrics evaluate(const simeon::EncoderConfig& cfg, const Corpus& corpus) {
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();

    // Encode all docs.
    std::vector<float> embs(corpus.docs.size() * dim, 0.0f);
    for (std::size_t i = 0; i < corpus.docs.size(); ++i) {
        enc.encode(corpus.docs[i].text, embs.data() + i * dim);
    }

    // Encode one query per cluster (the cluster seed sentence).
    std::vector<float> qembs(corpus.clusters.size() * dim, 0.0f);
    for (std::size_t k = 0; k < corpus.clusters.size(); ++k) {
        enc.encode(corpus.clusters[k].seed, qembs.data() + k * dim);
    }

    // Per-query retrieval. Score every doc, rank by cosine descending.
    double recall10_sum = 0.0, recall100_sum = 0.0, mrr10_sum = 0.0;
    const std::size_t num_queries = corpus.clusters.size();
    const std::size_t num_docs = corpus.docs.size();
    std::vector<std::pair<double, std::uint32_t>> scored;
    scored.resize(num_docs);
    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* qv = qembs.data() + q * dim;
        std::size_t relevant_count = 0;
        for (std::size_t i = 0; i < num_docs; ++i) {
            scored[i] = {cosine(qv, embs.data() + i * dim, dim), corpus.docs[i].cluster_id};
            if (corpus.docs[i].cluster_id == q) ++relevant_count;
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        std::size_t hits10 = 0, hits100 = 0;
        double first_rel_rank = 0.0;
        for (std::size_t r = 0; r < num_docs; ++r) {
            if (scored[r].second == q) {
                if (r < 10) ++hits10;
                if (r < 100) ++hits100;
                if (first_rel_rank == 0.0 && r < 10) {
                    first_rel_rank = static_cast<double>(r + 1);
                }
            }
        }
        // Cap denominator at K so Recall@K is not artificially ceiling-bound
        // when |relevant| > K (standard IR convention for "saturating" recall).
        const double denom10 = static_cast<double>(std::min<std::size_t>(10, relevant_count));
        const double denom100 = static_cast<double>(std::min<std::size_t>(100, relevant_count));
        recall10_sum += static_cast<double>(hits10) / denom10;
        recall100_sum += static_cast<double>(hits100) / denom100;
        mrr10_sum += (first_rel_rank > 0.0) ? (1.0 / first_rel_rank) : 0.0;
    }

    // Intra / inter cluster cosine means over all doc-doc pairs (i<j).
    double intra_sum = 0.0, inter_sum = 0.0;
    std::size_t intra_n = 0, inter_n = 0;
    for (std::size_t i = 0; i + 1 < num_docs; ++i) {
        const float* a = embs.data() + i * dim;
        const std::uint32_t ci = corpus.docs[i].cluster_id;
        for (std::size_t j = i + 1; j < num_docs; ++j) {
            const double c = cosine(a, embs.data() + j * dim, dim);
            if (corpus.docs[j].cluster_id == ci) { intra_sum += c; ++intra_n; }
            else                                 { inter_sum += c; ++inter_n; }
        }
    }
    return Metrics{
        recall10_sum / static_cast<double>(num_queries),
        recall100_sum / static_cast<double>(num_queries),
        mrr10_sum / static_cast<double>(num_queries),
        intra_n ? intra_sum / static_cast<double>(intra_n) : 0.0,
        inter_n ? inter_sum / static_cast<double>(inter_n) : 0.0,
    };
}

struct Row {
    std::uint32_t sketch_dim;
    std::uint32_t output_dim;
    simeon::ProjectionMode projection;
    const char* projection_name;
    std::uint32_t ngram_min;
    std::uint32_t ngram_max;
    simeon::NGramMode ngram_mode;
    const char* ngram_mode_name;
    bool matryoshka = false;
    std::uint32_t prefix_dim = 0;  // 0 = use full output_dim
    std::uint32_t pq_m = 0;        // 0 = no PQ (use float vectors)
};

Metrics evaluate_prefix(const simeon::EncoderConfig& cfg, const Corpus& corpus,
                        std::uint32_t prefix) {
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();

    std::vector<float> embs(corpus.docs.size() * dim, 0.0f);
    for (std::size_t i = 0; i < corpus.docs.size(); ++i) {
        enc.encode(corpus.docs[i].text, embs.data() + i * dim);
    }
    std::vector<float> qembs(corpus.clusters.size() * dim, 0.0f);
    for (std::size_t k = 0; k < corpus.clusters.size(); ++k) {
        enc.encode(corpus.clusters[k].seed, qembs.data() + k * dim);
    }

    // Truncate + L2-renormalize each vector to `prefix` dims, then run the
    // standard cosine ranking.
    std::vector<float> dprefix(corpus.docs.size() * prefix, 0.0f);
    for (std::size_t i = 0; i < corpus.docs.size(); ++i) {
        std::copy_n(embs.data() + i * dim, prefix, dprefix.data() + i * prefix);
        simeon::matryoshka_prefix_normalize(dprefix.data() + i * prefix, prefix);
    }
    std::vector<float> qprefix(corpus.clusters.size() * prefix, 0.0f);
    for (std::size_t k = 0; k < corpus.clusters.size(); ++k) {
        std::copy_n(qembs.data() + k * dim, prefix, qprefix.data() + k * prefix);
        simeon::matryoshka_prefix_normalize(qprefix.data() + k * prefix, prefix);
    }

    double recall10_sum = 0.0, recall100_sum = 0.0, mrr10_sum = 0.0;
    const std::size_t num_queries = corpus.clusters.size();
    const std::size_t num_docs = corpus.docs.size();
    std::vector<std::pair<double, std::uint32_t>> scored(num_docs);
    for (std::size_t q = 0; q < num_queries; ++q) {
        const float* qv = qprefix.data() + q * prefix;
        std::size_t relevant_count = 0;
        for (std::size_t i = 0; i < num_docs; ++i) {
            scored[i] = {cosine(qv, dprefix.data() + i * prefix, prefix),
                         corpus.docs[i].cluster_id};
            if (corpus.docs[i].cluster_id == q) ++relevant_count;
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t hits10 = 0, hits100 = 0;
        double first_rel_rank = 0.0;
        for (std::size_t r = 0; r < num_docs; ++r) {
            if (scored[r].second == q) {
                if (r < 10) ++hits10;
                if (r < 100) ++hits100;
                if (first_rel_rank == 0.0 && r < 10) {
                    first_rel_rank = static_cast<double>(r + 1);
                }
            }
        }
        const double denom10 = static_cast<double>(std::min<std::size_t>(10, relevant_count));
        const double denom100 = static_cast<double>(std::min<std::size_t>(100, relevant_count));
        recall10_sum += static_cast<double>(hits10) / denom10;
        recall100_sum += static_cast<double>(hits100) / denom100;
        mrr10_sum += (first_rel_rank > 0.0) ? (1.0 / first_rel_rank) : 0.0;
    }

    return Metrics{
        recall10_sum / static_cast<double>(num_queries),
        recall100_sum / static_cast<double>(num_queries),
        mrr10_sum / static_cast<double>(num_queries),
        0.0, 0.0,
    };
}

Metrics evaluate_pq(const simeon::EncoderConfig& cfg, const Corpus& corpus,
                    std::uint32_t pq_m) {
    simeon::Encoder enc(cfg);
    const std::uint32_t dim = enc.output_dim();

    std::vector<float> embs(corpus.docs.size() * dim, 0.0f);
    for (std::size_t i = 0; i < corpus.docs.size(); ++i) {
        enc.encode(corpus.docs[i].text, embs.data() + i * dim);
    }
    std::vector<float> qembs(corpus.clusters.size() * dim, 0.0f);
    for (std::size_t k = 0; k < corpus.clusters.size(); ++k) {
        enc.encode(corpus.clusters[k].seed, qembs.data() + k * dim);
    }

    simeon::PQConfig pcfg;
    pcfg.dim = dim;
    pcfg.m = pq_m;
    pcfg.k = 256;
    pcfg.seed = 0xBEEF1234ULL;
    simeon::ProductQuantizer pq(pcfg);
    pq.train(embs.data(), static_cast<std::uint32_t>(corpus.docs.size()),
             /*n_iters=*/15);

    std::vector<std::uint8_t> codes(corpus.docs.size() * pq_m, 0);
    pq.encode_batch(embs.data(), static_cast<std::uint32_t>(corpus.docs.size()),
                    codes.data());

    double recall10_sum = 0.0, recall100_sum = 0.0, mrr10_sum = 0.0;
    const std::size_t num_queries = corpus.clusters.size();
    const std::size_t num_docs = corpus.docs.size();
    std::vector<std::pair<double, std::uint32_t>> scored(num_docs);
    for (std::size_t q = 0; q < num_queries; ++q) {
        simeon::PQQuery pquery(pq, qembs.data() + q * dim);
        std::size_t relevant_count = 0;
        for (std::size_t i = 0; i < num_docs; ++i) {
            // Inner product is monotone with cosine for unit vectors; rank
            // descending by IP for retrieval.
            const float ip = pquery.inner_product(codes.data() + i * pq_m);
            scored[i] = {static_cast<double>(ip), corpus.docs[i].cluster_id};
            if (corpus.docs[i].cluster_id == q) ++relevant_count;
        }
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::size_t hits10 = 0, hits100 = 0;
        double first_rel_rank = 0.0;
        for (std::size_t r = 0; r < num_docs; ++r) {
            if (scored[r].second == q) {
                if (r < 10) ++hits10;
                if (r < 100) ++hits100;
                if (first_rel_rank == 0.0 && r < 10) {
                    first_rel_rank = static_cast<double>(r + 1);
                }
            }
        }
        const double denom10 = static_cast<double>(std::min<std::size_t>(10, relevant_count));
        const double denom100 = static_cast<double>(std::min<std::size_t>(100, relevant_count));
        recall10_sum += static_cast<double>(hits10) / denom10;
        recall100_sum += static_cast<double>(hits100) / denom100;
        mrr10_sum += (first_rel_rank > 0.0) ? (1.0 / first_rel_rank) : 0.0;
    }

    return Metrics{
        recall10_sum / static_cast<double>(num_queries),
        recall100_sum / static_cast<double>(num_queries),
        mrr10_sum / static_cast<double>(num_queries),
        0.0, 0.0,
    };
}

void emit(const Row& r, std::size_t clusters, std::size_t dpc, double leakage,
          const Metrics& m) {
    std::printf(
        "{\"bench\":\"accuracy\",\"projection\":\"%s\",\"sketch_dim\":%u,"
        "\"output_dim\":%u,\"ngram_min\":%u,\"ngram_max\":%u,\"ngram_mode\":\"%s\","
        "\"matryoshka\":%s,\"prefix_dim\":%u,\"pq_m\":%u,"
        "\"clusters\":%zu,\"docs_per_cluster\":%zu,\"leakage\":%.3f,"
        "\"recall_at_10\":%.4f,\"recall_at_100\":%.4f,\"mrr_at_10\":%.4f,"
        "\"intra_cos_mean\":%.4f,\"inter_cos_mean\":%.4f,\"separation\":%.4f,"
        "\"simd_tier\":\"%s\"}\n",
        r.projection_name, r.sketch_dim, r.output_dim, r.ngram_min, r.ngram_max,
        r.ngram_mode_name, r.matryoshka ? "true" : "false", r.prefix_dim, r.pq_m,
        clusters, dpc, leakage,
        m.recall_at_10, m.recall_at_100, m.mrr_at_10,
        m.intra_cos_mean, m.inter_cos_mean, m.intra_cos_mean - m.inter_cos_mean,
        simeon::simd_tier_name(simeon::active_simd_tier()));
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t dpc =
        argc > 1 ? static_cast<std::size_t>(std::atoll(argv[1])) : 50;
    const std::size_t words_per_doc =
        argc > 2 ? static_cast<std::size_t>(std::atoll(argv[2])) : 60;
    const double leakage =
        argc > 3 ? std::atof(argv[3]) : 0.05;

    std::mt19937 rng(0xC0FFEE);
    Corpus corpus = make_corpus(rng, dpc, leakage, words_per_doc);

    using PM = simeon::ProjectionMode;
    using NM = simeon::NGramMode;
    const Row rows[] = {
        // Raw sketch — upper bound on structure preservation.
        {1024, 0, PM::None, "none", 3, 5, NM::CharOnly, "char"},
        {4096, 0, PM::None, "none", 3, 5, NM::CharOnly, "char"},

        // Achlioptas × (sketch_dim × output_dim) cross product.
        {2048,  256, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {2048,  384, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {4096,  128, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {4096,  256, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {4096,  384, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {4096,  512, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {4096,  768, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},
        {8192,  384, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharOnly, "char"},

        // N-gram width at the sweet-spot dim.
        {4096,  384, PM::AchlioptasSparse, "achlioptas", 3, 3, NM::CharOnly, "char"},
        {4096,  384, PM::AchlioptasSparse, "achlioptas", 3, 7, NM::CharOnly, "char"},
        {4096,  384, PM::AchlioptasSparse, "achlioptas", 4, 6, NM::CharOnly, "char"},

        // N-gram mode — does adding word n-grams help?
        {4096,  384, PM::AchlioptasSparse, "achlioptas", 3, 5, NM::CharAndWord, "char+word"},
        {4096,  384, PM::AchlioptasSparse, "achlioptas", 1, 2, NM::WordOnly,    "word"},

        // Projection comparison at sweet-spot dim.
        {4096,  256, PM::DenseGaussian,    "gaussian",    3, 5, NM::CharOnly, "char"},
        {4096,  384, PM::DenseGaussian,    "gaussian",    3, 5, NM::CharOnly, "char"},
        {4096,  768, PM::DenseGaussian,    "gaussian",    3, 5, NM::CharOnly, "char"},
        {4096,  256, PM::VerySparse,       "very_sparse", 3, 5, NM::CharOnly, "char"},
        {4096,  384, PM::VerySparse,       "very_sparse", 3, 5, NM::CharOnly, "char"},
        {4096,  768, PM::VerySparse,       "very_sparse", 3, 5, NM::CharOnly, "char"},

        // Matryoshka: same projection, prefix-evaluated at decreasing widths.
        // Compares "store one 384-d vector, query at coarse 64/128 prefix" vs
        // a vanilla 384-d projection (rows above) at the matching prefix width.
        {4096,  384, PM::AchlioptasSparse, "achlioptas+matryoshka", 3, 5, NM::CharOnly, "char", true,   0, 0},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+matryoshka", 3, 5, NM::CharOnly, "char", true,  64, 0},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+matryoshka", 3, 5, NM::CharOnly, "char", true, 128, 0},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+matryoshka", 3, 5, NM::CharOnly, "char", true, 192, 0},

        // PQ ablation at sweet-spot dim. m subdims of dim/m floats each, k=256
        // centroids per subdim → m bytes per code (vs 384*4 = 1536 floats).
        // Compression ratios at output_dim=384: m=8 → 192×, m=16 → 96×,
        // m=32 → 48×, m=48 → 32×, m=64 → 24×.
        {4096,  384, PM::AchlioptasSparse, "achlioptas+pq", 3, 5, NM::CharOnly, "char", false, 0,  8},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+pq", 3, 5, NM::CharOnly, "char", false, 0, 16},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+pq", 3, 5, NM::CharOnly, "char", false, 0, 32},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+pq", 3, 5, NM::CharOnly, "char", false, 0, 48},
        {4096,  384, PM::AchlioptasSparse, "achlioptas+pq", 3, 5, NM::CharOnly, "char", false, 0, 64},
    };

    for (const auto& r : rows) {
        simeon::EncoderConfig cfg;
        cfg.ngram_mode = r.ngram_mode;
        cfg.ngram_min = r.ngram_min;
        cfg.ngram_max = r.ngram_max;
        cfg.sketch_dim = r.sketch_dim;
        cfg.output_dim = r.output_dim;
        cfg.projection = r.projection;
        cfg.l2_normalize = true;
        cfg.matryoshka = r.matryoshka;

        Metrics m;
        if (r.pq_m > 0) {
            // PQ training requires >= k=256 samples per subspace. Skip PQ
            // rows on tiny corpora (e.g. the meson smoke run) rather than
            // throwing.
            if (corpus.docs.size() < 256) {
                continue;
            }
            m = evaluate_pq(cfg, corpus, r.pq_m);
        } else if (r.prefix_dim > 0) {
            m = evaluate_prefix(cfg, corpus, r.prefix_dim);
        } else {
            m = evaluate(cfg, corpus);
        }
        emit(r, corpus.clusters.size(), dpc, leakage, m);
    }
    return 0;
}
