#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "simeon/bm25.hpp"
#include "simeon/concept_mining.hpp"
#include "simeon/fragment_geometry.hpp"
#include "simeon/pmi.hpp"
#include "simeon/query_router.hpp"
#include "simeon/simeon.hpp"

namespace {

bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

void test_bm25_and_router_case_invariance() {
    simeon::Bm25Index idx;
    idx.add_doc("RNA polymerase transcribes DNA into messenger RNA");
    idx.add_doc("T cells mediate adaptive immunity against viral antigens");
    idx.add_doc("RNA polymerase II requires general transcription factors");
    idx.add_doc("Cytotoxic T cells recognize MHC class I peptides");
    idx.add_doc("Baseline filler document with unrelated vocabulary alpha beta");
    idx.finalize();

    // Bm25Index lookup methods must fold ASCII case consistently with add_doc/score.
    for (const auto& [upper, lower] : std::vector<std::pair<std::string_view, std::string_view>>{
             {"RNA", "rna"},
             {"Polymerase", "polymerase"},
             {"T", "t"},
             {"Cells", "cells"},
             {"MHC", "mhc"},
         }) {
        assert(idx.hash_term(upper) == idx.hash_term(lower));
        assert(idx.df(upper) == idx.df(lower));
        assert(idx.df(upper) > 0u);
        assert(approx(idx.idf(upper), idx.idf(lower)));
        assert(idx.idf(upper) > 0.0f);
        assert(idx.total_tf(upper) == idx.total_tf(lower));
        assert(idx.total_tf(upper) > 0u);
    }

    // QueryRouter features must be identical across ASCII case variants.
    simeon::QueryRouter router(idx);
    const auto f_mixed = router.features("RNA Polymerase and T Cells");
    const auto f_lower = router.features("rna polymerase and t cells");
    const auto f_upper = router.features("RNA POLYMERASE AND T CELLS");

    assert(f_mixed.n_terms == f_lower.n_terms);
    assert(f_mixed.n_terms == f_upper.n_terms);
    assert(approx(f_mixed.oov_rate, f_lower.oov_rate));
    assert(approx(f_mixed.oov_rate, f_upper.oov_rate));
    assert(f_mixed.oov_rate < 0.5f); // "RNA", "Polymerase", "T", "Cells" are all in-vocab
    assert(approx(f_mixed.avg_idf, f_lower.avg_idf));
    assert(approx(f_mixed.avg_idf, f_upper.avg_idf));
    assert(approx(f_mixed.max_idf, f_lower.max_idf));
    assert(approx(f_mixed.min_idf, f_lower.min_idf));
    assert(approx(f_mixed.idf_stddev, f_lower.idf_stddev));
    assert(approx(f_mixed.scq_sum, f_lower.scq_sum));
    assert(approx(f_mixed.scq_sum, f_upper.scq_sum));
    assert(approx(f_mixed.simplified_clarity, f_lower.simplified_clarity));
    assert(approx(f_mixed.simplified_clarity, f_upper.simplified_clarity));
}

void test_concept_mining_case_invariance() {
    const std::vector<std::string> docs = {
        "RNA polymerase transcribes templates accurately",
        "RNA polymerase recruits elongation complexes",
        "RNA polymerase pauses near promoter regions",
        "T cells target infected host cells",
        "T cells proliferate during immune response",
        "T cells secrete inflammatory cytokines",
        "unrelated filler one two three four five",
        "another filler six seven eight nine ten",
    };
    std::vector<std::string_view> views;
    views.reserve(docs.size());
    simeon::Bm25Index idx;
    for (const auto& d : docs) {
        views.emplace_back(d);
        idx.add_doc(d);
    }
    idx.finalize();

    simeon::ConceptConfig cfg;
    cfg.min_ttf = 2;
    cfg.pmi_floor = 0.2f;
    cfg.concept_weight = 1.0f;

    const auto concepts = simeon::mine_concepts(idx, views, cfg);
    const auto h_rna_pol =
        simeon::ConceptIndex::hash_bigram(idx.hash_term("RNA"), idx.hash_term("polymerase"));
    const auto h_t_cells =
        simeon::ConceptIndex::hash_bigram(idx.hash_term("T"), idx.hash_term("cells"));
    assert(concepts.find(h_rna_pol) != nullptr);
    assert(concepts.find(h_t_cells) != nullptr);

    std::vector<float> s_mixed(docs.size(), 0.0f);
    std::vector<float> s_lower(docs.size(), 0.0f);
    std::vector<float> s_upper(docs.size(), 0.0f);
    simeon::score_bm25_with_concepts(idx, concepts, "RNA Polymerase", 0.5f, s_mixed);
    simeon::score_bm25_with_concepts(idx, concepts, "rna polymerase", 0.5f, s_lower);
    simeon::score_bm25_with_concepts(idx, concepts, "RNA POLYMERASE", 0.5f, s_upper);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        assert(approx(s_mixed[i], s_lower[i]));
        assert(approx(s_mixed[i], s_upper[i]));
    }
    assert(s_mixed[0] > s_mixed[6]);

    std::vector<float> tc_mixed(docs.size(), 0.0f);
    std::vector<float> tc_lower(docs.size(), 0.0f);
    concepts.score("T Cells", tc_mixed);
    concepts.score("t cells", tc_lower);
    for (std::size_t i = 0; i < docs.size(); ++i) {
        assert(approx(tc_mixed[i], tc_lower[i]));
    }
    assert(tc_mixed[3] > 0.0f);
}

void test_fragment_geometry_signature_and_sif_case_invariance() {
    const std::vector<std::string> docs = {
        "RNA polymerase catalyzes transcription of messenger RNA in eukaryotic nuclei.",
        "T cells recognize peptide antigens presented by major histocompatibility complexes.",
        "Weather patterns shift across coastal mountains during winter storms.",
    };
    std::vector<std::string_view> views;
    views.reserve(docs.size());
    simeon::Bm25Index idx;
    for (const auto& d : docs) {
        views.emplace_back(d);
        idx.add_doc(d);
    }
    idx.finalize();

    simeon::PmiConfig pcfg;
    pcfg.target_rank = 16;
    pcfg.min_token_count = 1;
    auto pmi = simeon::PmiEmbeddings::learn(views, pcfg);

    simeon::EncoderConfig ecfg;
    ecfg.ngram_mode = simeon::NGramMode::WordOnly;
    ecfg.ngram_min = 1;
    ecfg.ngram_max = 1;
    ecfg.sketch_dim = 0;
    ecfg.output_dim = pmi.dim();
    ecfg.projection = simeon::ProjectionMode::None;
    ecfg.l2_normalize = true;
    ecfg.pmi_rows = &pmi;
    simeon::Encoder enc(ecfg);

    const auto frags_mixed = simeon::build_doc_semantic_fragments_richcov_sif(
        enc, "RNA Polymerase catalyzes transcription.", idx, 2, 8, 0.8f, 0.8f);
    const auto frags_lower = simeon::build_doc_semantic_fragments_richcov_sif(
        enc, "rna polymerase catalyzes transcription.", idx, 2, 8, 0.8f, 0.8f);
    assert(frags_mixed.size() == frags_lower.size());
    assert(!frags_mixed.empty());
    for (std::size_t i = 0; i < frags_mixed.size(); ++i) {
        assert(frags_mixed[i].vec.size() == frags_lower[i].vec.size());
        for (std::size_t d = 0; d < frags_mixed[i].vec.size(); ++d) {
            assert(approx(frags_mixed[i].vec[d], frags_lower[i].vec[d]));
        }
        assert(frags_mixed[i].signature.terms.size() == frags_lower[i].signature.terms.size());
        for (std::size_t t = 0; t < frags_mixed[i].signature.terms.size(); ++t) {
            assert(frags_mixed[i].signature.terms[t].hash ==
                   frags_lower[i].signature.terms[t].hash);
            assert(approx(frags_mixed[i].signature.terms[t].weight,
                          frags_lower[i].signature.terms[t].weight));
        }
    }
}

} // namespace

int main() {
    test_bm25_and_router_case_invariance();
    test_concept_mining_case_invariance();
    test_fragment_geometry_signature_and_sif_case_invariance();
    return 0;
}
