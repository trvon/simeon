# Research notes

This directory contains the benchmark, routing, and ablation notes behind
the published simeon numbers.

## What to cite

- **Software use:** cite the repository via [`CITATION.cff`](../CITATION.cff).
- **Benchmark claims:** cite [benchmarks.md](benchmarks.md).
- **Training-free search-space definition:** cite [training_free_space_redefinition.md](training_free_space_redefinition.md).
- **Post-retrieval gate features:** see [phase36_post_retrieval_gate_features.md](phase36_post_retrieval_gate_features.md).
- **Corpus-agnostic router invariance:** see [phase37_general_router_invariance.md](phase37_general_router_invariance.md).
- **Oracle-external theorem:** see [phase38_oracle_external_theorem.md](phase38_oracle_external_theorem.md).
- **Rank-shape risk features:** see [phase39_rank_shape_risk_features.md](phase39_rank_shape_risk_features.md).
- **Observed shape-risk fusion:** see [phase40_observed_shape_risk_fusion.md](phase40_observed_shape_risk_fusion.md).
- **Ordering-gap lexical negative:** see [phase41_ordering_gap_negative_lexical_evidence.md](phase41_ordering_gap_negative_lexical_evidence.md).
- **Structural centrality rows:** see [phase42_structural_centrality_negative.md](phase42_structural_centrality_negative.md).
- **Structural-risk diagnostic:** see [phase43_structural_risk_diagnostic.md](phase43_structural_risk_diagnostic.md).
- **Simeon embedding generator slice:** see [phase44_simeon_embedding_generator_slice.md](phase44_simeon_embedding_generator_slice.md).
- **Risk-aware 4-generator fusion:** see [phase45_risk_aware_4gen_fusion.md](phase45_risk_aware_4gen_fusion.md).
- **Continuous 4-generator dampening:** see [phase46_continuous_dampening.md](phase46_continuous_dampening.md).
- **Dynamic multi-signal dampening (negative):** see [phase47_dynamic_dampening_negative.md](phase47_dynamic_dampening_negative.md).
- **Cross-generator consensus booster (negative):** see [phase48_consensus_booster_negative.md](phase48_consensus_booster_negative.md).
- **Embedding-weighted RM3 query expansion:** see [phase49_embedding_weighted_rm3.md](phase49_embedding_weighted_rm3.md).
- **Weighted RM3 in 4-way fusion:** see [phase50_weighted_rm3_4way_fusion.md](phase50_weighted_rm3_4way_fusion.md).
- **Adaptive α weighted RM3:** see [phase51_adaptive_alpha_rm3.md](phase51_adaptive_alpha_rm3.md).
- **Diversity-aware MMR RM3:** see [phase52_diversity_aware_rm3.md](phase52_diversity_aware_rm3.md).
- **MMR β sensitivity sweep:** see [phase53_beta_sensitivity_sweep.md](phase53_beta_sensitivity_sweep.md).
- **Query-type gate + diverse RM3 fusion:** see [phase54_query_type_gate.md](phase54_query_type_gate.md).
- **Embedding-based scoring (negative):** see [phase55_embedding_scoring_negative.md](phase55_embedding_scoring_negative.md).
- **Gated 3-way ensemble:** see [phase56_gated_ensemble.md](phase56_gated_ensemble.md).
- **PPR graph re-ranking:** see [phase57_ppr_graph_rerank.md](phase57_ppr_graph_rerank.md).
- **Theorem correction (LM framework):** see [theorem_correction_lm_framework.md](theorem_correction_lm_framework.md).
- **LM-inspired weighted mixture (inert):** see [phase58_lm_mixture_inert.md](phase58_lm_mixture_inert.md).
- **Term-level LM interpolation:** see [phase59_term_level_lm.md](phase59_term_level_lm.md).
- **Pseudo-label likelihood ratio scoring:** see [phase60_pseudo_label_scoring.md](phase60_pseudo_label_scoring.md).
- **Term-concentration structural boost (negative):** see [phase61_concentration_negative.md](phase61_concentration_negative.md).
- **Stance + entity scoring (negative):** see phase62_stance_entity_negative.md
- **Contrastive scoring + query router:** see [phase63_contrastive_router.md](phase63_contrastive_router.md).
- **Split-query debate scorer (negative):** see phase64_split_query_negative.md
- **ArguAna corpus adapter:** see [phase65_arguana_adapter.md](phase65_arguana_adapter.md).
- **Simeon-embedding PRF (negative):** see [phase66_simeon_prf_negative.md](phase66_simeon_prf_negative.md).
- **Adapter-aware ensemble:** see [phase67_adapter_ensemble.md](phase67_adapter_ensemble.md).
- **Generalizable adaptive adapter design:** see [generalizable_adapter_design.md](generalizable_adapter_design.md).
- **Self-assessed score-list router:** see [phase68_self_assess_router.md](phase68_self_assess_router.md).
- **Hard-select score-list router (negative):** see [phase69_hard_select_router.md](phase69_hard_select_router.md).
- **Entropy+length 2-way router:** see [phase71_entropy_length_router.md](phase71_entropy_length_router.md).
- **Research dashboard (all numbers):** see [research_dashboard.md](research_dashboard.md).
- **Paperbridge synthesis + next directions:** see [paperbridge_synthesis_next.md](paperbridge_synthesis_next.md).
- **Corpus adapter library design:** see [corpus_adapter_design.md](corpus_adapter_design.md).
- **Router behavior and transfer limits:** cite [router_design.md](router_design.md).
- **PMI / co-occurrence projection analysis:** cite [pmi_projection.md](pmi_projection.md).
- **Prior-work list used by this repo:** see [works_cited.md](../works_cited.md).

## Scope of the shipped research artifacts

The in-tree documents preserve:

- the headline benchmark tables used in the public README
- the routing rules, tuning workflow, and honest failure modes
- negative-result documentation for PMI and cross-corpus transfer
- the prior-work list most directly used in the implementation and docs

The repository intentionally does **not** ship large raw benchmark dumps,
temporary telemetry, or local helper scripts used during development.

## Framing guidance

When describing simeon publicly:

- treat it as a **training-free retrieval backend**, not a learned semantic replacement
- separate **headline claims** from **corpus-specific ablations**
- preserve negative results alongside positive ones
- avoid treating simeon as blanket validation of any single upstream paper

NUMEN is one important antecedent, but not the only one. The library also
draws on late interaction, product quantization, sparse random projection,
nested representations, and query-difficulty routing. The notes in this folder
also document where the measured behavior does **not** support a simplistic
"this validates NUMEN" reading.
