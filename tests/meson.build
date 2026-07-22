simeon_test_args = ['-DSIMEON_TESTING=1']

simeon_tests = {
  'test_tokenizer': 'test_tokenizer.cpp',
  'test_tokenizer_modes': 'test_tokenizer_modes.cpp',
  'test_hasher': 'test_hasher.cpp',
  'test_projection': 'test_projection.cpp',
  'test_projection_modes': 'test_projection_modes.cpp',
  'test_pq': 'test_pq.cpp',
  'test_matryoshka': 'test_matryoshka.cpp',
  'test_encoder': 'test_encoder.cpp',
  'test_simd_parity': 'test_simd_parity.cpp',
  'test_coverage': 'test_coverage.cpp',
  'test_bm25': 'test_bm25.cpp',
  'test_fusion': 'test_fusion.cpp',
  'test_minhash': 'test_minhash.cpp',
  'test_tokenizer_bpe': 'test_tokenizer_bpe.cpp',
  'test_sparse_jl': 'test_sparse_jl.cpp',
  'test_matryoshka_tuner': 'test_matryoshka_tuner.cpp',
  'test_projection_fwht': 'test_projection_fwht.cpp',
  'test_bm25_variants': 'test_bm25_variants.cpp',
  'test_query_router': 'test_query_router.cpp',
  'test_query_router_post_retrieval': 'test_query_router_post_retrieval.cpp',
  'test_pmi': 'test_pmi.cpp',
  'test_hashed_idf': 'test_hashed_idf.cpp',
  'test_prf': 'test_prf.cpp',
  'test_sdm': 'test_sdm.cpp',
  'test_concept_mining': 'test_concept_mining.cpp',
  'test_aho_corasick': 'test_aho_corasick.cpp',
  'test_text_rank': 'test_text_rank.cpp',
  'test_persistent_homology': 'test_persistent_homology.cpp',
  'test_fragment_geometry': 'test_fragment_geometry.cpp',
}

foreach name, src : simeon_tests
  exe = executable(name, src,
    dependencies: simeon_dep,
    cpp_args: simeon_test_args,
  )
  test(name, exe, suite: 'simeon')
endforeach

experiment_support_test = executable('test_experiment_support',
  ['test_experiment_support.cpp', '../benchmarks/experiment_support.cpp',
   '../benchmarks/fusion_experiment_support.cpp',
   '../benchmarks/coordinate_experiment_support.cpp',
   '../benchmarks/family_atlas_experiment_support.cpp'],
  dependencies: simeon_dep,
  include_directories: include_directories('../benchmarks'),
  cpp_args: simeon_test_args,
)
test('test_experiment_support', experiment_support_test, suite: 'simeon')

reranker_experiment_support_test = executable('test_reranker_experiment_support',
  ['test_reranker_experiment_support.cpp', '../benchmarks/experiment_support.cpp',
   '../benchmarks/reranker_experiment_support.cpp'],
  dependencies: simeon_dep,
  include_directories: include_directories('../benchmarks'),
  cpp_args: simeon_test_args,
)
test('test_reranker_experiment_support', reranker_experiment_support_test, suite: 'simeon')

ffi_rerank_test = executable('test_ffi_rerank',
  ['test_ffi_rerank.cpp', '../ffi/simeon_ffi.cpp'],
  dependencies: simeon_dep,
  include_directories: include_directories('../ffi'),
  cpp_args: simeon_test_args,
)
test('test_ffi_rerank', ffi_rerank_test, suite: 'simeon')

python = import('python').find_installation()
test('test_prepare_beir_fixture', python,
  args: [files('test_prepare_beir_fixture.py')],
  suite: 'simeon',
)

lean = find_program('lean', required: false)
if lean.found()
  test('check_experiment_contract', lean,
    args: [meson.project_source_root() / 'formal/ExperimentContract.lean'],
    suite: 'simeon',
  )
endif
