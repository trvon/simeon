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
  'test_prf': 'test_prf.cpp',
  'test_sdm': 'test_sdm.cpp',
  'test_concept_mining': 'test_concept_mining.cpp',
  'test_aho_corasick': 'test_aho_corasick.cpp',
  'test_text_rank': 'test_text_rank.cpp',
  'test_persistent_homology': 'test_persistent_homology.cpp',
}

foreach name, src : simeon_tests
  exe = executable(name, src,
    dependencies: simeon_dep,
    cpp_args: simeon_test_args,
  )
  test(name, exe, suite: 'simeon')
endforeach
