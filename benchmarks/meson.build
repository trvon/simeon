microbench_exe = executable('simeon_microbench',
  'microbench.cpp',
  dependencies: simeon_dep,
  install: false,
)

accuracy_exe = executable('simeon_accuracy_bench',
  'accuracy_bench.cpp',
  dependencies: simeon_dep,
  install: false,
)

pq_query_exe = executable('simeon_pq_query_bench',
  'pq_query_bench.cpp',
  dependencies: simeon_dep,
  install: false,
)

# Isolated SAB-smooth profile harness. Builds one Bm25Index with
# SubwordAwareBackoff/γ=5 and runs N repeats of the query loop, so
# `xctrace record --launch` / `sample` see a single busy phase.
profile_sab_exe = executable('simeon_profile_sab_smooth',
  'profile_sab_smooth.cpp',
  dependencies: simeon_dep,
  install: false,
)

profile_fragment_geometry_exe = executable('simeon_profile_fragment_geometry',
  'profile_fragment_geometry.cpp',
  dependencies: simeon_dep,
  install: false,
)

aho_corasick_exe = executable('simeon_aho_corasick_bench',
  'aho_corasick_bench.cpp',
  dependencies: simeon_dep,
  install: false,
)

text_rank_exe = executable('simeon_text_rank_bench',
  'text_rank_bench.cpp',
  dependencies: simeon_dep,
  install: false,
)

# Recipe-level accuracy bench: evaluates the promoted retrieval recipes
# (convex WSDM z-fusion, fused-feedback RM3, fragment geometry) end-to-end
# against a reference fixture. Needs an out-of-tree fixture dir; no CI test.
recipe_accuracy_exe = executable('simeon_recipe_accuracy_bench',
  'recipe_accuracy_bench.cpp',
  dependencies: simeon_dep,
  install: false,
)

# Reusable, manifest-driven exact retrieval instrument. The support layer is
# shared with a unit-test target; fixtures and result artifacts stay out of CI.
embedding_experiment_exe = executable('simeon_embedding_experiment',
  ['embedding_experiment_bench.cpp', 'experiment_support.cpp',
   'fusion_experiment_support.cpp', 'coordinate_experiment_support.cpp',
   'family_atlas_experiment_support.cpp'],
  dependencies: simeon_dep,
  install: false,
)

experiment_smoke_manifest = meson.project_source_root() / 'tests/data/experiment_smoke.exp'
experiment_smoke_fixture = meson.project_source_root() / 'tests/data/experiment_fixture'
test('simeon_embedding_experiment_smoke', embedding_experiment_exe,
  args: [experiment_smoke_manifest, '--fixture', experiment_smoke_fixture,
         '--split', 'dev', '--code-revision', 'ci'],
  suite: 'simeon',
  timeout: 30,
)

# RM1-build microbench: forward-index build_relevance_model vs the reference
# O(corpus) inverted scan, across corpus scales. Doubles as a bit-identity
# regression gate (exits non-zero on RM mismatch).
bench_prf_exe = executable('simeon_bench_prf',
  'bench_prf.cpp',
  dependencies: simeon_dep,
  install: false,
)

# Smoke-mode target that runs a very small sweep during meson test so CI
# can catch regressions without the full throughput run.
test('simeon_microbench_smoke', microbench_exe,
  args: ['200', '256'],
  suite: 'simeon',
  timeout: 120,
)

# Accuracy smoke — tiny corpus, one run per projection. Catches the "did
# we break topical structure?" regression without the full sweep.
test('simeon_accuracy_smoke', accuracy_exe,
  args: ['10', '40', '0.05'],
  suite: 'simeon',
  timeout: 120,
)

# AC smoke — 64 KiB input, small dictionaries only, catches regressions in
# build or match kernels without the full 1 MiB sweep.
test('simeon_aho_corasick_smoke', aho_corasick_exe,
  args: ['65536'],
  suite: 'simeon',
  timeout: 60,
)

# TextRank smoke — 20 docs × {4, 8, 16} sentences. Runs under a second.
test('simeon_text_rank_smoke', text_rank_exe,
  args: ['20'],
  suite: 'simeon',
  timeout: 30,
)

# PRF RM1-build smoke — two tiny scales; asserts forward/inverted RM bit-identity.
test('simeon_bench_prf_smoke', bench_prf_exe,
  args: ['500', '2000'],
  suite: 'simeon',
  timeout: 60,
)
