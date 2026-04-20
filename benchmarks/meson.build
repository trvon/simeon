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

# Comparison vs a learned reference embedding model. Requires an external
# fixture (corpus.tsv, queries.tsv, qrels.tsv, reference.bin); not run as a
# meson test because the fixture is not committed. See docs/reference_fixture.md.
vs_reference_exe = executable('simeon_bench_vs_reference',
  'bench_vs_reference.cpp',
  dependencies: simeon_dep,
  install: false,
)

# Isolated SAB-smooth profile harness. Builds one Bm25Index with
# SubwordAwareBackoff/γ=5 and runs N repeats of the query loop, so
# `xctrace record --launch` / `sample` see a single busy phase without
# dilution from the 40+ other configs in simeon_bench_vs_reference.
profile_sab_exe = executable('simeon_profile_sab_smooth',
  'profile_sab_smooth.cpp',
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
