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
# meson test because the fixture is not committed. Gated behind enable_research
# because it depends on research-only components (GloVe, SelfAssessRouter, etc.).
if get_option('enable_research')
vs_reference_exe = executable('simeon_bench_vs_reference',
  'bench_vs_reference.cpp',
  dependencies: simeon_dep,
  install: false,
)

vs_reference_research_exe = executable('simeon_bench_vs_reference_research',
  'bench_vs_reference_research.cpp',
  dependencies: simeon_dep,
  install: false,
)
endif

# Isolated SAB-smooth profile harness. Builds one Bm25Index with
# SubwordAwareBackoff/γ=5 and runs N repeats of the query loop, so
# `xctrace record --launch` / `sample` see a single busy phase without
# dilution from the 40+ other configs in simeon_bench_vs_reference.
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
