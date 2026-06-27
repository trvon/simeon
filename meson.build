project('simeon', 'cpp',
  version: '0.1.0',
  license: 'GPL-3.0-or-later',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=false',
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
  ],
  meson_version: '>=0.63.0',
)

cpp = meson.get_compiler('cpp')
host_cpu = host_machine.cpu_family()

extra_cpp_args = []
extra_link_args = []

if get_option('enable_coverage')
  if cpp.get_id() == 'clang'
    extra_cpp_args += ['-fprofile-instr-generate', '-fcoverage-mapping']
    extra_link_args += ['-fprofile-instr-generate', '-fcoverage-mapping']
    message('simeon: clang source-based coverage enabled')
  elif cpp.get_id() == 'gcc'
    extra_cpp_args += ['--coverage']
    extra_link_args += ['--coverage']
    message('simeon: gcov coverage enabled')
  else
    warning('simeon: enable_coverage=true ignored; unsupported compiler @0@'.format(cpp.get_id()))
  endif
endif

simd_defines = []
simd_flags_neon = []
simd_flags_avx2 = []

has_neon = false
has_avx2 = false

if host_cpu == 'aarch64' and get_option('enable_simd_neon')
  neon_probe = '''
  #include <arm_neon.h>
  int main() {
      float32x4_t v = vdupq_n_f32(1.0f);
      (void)v;
      return 0;
  }
  '''
  if cpp.compiles(neon_probe, name: 'aarch64 NEON')
    has_neon = true
    simd_defines += '-DSIMEON_HAS_NEON=1'
    message('simeon: NEON enabled (aarch64 baseline)')
  endif
elif host_cpu in ['x86', 'x86_64'] and get_option('enable_simd_avx2')
  avx2_probe = '''
  #include <immintrin.h>
  int main() {
      __m256 v = _mm256_set1_ps(1.0f);
      __m256 r = _mm256_fmadd_ps(v, v, v);
      (void)r;
      return 0;
  }
  '''
  if cpp.compiles(avx2_probe, args: ['-mavx2', '-mfma'], name: 'AVX2+FMA')
    has_avx2 = true
    simd_flags_avx2 += ['-mavx2', '-mfma']
    simd_defines += '-DSIMEON_HAS_AVX2=1'
    message('simeon: AVX2+FMA enabled')
  endif
endif

simeon_inc = include_directories('include')

common_srcs = [
  'src/tokenizer.cpp',
  'src/hasher.cpp',
  'src/hash_xxh64.cpp',
  'src/hash_crc32c.cpp',
  'src/hash_tabulation.cpp',
  'src/projection.cpp',
  'src/pq.cpp',
  'src/simeon.cpp',
  'src/bm25.cpp',
  'src/fusion.cpp',
  'src/minhash.cpp',
  'src/tokenizer_bpe.cpp',
  'src/matryoshka.cpp',
  'src/pmi.cpp',
  'src/query_router.cpp',
  'src/prf.cpp',
  'src/concept_mining.cpp',
  'src/corpus_adapter.cpp',
  'src/retrieval_strategy.cpp',
  'src/aho_corasick.cpp',
  'src/text_rank.cpp',
  'src/fragment_geometry.cpp',
  'src/arch/scalar.cpp',
  'src/persistent_homology.cpp',
]

arch_srcs = []
if has_neon
  arch_srcs += 'src/arch/neon.cpp'
endif

simeon_avx2_link_whole = []
if has_avx2
  simeon_avx2_obj = static_library('simeon_avx2',
    'src/arch/avx2.cpp',
    include_directories: simeon_inc,
    cpp_args: simd_defines + simd_flags_avx2 + extra_cpp_args,
    install: false,
  )
  simeon_avx2_link_whole += simeon_avx2_obj
endif

simeon_lib = static_library('simeon',
  common_srcs + arch_srcs,
  include_directories: simeon_inc,
  cpp_args: simd_defines + extra_cpp_args,
  link_args: extra_link_args,
  link_whole: simeon_avx2_link_whole,
  install: true,
)

simeon_dep = declare_dependency(
  link_with: simeon_lib,
  include_directories: simeon_inc,
  compile_args: simd_defines + extra_cpp_args,
  link_args: extra_link_args,
)

meson.override_dependency('simeon', simeon_dep)

pkg = import('pkgconfig')
pkg.generate(simeon_lib,
  name: 'simeon',
  description: 'Simeon: training-free embedding backend',
  subdirs: ['simeon'],
)

install_headers(
  [
    'include/simeon/simeon.hpp',
    'include/simeon/tokenizer.hpp',
    'include/simeon/hasher.hpp',
    'include/simeon/projection.hpp',
    'include/simeon/pq.hpp',
    'include/simeon/simd.hpp',
    'include/simeon/bm25.hpp',
    'include/simeon/fusion.hpp',
    'include/simeon/minhash.hpp',
    'include/simeon/tokenizer_bpe.hpp',
    'include/simeon/hash_tabulation.hpp',
    'include/simeon/matryoshka.hpp',
    'include/simeon/pmi.hpp',
    'include/simeon/manifold.hpp',
    'include/simeon/query_router.hpp',
    'include/simeon/prf.hpp',
    'include/simeon/concept_mining.hpp',
    'include/simeon/corpus_adapter.hpp',
    'include/simeon/retrieval_strategy.hpp',
    'include/simeon/aho_corasick.hpp',
    'include/simeon/text_rank.hpp',
    'include/simeon/fragment_geometry.hpp',
    'include/simeon/persistent_homology.hpp',
  ],
  subdir: 'simeon',
)

if get_option('enable_tests')
  subdir('tests')
endif

if get_option('enable_benchmarks')
  subdir('benchmarks')
endif

if get_option('enable_fuzz')
  subdir('tests/fuzz')
endif

summary({
  'C++ Standard': get_option('cpp_std'),
  'Build Type': get_option('buildtype'),
  'NEON': has_neon,
  'AVX2+FMA': has_avx2,
  'Tests': get_option('enable_tests'),
  'Benchmarks': get_option('enable_benchmarks'),
  'Fuzz harnesses': get_option('enable_fuzz'),
  'Coverage': get_option('enable_coverage'),
}, section: 'simeon configuration')
