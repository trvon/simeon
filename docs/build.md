# Building simeon

## Requirements

- C++20 toolchain — Clang 16+ or GCC 13+ (MSVC 19.36+ untested but expected to work).
- Meson ≥ 0.63.
- Ninja (or any other Meson backend).

simeon has zero third-party runtime dependencies.

## Standard build

```sh
meson setup build
meson compile -C build
meson test -C build
```

Output:

- `build/libsimeon.a` — the static library.
- `build/tests/test_*` — unit tests, runnable individually or via `meson test`.
- `build/benchmarks/simeon_microbench` — throughput microbench.
- `build/benchmarks/simeon_accuracy_bench` — clustered-corpus retrieval bench (emits JSONL).

## Build options

| Option                | Default | Effect                                                   |
|-----------------------|--------:|----------------------------------------------------------|
| `enable_simd_neon`    |  `true` | Enable NEON kernels on aarch64 (auto-probed).            |
| `enable_simd_avx2`    |  `true` | Enable AVX2+FMA kernels on x86_64 (auto-probed). Set `false` for distribution builds targeting older x86. |
| `enable_tests`        |  `true` | Build the unit-test suite.                               |
| `enable_benchmarks`   |  `true` | Build the microbench + accuracy bench.                   |
| `enable_fuzz`         | `false` | Build libFuzzer harnesses (clang only).                  |
| `enable_coverage`     | `false` | Inject `-fprofile-instr-generate -fcoverage-mapping` (clang) or `--coverage` (gcc). |

Examples:

```sh
# x86 native, AVX2 on
meson setup build -Denable_simd_avx2=true

# library only, no tests / benches
meson setup build -Denable_tests=false -Denable_benchmarks=false

# clang coverage build
CC=clang CXX=clang++ meson setup build-cov -Denable_coverage=true --buildtype=debug
meson compile -C build-cov
LLVM_PROFILE_FILE=build-cov/meson-logs/profraw/test-%p-%m.profraw meson test -C build-cov
```

To render reports, merge the `.profraw` files with `llvm-profdata` and feed the test binaries to `llvm-cov`. Latest run on Apple M-series, NEON, 12 tests:

| Scope                           | Region | Function | Line   | Branch |
|---------------------------------|-------:|---------:|-------:|-------:|
| `src/` + `include/` overall     |  91.8% |    83.7% |  90.9% |  92.5% |

100% coverage: `tokenizer.cpp`, `hash_xxh64.cpp`, `arch/scalar.cpp`, `arch/neon.cpp`. Weakest is `hash_crc32c.cpp` (60% function / 59.5% line) — the slice-by-1 software fallback never fires on aarch64 because the hardware `__crc32cd` path takes everything. A CI run on x86_64 with hardware CRC disabled would close that gap.

## Sanitizers

Standard Meson `b_sanitize` works because simeon has no external libs:

```sh
# AddressSanitizer + UndefinedBehaviorSanitizer
meson setup build-asan -Db_sanitize=address,undefined -Db_lundef=false
meson test -C build-asan

# ThreadSanitizer (validates the thread-local sketch buffer in encode_one)
meson setup build-tsan -Db_sanitize=thread -Db_lundef=false
meson test -C build-tsan
```

Both should run clean. simeon is small enough that the full test suite finishes in <30 s under any sanitizer.

## Consuming simeon from another Meson project

```meson
simeon_proj = subproject('simeon')
simeon_dep  = simeon_proj.get_variable('simeon_dep')

executable('myapp', 'main.cpp', dependencies: simeon_dep)
```

Or, after `meson install`:

```meson
simeon_dep = dependency('simeon')
```

## Vendoring as plain sources

If your build system isn't Meson, copy `include/simeon/` and `src/` into your tree and compile every `.cpp` under `src/` with `-Iinclude`. Architecture-specific files (`src/arch/neon.cpp`, `src/arch/avx2.cpp`) should only be compiled when the corresponding ISA is available; gate them with `__aarch64__` / `__AVX2__` or your build system's equivalent. AVX2 sources require `-mavx2 -mfma`.

## Reproducing benchmarks

```sh
./build/benchmarks/simeon_microbench 1000 256 > microbench.jsonl
./build/benchmarks/simeon_accuracy_bench 50 60 0.05 > accuracy.jsonl
```

The accuracy bench takes `(docs_per_cluster, words_per_doc, leakage)` as positional args. See [benchmarks.md](benchmarks.md) for interpretation and published summary tables.
