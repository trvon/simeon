# simeon CI

These workflows run in the standalone `trvon/simeon` repo on push to
`master`/`main`, on pull requests, and via `workflow_dispatch`.

To validate them locally:

```sh
# Matrix build (subset)
meson setup build && meson compile -C build && meson test -C build

# Sanitizers (each one, separately — they don't compose)
for s in address undefined thread; do
  meson setup build-$s -Db_sanitize=$s -Db_lundef=false --buildtype=debug
  meson compile -C build-$s
  meson test -C build-$s --print-errorlogs
done

# Coverage (clang)
CC=clang CXX=clang++ meson setup buildcov -Denable_coverage=true
meson compile -C buildcov
LLVM_PROFILE_FILE=buildcov/test-%p.profraw meson test -C buildcov
llvm-profdata merge -sparse buildcov/*.profraw -o merged.profdata
llvm-cov report buildcov/tests/test_* -instr-profile=merged.profdata
```
