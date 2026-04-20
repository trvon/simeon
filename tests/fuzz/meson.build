# libFuzzer harnesses. Clang-only, and only when the runtime is actually
# available. Apple's bundled clang ships without libFuzzer; on macOS install
# llvm via Homebrew and point `CXX` / `LDFLAGS` at it before configuring.

if cpp.get_id() != 'clang'
  warning('simeon: enable_fuzz=true ignored; libFuzzer requires clang')
  subdir_done()
endif

if not cpp.has_link_argument('-fsanitize=fuzzer')
  warning('simeon: enable_fuzz=true ignored; libFuzzer runtime unavailable for this clang')
  subdir_done()
endif

fuzz_args = ['-fsanitize=fuzzer']
fuzz_link_args = ['-fsanitize=fuzzer']

fuzz_targets = {
  'fuzz_tokenize': 'fuzz_tokenize.cpp',
  'fuzz_encode': 'fuzz_encode.cpp',
}

foreach name, src : fuzz_targets
  executable(name, src,
    dependencies: simeon_dep,
    cpp_args: fuzz_args,
    link_args: fuzz_link_args,
  )
endforeach
