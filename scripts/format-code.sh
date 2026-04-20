#!/usr/bin/env bash
# Minimal clang-format runner for simeon. Supports --check / --git / --serial /
# --stdin-files / -v and is scoped to the simeon source tree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="format"
GIT_ONLY=false
READ_STDIN=false
VERBOSE=false

while (($#)); do
  case "$1" in
    -c|--check) MODE="check";;
    -g|--git) GIT_ONLY=true;;
    -s|--serial) :;;
    --stdin-files) READ_STDIN=true;;
    -v|--verbose) VERBOSE=true;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--check] [--git] [--stdin-files] [-v]
  --check        Verify formatting; exit 1 if changes would be made.
  --git          Format files changed in git (staged + unstaged).
  --stdin-files  Read newline-separated file list from stdin.
  -v, --verbose  Print each file.
EOF
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
  shift
done

command -v clang-format >/dev/null || { echo "clang-format not installed" >&2; exit 1; }

collect() {
  if $READ_STDIN; then
    while IFS= read -r f; do [[ -n "$f" ]] && echo "$f"; done
  elif $GIT_ONLY; then
    git -C "$PROJECT_ROOT" diff --name-only HEAD -- '*.cpp' '*.hpp' '*.h' 2>/dev/null || true
  else
    find "$PROJECT_ROOT" \
      \( -path "*/build*" -o -path "*/third_party/*" \) -prune -o \
      \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.cc' \) -print
  fi
}

rc=0
while IFS= read -r f; do
  [[ -f "$f" ]] || continue
  $VERBOSE && echo "fmt: $f"
  if [[ "$MODE" == "check" ]]; then
    if ! clang-format --style=file --dry-run --Werror "$f" >/dev/null 2>&1; then
      echo "needs formatting: $f"
      rc=1
    fi
  else
    clang-format --style=file -i "$f"
  fi
done < <(collect)

exit "$rc"
