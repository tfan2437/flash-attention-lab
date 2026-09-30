#!/usr/bin/env bash
# Runs the sanitize test subset (tests/test_sanitize.py) under each compute-sanitizer tool.
# PyTorch's caching allocator is turned off so that an out-of-bounds access cannot hide inside a
# larger cached block.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

sanitize() {
  export PYTORCH_NO_CUDA_MEMORY_CACHING=1
  for tool in memcheck racecheck synccheck initcheck; do
    echo "== compute-sanitizer --tool $tool"
    compute-sanitizer --tool "$tool" --error-exitcode 1 \
      python -m pytest -q -p no:cacheprovider -m sanitize tests/test_sanitize.py "$@"
  done
}

run_logged sanitize sanitize "$@"
mkdir -p profiling/sanitize
grep -E "^== |^=+ (ERROR|RACECHECK) SUMMARY|passed|failed" "$RUN_DIR/stdout.log" \
  >"profiling/sanitize/${GIT_SHA}.txt" || true
echo "sanitizer summary: profiling/sanitize/${GIT_SHA}.txt"
