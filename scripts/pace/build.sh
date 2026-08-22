#!/usr/bin/env bash
# Builds flash_lab._C in place with ptxas resource reporting. The per-kernel register, spill, and
# shared-memory lines are saved to profiling/ptxas/<sha>.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

build() {
  nvcc --version | tail -n 2
  FLASH_LAB_PTXAS_VERBOSE=1 python setup.py build_ext --inplace
}

run_logged build build "$@"
grep -E "ptxas (info|warning)" "$RUN_DIR/stdout.log" >"$RUN_DIR/ptxas.txt" || true
mkdir -p profiling/ptxas
cp "$RUN_DIR/ptxas.txt" "profiling/ptxas/${GIT_SHA}.txt"
echo "ptxas summary: profiling/ptxas/${GIT_SHA}.txt ($(wc -l <"$RUN_DIR/ptxas.txt") lines)"
