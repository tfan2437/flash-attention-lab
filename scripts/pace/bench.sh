#!/usr/bin/env bash
# Runs one benchmark suite on a GPU node and writes bench/results/<gpu>/<date>-<sha>-<suite>.json.
# Usage: scripts/pace/gpu.sh scripts/pace/bench.sh prefill_bf16 [--impls a,b] [--configs 0,1]
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

suite="$1"
shift
run_logged "bench-${suite}" python -m bench.run --suite "$suite" "$@"
