#!/usr/bin/env bash
# Produces the GPU results the docs cite, in one allocation and from one clean commit: the full
# tests, the sanitizer subset, three interleaved rounds of every benchmark suite (tagged
# final1..final3), three runs of the end-to-end suite, Nsight Compute digests for every kernel
# (profiles.sh), and the generate demo. Each step logs to its own runs/ directory; a failed step
# is reported at the end and the others still run.
#
#   scripts/pace/gpu.sh scripts/pace/final.sh
# or, so that it survives a dropped SSH connection, from a login node:
#   setsid nohup scripts/pace/gpu.sh scripts/pace/final.sh >runs/final.log 2>&1 </dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

if code_dirty; then
  echo "final.sh records results, so it needs a clean checkout" >&2
  exit 2
fi
echo "commit $(git rev-parse --short=7 HEAD), job ${SLURM_JOB_ID:-none} on $(hostname)"

failed=()
step() {
  echo "=== $(date +%H:%M:%S) $*"
  "$@" || failed+=("$*")
}

# A full rebuild, so profiling/ptxas/<sha>.txt lists every kernel's registers, spills, and shared
# memory for the commit that produced the numbers.
rm -rf build flash_lab/_C*.so
step scripts/pace/build.sh
step scripts/pace/test.sh
step scripts/pace/sanitize.sh

# Suites are interleaved rather than repeated back to back, so slow drift in clocks or
# temperature spreads over all of them.
for round in 1 2 3; do
  for suite in prefill_fp32 prefill_bf16 decode_ctx decode_batch; do
    step scripts/pace/bench.sh "$suite" --tag "final$round"
  done
done
for round in 1 2 3; do
  step scripts/pace/bench.sh e2e_llama --tag "final$round"
done

# Nsight Compute digests for every kernel, after the timed runs (profiles.sh).
step scripts/pace/profiles.sh

demo() {
  python examples/generate.py --model meta-llama/Llama-3.1-8B-Instruct --compare sdpa
}
step run_logged generate-llama8b demo

echo "=== $(date +%H:%M:%S) done"
if [ "${#failed[@]}" -gt 0 ]; then
  printf 'failed: %s\n' "${failed[@]}"
  exit 1
fi
