#!/usr/bin/env bash
# Produces the GPU results the docs cite, in one allocation and from one clean commit: the full
# tests, the sanitizer subset, three interleaved rounds of every benchmark suite (tagged
# final1..final3), three runs of the end-to-end suite, Nsight Compute digests for every kernel,
# and the generate demo. Each step logs to its own runs/ directory; a failed step
# is reported at the end and the others still run.
#
#   scripts/pace/gpu.sh scripts/pace/final.sh
# or, so that it survives a dropped SSH connection, from a login node:
#   setsid nohup scripts/pace/gpu.sh scripts/pace/final.sh >runs/final.log 2>&1 </dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
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

# Nsight Compute for every kernel at this commit, so the digests and the roofline describe the
# code that produced the numbers. Prefill runs at (4, 32, 32, 4096, 128) non-causal; decode at a
# 32K context, one sequence with 32 KV heads and eight with 8. The fp32 kernels are slow, so they
# run fewer calls. The Triton kernel is pinned to the configuration autotuning chose in round 1,
# so no trial launches are profiled. Timings taken under the profiler go to runs/.
profile() {  # <report> <kernel regex> <kernels per call> <bench.run args...>
  local report="$1" regex="$2" count="$3"
  shift 3
  step env NCU_COUNT="$count" NCU_SKIP=$((count * 5)) scripts/pace/ncu.sh "$report" "$regex" \
    -m bench.run --out-dir runs/bench "$@"
}
fp32=(--suite prefill_fp32 --configs 1 --causal false --n-warmup 2 --n-iters 4)
bf16=(--suite prefill_bf16 --configs 1 --causal false --n-iters 12)
profile naive-s4096-full "gemm_kernel|scale_mask_kernel|softmax_kernel" 4 "${fp32[@]}" --impls naive
profile fp32-fused-s4096-full fp32_fused_kernel 1 "${fp32[@]}" --impls fp32_fused
profile fp32-regtile-s4096-full fp32_regtile_kernel 1 "${fp32[@]}" --impls fp32_regtile
profile mma-s4096-full mma_attention_kernel 1 "${bf16[@]}" --impls mma
profile mma-pipelined-s4096-full mma_pipelined_kernel 1 "${bf16[@]}" --impls mma_pipelined
profile flash-attn-s4096-full flash_fwd_kernel 1 "${bf16[@]}" --impls flash_attn
triton_config="$(python - <<'PY'
import glob, json
path = sorted(glob.glob("bench/results/*/*-prefill_bf16-final1.json"))[-1]
for r in json.load(open(path))["results"]:
    c = r["config"]
    if r["impl"] == "triton" and c["seqlen"] == 4096 and c["heads_kv"] == 32 and not c["causal"]:
        t = r.get("triton_config") or {}
        print(f"{t['BLOCK_M']},{t['BLOCK_N']},{t['num_warps']},{t['num_stages']}")
PY
)"
echo "Triton configuration for the profile: ${triton_config:-none found}"
if [ -n "$triton_config" ]; then
  export FLASH_LAB_TRITON_CONFIG="$triton_config"
  profile triton-s4096-full _attention_fwd_kernel 1 "${bf16[@]}" --impls triton
  unset FLASH_LAB_TRITON_CONFIG
else
  failed+=("Triton profile: no configuration recorded in round 1")
fi
decode=(--suite decode_ctx --n-iters 12)
profile decode-inplace-b1-ctx32k decode_kernel 1 "${decode[@]}" --impls decode_inplace --configs 3
profile splitkv-b1-ctx32k "decode_kernel|merge_kernel" 2 "${decode[@]}" --impls splitkv --configs 3
profile flash-attn-decode-b1-ctx32k flash_fwd_splitkv 2 "${decode[@]}" --impls flash_attn \
  --configs 3
profile splitkv-b8-gqa-ctx32k "decode_kernel|merge_kernel" 2 "${decode[@]}" --impls splitkv \
  --configs 15
profile flash-attn-decode-b8-gqa-ctx32k flash_fwd_splitkv 2 "${decode[@]}" --impls flash_attn \
  --configs 15

source scripts/pace/common.sh
demo() {
  python examples/generate.py --model meta-llama/Llama-3.1-8B-Instruct --compare sdpa
}
step run_logged generate-llama8b demo

echo "=== $(date +%H:%M:%S) done"
if [ "${#failed[@]}" -gt 0 ]; then
  printf 'failed: %s\n' "${failed[@]}"
  exit 1
fi
