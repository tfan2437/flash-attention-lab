#!/usr/bin/env bash
# Nsight Compute digests for every kernel, for profiling/metrics.md and the roofline. Prefill at
# (4, 32, 32, 4096, 128), non-causal and causal; decode at a 32K context, one sequence with 32 KV
# heads and eight with 8. The fp32 kernels are slow, so they run fewer calls. The Triton kernel is
# pinned to the configuration autotuning chose for the same shape in the newest final1 run, so no
# trial launches are profiled. Timings taken under the profiler go to runs/. A failed profile is
# reported at the end and the others still run.
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh

failed=()
profile() {  # <report> <kernel regex> <kernels per call> <bench.run args...>
  local report="$1" regex="$2" count="$3"
  shift 3
  echo "=== $(date +%H:%M:%S) $report"
  env NCU_COUNT="$count" NCU_SKIP=$((count * 5)) scripts/pace/ncu.sh "$report" "$regex" \
    -m bench.run --out-dir runs/bench "$@" || failed+=("$report")
}

triton_config() {  # <causal: True|False>
  python - "$1" <<'PY'
import glob, json, sys
path = sorted(glob.glob("bench/results/*/*-prefill_bf16-final1.json"))[-1]
for r in json.load(open(path))["results"]:
    c = r["config"]
    if r["impl"] == "triton" and c["seqlen"] == 4096 and c["heads_kv"] == 32 \
            and str(c["causal"]) == sys.argv[1]:
        t = r.get("triton_config") or {}
        print(f"{t['BLOCK_M']},{t['BLOCK_N']},{t['num_warps']},{t['num_stages']}")
PY
}

fp32=(--suite prefill_fp32 --configs 1 --causal false --n-warmup 2 --n-iters 4)
profile naive-s4096-full "gemm_kernel|scale_mask_kernel|softmax_kernel" 4 "${fp32[@]}" --impls naive
profile fp32-fused-s4096-full fp32_fused_kernel 1 "${fp32[@]}" --impls fp32_fused
profile fp32-regtile-s4096-full fp32_regtile_kernel 1 "${fp32[@]}" --impls fp32_regtile

for causal in false true; do
  suffix=$([ "$causal" = true ] && echo causal || echo full)
  bf16=(--suite prefill_bf16 --configs 1 --causal "$causal" --n-iters 12)
  if [ "$causal" = false ]; then
    profile mma-s4096-full mma_attention_kernel 1 "${bf16[@]}" --impls mma
  fi
  profile "mma-pipelined-s4096-$suffix" mma_pipelined_kernel 1 "${bf16[@]}" --impls mma_pipelined
  profile "flash-attn-s4096-$suffix" flash_fwd_kernel 1 "${bf16[@]}" --impls flash_attn
  config="$(triton_config "$([ "$causal" = true ] && echo True || echo False)")"
  if [ -n "$config" ]; then
    echo "Triton configuration for the $suffix profile: $config"
    FLASH_LAB_TRITON_CONFIG="$config" profile "triton-s4096-$suffix" _attention_fwd_kernel 1 \
      "${bf16[@]}" --impls triton
  else
    failed+=("triton-s4096-$suffix: no configuration recorded in a final1 run")
  fi
done

decode=(--suite decode_ctx --n-iters 12)
profile decode-inplace-b1-ctx32k decode_kernel 1 "${decode[@]}" --impls decode_inplace --configs 3
profile splitkv-b1-ctx32k "decode_kernel|merge_kernel" 2 "${decode[@]}" --impls splitkv --configs 3
profile flash-attn-decode-b1-ctx32k flash_fwd_splitkv 2 "${decode[@]}" --impls flash_attn \
  --configs 3
profile splitkv-b8-gqa-ctx32k "decode_kernel|merge_kernel" 2 "${decode[@]}" --impls splitkv \
  --configs 15
profile flash-attn-decode-b8-gqa-ctx32k flash_fwd_splitkv 2 "${decode[@]}" --impls flash_attn \
  --configs 15

if [ "${#failed[@]}" -gt 0 ]; then
  printf 'profile failed: %s\n' "${failed[@]}"
  exit 1
fi
