#!/usr/bin/env bash
# Measurements beyond final.sh, for the analysis in the docs, from one clean commit: the machine's
# configuration, Nsight Systems traces and causal Nsight Compute profiles, the split-count and
# heads-per-block sweeps, fp16 prefill, prefill against sequence length, D = 64 and 64K/128K
# decode, the fp32 tile shapes, the end-to-end suite at longer prompts and at batch 8, and the
# compiled demo. Ordered so the short and most-cited runs come first. Same conventions as
# final.sh: a failed step is reported at the end and the others still run.
#
#   setsid nohup scripts/pace/gpu.sh scripts/pace/extras.sh >runs/extras.log 2>&1 </dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "extras.sh records results, so it needs a clean checkout" >&2
  exit 2
fi
echo "commit $(git rev-parse --short=7 HEAD), job ${SLURM_JOB_ID:-none} on $(hostname)"

failed=()
step() {
  echo "=== $(date +%H:%M:%S) $*"
  "$@" || failed+=("$*")
}

# The machine, for docs/environment.md. Kept in runs/ (it includes device serial numbers).
machine() {
  nvidia-smi -q >"$RUN_DIR/nvidia-smi-q.txt"
  nvidia-smi topo -m >"$RUN_DIR/nvidia-smi-topo.txt"
  lscpu >"$RUN_DIR/lscpu.txt"
  free -g >"$RUN_DIR/free-g.txt"
  uname -a >"$RUN_DIR/uname.txt"
  { nvcc --version; ncu --version; nsys --version; compute-sanitizer --version; } \
    >"$RUN_DIR/tool-versions.txt" 2>&1
  python -m torch.utils.collect_env >"$RUN_DIR/torch-collect-env.txt" 2>&1
  uv pip freeze >"$RUN_DIR/pip-freeze.txt"
  ls "$RUN_DIR"
}
step run_logged machine machine

step scripts/pace/nsys.sh decode-steps-b8-ctx8k -m bench.decode_trace \
  --impls decode_copy,decode_inplace,splitkv --batch 8 --heads 32 --heads-kv 32 --context 8192 \
  --steps 20
step scripts/pace/nsys.sh llama8b-steps-ctx8k -m bench.e2e_trace --impls sdpa,flash_lab

# Causal prefill at (4, 32, 32, 4096, 128), next to final.sh's non-causal digests. The Triton
# configuration is the one autotuning chose for this shape in final.sh's first round.
profile() {  # <report> <kernel regex> <kernels per call> <bench.run args...>
  local report="$1" regex="$2" count="$3"
  shift 3
  step env NCU_COUNT="$count" NCU_SKIP=$((count * 5)) scripts/pace/ncu.sh "$report" "$regex" \
    -m bench.run --out-dir runs/bench "$@"
}
causal=(--suite prefill_bf16 --configs 1 --causal true --n-iters 12)
profile mma-pipelined-s4096-causal mma_pipelined_kernel 1 "${causal[@]}" --impls mma_pipelined
profile flash-attn-s4096-causal flash_fwd_kernel 1 "${causal[@]}" --impls flash_attn
triton_config="$(python - <<'PY'
import glob, json
path = sorted(glob.glob("bench/results/*/*-prefill_bf16-final1.json"))[-1]
for r in json.load(open(path))["results"]:
    c = r["config"]
    if r["impl"] == "triton" and c["seqlen"] == 4096 and c["heads_kv"] == 32 and c["causal"]:
        t = r.get("triton_config") or {}
        print(f"{t['BLOCK_M']},{t['BLOCK_N']},{t['num_warps']},{t['num_stages']}")
PY
)"
if [ -n "$triton_config" ]; then
  export FLASH_LAB_TRITON_CONFIG="$triton_config"
  profile triton-s4096-causal _attention_fwd_kernel 1 "${causal[@]}" --impls triton
  unset FLASH_LAB_TRITON_CONFIG
else
  failed+=("Triton causal profile: no configuration recorded in final1")
fi

# Split-KV: every split count on one and eight sequences, MHA and GQA, 32K keys; then the GQA
# head grouping (1, 2, or 4 query heads per block) on the H_kv = 8 configurations.
splits="$(printf 'splitkv@%s,' 1 2 4 8 16 32 64 128)"
step scripts/pace/bench.sh decode_ctx --impls "${splits%,}" --configs 3,7,11,15 --tag splits
for heads in 1 2 4; do
  step env FLASH_LAB_DECODE_HEADS_PER_BLOCK="$heads" scripts/pace/bench.sh decode_ctx \
    --impls decode_inplace,splitkv --configs 4,5,6,7,12,13,14,15 --tag "heads$heads"
done

step scripts/pace/bench.sh prefill_bf16 --dtypes fp16 --tag fp16
step scripts/pace/bench.sh prefill_seqlen
step scripts/pace/bench.sh decode_d64
step scripts/pace/bench.sh decode_long
for tile in 16x16 8x32; do
  step env FLASH_LAB_FP32_TILE="$tile" scripts/pace/bench.sh prefill_fp32 --impls fp32_fused \
    --tag "tile-$tile"
done

e2e=(--impls sdpa,flash_lab,flash_lab_cuda)
step scripts/pace/bench.sh e2e_llama "${e2e[@]}" --prompt-lens 2048,4096,16384,32768 --tag context
step scripts/pace/bench.sh e2e_llama "${e2e[@]}" --batch 8 --prompt-lens 512,4096 --tag batch8

demo() {
  python examples/generate.py --model meta-llama/Llama-3.1-8B-Instruct --compare sdpa \
    --cache static
}
step run_logged generate-llama8b-static demo

echo "=== $(date +%H:%M:%S) done"
if [ "${#failed[@]}" -gt 0 ]; then
  printf 'failed: %s\n' "${failed[@]}"
  exit 1
fi
