#!/usr/bin/env bash
# Measurements beyond final.sh, for the analysis in the docs, from one clean commit: the machine's
# configuration, the split-count and heads-per-block sweeps, fp16 prefill, prefill against
# sequence length, D = 64 and 64K/128K decode, the fp32 tile shapes, the end-to-end suite at
# longer prompts and at batch 8, the compiled demo, and then the profilers: Nsight Compute for
# every kernel (profiles.sh) and the Nsight Systems traces. Same conventions as final.sh: a
# failed step is reported at the end and the others still run.
#
#   setsid nohup scripts/pace/gpu.sh scripts/pace/extras.sh >runs/extras.log 2>&1 </dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/pace/env.sh
source scripts/pace/common.sh

if code_dirty; then
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

# Profilers last: they rewrite committed digests, which the timed runs above must not see.
step scripts/pace/profiles.sh
step scripts/pace/nsys.sh decode-steps-b8-ctx8k -m bench.decode_trace \
  --impls decode_copy,decode_inplace,splitkv --batch 8 --heads 32 --heads-kv 32 --context 8192 \
  --steps 20
step scripts/pace/nsys.sh llama8b-steps-ctx8k -m bench.e2e_trace --impls sdpa,flash_lab

echo "=== $(date +%H:%M:%S) done"
if [ "${#failed[@]}" -gt 0 ]; then
  printf 'failed: %s\n' "${failed[@]}"
  exit 1
fi
