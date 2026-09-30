"""Llama prefill and decode steps under NVTX ranges, for Nsight Systems: where the GPU time of
a generation step goes, kernel by kernel.

    nsys profile -t cuda,nvtx python -m bench.e2e_trace --impls sdpa,flash_lab

The model runs eagerly (no torch.compile, no CUDA graphs), so every kernel launch falls inside
the range of its phase: `<impl>:prefill` for the prompt and `<impl>:step` for each greedy decode
step, each ending with a device synchronize. Step times include eager-mode host overhead; the
kernel shares are the point. bench/e2e.py measures the compiled steps.
"""

import argparse
import contextlib

import torch
from transformers import AutoModelForCausalLM, StaticCache

import flash_lab.hf


@torch.no_grad()
def run(model, cache, input_ids, steps: int, label: str | None) -> None:
    def phase(name):
        return torch.cuda.nvtx.range(f"{label}:{name}") if label else contextlib.nullcontext()

    prompt_len = input_ids.shape[1]
    cache.reset()
    with phase("prefill"):
        positions = torch.arange(prompt_len, device="cuda")
        logits = model(
            input_ids, past_key_values=cache, cache_position=positions, logits_to_keep=1
        ).logits
        torch.cuda.synchronize()
    token = logits[:, -1:].argmax(-1)
    for step in range(steps):
        with phase("step"):
            position = torch.tensor([prompt_len + step], device="cuda")
            logits = model(token, past_key_values=cache, cache_position=position).logits
            token = logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--impls", default="sdpa,flash_lab")
    parser.add_argument("--prompt-len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    flash_lab.hf.register()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, 128000, (1, args.prompt_len), generator=gen).cuda()
    cache = StaticCache(config=model.config, max_cache_len=args.prompt_len + args.steps)
    for impl in args.impls.split(","):
        model.set_attn_implementation(impl)
        run(model, cache, input_ids, 3, label=None)  # warmup, including Triton autotuning
        run(model, cache, input_ids, args.steps, label=impl)


if __name__ == "__main__":
    main()
