"""End-to-end generation timing for a Llama checkpoint, by attention implementation.

    python -m bench.e2e --model meta-llama/Llama-3.1-8B-Instruct
    python -m bench.e2e --model meta-llama/Llama-3.2-1B-Instruct --prompt-lens 512 --runs 1

Every implementation generates --new-tokens greedy tokens from the same synthetic prompt (random
token ids, fixed seed) with a static KV cache, so generate() compiles the decode step with
torch.compile (fullgraph, CUDA graphs) for each of them alike; untimed warmup runs pay for the
compilation and the Triton autotuning. A streamer timestamps every step: the first gap is the
time to first token (prefill), the rest are decode steps. Generated tokens are compared with the
first implementation's. The JSON goes where bench/run.py puts its results.

flash_lab runs the default kernels (Triton prefill, split-KV decode); flash_lab_cuda swaps in the
CUDA prefill kernel (mma_pipelined).
"""

import argparse
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, CompileConfig
from transformers.generation.streamers import BaseStreamer

import flash_lab.hf
from bench.env import collect_env
from bench.run import recordable, write_results

IMPLS = ["sdpa", "flash_attention_2", "flash_lab", "flash_lab_cuda"]


class StepClock(BaseStreamer):
    """generate() hands the streamer the prompt once, then each step's new tokens."""

    def __init__(self):
        self.times: list[float] = []

    def put(self, value):
        torch.cuda.synchronize()
        self.times.append(time.perf_counter())

    def end(self):
        pass


def generate(model, input_ids, new_tokens: int, eos_id: int) -> tuple[torch.Tensor, list[float]]:
    clock = StepClock()
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=eos_id,
        cache_implementation="static",
        compile_config=CompileConfig(fullgraph=True),
        streamer=clock,
    )
    gaps = [(b - a) * 1e3 for a, b in zip(clock.times, clock.times[1:], strict=False)]
    return out[:, input_ids.shape[1] :], gaps


def first_difference(tokens: torch.Tensor, reference: torch.Tensor) -> int | None:
    differ = (tokens != reference).any(dim=0).nonzero()
    return int(differ[0]) if len(differ) else None


def bench_impl(model, impl, prompts, args, eos_id, reference_tokens) -> list[dict]:
    model.set_attn_implementation(impl)
    torch._dynamo.reset()
    results = []
    for prompt_len, input_ids in prompts.items():
        config = {"prompt_len": prompt_len, "batch": args.batch, "new_tokens": args.new_tokens}
        try:
            for _ in range(args.n_warmup):
                generate(model, input_ids, args.new_tokens, eos_id)
            runs = [generate(model, input_ids, args.new_tokens, eos_id) for _ in range(args.runs)]
        except (RuntimeError, ValueError, NotImplementedError, ImportError) as exc:
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            results.append({"config": config, "impl": impl, "status": "error", "reason": reason})
            print(f"{impl:<18} prompt {prompt_len:>6}: error: {reason}", flush=True)
            continue

        tokens = runs[0][0]
        reference = reference_tokens.setdefault(prompt_len, (impl, tokens))
        ttft = [gaps[0] for _, gaps in runs]
        steps = [step for _, gaps in runs for step in gaps[1:]]
        step_ms = statistics.median(steps)
        result = {
            "config": config,
            "impl": impl,
            "status": "ok",
            "ttft_ms": {"median": statistics.median(ttft), "min": min(ttft), "max": max(ttft)},
            "decode_step_ms": {
                "median": step_ms,
                "p10": statistics.quantiles(steps, n=10)[0],
                "p90": statistics.quantiles(steps, n=10)[-1],
            },
            "decode_tokens_per_s": args.batch * 1e3 / step_ms,
            "compared_with": reference[0],
            "first_token_difference": first_difference(tokens, reference[1]),
        }
        results.append(result)
        print(
            f"{impl:<18} prompt {prompt_len:>6}: ttft {result['ttft_ms']['median']:8.1f} ms, "
            f"decode {step_ms:6.2f} ms/step ({result['decode_tokens_per_s']:7.1f} tokens/s), "
            f"first difference from {reference[0]}: {result['first_token_difference']}",
            flush=True,
        )
    return results


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--prompt-lens", default="512,8192")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, help="default: bench/results/<gpu>")
    parser.add_argument("--allow-dirty", action="store_true", help="write to runs/ instead")
    parser.add_argument("--tag", help="appended to the output file name")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env = collect_env()
    if not recordable(env, args.allow_dirty):
        return 2

    flash_lab.hf.register()
    flash_lab.hf.register("flash_lab_cuda", prefill_impl="mma_pipelined")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    eos = model.generation_config.eos_token_id
    eos_id = eos[0] if isinstance(eos, list) else eos

    # Random ids below the special tokens; the same prompt for every implementation.
    gen = torch.Generator().manual_seed(args.seed)
    vocab = min(model.config.vocab_size, 128000)
    prompts = {
        int(n): torch.randint(0, vocab, (args.batch, int(n)), generator=gen).cuda()
        for n in args.prompt_lens.split(",")
    }

    results = []
    reference_tokens: dict[int, tuple[str, torch.Tensor]] = {}
    for impl in args.impls.split(","):
        results += bench_impl(model, impl, prompts, args, eos_id, reference_tokens)
        torch.cuda.empty_cache()

    out = {
        "schema": 1,
        "suite": "e2e_llama",
        "kind": "e2e",
        "env": env,
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": "bfloat16",
        "cache": "static",
        "n_warmup": args.n_warmup,
        "runs": args.runs,
        "seed": args.seed,
        "tag": args.tag,
        "results": results,
    }
    write_results(out, env, "e2e_llama", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
