"""Greedy generation with a Llama checkpoint on the flash_lab kernels.

    python examples/generate.py --model meta-llama/Llama-3.1-8B-Instruct
    python examples/generate.py --compare sdpa --cache static

The model runs with attn_implementation="flash_lab" (flash_lab.hf): the prompt goes through
flash_lab.attention and every generated token through flash_lab.decode. --compare repeats the
generation with another implementation and reports the first generated token that differs.
Each implementation first runs once untimed, which pays for Triton's compilation and autotuning
and, with --cache static, for torch.compile of the decode step (CUDA graphs). bench/e2e.py times
prefill and decode separately.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flash_lab.hf

PROMPT = "In three sentences, why is generating one token at a time limited by memory bandwidth?"


def generate(model, tokenizer, inputs, args) -> tuple[torch.Tensor, float]:
    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        cache_implementation=args.cache,
    )
    model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(**inputs, **kwargs)
    torch.cuda.synchronize()
    return out[0, inputs["input_ids"].shape[1] :], time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--compare", metavar="IMPL", help="also generate with sdpa, eager, ...")
    args = parser.parse_args()

    flash_lab.hf.register()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="flash_lab", device_map="cuda"
    ).eval()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    tokens = {}
    for impl in ["flash_lab"] + ([args.compare] if args.compare else []):
        model.set_attn_implementation(impl)
        tokens[impl], seconds = generate(model, tokenizer, inputs, args)
        count = len(tokens[impl])
        print(f"--- {impl}: prompt {prompt_len} tokens, generated {count} in {seconds:.2f} s")
        print(tokenizer.decode(tokens[impl], skip_special_tokens=True))

    if args.compare:
        ours, theirs = tokens["flash_lab"], tokens[args.compare]
        n = min(len(ours), len(theirs))
        differ = (ours[:n] != theirs[:n]).nonzero()
        if len(differ):
            print(f"--- first difference from {args.compare} at generated token {int(differ[0])}")
        elif len(ours) != len(theirs):
            print(f"--- same first {n} tokens as {args.compare}, then one run stopped")
        else:
            print(f"--- identical tokens to {args.compare}")


if __name__ == "__main__":
    main()
