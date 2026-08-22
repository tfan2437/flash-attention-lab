"""Benchmark harness: times each implementation with CUDA events and writes one JSON per run.

    python -m bench.run --suite prefill_bf16
    python -m bench.run --suite decode_ctx --impls splitkv,flash_attn --configs 0,3

Every implementation's output is checked against an fp64 reference before it is timed (the same
rule as the tests, on a sample of query rows for long sequences); an implementation that fails
is recorded as "incorrect" and not timed. Results go to bench/results/<gpu>/ only from a clean
checkout, so every committed number names an exact commit.
"""

import argparse
import json
import math
import re
import statistics
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench import baselines
from bench.env import collect_env
from bench.flops import CEILINGS, attention_flops, decode_bytes, prefill_min_bytes
from bench.suites import SUITES, DecodeConfig, PrefillConfig
from flash_lab import reference

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
FLOOR = {torch.float32: 1e-6, torch.bfloat16: 1e-3, torch.float16: 1e-3}
CHECK_ROWS = 256
FLUSH_BYTES = 256 * 2**20  # larger than the 50 MB L2 of an H100


def gpu_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace("nvidia", "")).strip("-")


def summarize(times_ms: list[float]) -> dict:
    ordered = sorted(times_ms)

    def pct(p):
        return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]

    return {
        "median": statistics.median(ordered),
        "p10": pct(10),
        "p90": pct(90),
        "min": ordered[0],
    }


def time_cuda(run, n_warmup: int, n_iters: int, flush_l2: bool) -> list[float]:
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda") if flush_l2 else None
    for _ in range(n_warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        if flush is not None:
            flush.zero_()
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]


def sample_rows(seqlen: int, generator: torch.Generator) -> torch.Tensor:
    if seqlen <= CHECK_ROWS:
        return torch.arange(seqlen)
    middle = torch.randperm(seqlen - 2, generator=generator)[: CHECK_ROWS - 2] + 1
    return torch.cat([torch.tensor([0, seqlen - 1]), middle]).sort().values


def sdpa_math_rows(q, k, v, rows, causal, scale):
    """PyTorch's math backend at the input dtype, for the sampled rows only."""
    group = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(group, dim=2)
    v = v.repeat_interleave(group, dim=2)
    mask = None
    if causal:
        cols = torch.arange(k.shape[1], device=q.device)[None, :]
        mask = cols <= rows.to(q.device)[:, None] + (k.shape[1] - q.shape[1])
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(
            q[:, rows].transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            scale=scale,
        )
    return out.transpose(1, 2)


def check_prefill(out, q, k, v, causal, scale, rows) -> dict:
    ref = reference.attention_rows(q, k, v, rows, causal=causal, softmax_scale=scale)
    err = (out[:, rows].double() - ref).abs().max().item()
    torch_out = sdpa_math_rows(q, k, v, rows, causal, scale)
    err_torch = (torch_out.double() - ref).nan_to_num(0.0).abs().max().item()
    finite = bool(torch.isfinite(out).all())
    ok = finite and err <= 2 * err_torch + FLOOR[q.dtype]
    return {"max_err_vs_ref64": err, "max_err_torch_vs_ref64": err_torch, "correct": ok}


def check_decode(out, q, k_cache, v_cache, seq_lens, scale) -> dict:
    ref = reference.decode(q, k_cache, v_cache, seq_lens, softmax_scale=scale)
    err = (out.double() - ref).abs().max().item()
    torch_out = reference.decode(q, k_cache, v_cache, seq_lens, softmax_scale=scale, dtype=q.dtype)
    err_torch = (torch_out.double() - ref).abs().max().item()
    finite = bool(torch.isfinite(out).all())
    ok = finite and err <= 2 * err_torch + FLOOR[q.dtype]
    return {"max_err_vs_ref64": err, "max_err_torch_vs_ref64": err_torch, "correct": ok}


def prefill_inputs(cfg: PrefillConfig, dtype, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (cfg.batch, cfg.seqlen, cfg.heads, cfg.head_dim)
    shape_kv = (cfg.batch, cfg.seqlen, cfg.heads_kv, cfg.head_dim)
    q = torch.randn(shape_q, generator=gen, device="cuda", dtype=dtype)
    k = torch.randn(shape_kv, generator=gen, device="cuda", dtype=dtype)
    v = torch.randn(shape_kv, generator=gen, device="cuda", dtype=dtype)
    return q, k, v


def decode_inputs(cfg: DecodeConfig, dtype, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        cfg.batch, 1, cfg.heads, cfg.head_dim, generator=gen, device="cuda", dtype=dtype
    )
    cache_shape = (cfg.batch, cfg.context, cfg.heads_kv, cfg.head_dim)
    k_cache = torch.randn(cache_shape, generator=gen, device="cuda", dtype=dtype)
    v_cache = torch.randn(cache_shape, generator=gen, device="cuda", dtype=dtype)
    seq_lens = torch.full((cfg.batch,), cfg.context, dtype=torch.int32, device="cuda")
    return q, k_cache, v_cache, seq_lens


def bench_one(kind, impl, cfg, dtype, causal, args, ceilings) -> dict:
    scale = 1.0 / math.sqrt(cfg.head_dim)
    elem = torch.finfo(dtype).bits // 8
    result = {"impl": impl, "status": "ok"}
    if kind == "prefill":
        q, k, v = prefill_inputs(cfg, dtype, args.seed)
        make = lambda: baselines.prefill_runner(impl, q, k, v, causal, scale)  # noqa: E731
        rows = sample_rows(cfg.seqlen, torch.Generator().manual_seed(args.seed))
        check = lambda out: check_prefill(out, q, k, v, causal, scale, rows)  # noqa: E731
    else:
        q, k_cache, v_cache, seq_lens = decode_inputs(cfg, dtype, args.seed)
        make = lambda: baselines.decode_runner(impl, q, k_cache, v_cache, seq_lens, scale)  # noqa: E731
        check = lambda out: check_decode(out, q, k_cache, v_cache, seq_lens, scale)  # noqa: E731

    try:
        run, context = make()
        # SDPA warns at length when a forced backend rejects the input; the status records it.
        with context, warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result.update(check(run()))
            if not result.pop("correct"):
                result["status"] = "incorrect"
                return result
            times = time_cuda(run, args.n_warmup, args.n_iters, args.flush_l2)
    except baselines.Unsupported as exc:
        return {"impl": impl, "status": "unsupported", "reason": str(exc)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"impl": impl, "status": "oom"}
    except RuntimeError as exc:
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        # SDPA raises "No available kernel" when the selected backend rejects the input.
        status = "unsupported" if "No available kernel" in str(exc) else "error"
        return {"impl": impl, "status": status, "reason": reason}

    stats = summarize(times)
    seconds = stats["median"] / 1e3
    result["op_ms"] = stats
    if kind == "prefill":
        flops = attention_flops(cfg.batch, cfg.heads, cfg.seqlen, cfg.seqlen, cfg.head_dim, causal)
        result["flops"] = flops
        result["tflops"] = flops / seconds / 1e12
        min_bytes = prefill_min_bytes(
            cfg.batch, cfg.heads, cfg.heads_kv, cfg.seqlen, cfg.seqlen, cfg.head_dim, elem
        )
        result["min_bytes"] = min_bytes
        if ceilings is not None:
            peak = ceilings.fp32_tflops if dtype == torch.float32 else ceilings.tensor_tflops
            result["pct_peak_compute"] = 100 * result["tflops"] / peak
    else:
        nbytes = decode_bytes(cfg.batch, cfg.heads, cfg.heads_kv, cfg.context, cfg.head_dim, elem)
        result["bytes"] = nbytes
        result["gbps"] = nbytes / seconds / 1e9
        if ceilings is not None:
            result["pct_peak_bw"] = 100 * result["gbps"] / ceilings.hbm_gbps
    return result


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--impls", help="comma-separated; default: the suite's list")
    parser.add_argument("--dtypes", help="comma-separated fp32,bf16,fp16; default: the suite's")
    parser.add_argument("--configs", help="comma-separated config indices; default: all")
    parser.add_argument("--causal", choices=["both", "true", "false"], default="both")
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-iters", type=int, help="default: the suite's")
    parser.add_argument("--flush-l2", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, help="default: bench/results/<gpu>")
    parser.add_argument("--allow-dirty", action="store_true", help="write to runs/ instead")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    suite = SUITES[args.suite]
    impls = args.impls.split(",") if args.impls else list(suite.impls)
    dtypes = [DTYPES[d] for d in (args.dtypes.split(",") if args.dtypes else suite.dtypes)]
    indices = (
        [int(i) for i in args.configs.split(",")] if args.configs else range(len(suite.configs))
    )
    causal_values = {"both": suite.causal, "true": (True,), "false": (False,)}[args.causal]
    causal_values = [c for c in causal_values if c in suite.causal]
    args.n_iters = args.n_iters or suite.n_iters
    args.flush_l2 = suite.flush_l2 if args.flush_l2 is None else args.flush_l2

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = collect_env()
    if env["git_dirty"] and not args.allow_dirty:
        print(
            "refusing to record results from a dirty checkout (use --allow-dirty)", file=sys.stderr
        )
        return 2
    ceilings = CEILINGS.get(env.get("gpu", ""))

    results = []
    for index in indices:
        cfg = suite.configs[index]
        for dtype in dtypes:
            for causal in causal_values:
                for impl in impls:
                    res = bench_one(suite.kind, impl, cfg, dtype, causal, args, ceilings)
                    config = {**vars(cfg), "dtype": str(dtype).removeprefix("torch.")}
                    if suite.kind == "prefill":
                        config["causal"] = causal
                    results.append({"config": config, **res})
                    print(format_row(index, config, res), flush=True)

    out = {
        "schema": 1,
        "suite": args.suite,
        "kind": suite.kind,
        "env": env,
        "ceilings": vars(ceilings) if ceilings else None,
        "flush_l2": args.flush_l2,
        "n_warmup": args.n_warmup,
        "n_iters": args.n_iters,
        "seed": args.seed,
        "results": results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    sha = (env["git_sha"] or "nogit")[:7]
    if args.allow_dirty and env["git_dirty"]:
        out_dir = Path("runs") / "bench"
        sha += "-dirty"
    else:
        out_dir = args.out_dir or Path("bench/results") / gpu_slug(env.get("gpu", "cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}-{sha}-{args.suite}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


def format_row(index, config, res) -> str:
    name = f"[{index}] " + " ".join(
        f"{key}={value}" for key, value in config.items() if key not in ("dtype",)
    )
    if res["status"] != "ok":
        return f"{name:<70} {res['impl']:<15} {res['status']}"
    rate = f"{res['tflops']:8.1f} TFLOP/s" if "tflops" in res else f"{res['gbps']:8.1f} GB/s"
    return f"{name:<70} {res['impl']:<15} {res['op_ms']['median']:9.3f} ms {rate}"


if __name__ == "__main__":
    raise SystemExit(main())
