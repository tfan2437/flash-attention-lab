"""Final numbers: the median and range over repeated runs of each suite at one commit.

    python -m bench.summarize bench/results/h100-80gb-hbm3/*-<sha>-*-final?.json \
        --json bench/results/h100-80gb-hbm3/summary-<sha>.json > summary-<sha>.md

For every (suite, config, implementation) it takes each run's median time, then reports the
median of those and their range across runs, with the throughput derived from that median.
Ratios to flash-attn are computed within each run first, because flash-attn's timing varies
more between runs than within one.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict

from bench.flops import CEILINGS

PREFILL_KEYS = ("batch", "heads", "heads_kv", "seqlen", "head_dim", "dtype", "causal")
DECODE_KEYS = ("batch", "heads", "heads_kv", "context", "head_dim", "dtype")


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def kernel_suites(runs: list[dict]) -> dict:
    """Aggregates prefill and decode suites, keyed by suite name. vs_flash_attn is flash-attn's
    time divided by the implementation's, so above 1 is faster."""
    by_suite = defaultdict(list)
    for run in runs:
        by_suite[run["suite"]].append(run)
    out = {}
    for suite, suite_runs in sorted(by_suite.items()):
        kind = suite_runs[0]["kind"]
        keys = PREFILL_KEYS if kind == "prefill" else DECODE_KEYS
        times = defaultdict(list)  # (config, impl) -> per-run median ms
        ratios = defaultdict(list)  # (config, impl) -> per-run flash_attn_ms / impl_ms
        work = {}  # config -> flops or bytes
        statuses = defaultdict(set)
        for run in suite_runs:
            flash_ms = {}
            for r in run["results"]:
                config = tuple(r["config"].get(k) for k in keys)
                statuses[(config, r["impl"])].add(r["status"])
                if r["status"] != "ok":
                    continue
                times[(config, r["impl"])].append(r["op_ms"]["median"])
                work[config] = r.get("flops") or r.get("bytes")
                if r["impl"] == "flash_attn":
                    flash_ms[config] = r["op_ms"]["median"]
            for r in run["results"]:
                config = tuple(r["config"].get(k) for k in keys)
                if r["status"] == "ok" and config in flash_ms:
                    ratios[(config, r["impl"])].append(flash_ms[config] / r["op_ms"]["median"])
        rows = []
        for (config, impl), status in statuses.items():
            row = {"config": dict(zip(keys, config, strict=True)), "impl": impl}
            measured = times.get((config, impl))
            if not measured or len(measured) < len(suite_runs):
                row["status"] = sorted(status)
            if measured:
                row["runs"] = len(measured)
                row["ms"] = spread(measured)
                seconds = row["ms"]["median"] / 1e3
                ceilings = CEILINGS.get(suite_runs[0]["env"].get("gpu", ""))
                if kind == "prefill":
                    row["tflops"] = work[config] / seconds / 1e12
                    if ceilings:
                        fp32 = row["config"]["dtype"] == "float32"
                        peak = ceilings.fp32_tflops if fp32 else ceilings.tensor_tflops
                        row["pct_peak"] = 100 * row["tflops"] / peak
                else:
                    row["gbps"] = work[config] / seconds / 1e9
                    if ceilings:
                        row["pct_peak"] = 100 * row["gbps"] / ceilings.hbm_gbps
                if ratios.get((config, impl)):
                    row["vs_flash_attn"] = spread(ratios[(config, impl)])
            rows.append(row)
        out[suite] = {"kind": kind, "runs": len(suite_runs), "rows": rows}
    return out


def e2e_suite(runs: list[dict]) -> dict:
    ttft, step, err = defaultdict(list), defaultdict(list), defaultdict(list)
    statuses = defaultdict(set)
    for run in runs:
        for r in run["results"]:
            key = (r["config"]["prompt_len"], r["impl"])
            statuses[key].add(r["status"])
            if "max_logit_err_vs_fp32" in r:
                err[key].append(r["max_logit_err_vs_fp32"])
            if r["status"] == "ok":
                ttft[key].append(r["ttft_ms"]["median"])
                step[key].append(r["decode_step_ms"]["median"])
    rows = []
    for (prompt_len, impl), status in sorted(statuses.items()):
        key = (prompt_len, impl)
        row = {"prompt_len": prompt_len, "impl": impl, "status": sorted(status)}
        if step.get(key):
            row["ttft_ms"] = spread(ttft[key])
            row["decode_step_ms"] = spread(step[key])
            row["decode_tokens_per_s"] = 1e3 / row["decode_step_ms"]["median"]
        if err.get(key):
            row["max_logit_err_vs_fp32"] = max(err[key])
        rows.append(row)
    return {"kind": "e2e", "runs": len(runs), "model": runs[0]["model"], "rows": rows}


def fmt(spread_ms: dict, digits: int = 3) -> str:
    return (
        f"{spread_ms['median']:.{digits}f} ({spread_ms['min']:.{digits}f}-"
        f"{spread_ms['max']:.{digits}f})"
    )


def markdown(summary: dict) -> str:
    lines = [f"# Final results, commit {summary['commit']}", ""]
    lines.append(
        "Medians of per-run medians across the runs listed; ranges in parentheses are the "
        "smallest and largest run. Ratios to flash-attn are taken within each run."
    )
    for suite, data in summary["suites"].items():
        lines += ["", f"## {suite} ({data['runs']} runs)", ""]
        if data["kind"] == "e2e":
            lines += [
                f"Model {data['model']}.",
                "",
                "| prompt | impl | time to first token (ms) | decode step (ms) | tokens/s | "
                "max logit error vs fp32 |",
                "|---|---|---|---|---|---|",
            ]
            for r in data["rows"]:
                if "decode_step_ms" not in r:
                    err = r.get("max_logit_err_vs_fp32", float("nan"))
                    status = ",".join(r["status"])
                    lines.append(f"| {r['prompt_len']} | {r['impl']} | {status} | | | {err:.3f} |")
                    continue
                lines.append(
                    f"| {r['prompt_len']} | {r['impl']} | {fmt(r['ttft_ms'], 1)} | "
                    f"{fmt(r['decode_step_ms'], 2)} | {r['decode_tokens_per_s']:.1f} | "
                    f"{r.get('max_logit_err_vs_fp32', float('nan')):.3f} |"
                )
            continue
        rate = "TFLOP/s" if data["kind"] == "prefill" else "GB/s"
        lines += [
            f"| config | impl | ms | {rate} | % of peak | speed vs flash-attn |",
            "|---|---|---|---|---|---|",
        ]
        for r in data["rows"]:
            config = ", ".join(f"{k}={v}" for k, v in r["config"].items() if k != "dtype")
            if "ms" not in r:
                lines.append(f"| {config} | {r['impl']} | {','.join(r['status'])} | | | |")
                continue
            value = r.get("tflops", r.get("gbps"))
            pct = f"{r['pct_peak']:.1f}" if "pct_peak" in r else ""
            ratio = fmt(r["vs_flash_attn"], 2) if "vs_flash_attn" in r else ""
            lines.append(
                f"| {config} | {r['impl']} | {fmt(r['ms'])} | {value:.1f} | {pct} | {ratio} |"
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json", help="also write the aggregates here")
    args = parser.parse_args(argv)

    runs = [json.load(open(path)) for path in args.files]
    commits = {run["env"]["git_sha"][:7] for run in runs}
    if len(commits) != 1 or any(run["env"]["git_dirty"] for run in runs):
        print(f"runs must come from one clean commit, got {sorted(commits)}", file=sys.stderr)
        return 2
    e2e = [run for run in runs if run.get("kind") == "e2e"]
    kernels = [run for run in runs if run.get("kind") != "e2e"]
    summary = {
        "commit": commits.pop(),
        "gpu": runs[0]["env"].get("gpu"),
        "files": sorted(args.files),
        "suites": kernel_suites(kernels),
    }
    if e2e:
        summary["suites"]["e2e_llama"] = e2e_suite(e2e)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)
            f.write("\n")
    sys.stdout.write(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
