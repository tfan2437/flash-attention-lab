"""Figures for the docs, drawn from committed results only.

    python -m bench.plot roofline --summary bench/results/h100-80gb-hbm3/summary-<sha>.json

roofline: each kernel at its measured arithmetic intensity (analytic FLOPs over the DRAM bytes
Nsight Compute counted) and its benchmarked throughput (analytic FLOPs over the median time in
the final summary), with the analytic intensity (FLOPs over the bytes that must move at least
once) as a hollow marker on the same row. Writes a light and a dark SVG to docs/figures/.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from bench.flops import CEILINGS, attention_flops, decode_bytes, prefill_min_bytes

REPORTS = Path("profiling/reports/h100-80gb-hbm3")
FIGURES = Path("docs/figures")
GPU = "NVIDIA H100 80GB HBM3"
UNITS = {
    "byte": 1,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1,
}
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "ours": "#2a78d6",
        "baseline": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "ours": "#3987e5",
        "baseline": "#d95926",
    },
}


@dataclass(frozen=True)
class Point:
    report: str  # Nsight Compute digest name in REPORTS
    label: str
    suite: str
    impl: str
    shape: tuple  # (batch, heads, heads_kv, seqlen or context, head_dim)
    elem: int  # bytes per element
    ours: bool = True
    label_offset: tuple = (6, 4)  # points, from the measured marker


PREFILL = (4, 32, 32, 4096, 128)
DECODE_1 = (1, 32, 32, 32768, 128)
DECODE_8 = (8, 32, 8, 32768, 128)
POINTS = [
    Point("naive-s4096-full", "naive", "prefill_fp32", "naive", PREFILL, 4),
    Point("fp32-fused-s4096-full", "fp32_fused", "prefill_fp32", "fp32_fused", PREFILL, 4),
    Point("fp32-regtile-s4096-full", "fp32_regtile", "prefill_fp32", "fp32_regtile", PREFILL, 4),
    Point("mma-s4096-full", "mma", "prefill_bf16", "mma", PREFILL, 2),
    Point("mma-pipelined-s4096-full", "mma_pipelined", "prefill_bf16", "mma_pipelined", PREFILL, 2),
    Point("triton-s4096-full", "triton", "prefill_bf16", "triton", PREFILL, 2),
    Point("flash-attn-s4096-full", "flash-attn", "prefill_bf16", "flash_attn", PREFILL, 2, False),
    Point(
        "decode-inplace-b1-ctx32k", "decode_inplace", "decode_ctx", "decode_inplace", DECODE_1, 2
    ),
    Point("splitkv-b1-ctx32k", "splitkv", "decode_ctx", "splitkv", DECODE_1, 2),
    Point(
        "flash-attn-decode-b1-ctx32k", "flash-attn", "decode_ctx", "flash_attn", DECODE_1, 2, False
    ),
    Point("splitkv-b8-gqa-ctx32k", "splitkv, GQA", "decode_ctx", "splitkv", DECODE_8, 2),
    Point(
        "flash-attn-decode-b8-gqa-ctx32k",
        "flash-attn, GQA",
        "decode_ctx",
        "flash_attn",
        DECODE_8,
        2,
        False,
    ),
]


def dram_bytes(report: str) -> float:
    """DRAM bytes read and written by all profiled launches (one call of the implementation)."""
    with open(REPORTS / f"{report}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    units, kernels = rows[0], [row for row in rows if row.get("ID")]
    total = 0.0
    for row in kernels:
        for key in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
            total += float(row[key].replace(",", "")) * UNITS[units[key]]
    return total


def analytic(point: Point) -> tuple[float, float]:
    """FLOPs per call and the bytes that must move at least once."""
    batch, heads, heads_kv, length, head_dim = point.shape
    if point.suite.startswith("prefill"):
        flops = attention_flops(batch, heads, length, length, head_dim, causal=False)
        min_bytes = prefill_min_bytes(batch, heads, heads_kv, length, length, head_dim, point.elem)
    else:
        flops = attention_flops(batch, heads, 1, length, head_dim, causal=False)
        min_bytes = decode_bytes(batch, heads, heads_kv, length, head_dim, point.elem)
    return flops, min_bytes


def median_ms(summary: dict, point: Point) -> float | None:
    batch, heads, heads_kv, length, head_dim = point.shape
    length_key = "seqlen" if point.suite.startswith("prefill") else "context"
    for row in summary["suites"][point.suite]["rows"]:
        c = row["config"]
        same = (c["batch"], c["heads"], c["heads_kv"], c[length_key], c["head_dim"])
        if row["impl"] == point.impl and same == point.shape and not c.get("causal"):
            return row["ms"]["median"] if "ms" in row else None
    return None


def roofline(summary: dict, theme: str, path: Path) -> list[dict]:
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    colors = THEMES[theme]
    ceilings = CEILINGS[GPU]
    bandwidth = ceilings.hbm_gbps / 1e3  # TB/s, so TB/s x FLOP/byte = TFLOP/s
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=colors["surface"])
    ax.set_facecolor(colors["surface"])
    x_min, x_max, y_min, y_max = 0.3, 6000, 0.1, 2000
    xs = [x_min * (x_max / x_min) ** (i / 400) for i in range(401)]
    for peak, name in (
        (ceilings.tensor_tflops, "bf16 tensor cores"),
        (ceilings.fp32_tflops, "fp32"),
    ):
        ax.plot(xs, [min(peak, bandwidth * x) for x in xs], color=colors["muted"], lw=1.5)
        # Labelled just right of the ridge point, clear of the prefill kernels near 2000.
        ax.text(
            peak / bandwidth * 1.5,
            peak * 1.12,
            f"{name}, {peak:g} TFLOP/s",
            color=colors["muted"],
            ha="left",
            va="bottom",
        )
    ax.text(
        0.45,
        bandwidth * 0.45 * 1.6,
        f"HBM {ceilings.hbm_gbps / 1e3:.2f} TB/s",
        color=colors["muted"],
        rotation=33,
        rotation_mode="anchor",
        va="bottom",
    )

    table = []
    for point in POINTS:
        ms = median_ms(summary, point)
        if ms is None or not (REPORTS / f"{point.report}.csv").exists():
            continue
        flops, min_bytes = analytic(point)
        measured = dram_bytes(point.report)
        tflops = flops / (ms / 1e3) / 1e12
        color = colors["ours" if point.ours else "baseline"]
        ax.plot(
            [flops / measured, flops / min_bytes],
            [tflops, tflops],
            color=color,
            lw=1,
            alpha=0.5,
            zorder=2,
        )
        ax.scatter(
            [flops / min_bytes],
            [tflops],
            s=40,
            facecolors=colors["surface"],
            edgecolors=color,
            linewidths=1.5,
            zorder=3,
        )
        ax.scatter(
            [flops / measured],
            [tflops],
            s=48,
            color=color,
            edgecolors=colors["surface"],
            linewidths=1.5,
            zorder=4,
        )
        ax.annotate(
            point.label,
            (flops / measured, tflops),
            xytext=point.label_offset,
            textcoords="offset points",
            color=colors["text"],
            fontsize=8,
        )
        table.append(
            {
                "report": point.report,
                "label": point.label,
                "tflops": tflops,
                "measured_intensity": flops / measured,
                "analytic_intensity": flops / min_bytes,
                "dram_bytes": measured,
                "min_bytes": min_bytes,
            }
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("arithmetic intensity (FLOP per DRAM byte)", color=colors["text"])
    ax.set_ylabel("TFLOP/s", color=colors["text"])
    ax.set_title(
        "H100 roofline: prefill at (4, 32, 32, 4096, 128), decode at a 32K context",
        color=colors["text"],
        fontsize=10,
        loc="left",
    )
    ax.grid(True, which="major", color=colors["grid"], lw=0.8)
    ax.tick_params(colors=colors["muted"], which="both")
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        ax.scatter([], [], s=48, color=colors["ours"], label="flash_lab, measured bytes"),
        ax.scatter([], [], s=48, color=colors["baseline"], label="flash-attn, measured bytes"),
        ax.scatter(
            [],
            [],
            s=40,
            facecolors=colors["surface"],
            edgecolors=colors["muted"],
            linewidths=1.5,
            label="same kernel, minimum bytes",
        ),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(colors["text"])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=colors["surface"])
    plt.close(fig)
    return table


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figure", choices=["roofline"])
    parser.add_argument("--summary", required=True, help="bench.summarize --json output")
    parser.add_argument("--table", help="also write the plotted values as JSON")
    args = parser.parse_args(argv)
    summary = json.load(open(args.summary))
    table = []
    for theme in THEMES:
        suffix = "" if theme == "light" else "_dark"
        table = roofline(summary, theme, FIGURES / f"roofline_h100{suffix}.svg")
    if args.table:
        with open(args.table, "w") as f:
            json.dump(table, f, indent=1)
            f.write("\n")
    for row in table:
        print(
            f"{row['label']:<18} {row['tflops']:8.1f} TFLOP/s  intensity measured "
            f"{row['measured_intensity']:8.1f}, minimum-bytes {row['analytic_intensity']:8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
