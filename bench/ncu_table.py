"""One table across the Nsight Compute digests, a row per profiled kernel, with the metrics
profiling/metrics.md tracks.

    python -m bench.ncu_table > table.md

Durations are under the profiler, which serializes launches and flushes caches before each
one; they describe a single cold launch, not the benchmarked time.
"""

import csv
import re

from bench.plot import POINTS, REPORTS, UNITS

STALL = re.compile(r"smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio")


def value(row: dict, units: dict, key: str) -> float | None:
    text = row.get(key, "").replace(",", "")
    if not text:
        return None
    return float(text) * UNITS.get(units.get(key, ""), 1)


def pct(row: dict, key: str) -> str:
    text = row.get(key, "")
    return f"{float(text):.1f}" if text else ""


def kernel_name(full: str) -> str:
    name = full.split("(")[0].removeprefix("void ").split("::")[-1]
    return name.split("<")[0]


def rows() -> list[list[str]]:
    table = []
    for point in POINTS:
        path = REPORTS / f"{point.report}.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            data = list(csv.DictReader(f))
        units, kernels = data[0], [row for row in data if row.get("ID")]
        for row in kernels:
            seconds = value(row, units, "gpu__time_duration.sum")
            dram = sum(value(row, units, f"dram__bytes_{d}.sum") or 0 for d in ("read", "write"))
            dram_pct = sum(
                float(row.get(f"dram__bytes_{d}.sum.pct_of_peak_sustained_elapsed") or 0)
                for d in ("read", "write")
            )
            stalls = sorted(
                (
                    (float(v), STALL.fullmatch(k).group(1))
                    for k, v in row.items()
                    if STALL.fullmatch(k) and v
                ),
                reverse=True,
            )[:3]

            table.append(
                [
                    point.label if point.ours else f"{point.label} (baseline)",
                    kernel_name(row.get("Kernel Name", "?")),
                    f"{seconds * 1e3:.3f}" if seconds else "",
                    f"{float(row['gpc__cycles_elapsed.avg.per_second']):.2f}"
                    if row.get("gpc__cycles_elapsed.avg.per_second")
                    else "",
                    pct(row, "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                    pct(row, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"),
                    pct(row, "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"),
                    f"{dram_pct:.1f}",
                    f"{dram / 1e6:.0f}",
                    pct(row, "lts__t_sector_hit_rate.pct"),
                    pct(row, "sm__warps_active.avg.pct_of_peak_sustained_active"),
                    row.get("launch__registers_per_thread", ""),
                    ", ".join(f"{name} {cpi:.2f}" for cpi, name in stalls),
                ]
            )
    return table


HEADER = [
    "implementation",
    "kernel",
    "duration (ms)",
    "SM clock (GHz)",
    "SM throughput %",
    "tensor pipe %",
    "FMA pipe %",
    "DRAM bandwidth %",
    "DRAM MB",
    "L2 hit %",
    "achieved occupancy %",
    "registers",
    "top stalls (cycles per issued instruction)",
]


def main() -> None:
    print("| " + " | ".join(HEADER) + " |")
    print("|" + "---|" * len(HEADER))
    for row in rows():
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
