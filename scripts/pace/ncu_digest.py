"""Markdown digest of an Nsight Compute raw-metric CSV.

    ncu --import report.ncu-rep --page raw --csv > report.csv
    python scripts/pace/ncu_digest.py report.csv > report.md

Lists the metrics tracked for every kernel (profiling/metrics.md), the tensor and FMA pipe
utilization, and the warp stall reasons sorted by cycles per issued instruction.
"""

import csv
import sys

METRICS = [
    ("gpu__time_duration.sum", "duration"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput (% of peak)"),
    ("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "memory throughput (%)"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput (%)"),
    ("dram__bytes_read.sum", "DRAM bytes read"),
    ("dram__bytes_write.sum", "DRAM bytes written"),
    ("lts__t_sector_hit_rate.pct", "L2 hit rate (%)"),
    ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue slots busy (%)"),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy (%)"),
    ("sm__maximum_warps_per_active_cycle_pct", "theoretical occupancy (%)"),
    ("launch__registers_per_thread", "registers per thread"),
    ("launch__shared_mem_per_block_static", "static shared memory per block"),
    ("launch__shared_mem_per_block_dynamic", "dynamic shared memory per block"),
    ("launch__occupancy_limit_registers", "blocks per SM allowed by registers"),
    ("launch__occupancy_limit_shared_mem", "blocks per SM allowed by shared memory"),
    ("launch__waves_per_multiprocessor", "waves per SM"),
    ("l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "shared-memory wavefronts"),
    ("l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "shared-memory bank conflicts"),
]
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


def number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def digest(row: dict, units: dict) -> str:
    lines = [f"## {row.get('Kernel Name', '?')}", ""]
    lines.append(f"- grid {row.get('Grid Size', '?')}, block {row.get('Block Size', '?')}")
    lines += ["", "| metric | value | unit |", "|---|---|---|"]
    for key, label in METRICS:
        if row.get(key):
            lines.append(f"| {label} (`{key}`) | {row[key]} | {units.get(key, '')} |")

    pipes = sorted(
        key
        for key in row
        if ("pipe_tensor" in key or "pipe_fma" in key)
        and key.endswith("pct_of_peak_sustained_active")
    )
    if pipes:
        lines += ["", "| pipe utilization | % of peak (active cycles) |", "|---|---|"]
        lines += [f"| `{key}` | {row[key]} |" for key in pipes if row[key]]

    stalls = []
    for key, value in row.items():
        if key.startswith(STALL_PREFIX) and key.endswith(STALL_SUFFIX):
            parsed = number(value)
            if parsed is not None and parsed > 0:
                stalls.append((parsed, key[len(STALL_PREFIX) : -len(STALL_SUFFIX)]))
    if stalls:
        lines += ["", "| stall reason | cycles per issued instruction |", "|---|---|"]
        lines += [
            f"| {reason} | {value:.2f} |" for value, reason in sorted(stalls, reverse=True)[:10]
        ]
    return "\n".join(lines)


def main(path: str) -> None:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # The first data row of a raw-page export holds the units.
    units = rows[0] if rows and not rows[0].get("ID") else {}
    kernels = [row for row in rows if row.get("ID")]
    print("\n\n".join(digest(row, units) for row in kernels))


if __name__ == "__main__":
    main(sys.argv[1])
