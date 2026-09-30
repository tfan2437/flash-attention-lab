"""Markdown digest of `nsys stats` CSV exports for runs whose NVTX ranges are named
`<impl>:<phase>` (bench/decode_trace.py, bench/e2e_trace.py): time per range and where its GPU
time went.

    python scripts/pace/nsys_digest.py <prefix>    # reads <prefix>_nvtx_sum.csv and
                                                   # <prefix>_nvtx_kern_sum.csv
"""

import csv
import sys
from collections import defaultdict

TOP = 12  # kernels listed per range; the rest are summed


def rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def column(row: dict, *names: str) -> str:
    for key in row:
        if any(name.lower() == key.lower().strip() for name in names):
            return row[key]
    raise KeyError(f"none of {names} in {list(row)}")


def main(prefix: str) -> None:
    print("## Time per range (NVTX, includes host work)")
    print()
    print("| range | instances | mean (us) |")
    print("|---|---|---|")
    for row in rows(f"{prefix}_nvtx_sum.csv"):
        name = column(row, "Range").lstrip(":")
        if ":" not in name:
            continue
        avg_us = float(column(row, "Avg (ns)")) / 1e3
        print(f"| {name} | {column(row, 'Instances')} | {avg_us:.1f} |")

    per_range = defaultdict(list)
    for row in rows(f"{prefix}_nvtx_kern_sum.csv"):
        name = column(row, "NVTX Range").lstrip(":")
        if ":" in name:
            total = float(column(row, "Total Time (ns)"))
            per_range[name].append((total, column(row, "Kernel Name")))

    print()
    print("## GPU time by kernel within each range")
    for name, kernels in per_range.items():
        grand = sum(total for total, _ in kernels)
        kernels.sort(reverse=True)
        print()
        print(f"### {name}")
        print()
        print("| kernel | share of GPU time | total (us) |")
        print("|---|---|---|")
        for total, kernel in kernels[:TOP]:
            short = kernel if len(kernel) <= 90 else kernel[:87] + "..."
            print(f"| `{short}` | {100 * total / grand:.1f}% | {total / 1e3:.1f} |")
        if len(kernels) > TOP:
            rest = sum(total for total, _ in kernels[TOP:])
            others = f"{len(kernels) - TOP} other kernels"
            print(f"| {others} | {100 * rest / grand:.1f}% | {rest / 1e3:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1])
