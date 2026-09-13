"""Markdown digest of `nsys stats` CSV exports for runs whose NVTX ranges are named
`<impl>:step` (bench/decode_trace.py): step time per implementation and where its GPU time went.

    python scripts/pace/nsys_digest.py <prefix>    # reads <prefix>_nvtx_sum.csv and
                                                   # <prefix>_nvtx_kern_sum.csv
"""

import csv
import sys
from collections import defaultdict


def rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def column(row: dict, *names: str) -> str:
    for key in row:
        if any(name.lower() == key.lower().strip() for name in names):
            return row[key]
    raise KeyError(f"none of {names} in {list(row)}")


def main(prefix: str) -> None:
    print("## Step time per implementation (NVTX range, includes host work)")
    print()
    print("| implementation | steps | mean step (us) |")
    print("|---|---|---|")
    for row in rows(f"{prefix}_nvtx_sum.csv"):
        name = column(row, "Range").lstrip(":")
        if not name.endswith(":step"):
            continue
        avg_us = float(column(row, "Avg (ns)")) / 1e3
        print(f"| {name.removesuffix(':step')} | {column(row, 'Instances')} | {avg_us:.1f} |")

    per_impl = defaultdict(list)
    for row in rows(f"{prefix}_nvtx_kern_sum.csv"):
        name = column(row, "NVTX Range").lstrip(":")
        if name.endswith(":step"):
            total = float(column(row, "Total Time (ns)"))
            per_impl[name.removesuffix(":step")].append((total, column(row, "Kernel Name")))

    print()
    print("## GPU time by kernel within the steps")
    for impl, kernels in per_impl.items():
        grand = sum(total for total, _ in kernels)
        print()
        print(f"### {impl}")
        print()
        print("| kernel | share of GPU time | total (us) |")
        print("|---|---|---|")
        for total, kernel in sorted(kernels, reverse=True):
            short = kernel if len(kernel) <= 90 else kernel[:87] + "..."
            print(f"| `{short}` | {100 * total / grand:.1f}% | {total / 1e3:.1f} |")


if __name__ == "__main__":
    main(sys.argv[1])
