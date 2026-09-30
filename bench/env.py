"""Environment block stored with every benchmark result and PACE run log.

Usage: python -m bench.env
"""

import importlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone


def _run(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:  # optional packages can fail to import for reasons other than absence
        return None


def _nvidia_smi(fields: str) -> list[str] | None:
    # Index 0 is the first GPU visible to this job; Slurm's cgroups hide the others.
    out = _run(["nvidia-smi", "-i", "0", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    return [part.strip() for part in out.split(",")] if out else None


def collect_env() -> dict:
    import torch

    env = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": _version("triton"),
        "flash_attn": _version("flash_attn"),
        "transformers": _version("transformers"),
        # Kernel switches such as FLASH_LAB_FP32_TILE change what is being measured.
        "flash_lab_env": {k: v for k, v in os.environ.items() if k.startswith("FLASH_LAB_")},
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env.update(
            gpu=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            sm_count=props.multi_processor_count,
            mem_gb=round(props.total_memory / 2**30, 1),
        )
        smi = _nvidia_smi("driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem")
        if smi:
            env["driver"] = smi[0]
            env["clocks_mhz"] = dict(zip(["sm", "mem", "max_sm", "max_mem"], smi[1:], strict=False))
    return env


if __name__ == "__main__":
    print(json.dumps(collect_env(), indent=2))
