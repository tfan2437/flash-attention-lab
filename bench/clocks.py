"""GPU clock, power, and temperature samples taken while benchmarks run.

The clocks in env.json are read once before the work starts, when an idle GPU can sit at a low
SM clock. A ClockSampler runs `nvidia-smi` in a loop for the whole suite, and window() reports
the samples taken while one implementation ran. Samples carry nvidia-smi's own timestamps, so
buffered output does not shift them.
"""

import statistics
import subprocess
import threading
from datetime import datetime

FIELDS = ("clocks.sm", "clocks.mem", "power.draw", "temperature.gpu")
NAMES = ("sm_mhz", "mem_mhz", "power_w", "temp_c")


class ClockSampler:
    def __init__(self, interval_ms: int = 50):
        self.interval_ms = interval_ms
        self.samples: list[tuple[float, list[float]]] = []
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def __enter__(self):
        query = ",".join(("timestamp", *FIELDS))
        cmd = ["nvidia-smi", "-i", "0", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        try:
            self._proc = subprocess.Popen(
                [*cmd, "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:  # no nvidia-smi: every window comes back empty
            return self
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        return self

    def _read(self) -> None:
        for line in self._proc.stdout:
            parts = [part.strip() for part in line.split(",")]
            try:
                stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                values = [float(part) for part in parts[1:]]
            except ValueError:
                continue
            self.samples.append((stamp, values))

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
            self._reader.join(timeout=5)

    def window(self, start: float, end: float) -> dict | None:
        """Min, median, and max of each field between two time.time() stamps. A window shorter
        than the sampling interval gets the sample nearest to its middle."""
        rows = [values for stamp, values in self.samples if start <= stamp <= end]
        if not rows and self.samples:
            middle = (start + end) / 2
            rows = [min(self.samples, key=lambda sample: abs(sample[0] - middle))[1]]
        if not rows:
            return None
        out: dict = {"samples": len(rows)}
        for i, name in enumerate(NAMES):
            column = [row[i] for row in rows]
            median = statistics.median(column)
            out[name] = {"min": min(column), "median": median, "max": max(column)}
        return out
