"""RSS / heap sampling with an optional RSS ceiling (moved out of cli/main.py, p6-08).

One sample = the RSS reading, plus (streaming emit) the traced heap peak at that moment.
`over_limit()` is the `--max-rss-gb` abort check; the caller returns exit code 3.
An unmeasurable RSS sample is kept as None (JSON null), never 0 (p6-10); the CLI refuses
`--max-rss-gb` up front on such an OS, so a ceiling never meets a None sample.
"""
from __future__ import annotations

from stratashift.monitor.heap import heap_peak_bytes
from stratashift.monitor.rss import current_rss_bytes

GB = 1 << 30


class Sampler:
    def __init__(self, max_rss_gb: float | None) -> None:
        self.limit = int(max_rss_gb * GB) if max_rss_gb else None
        self.rss: list[int | None] = []
        self.heap: list[int] = []

    def record(self, heap: bool = False) -> None:
        self.rss.append(current_rss_bytes())
        if heap:
            self.heap.append(heap_peak_bytes())

    def over_limit(self) -> bool:
        last = self.rss[-1] if self.rss else None
        return self.limit is not None and last is not None and last > self.limit

    def sample(self) -> bool:
        """Record RSS + heap and report whether the RSS ceiling is exceeded."""
        self.record(heap=True)
        return self.over_limit()

    def rss_peak(self) -> int | None:
        measured = [r for r in self.rss if r is not None]
        return max(measured) if measured else None

    def slice_heap_peak(self) -> int:
        return max(self.heap) if self.heap else 0
