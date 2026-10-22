"""RSS / heap sampling with an optional RSS ceiling (moved out of cli/main.py, p6-08).

One sample = the RSS reading, plus (streaming emit) the traced heap peak at that moment.
`over_limit()` is the `--max-rss-gb` abort check; the caller returns exit code 3.
"""
from __future__ import annotations

from stratashift.monitor.heap import heap_peak_bytes
from stratashift.monitor.rss import current_rss_bytes

GB = 1 << 30


class Sampler:
    def __init__(self, max_rss_gb: float | None) -> None:
        self.limit = int(max_rss_gb * GB) if max_rss_gb else None
        self.rss: list[int] = []
        self.heap: list[int] = []

    def record(self, heap: bool = False) -> None:
        self.rss.append(current_rss_bytes())
        if heap:
            self.heap.append(heap_peak_bytes())

    def over_limit(self) -> bool:
        return self.limit is not None and bool(self.rss) and self.rss[-1] > self.limit

    def sample(self) -> bool:
        """Record RSS + heap and report whether the RSS ceiling is exceeded."""
        self.record(heap=True)
        return self.over_limit()

    def rss_peak(self) -> int:
        return max(self.rss) if self.rss else 0

    def slice_heap_peak(self) -> int:
        return max(self.heap) if self.heap else 0
