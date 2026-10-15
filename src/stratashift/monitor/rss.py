"""RSS sampling from /proc/self/status (VmRSS). Linux-only; returns 0 elsewhere."""
from __future__ import annotations


def current_rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0


def sample_rss(n: int) -> list[int]:
    """Sample RSS n times (busy-loop between samples to expose growth)."""
    samples = []
    acc = 0
    for i in range(n):
        acc += i
        samples.append(current_rss_bytes())
    return samples
