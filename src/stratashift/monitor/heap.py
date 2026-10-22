"""Python heap peak gate (tracemalloc), moved out of cli/main.py (p6-08).

Heap peak, not VmRSS, gates the streaming slice (p3-03 AC5): VmRSS counts mmap/page-cache
pages of the streamed file, which are reclaimable and inflate the reading.
"""
from __future__ import annotations

import tracemalloc
from contextlib import contextmanager


@contextmanager
def heap_gate():
    """p5-02 AC4 root cause fix (2026-10-18): tracemalloc start/stop must be PAIRED on
    every exit path. Observed in the full suite: the BF16 emit `ValueError` raised after
    `tracemalloc.start()` bypassed the trailing `stop()`, leaving the tracer active for
    the rest of the session — `test_no_full_load` header peak budget then measured the
    cumulative session peak (1,123,499 B vs 133,770 B standalone) and failed. Bodies
    run inside `with heap_gate():` so exceptions, RSS aborts (return 3) and normal
    returns all stop the tracer.
    """
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


def heap_peak_bytes() -> int:
    """Peak traced heap since `heap_gate()` started (valid only inside the gate)."""
    return tracemalloc.get_traced_memory()[1]
