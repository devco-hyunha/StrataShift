"""Streaming slice readers — mmap kept open, yields ONE tensor payload at a time.

CONSTRAINTS C1/C2: the whole file is never copied; peak live data is a single
slice. The generator releases each slice before yielding the next.
"""
from __future__ import annotations

import mmap
from collections.abc import Iterator

from stratashift.ingest.safetensors_header import ModelMeta, TensorMeta


def iter_tensor_payloads(path: str, meta: ModelMeta) -> Iterator[tuple[TensorMeta, bytes]]:
    """Yield (entry, slice_bytes) in header order — one slice alive at a time."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for entry in meta.tensors:
                begin, end = entry.data_offsets
                yield entry, mm[meta.data_start + begin: meta.data_start + end]
        finally:
            mm.close()
