"""Slice readers for streamed model files (seek+read, NOT mmap).

Moved out of cli/main.py (p6-08). Contract unchanged from p4-01/p4-02:

Reading is a plain seek+read, never mmap (p4-01 streaming hygiene). The streamed
fixture is read once, sequentially, so its pages are never reused. mmap keeps every
touched page resident: on a 10 GB fixture the page cache grows VmRSS toward the fixture
size and the process is OOM-killed (RC=137, observed 2026-10-17) even with fadvise
DONTNEED per slice, on both 9p and ext4. A plain seek+read leaves no page cache behind,
so VmRSS stays at the heap peak.
"""
from __future__ import annotations

import pathlib

from stratashift.ingest.safetensors_header import DTYPE_SIZES
from stratashift.ingest.shard_index import abs_offset


class SliceReader:
    """Open files on demand and cache the handles (handles, not pages); one slice per read."""

    def __init__(self) -> None:
        self._handles: dict[str, "object"] = {}

    def read(self, path: str, lo: int, n: int) -> bytes:
        fh = self._handles.get(path)
        if fh is None:
            fh = open(path, "rb")
            self._handles[path] = fh
        fh.seek(lo)
        return fh.read(n)

    def close(self) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()

    def __enter__(self) -> "SliceReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TensorReader(SliceReader):
    """safetensors tensor payloads of one model (single file or multi-shard directory)."""

    def __init__(self, model: str, meta) -> None:
        super().__init__()
        self._model = model
        self._meta = meta

    def shard_of(self, entry) -> str:
        # weight_map names shards as basenames — resolve against the model directory.
        return str(pathlib.Path(self._model) / entry.shard) if entry.shard else self._model

    def payload(self, entry) -> bytes:
        lo = abs_offset(self._meta, entry)
        return self.read(self.shard_of(entry), lo, entry.nbytes())

    def payload_chunks(self, entry, chunk_elems: int = 1 << 19):
        """Stream one tensor's payload in 32-element-aligned chunks. Quantizer blocks
        are 16/32 elements (Q3_K/Q6_K: 16, Q4_0/Q4_K/Q8_0: 32), so concatenating
        chunk-boundary outputs is byte-identical to quantizing the full tensor.
        p4-03 root cause: full payload of embed/lm_head (622 MB BF16 → 1.24 GB F32)
        pushed the heap peak to 2.9 GiB > 2 GiB gate (observed 2026-10-17, second
        emit OOM RC=137)."""
        esz = DTYPE_SIZES[entry.dtype]
        step = chunk_elems * esz
        lo = abs_offset(self._meta, entry)
        shard = self.shard_of(entry)
        remaining = entry.nbytes()
        while remaining > 0:
            n = min(step, remaining)
            yield self.read(shard, lo, n)
            lo += n
            remaining -= n
