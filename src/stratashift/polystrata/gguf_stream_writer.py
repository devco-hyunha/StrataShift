"""Harness-side streaming GGUF v3 writer (moved out of cli/main.py, p6-08).

The engine tools are imported through `polystrata.engine_tools` (read-only checkout,
`gguf_writer` constants are reused; engine files are never edited).
"""
from __future__ import annotations

import os
import pathlib
import struct

import numpy as np

from stratashift.ingest.gguf import GGUF_MAGIC
from stratashift.polystrata import engine_tools


# 256 MiB periodic fsync: harness tuning, not an engine format value. Measured: ENOMEM
# (OSError Errno 12) at 3.57 GB of unflushed tmp on the 7.7 GB WSL box (2026-10-17,
# docs/work/p4-03.md:113-115); the 256 MB period completed the p5-08 real-model emit
# (docs/work/p5-08.md:267).
FSYNC_EVERY_BYTES = 1 << 28


class StreamingGGUFWriter:
    """Harness-side GGUF v3 writer that streams tensor payloads to a temp file as they
    are produced (p4-01 AC5: heap peak = ONE tensor, not the whole GGUF).

    Byte-identical to the engine's GGUFWriter output (tools/gguf_writer.py:151-180,
    pin c566ff4, read-only): same magic/version/counts, metadata kv encoding, tensor
    directory, alignment 32, per-tensor padding. The engine writer accumulates ALL
    tensor bytes in w.tensors and write() copies the full body — the p3 fixtures fit;
    the p4 qwen4exp fixture (1.8 GB emitted) OOM-killed the process on the 7.7 GB
    WSL box (observed 2026-10-17, RC=137), so the p4 emit path streams. Engine files
    are never edited; this is the harness-side writer.
    """

    def __init__(self, alignment: int = 32, tmp_path: str | None = None):
        self.alignment = alignment
        self.metadata: dict[str, tuple[str, object]] = {}
        self.tensors: list[tuple] = []  # (name, shape, type_name, offset, nbytes)
        self._tmp = pathlib.Path(tmp_path) if tmp_path else \
            pathlib.Path("/opt/data/cache/scratch/stratashift-gguf-tmp.bin")
        self._fh = open(self._tmp, "wb")
        self._off = 0
        self._since_fsync = 0

    def __enter__(self) -> "StreamingGGUFWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.discard()

    def discard(self) -> None:
        """Close the temp handle and remove the temp file. No-op after a successful
        write() (which already closed and unlinked it); on an early exit (RSS abort
        rc=3, refusal, exception) it keeps a multi-GB `<out>.tmp` from being left
        behind with an open handle (p6-08 review)."""
        if not self._fh.closed:
            self._fh.close()
        self._tmp.unlink(missing_ok=True)

    def add(self, key: str, value, type_name: str | None = None) -> None:
        """Metadata kv, same inference table as the engine writer (gguf_writer.py:93)."""
        if type_name is None:
            type_name = {bool: "bool", int: "i64", float: "f64", str: "string"}.get(type(value))
        if type_name is None:
            if isinstance(value, list) and value:
                elem = {bool: "bool", int: "i32", float: "f32", str: "string"}.get(type(value[0]))
                if elem is None:
                    raise TypeError(f"unsupported array element {type(value[0])} for {key}")
                type_name = "array:" + elem
            else:
                raise TypeError(f"cannot infer a GGUF type for {key}")
        self.metadata[key] = (type_name, value)

    def add_tensor(self, name: str, shape: list[int], type_name: str, data: bytes) -> None:
        """Stream one tensor payload to the temp file; keep only its directory entry."""
        pad = (-len(data)) % self.alignment
        self._fh.write(data)
        if pad:
            self._fh.write(b"\0" * pad)
        self._flush_periodic(len(data) + pad)
        self.tensors.append((name, list(shape), type_name, self._off, len(data)))
        self._off += len(data) + pad

    def _flush_periodic(self, written: int) -> None:
        """p3-09 convention: fsync every 256 MB so dirty pages of the streamed GGUF
        write back incrementally. ENOMEM on a later small read (OSError Errno 12,
        observed 2026-10-17 at 3.57 GB of unflushed tmp on the 7.7 GB WSL box) is
        the dirty-page accumulation, not heap growth."""
        self._since_fsync += written
        if self._since_fsync >= FSYNC_EVERY_BYTES:
            os.fsync(self._fh.fileno())
            self._since_fsync = 0

    def add_tensor_stream(self, name: str, shape: list[int], type_name: str, chunks) -> int:
        """Stream a tensor payload chunk-by-chunk to the temp file; keep only its
        directory entry. Heap peak = ONE chunk, not the whole tensor (p4-01 AC5).
        Returns the total payload bytes written."""
        total = 0
        for c in chunks:
            self._fh.write(c)
            total += len(c)
            self._flush_periodic(len(c))
        pad = (-total) % self.alignment
        if pad:
            self._fh.write(b"\0" * pad)
        self._flush_periodic(pad)
        self.tensors.append((name, list(shape), type_name, self._off, total))
        self._off += total + pad
        return total

    def add_f32(self, name: str, arr) -> None:
        a = np.ascontiguousarray(arr, dtype="<f4")
        self.add_tensor(name, list(a.shape), "F32", a.tobytes())

    def write(self, path) -> pathlib.Path:
        """Header + tensor directory + padded data, streamed from the temp file."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        out = pathlib.Path(path)
        meta_body = bytearray()
        for k, (t, v) in self.metadata.items():
            meta_body += self._kv_bytes(k, t, v)
        head = struct.pack("<I", GGUF_MAGIC) + struct.pack("<I", 3)
        head += struct.pack("<Q", len(self.tensors)) + struct.pack("<Q", len(self.metadata))
        infos = bytearray()
        for name, shape, type_name, off, _n in self.tensors:
            infos += struct.pack("<Q", len(name)) + name.encode("utf-8")
            infos += struct.pack("<I", len(shape))
            infos += struct.pack(f"<{len(shape)}Q", *shape)
            infos += struct.pack("<I", engine_tools.gguf_writer.TYPE_IDS[type_name])
            infos += struct.pack("<Q", off)
        body = bytes(head) + bytes(meta_body) + bytes(infos)
        pad = (-len(body)) % self.alignment
        with open(out, "wb") as dst:
            dst.write(body)
            dst.write(b"\0" * pad)
            with open(self._tmp, "rb") as src:
                while True:
                    chunk = src.read(1 << 24)
                    if not chunk:
                        break
                    dst.write(chunk)
        self._tmp.unlink()
        return out

    def _kv_bytes(self, key: str, type_name: str, value) -> bytes:
        """Metadata kv bytes, same encoding as the engine writer (gguf_writer.py:127)."""
        out = bytearray()
        out += struct.pack("<Q", len(key)) + key.encode("utf-8")
        if type_name.startswith("array:"):
            elem = type_name.split(":", 1)[1]
            out += struct.pack("<I", engine_tools.gguf_writer._META_IDS["array"])
            out += struct.pack("<I", engine_tools.gguf_writer._META_IDS[elem])
            out += struct.pack("<Q", len(value))
            for v in value:
                out += self._scalar(elem, v)
            return bytes(out)
        out += struct.pack("<I", engine_tools.gguf_writer._META_IDS[type_name])
        out += self._scalar(type_name, value)
        return bytes(out)

    @staticmethod
    def _scalar(type_name: str, v) -> bytes:
        if type_name == "string":
            b = v.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        fmt = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i",
               "f32": "<f", "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}[type_name]
        return struct.pack(fmt, v)
