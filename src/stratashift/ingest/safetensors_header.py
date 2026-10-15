"""safetensors header view — reads the header bytes only, never the weight body.

Format (https://huggingface.co/docs/safetensors):
    [u64 LE header_len][JSON header][data blob, 8-byte aligned]
Tensor ``data_offsets`` are relative to ``data_start`` (the first 8-byte-aligned
offset after the JSON header).

CONSTRAINTS C1/C2 (docs/CONSTRAINTS.md): this module must never copy the data
blob. Only the 8-byte length field and ``header_len`` JSON bytes are read.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

# safetensors dtype → element size in bytes (spec table, HF docs).
DTYPE_SIZES: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "U64": 8, "I32": 4, "U32": 4, "I16": 2, "U16": 2,
    "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E4M3FN": 1, "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1, "F8_E434": 1,
}

ALIGN = 8  # safetensors data section alignment (HF spec)


@dataclass(frozen=True)
class TensorMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]  # begin, end — relative to data_start

    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True)
class ModelMeta:
    path: str
    tensors: list[TensorMeta]
    header_size: int      # header_len (JSON bytes, excluding the u64 prefix)
    data_start: int       # absolute file offset of the data blob
    file_size: int
    metadata: dict = field(default_factory=dict)


def parse_header(path: str) -> ModelMeta:
    """Parse the safetensors header. Reads exactly 8 + header_len bytes."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: too short for a safetensors header")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len == 0:
            raise ValueError(f"{path}: empty header")
        header_bytes = f.read(header_len)
        file_size = f.seek(0, 2)
    if len(header_bytes) != header_len:
        raise ValueError(f"{path}: header truncated ({len(header_bytes)}/{header_len})")

    header = json.loads(header_bytes)
    metadata = header.get("__metadata__", {})
    data_start = 8 + header_len
    data_start += (-data_start) % ALIGN  # pad to 8-byte boundary

    tensors: list[TensorMeta] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = entry["dtype"]
        if dtype not in DTYPE_SIZES:
            raise ValueError(f"{path}: unknown dtype {dtype!r} for {name!r}")
        begin, end = entry["data_offsets"]
        if begin < 0 or end < begin:
            raise ValueError(f"{path}: bad data_offsets [{begin},{end}] for {name!r}")
        shape = tuple(entry["shape"])
        expected = DTYPE_SIZES[dtype]
        for dim in shape:
            expected *= dim
        if end - begin != expected:
            raise ValueError(
                f"{path}: {name!r} payload {end - begin} B != shape*dtype {expected} B"
            )
        tensors.append(TensorMeta(name=name, dtype=dtype, shape=shape,
                                  data_offsets=(begin, end)))

    if tensors:
        max_end = max(t.data_offsets[1] for t in tensors)
        if data_start + max_end > file_size:
            raise ValueError(
                f"{path}: data section exceeds file size ({data_start + max_end} > {file_size})"
            )
    return ModelMeta(path=path, tensors=tensors, header_size=header_len,
                     data_start=data_start, file_size=file_size, metadata=metadata)
