"""GGUF v3 header view — reads the header bytes only, never the weight body.

Format (docs/rea/struct_layouts.md §1, pin c566ff4):
    [u32 magic "GGUF"][u32 version=3][u64 n_tensors][u64 n_kv]
    [KV metadata][tensor dir: name,n_dims,dtype,offset,shape]
    data_start = ceil(header_end, alignment); tensor offsets are relative to data_start.

Evidence: SOURCE gguf_reader.hpp:442-482 + DECOMPILED strata-gguf main
(magic 0x46554747 gate, v3 gate, alignment default 0x20=32, `general.alignment`).

CONSTRAINTS C1/C2: the data blob is never copied; only header bytes are read.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from stratashift.polystrata.headers import GGUF_MAGIC, GGUF_VERSION, GGUF_ALIGN_DEFAULT

# ggml type ids → names. Table from the pinned engine's tools/gguf_reader.py
# (docs/rea/struct_layouts.md §2; type 42 Q2_0 is ahead of gguf-py — REA P0.S6 finding).
GGML_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64",
    28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
    34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4", 37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8",
    39: "MXFP4", 40: "NVFP4", 41: "Q4_0_8_8", 42: "Q2_0",
}

# ggml block geometry (elems per block, bytes per block) — struct_layouts §2 table
# (SOURCE gguf_reader.hpp:125-175) cross-checked against tools/gguf_reader.py BLOCK_GEOMETRY.
BLOCK_GEOMETRY: dict[str, tuple[int, int]] = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "F64": (1, 8),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176),
    "Q6_K": (256, 210), "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98), "IQ1_S": (256, 50),
    "IQ4_NL": (32, 18), "IQ3_S": (256, 110), "IQ2_S": (256, 82), "IQ4_XS": (256, 136),
    "IQ1_M": (256, 56), "Q2_0": (64, 18), "MXFP4": (32, 17), "NVFP4": (64, 36),
}

# GGUF metadata value types (struct_layouts §1: MetaType 0..11)
_META_SIZES: dict[int, int] = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_META_STRING, _META_ARRAY = 8, 9
_ARRAY_ELEM_FMT: dict[int, str] = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                                   6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


@dataclass(frozen=True)
class GgufTensor:
    name: str
    shape: tuple[int, ...]
    dtype_id: int
    type_name: str
    offset: int  # relative to data_start

    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def expected_bytes(self) -> int:
        elems, block = BLOCK_GEOMETRY[self.type_name]
        return self.elements() * block // elems


@dataclass(frozen=True)
class GgufMeta:
    path: str
    version: int
    tensors: list[GgufTensor]
    metadata: dict
    alignment: int
    data_start: int
    file_size: int


def _align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def parse_gguf_header(path: str) -> GgufMeta:
    """Parse the GGUF v3 header. Reads header bytes only (C1/C2)."""
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) != 24:
            raise ValueError(f"{path}: too short for a GGUF header")
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", head)
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file (bad magic {magic:#x})")
        if version != GGUF_VERSION:
            raise ValueError(f"{path}: GGUF v{version}, this reader handles v3")

        metadata: dict = {}
        pos = 16

        def read(n: int) -> bytes:
            nonlocal pos
            b = f.read(n)
            if len(b) != n:
                raise ValueError(f"{path}: truncated header at offset {pos}")
            pos += n
            return b

        def read_string() -> str:
            raw = read(struct.unpack("<Q", read(8))[0])
            return raw.decode("utf-8")

        for _ in range(n_kv):
            key = read_string()
            vtype = struct.unpack("<I", read(4))[0]
            if vtype == _META_STRING:
                metadata[key] = read_string()
            elif vtype == _META_ARRAY:
                etype = struct.unpack("<I", read(4))[0]
                count = struct.unpack("<Q", read(8))[0]
                fmt = _ARRAY_ELEM_FMT[etype]
                metadata[key] = [struct.unpack(fmt, read(struct.calcsize(fmt)))[0]
                                 for _ in range(count)]
            else:
                size = _META_SIZES[vtype]
                raw = read(size)
                fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                       6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[vtype]
                metadata[key] = struct.unpack(fmt, raw)[0]

        alignment = int(metadata.get("general.alignment", GGUF_ALIGN_DEFAULT))

        tensors: list[GgufTensor] = []
        for _ in range(n_tensors):
            name = read_string()
            n_dims = struct.unpack("<I", read(4))[0]
            # GGUF tensor record order (engine gguf_reader.hpp:463-469): n_dims, shape, dtype, offset.
            shape = tuple(struct.unpack("<Q", read(8))[0] for _ in range(n_dims))
            dtype_id = struct.unpack("<I", read(4))[0]
            offset = struct.unpack("<Q", read(8))[0]
            type_name = GGML_TYPES.get(dtype_id)
            if type_name is None:
                raise ValueError(f"{path}: unknown ggml type id {dtype_id} in {name}")
            tensors.append(GgufTensor(name, shape, dtype_id, type_name, offset))

        file_size = f.seek(0, 2)
        data_start = _align_up(pos, alignment)
        return GgufMeta(path=path, version=version, tensors=tensors, metadata=metadata,
                        alignment=alignment, data_start=data_start, file_size=file_size)
