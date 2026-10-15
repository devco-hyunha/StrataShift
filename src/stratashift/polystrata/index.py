"""`index.txt` schema — dense tensor index written by tools/pack_index.py and
read by WeightTable::load (docs/rea/struct_layouts.md §4; DECOMPILED F9).

Header line:  `# align <align> pool <bytes> tensors <count>`
Rows: 19 space-separated fields, no quoting. A row that does not yield exactly
19 fields is REFUSED (weights.cpp:204-207: a wrong byte offset decodes to a
plausible weight — refuse loudly, never partial-parse).
"""
from __future__ import annotations

from dataclasses import dataclass

from stratashift.polystrata.headers import INDEX_FIELDS

# field order per pack_index.py:44-60 / weights.cpp:190-205 (DECOMPILED sscanf)
ROW_FIELDS: tuple[str, ...] = (
    "name", "file", "kind", "src_off", "src_bytes", "dst_off", "dst_bytes",
    "ne0", "ne1", "code_bits", "code_bias", "group_elems", "codebook",
    "has_offset", "codes_bytes", "scales_bytes", "offset_bytes",
    "scales_fp16", "act_kind",
)

# kind codes (pack_index.py:81, weights.cpp:211-214)
KIND_VERBATIM = 0
KIND_BF16_IN_F32 = 1
KIND_F32_COPY = 2
KIND_F16_IN_F32 = 3
KIND_BF16_RAW16 = 4
KIND_F16_RAW16 = 5

_INT_FIELDS = {  # field → signed? (sscanf %d/%lld are signed, %llu unsigned)
    "file": False, "kind": True, "src_off": False, "src_bytes": False,
    "dst_off": False, "dst_bytes": False, "ne0": True, "ne1": True,
    "code_bits": True, "code_bias": True, "group_elems": True, "codebook": True,
    "has_offset": True, "codes_bytes": False, "scales_bytes": False,
    "offset_bytes": False, "scales_fp16": True, "act_kind": True,
}


@dataclass(frozen=True)
class IndexHeader:
    align: int
    pool: int
    tensors: int


@dataclass(frozen=True)
class IndexRow:
    values: dict  # field → int/str, 19 keys


def parse_index_header(line: str) -> IndexHeader:
    parts = line.split()
    if len(parts) != 7 or parts[0] != "#" or parts[1] != "align" \
            or parts[3] != "pool" or parts[5] != "tensors":
        raise ValueError(f"index.txt: bad header line {line!r}")
    try:
        align = int(parts[2])
        pool = int(parts[4])
        tensors = int(parts[6])
    except ValueError as exc:
        raise ValueError(f"index.txt: header not numeric: {line!r}") from exc
    return IndexHeader(align=align, pool=pool, tensors=tensors)


def parse_index_row(line: str) -> IndexRow:
    tokens = line.split()
    if len(tokens) != INDEX_FIELDS:
        raise ValueError(
            f"index.txt: row has {len(tokens)} fields, need exactly {INDEX_FIELDS} "
            f"(refused — partial parse means a wrong byte offset)"
        )
    values: dict = {"name": tokens[0]}
    for field, token in zip(ROW_FIELDS[1:], tokens[1:]):
        try:
            values[field] = int(token)
        except ValueError as exc:
            raise ValueError(f"index.txt: field {field!r} not numeric: {token!r}") from exc
    return IndexRow(values=values)
