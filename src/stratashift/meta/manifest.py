"""Writers for conversion outputs: experts.bin, index.txt, manifest.json.

`index.txt` rows follow the pinned 19-field schema (docs/rea/struct_layouts.md §4,
DECOMPILED F9): name file kind src_off src_bytes dst_off dst_bytes ne0 ne1
code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes
offset_bytes scales_fp16 act_kind.

Q2_0 pack rows: kind=0 VERBATIM (codes/scales copied verbatim), code_bits=2,
code_bias=1, group_elems=QK64, scales_fp16=1, has_offset=0.

DRAFT limitation (Phase 3): fused-tensor rows share the source tensor name;
engine `WeightTable::find` is first-match, so name uniqueness must be settled
with a real engine load test (CONSTRAINTS C4 — not finalized here).
"""
from __future__ import annotations

import json
from pathlib import Path

from stratashift.polystrata.headers import INDEX_FIELDS, QK2_0
from stratashift.polystrata.index import ROW_FIELDS


def build_index_row(name: str, src_off: int, src_bytes: int, dst_off: int,
                    codes_bytes: int, scales_bytes: int, blocks: int) -> str:
    """One 19-field row for a Q2_0 VERBATIM pack slice."""
    values = {
        "name": name, "file": 0, "kind": 0,
        "src_off": src_off, "src_bytes": src_bytes,
        "dst_off": dst_off, "dst_bytes": codes_bytes + scales_bytes,
        "ne0": blocks, "ne1": 1,
        "code_bits": 2, "code_bias": 1, "group_elems": QK2_0, "codebook": 0,
        "has_offset": 0, "codes_bytes": codes_bytes, "scales_bytes": scales_bytes,
        "offset_bytes": 0, "scales_fp16": 1, "act_kind": 0,
    }
    if len(values) != INDEX_FIELDS:
        raise ValueError(f"row schema drift: {len(values)} != {INDEX_FIELDS}")
    return " ".join(str(values[f]) for f in ROW_FIELDS)


def write_index_txt(path: Path, align: int, rows: list[str]) -> int:
    """Write header + rows; returns total pool bytes referenced by rows."""
    pool = sum(int(r.split()[6]) for r in rows)  # dst_bytes column
    lines = [f"# align {align} pool {pool} tensors {len(rows)}"] + rows
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pool


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
