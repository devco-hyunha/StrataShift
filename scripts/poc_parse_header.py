#!/usr/bin/env python3
"""PoC: parse a safetensors header without loading the model, print the expert
map, and (with --slice) read exactly ONE expert tensor slice via mmap.

Usage:
    python3 scripts/poc_parse_header.py <model.safetensors> [--slice] [--json]

CONSTRAINTS C1/C2: the data blob is mmap'd read-only; at most one tensor slice
is touched, never the whole file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stratashift.ingest.safetensors_header import ModelMeta, parse_header
from stratashift.meta.expert_map import build_expert_map, group_by_expert


def read_slice(path: str, meta: ModelMeta, begin: int, end: int) -> bytes:
    """Read one tensor slice via mmap (read-only, no full-file copy)."""
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return mm[meta.data_start + begin: meta.data_start + end]
        finally:
            mm.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="model.safetensors path")
    ap.add_argument("--slice", action="store_true",
                    help="read the first expert tensor slice via mmap")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    meta = parse_header(args.model)
    refs = build_expert_map(meta)
    groups = group_by_expert(refs)

    out: dict = {
        "path": args.model,
        "header_size": meta.header_size,
        "data_start": meta.data_start,
        "file_size": meta.file_size,
        "tensors": len(meta.tensors),
        "expert_candidates": len(refs),
        "experts": len(groups),
        "layers": sorted({r.layer_id for r in refs}),
        "metadata": meta.metadata,
    }
    if args.slice and refs:
        first = refs[0]
        blob = read_slice(args.model, meta, *first.data_offsets)
        out["slice"] = {
            "tensor": first.tensor_name,
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
        }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"header {meta.header_size} B, data_start {meta.data_start}, "
              f"file {meta.file_size} B, tensors {len(meta.tensors)}")
        print(f"expert candidates: {len(refs)} tensors in {len(groups)} experts, "
              f"layers {out['layers']}")
        for (layer, expert), rows in list(groups.items())[:3]:
            print(f"  layer {layer} expert {expert}: "
                  + ", ".join(f"{r.role}={r.data_offsets}" for r in rows))
        if "slice" in out:
            print(f"slice {out['slice']['tensor']}: {out['slice']['bytes']} B "
                  f"sha256={out['slice']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
