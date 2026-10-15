"""CLI entry: `stratashift inspect` / `stratashift convert`.

CLI holds argument parsing only; conversion logic lives in ingest/meta/stream/quant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from stratashift.ingest.safetensors_header import parse_header
from stratashift.meta.expert_map import build_expert_map
from stratashift.meta.manifest import build_index_row, write_index_txt, write_manifest
from stratashift.monitor.rss import current_rss_bytes
from stratashift.polystrata.headers import GGUF_ALIGN_DEFAULT, Q2_0_BLOCK_BYTES
from stratashift.stream.mmap_reader import iter_tensor_payloads

GB = 1 << 30


def cmd_inspect(model: str, as_json: bool) -> int:
    meta = parse_header(model)
    refs = build_expert_map(meta)
    out = {
        "path": model,
        "header_size": meta.header_size,
        "data_start": meta.data_start,
        "file_size": meta.file_size,
        "tensors": len(meta.tensors),
        "expert_candidates": len(refs),
        "layers": sorted({r.layer_id for r in refs}),
        "metadata": meta.metadata,
    }
    if as_json:
        print(json.dumps(out, indent=2))
    else:
        print(f"tensors={len(meta.tensors)} experts={len(refs)} layers={out['layers']} "
              f"data_start={meta.data_start}")
    return 0


def cmd_convert(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    meta = parse_header(model)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    dst = 0
    rss_samples: list[int] = []
    aborted = False
    limit = int(max_rss_gb * GB) if max_rss_gb else None

    # engine index is tensor-keyed (WeightTable::find by name) → one row per tensor;
    # the expert map drives slicing/selection, not index rows.
    with open(out / "experts.bin", "wb") as fh:
        pass  # truncate
    for entry, blob in iter_tensor_payloads(model, meta):
        if len(blob) % Q2_0_BLOCK_BYTES:
            # not a Q2_0-shaped tensor (e.g. F32) — skip, record in manifest
            rss_samples.append(current_rss_bytes())
            continue
        codes = bytearray()
        scales = bytearray()
        for i in range(0, len(blob), Q2_0_BLOCK_BYTES):
            scales += blob[i:i + 2]
            codes += blob[i + 2:i + Q2_0_BLOCK_BYTES]
        with open(out / "experts.bin", "ab") as fh:
            fh.write(codes)
            fh.write(scales)
        blocks = len(blob) // Q2_0_BLOCK_BYTES
        rows.append(build_index_row(entry.name, entry.data_offsets[0], len(blob), dst,
                                     len(codes), len(scales), blocks))
        dst += len(codes) + len(scales)
        rss_samples.append(current_rss_bytes())
        del blob
        if limit is not None and rss_samples[-1] > limit:
            aborted = True
            break

    pool = write_index_txt(out / "index.txt", GGUF_ALIGN_DEFAULT, rows)
    experts = out / "experts.bin"
    manifest = {
        "model": model,
        "out_dir": str(out),
        "packed_rows": len(rows),
        "skipped_slices": len(rss_samples) - len(rows),
        "pool_bytes": pool,
        "experts_bin_sha256": hashlib.sha256(experts.read_bytes()).hexdigest(),
        "rss_samples_bytes": rss_samples,
        "rss_peak_bytes": max(rss_samples) if rss_samples else 0,
        "aborted_max_rss": aborted,
    }
    write_manifest(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 3 if aborted else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stratashift")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ins = sub.add_parser("inspect", help="parse header + expert map (no full load)")
    p_ins.add_argument("model")
    p_ins.add_argument("--json", action="store_true")
    p_conv = sub.add_parser("convert", help="stream-pack Q2_0 slices")
    p_conv.add_argument("model")
    p_conv.add_argument("--out", required=True)
    p_conv.add_argument("--max-rss-gb", type=float, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        return cmd_inspect(args.model, args.json)
    return cmd_convert(args.model, args.out, args.max_rss_gb)


if __name__ == "__main__":
    sys.exit(main())
