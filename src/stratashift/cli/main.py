"""CLI entry: `stratashift inspect` / `convert` / `verify`.

CLI holds argument parsing only; conversion logic lives in ingest/meta/stream/quant/polystrata.
`convert` accepts safetensors (Phase 2) and GGUF v3 (Phase 3 engine path,
docs/rea/struct_layouts.md §1). GGUF convert emits the engine pack layout
(experts.bin + dense.bin + embd.bin + engine-format manifest.json + index.txt)
so the pinned engine's own verifier (tools/strata_pack.py verify) can accept it.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

from stratashift.ingest.gguf import GGUF_MAGIC
from stratashift.ingest.safetensors_header import parse_header
from stratashift.ingest.shard_index import load_sharded_model
from stratashift.meta.expert_map import build_expert_map
from stratashift.meta.hf_gguf_map import is_hf_naming
from stratashift.monitor.rss import current_rss_bytes
from stratashift.polystrata import engine_tools
from stratashift.polystrata.verify_pack import verify_pack
from stratashift.stream.convert_pack import convert_gguf, convert_safetensors
from stratashift.stream.emit_gguf import emit_gguf as cmd_emit_gguf
from stratashift.stream.emit_hf import emit_gguf_hf as cmd_emit_gguf_hf


def cmd_inspect(model: str, as_json: bool) -> int:
    if pathlib.Path(model).is_dir():
        meta = load_sharded_model(model)
        refs = build_expert_map(meta)
        out = {
            "path": model,
            "shards": len(meta.shard_data_start),
            "header_size": meta.header_size,
            "file_size": meta.file_size,
            "tensors": len(meta.tensors),
            "expert_candidates": len(refs),
            "layers": sorted({r.layer_id for r in refs}),
            "metadata": meta.metadata,
        }
        if as_json:
            print(json.dumps(out, indent=2))
        else:
            print(f"shards={len(meta.shard_data_start)} tensors={len(meta.tensors)} "
                  f"experts={len(refs)} layers={out['layers']}")
        return 0
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
    if pathlib.Path(model).is_dir():
        # p4-02 scope: multi-shard input is accepted by the HF emit path (engine GGUF
        # acceptance); the Q2_0 slice convert pack path is single-file only.
        print(f"{model}: multi-shard convert is not supported — use --emit-gguf --hf-emit "
              f"(engine GGUF acceptance is the Phase 4 target)", file=sys.stderr)
        return 2
    with open(model, "rb") as fh:
        head = fh.read(4)
    if struct.unpack("<I", head)[0] == GGUF_MAGIC:
        return convert_gguf(model, out_dir, max_rss_gb)
    return convert_safetensors(model, out_dir, max_rss_gb)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="stratashift")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_ins = sub.add_parser("inspect", help="parse header + expert map (no full load)")
    p_ins.add_argument("model")
    p_ins.add_argument("--json", action="store_true")
    p_conv = sub.add_parser("convert", help="stream-pack Q2_0 slices (safetensors or GGUF)")
    p_conv.add_argument("model")
    p_conv.add_argument("--out", required=True)
    p_conv.add_argument("--max-rss-gb", type=float, default=None)
    p_conv.add_argument("--emit-gguf", default=None,
                        help="safetensors float experts: also write an engine GGUF v3 "
                             "(Q2_0 tensors) at this path")
    p_conv.add_argument("--guard-keys", default=None,
                        help="JSON preset of compiled architecture guard keys for --emit-gguf "
                             "(never invented; struct_layouts §1)")
    p_conv.add_argument("--hf-emit", action="store_true",
                        help="safetensors uses HF naming (model.layers.*, fused gate_up): "
                             "map to engine GGUF keys and emit (p4-01); auto-detected, "
                             "flag forces the HF path")
    p_conv.add_argument("--dense-type", default="F32",
                        choices=("F32", "Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q8_0"),
                        help="emit-gguf dense tensor type: F32 pass-through (default) "
                             "or quantized blocks (p3-03/p3-04/p3-05/p3-06/p3-07/p3-09)")
    p_conv.add_argument("--imatrix", default=None,
                        help="emit-gguf: GGUF imatrix stats file (calibration producer "
                             "output, engine-keyed planes). Expert slots: gate/up Q4_K + "
                             "down Q4_0 per plane unit; dense engine-name lookup. Opt-in — "
                             "the plain RTN path is byte-unchanged without it (p6-03 AC3).")
    p_conv.add_argument("--engine-src", default=None,
                        help="emit-gguf: pinned engine checkout (its tools/ holds gguf_writer). "
                             "Beats STRATASHIFT_ENGINE_SRC / STRATASHIFT_ENGINE_TOOLS; default "
                             ".local/polystrata/PolyStrata-main (p6-09)")
    p_ver = sub.add_parser("verify", help="local self-check of a GGUF-path pack")
    p_ver.add_argument("out")
    return ap


def _load_guard(path: str | None) -> dict:
    return json.loads(pathlib.Path(path).read_text()) if path else {}


def _convert_shard_dir(args: argparse.Namespace) -> int:
    # p4-02: directory = multi-shard HF input (index.json + shards). The Q2_0 slice
    # convert pack path is single-file only; the engine GGUF emit path (the Phase 4
    # acceptance target) takes the shard directory directly.
    if not args.emit_gguf:
        print(f"{args.model}: multi-shard convert is not supported — use "
              f"--emit-gguf (engine GGUF acceptance is the Phase 4 target)",
              file=sys.stderr)
        return 2
    # p6-05: the HF fused emit path consumes --imatrix planes (engine-key
    # lookup; expert slots gate/up Q4_K + down Q4_0, dense engine-name lookup).
    return cmd_emit_gguf_hf(args.model, args.emit_gguf, _load_guard(args.guard_keys),
                            args.max_rss_gb, args.dense_type, args.imatrix)


def _wants_hf_emit(args: argparse.Namespace) -> bool:
    if args.hf_emit:
        return True
    # p5-02 auto-detection: safetensors with HF naming routes to the fused HF emit path
    # WITHOUT the flag (p4-01 "auto-detected" claim, connected here — is_hf_naming existed
    # in the map module, dispatch never used it).
    with open(args.model, "rb") as f:
        gguf = struct.unpack("<I", f.read(4))[0] == GGUF_MAGIC
    return not gguf and is_hf_naming(parse_header(args.model))


def _convert_file(args: argparse.Namespace) -> int:
    rc = cmd_convert(args.model, args.out, args.max_rss_gb)
    if rc != 0 or not args.emit_gguf:
        return rc
    emit = cmd_emit_gguf_hf if _wants_hf_emit(args) else cmd_emit_gguf
    return emit(args.model, args.emit_gguf, _load_guard(args.guard_keys), args.max_rss_gb,
                args.dense_type, args.imatrix)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "inspect":
        return cmd_inspect(args.model, args.json)
    if args.cmd == "convert":
        if args.max_rss_gb and current_rss_bytes() is None:
            # p6-10: a ceiling that cannot be measured must not pass silently
            print(f"error: --max-rss-gb needs RSS measurement, unavailable on {sys.platform}",
                  file=sys.stderr)
            return 2
        if args.emit_gguf:
            # p6-09: the option must pick the checkout before the deferred engine import
            engine_tools.load(args.engine_src)
        if pathlib.Path(args.model).is_dir():
            return _convert_shard_dir(args)
        return _convert_file(args)
    return verify_pack(args.out)


def _run() -> int:
    """p5-02 exit-code contract (AC3): user-facing I/O and value failures print
    `error: <msg>` on stderr and map to exit 1 — no traceback for missing files,
    unreadable dirs, header parse failures, guard/geometry refusals. argparse usage
    errors keep SystemExit(2) (not caught); developer exceptions stay visible.
    """
    try:
        return main()
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def cli_entry() -> None:
    """Console-script entry: propagate main()'s exit code as process exit code.

    setuptools console scripts call the entry function and ignore its return value,
    so the exit-code contract (0/1/2) is enforced here (p5-01 AC1, p5-02 AC3).
    """
    raise SystemExit(_run())


if __name__ == "__main__":
    sys.exit(_run())
