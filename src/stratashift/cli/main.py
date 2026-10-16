"""CLI entry: `stratashift inspect` / `convert` / `verify`.

CLI holds argument parsing only; conversion logic lives in ingest/meta/stream/quant.
`convert` accepts safetensors (Phase 2) and GGUF v3 (Phase 3 engine path,
docs/rea/struct_layouts.md §1). GGUF convert emits the engine pack layout
(experts.bin + dense.bin + embd.bin + engine-format manifest.json + index.txt)
so the pinned engine's own verifier (tools/strata_pack.py verify) can accept it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import pathlib
import struct
import sys
import tracemalloc

import numpy as np

# pinned engine checkout (p0-01, .local/polystrata) — read-only tools, engine never edited.
ENGINE_TOOLS = os.environ.get("STRATASHIFT_ENGINE_TOOLS", "/opt/data/polystrata-src/tools")
if ENGINE_TOOLS not in sys.path:
    sys.path.append(ENGINE_TOOLS)

from stratashift.ingest.gguf import GGUF_MAGIC, parse_gguf_header
from stratashift.ingest.safetensors_header import parse_header
from stratashift.meta.engine_manifest import blob_offsets, build_manifest, tensor_entry
from stratashift.meta.expert_map import build_expert_map
from stratashift.meta.manifest import build_index_row, write_index_txt, write_manifest
from stratashift.monitor.rss import current_rss_bytes
from stratashift.polystrata.headers import GGUF_ALIGN_DEFAULT, Q2_0_BLOCK_BYTES, Q4_0_MIN_PACK_ELEMENTS, QK4_0
from stratashift.polystrata.index import engine_rows, write_engine_index
from stratashift.quant.pack import expert_blob, quantize_q2_0_stream, quantize_q4_0_stream
from stratashift.stream.mmap_reader import iter_tensor_payloads

# engine writer is imported lazily inside cmd_emit_gguf (engine tools are read-only,
# p3-01 harness convention); TYPE_IDS lacks IQ4_NL (id 20, gguf_reader GGML_TYPES) —
# harness-side extension only, engine files untouched.
try:
    from gguf_writer import GGUFWriter, _Tensor  # noqa: E402
    import gguf_writer  # noqa: E402
    gguf_writer.TYPE_IDS["IQ4_NL"] = 20
    gguf_writer.TYPE_IDS["Q4_0"] = 2  # GGML_TYPES id, engine gguf_reader.py:21 (pin c566ff4)
except ImportError:  # engine checkout absent: emit path unavailable, convert/verify unaffected
    GGUFWriter = _Tensor = gguf_writer = None

GB = 1 << 30
EMBD_NAME = "token_embd.weight"  # tools/strata_pack.py:51
PLE_NAME = "per_layer_token_embd.weight"  # tools/strata_pack.py:301 (shard2 PLE table)
PLE_HEAD_DIM = 160  # include/strata/kernels/ngram.hpp:32 (VERIFIED p3-01)
PLE_ROWS = 8  # p3-01 fixture GGUF shape [160, 8] — engine acceptance VERIFIED
PLE_BYTES = PLE_HEAD_DIM * PLE_ROWS // 32 * 18  # IQ4_NL: 32 elems per 18 B block = 720


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


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
    with open(model, "rb") as fh:
        head = fh.read(4)
    if struct.unpack("<I", head)[0] == GGUF_MAGIC:
        return cmd_convert_gguf(model, out_dir, max_rss_gb)
    return cmd_convert_safetensors(model, out_dir, max_rss_gb)


def cmd_convert_safetensors(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    meta = parse_header(model)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    dst = 0
    rss_samples: list[int] = []
    aborted = False
    tracemalloc.start()  # p3-03 AC5: heap peak, not VmRSS (page cache inflates VmRSS)
    limit = int(max_rss_gb * GB) if max_rss_gb else None

    # engine index is tensor-keyed (WeightTable::find by name) → one row per tensor;
    # the expert map drives slicing/selection, not index rows.
    with open(out / "experts.bin", "wb") as fh:
        pass  # truncate
    for entry, blob in iter_tensor_payloads(model, meta):
        if entry.name == PLE_NAME:
            # PLE table is IQ4_NL bytes, not Q2_0 (ngram.hpp PleFormat enum) — never pack it.
            rss_samples.append(current_rss_bytes())
            continue
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
        "experts_bin_sha256": _sha256(experts),
        "rss_samples_bytes": rss_samples,
        "rss_peak_bytes": max(rss_samples) if rss_samples else 0,
        "heap_peak_bytes": tracemalloc.get_traced_memory()[1],
        "aborted_max_rss": aborted,
    }
    tracemalloc.stop()
    write_manifest(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 3 if aborted else 0


def cmd_convert_gguf(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    meta = parse_gguf_header(model)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    limit = int(max_rss_gb * GB) if max_rss_gb else None
    rss_samples: list[int] = []
    tracemalloc.start()  # p3-03 AC5: heap peak, not VmRSS (page cache inflates VmRSS)

    by_name = {t.name: t for t in meta.tensors}
    layers: dict[int, dict[str, int]] = {}
    for t in meta.tensors:
        if t.name.startswith("blk.") and t.name.endswith("_exps.weight"):
            layer = int(t.name.split(".")[1])
            role = t.name.split(".")[2].removesuffix("_exps")  # ffn_gate / ffn_up / ffn_down
            layers.setdefault(layer, {})[role] = t.offset
    if not layers:
        raise ValueError(f"{model}: no blk.* ffn_*_exps.weight expert tensors found")
    layer_ids = sorted(layers)
    gate0 = by_name[f"blk.{layer_ids[0]}.ffn_gate_exps.weight"]
    down0 = by_name[f"blk.{layer_ids[0]}.ffn_down_exps.weight"]
    H, FF, NE = gate0.shape[0], gate0.shape[1], gate0.shape[2]
    if down0.shape[:2] != (FF, H):
        raise ValueError(f"{model}: down shape {down0.shape[:2]} != ({FF}, {H})")
    role_bytes = H * FF * Q2_0_BLOCK_BYTES // 64  # contiguous run per expert per role
    blob_bytes = 3 * role_bytes

    for name in ("experts.bin", "dense.bin", "embd.bin"):
        (out / name).write_bytes(b"")  # truncate

    expert_layers: list[dict] = []
    with open(model, "rb") as src:
        mm = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            # experts.bin: one blob per expert, streamed (C1/C2: one slice alive at a time)
            for layer in layer_ids:
                layer_off = (out / "experts.bin").stat().st_size
                role_offsets = {}
                for e in range(NE):
                    slices = []
                    for role in ("ffn_gate", "ffn_up", "ffn_down"):
                        off = meta.data_start + layers[layer][role] + e * role_bytes
                        slices.append(mm[off:off + role_bytes])
                    blob = expert_blob(slices[0], slices[1], slices[2], H, FF)
                    del slices
                    with open(out / "experts.bin", "ab") as fh:
                        fh.write(blob)
                    del blob
                    rss_samples.append(current_rss_bytes())
                    if limit is not None and rss_samples[-1] > limit:
                        return 3
                for role in ("ffn_gate", "ffn_up", "ffn_down"):
                    role_offsets[role] = by_name[f"blk.{layer}.{role}_exps.weight"].offset
                expert_layers.append({"layer": layer, "offset": layer_off,
                                      "role_offsets": role_offsets})

            # dense.bin / embd.bin: stream one tensor slice at a time, pack planes at ALIGN=64.
            # Engine insertion order is dense group first, then embd (tools/strata_pack.py:268-270
            # `for fname, group in (("dense.bin", dense), ("embd.bin", embd))`), and pack_index.py
            # emits manifest insertion order — so index.txt rows must follow the same grouping.
            tensor_entries: dict = {}
            seg = {"dense.bin": 0, "embd.bin": 0}
            groups = [("dense.bin", [t for t in meta.tensors
                                     if not (t.name.startswith("blk.")
                                             and t.name.endswith("_exps.weight"))
                                     and t.name != EMBD_NAME]),
                      ("embd.bin", [t for t in meta.tensors if t.name == EMBD_NAME])]
            for fname, group in groups:
                for t in group:
                    raw = mm[meta.data_start + t.offset:
                            meta.data_start + t.offset + t.expected_bytes()]
                    body, ent = tensor_entry(t.name, t, raw, seg[fname], fname)
                    with open(out / fname, "ab") as fh:
                        fh.write(body)
                    seg[fname] += len(body)
                    tensor_entries[t.name] = ent
                    rss_samples.append(current_rss_bytes())
                    del raw, body
                    if limit is not None and rss_samples[-1] > limit:
                        return 3
        finally:
            mm.close()

    experts = {
        "blob_bytes": blob_bytes,
        "offsets": blob_offsets(H, FF),
        "layers": expert_layers,
        "total_bytes": len(layer_ids) * NE * blob_bytes,
        "scales_fp16": True, "source_type": "Q2_0", "form": "S2",
    }
    files = {name: (out / name).stat().st_size
            for name in ("experts.bin", "dense.bin", "embd.bin")}
    shard2 = str(model).replace("00001-of-00002", "00002-of-00002")
    man = build_manifest(model, len(layer_ids), NE, tensor_entries, experts, files, shard2)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    pool = write_engine_index(out / "index.txt", man)

    manifest = {
        "model": model,
        "out_dir": str(out),
        "layers": layer_ids,
        "geometry": {"H": H, "FF": FF, "NE": NE},
        "pool_bytes": pool,
        "experts_bin_sha256": _sha256(out / "experts.bin"),
        "dense_bin_sha256": _sha256(out / "dense.bin"),
        "embd_bin_sha256": _sha256(out / "embd.bin"),
        "index_txt_sha256": _sha256(out / "index.txt"),
        "rss_samples_bytes": rss_samples,
        "rss_peak_bytes": max(rss_samples) if rss_samples else 0,
        "heap_peak_bytes": tracemalloc.get_traced_memory()[1],
    }
    tracemalloc.stop()
    write_manifest(out / "conversion.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_verify(out_dir: str) -> int:
    """Local self-check of a GGUF-path pack (engine acceptance is the harness's job).

    Checks (ACCEPTANCE Phase 3): index 19-field contract with plane sums and pool
    recomputed from the manifest, experts.bin size == layers x NE x blob, every
    layer blob inside the file, output hashes recomputed for regression.
    """
    out = pathlib.Path(out_dir)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    rows = engine_rows(man)
    lines = (out / "index.txt").read_text(encoding="utf-8").splitlines()
    data_rows = [ln for ln in lines if not ln.startswith("#")]
    if len(data_rows) != len(rows):
        errors.append(f"index.txt rows {len(data_rows)} != recomputed {len(rows)}")
    for got, want in zip(data_rows, rows):
        if got.split() != [str(x) for x in want]:
            errors.append(f"index.txt row mismatch: {got[:60]}")
            break
    align = int(man["align"])
    recomputed = 0
    for r in rows:
        recomputed = (recomputed + r[6] + align - 1) // align * align
    header = lines[1].split()
    if int(header[4]) != recomputed:
        errors.append(f"header pool {header[4]} != recomputed {recomputed}")

    exp = man["experts"]
    want_size = man["n_layers"] * man["n_experts_per_layer"] * exp["blob_bytes"]
    got_size = (out / "experts.bin").stat().st_size
    if got_size != want_size:
        errors.append(f"experts.bin {got_size} != layers*NE*blob {want_size}")
    if exp["offsets"]["down_scales"] + (exp["blob_bytes"] - exp["offsets"]["down_scales"]) \
            != exp["blob_bytes"]:
        errors.append("blob offsets exceed blob_bytes")
    for lay in exp["layers"]:
        if lay["offset"] + exp["blob_bytes"] > got_size:
            errors.append(f"layer {lay['layer']} blob runs past experts.bin")

    hashes = {name: _sha256(out / name) for name in
              ("experts.bin", "dense.bin", "embd.bin", "index.txt")}
    print(json.dumps({"out_dir": str(out), "ok": not errors, "errors": errors,
                      "pool_bytes": recomputed, "sha256": hashes}, indent=2))
    return 0 if not errors else 1


def cmd_emit_gguf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
                  dense_type: str = "F32") -> int:
    """safetensors float experts → engine GGUF v3 with Q2_0 tensors (p3-02).

    Expert tensors are named `blk.{L}.ffn_{gate,up,down}_exps.weight` with safetensors
    shape (out, NE, in) row-major; the GGUF fused shape is (in, out, NE) with blocks
    running along dim 0 (engine gguf_writer add_q2_0 contract). One tensor is alive at
    a time (CONSTRAINTS C1/C2). Guard keys come from the preset input — never invented
    (docs/rea/struct_layouts.md §1 architecture guard, VERIFIED p3-01).

    `dense_type` (p3-03): "F32" = pass-through (p3-02 behaviour); "Q4_0" quantizes each
    dense F32 tensor to Q4_0 blocks (18 B / 32 elements) with the SAME shape recorded —
    engine gguf_reader expected_bytes = elements × 18 / 32, so the element count must be
    a multiple of 32 (refused with an explicit error otherwise).
    """
    if GGUFWriter is None:
        print("emit-gguf requires the pinned engine tools checkout "
              f"(STRATASHIFT_ENGINE_TOOLS={ENGINE_TOOLS})", file=sys.stderr)
        return 2
    meta = parse_header(model)
    limit = int(max_rss_gb * GB) if max_rss_gb else None
    rss_samples: list[int] = []
    tracemalloc.start()  # p3-03 AC5: no-full-load gate measures the Python heap peak,
    # not VmRSS — VmRSS counts mmap page-cache pages of the streamed file, which are
    # reclaimable and inflate the reading to multi-GB on a 10 GB fixture (WSL measured).
    slice_heap_samples: list[int] = []  # heap at each slice sample: the streaming slice
    # peak is what AC5 gates. The engine GGUFWriter accumulates all tensor bytes in
    # w.tensors and write() copies the full body (tools/gguf_writer.py:151-180, pin
    # c566ff4, read-only), so the total heap peak includes that writer accumulation —
    # it is the engine's write design, not a full model load.
    w = GGUFWriter()
    w.add("general.architecture", guard_keys.get("architecture", "qwen4exp"))
    w.add("general.name", guard_keys.get("name", "stratashift emit"))
    for key, value in guard_keys.items():
        if key in ("architecture", "name"):
            continue
        w.add(key, value)
    with open(model, "rb") as src:
        mm = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for entry in meta.tensors:
                begin, end = entry.data_offsets
                if entry.name.startswith("blk.") and entry.name.endswith("_exps.weight"):
                    if entry.dtype != "F32":
                        raise ValueError(f"{entry.name}: expert tensor must be F32 float, "
                                         f"got {entry.dtype}")
                    out_dim, ne, in_dim = entry.shape
                    # AC6 no-full-load gate: slice ONE expert run (out_dim × in_dim floats),
                    # never the whole tensor (3.35 GB) — peak = header + single expert slice.
                    run_bytes = out_dim * in_dim * 4
                    q = bytearray()
                    for e in range(ne):
                        raw = mm[meta.data_start + begin + e * run_bytes:
                                 meta.data_start + begin + (e + 1) * run_bytes]
                        q += quantize_q2_0_stream(raw)
                        del raw
                        rss_samples.append(current_rss_bytes())
                        slice_heap_samples.append(tracemalloc.get_traced_memory()[1])
                        if limit is not None and rss_samples[-1] > limit:
                            return 3
                    w.tensors.append(_Tensor(entry.name, [in_dim, out_dim, ne], "Q2_0", bytes(q)))
                    del q
                elif entry.dtype == "U8" and entry.name == PLE_NAME:
                    # PLE table: raw IQ4_NL block bytes pass through verbatim. Engine
                    # ple_formats() has no F32 (ngram.hpp PleFormat enum); strata_pack
                    # build:301 requires per_layer_token_embd.weight in shard2, and
                    # ngram.hpp:32 requires GGUF shape[0] == PLE_HEAD_DIM (160).
                    # safetensors stores the 720 raw bytes as U8[720] (IQ4_NL: 32
                    # elements per 18 B block → GGUF shape [160, 8], p3-01 VERIFIED).
                    if list(entry.shape) != [PLE_BYTES]:
                        raise ValueError(f"{entry.name}: PLE payload must be U8[{PLE_BYTES}], "
                                         f"got U8{entry.shape}")
                    raw = mm[meta.data_start + begin: meta.data_start + end]
                    w.tensors.append(_Tensor(entry.name, [PLE_HEAD_DIM, PLE_ROWS],
                                             "IQ4_NL", raw))
                    del raw
                    rss_samples.append(current_rss_bytes())
                    slice_heap_samples.append(tracemalloc.get_traced_memory()[1])
                else:
                    if entry.dtype != "F32":
                        raise ValueError(f"{entry.name}: only F32 dense tensors supported "
                                         f"for emit, got {entry.dtype}")
                    raw = mm[meta.data_start + begin: meta.data_start + end]
                    if dense_type == "Q4_0":
                        n_el = 1
                        for d in entry.shape:
                            n_el *= d
                        if n_el % QK4_0:
                            raise ValueError(f"{entry.name}: Q4_0 emit needs a multiple of "
                                             f"{QK4_0} elements, shape {entry.shape} = {n_el}")
                        # Engine pack gate (p3-03): pack_index.py:144-152 requires the
                        # codes→scales planes to be CONTIGUOUS and strata_pack.py:43
                        # ALIGN=64 pads every plane start to 64 B. codes plane is
                        # n_el/2 bytes (4-bit, 2 codes/byte), so contiguity holds iff
                        # n_el/2 % 64 == 0, i.e. n_el % 128 == 0. Smaller tensors
                        # (e.g. attn_norm 64 elems) cannot be packed as Q4_0 by the
                        # engine itself — keep them F32 pass-through.
                        if n_el % Q4_0_MIN_PACK_ELEMENTS:
                            w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                        else:
                            q = quantize_q4_0_stream(raw)
                            w.tensors.append(_Tensor(entry.name, list(entry.shape), "Q4_0", q))
                            del q
                    else:
                        w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                    del raw
                    rss_samples.append(current_rss_bytes())
                    slice_heap_samples.append(tracemalloc.get_traced_memory()[1])
                if limit is not None and rss_samples[-1] > limit:
                    return 3
        finally:
            mm.close()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.write(out)
    # Engine convention: the data section must end flush at EOF. The engine gguf_writer pads
    # EVERY tensor to 64 B (tools/gguf_writer.py:177-179, pin c566ff4), so a last tensor that
    # is not a 64-multiple (e.g. PLE IQ4_NL 720 B) leaves trailing padding. strata_pack.py
    # verify computes data_section_offset = filelen - max(offset + expected_bytes)
    # (tools/canonical_xcheck.py:619-625), which then reads every tensor shifted by the
    # trailing pad -> all expert checks MISMATCH. p3-01 fixture GGUF was flush (last tensor
    # 262144 B = 64-multiple). Truncate to data_start + max(offset + expected_bytes).
    end_meta = parse_gguf_header(str(out))
    data_end = end_meta.data_start + max(t.offset + t.expected_bytes() for t in end_meta.tensors)
    if out.stat().st_size > data_end:
        import os
        os.truncate(str(out), data_end)
    heap_peak = tracemalloc.get_traced_memory()[1]
    slice_heap_peak = max(slice_heap_samples) if slice_heap_samples else 0
    tracemalloc.stop()
    print(json.dumps({
        "model": model,
        "gguf": str(out),
        "tensors": len(w.tensors),
        "guard_keys": len(guard_keys),
        "rss_samples_bytes": rss_samples,
        "rss_peak_bytes": max(rss_samples) if rss_samples else 0,
        "heap_peak_bytes": heap_peak,
        "slice_heap_peak_bytes": slice_heap_peak,
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
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
    p_conv.add_argument("--dense-type", default="F32", choices=("F32", "Q4_0"),
                        help="emit-gguf dense tensor type: F32 pass-through (default) "
                             "or Q4_0 quantized blocks (p3-03)")
    p_ver = sub.add_parser("verify", help="local self-check of a GGUF-path pack")
    p_ver.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        return cmd_inspect(args.model, args.json)
    if args.cmd == "convert":
        rc = cmd_convert(args.model, args.out, args.max_rss_gb)
        if rc == 0 and args.emit_gguf:
            guard = json.loads(pathlib.Path(args.guard_keys).read_text()) \
                if args.guard_keys else {}
            rc = cmd_emit_gguf(args.model, args.emit_gguf, guard, args.max_rss_gb,
                               args.dense_type)
        return rc
    return cmd_verify(args.out)


if __name__ == "__main__":
    sys.exit(main())
