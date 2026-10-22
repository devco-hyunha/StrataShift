"""`convert --out`: stream-pack Q2_0 slices into an experts.bin pack (moved out of
cli/main.py, p6-08).

Two input forms: safetensors (p2 pack, manifest.json `packed_rows` form) and GGUF v3
(engine pack layout experts.bin + dense.bin + embd.bin + engine-format manifest.json +
index.txt, so the pinned engine's own verifier can accept it). One slice alive at a time
(CONSTRAINTS C1/C2).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from stratashift.ingest.gguf import parse_gguf_header
from stratashift.ingest.safetensors_header import DTYPE_SIZES, numel, parse_header
from stratashift.meta.engine_manifest import blob_offsets, build_manifest, tensor_entry
from stratashift.meta.manifest import (build_index_row, sha256_file, write_index_txt,
                                       write_manifest)
from stratashift.monitor.heap import heap_gate, heap_peak_bytes
from stratashift.monitor.sampler import Sampler
from stratashift.polystrata.headers import EMBD_NAME, GGUF_ALIGN_DEFAULT, PLE_NAME, Q2_0_BLOCK_BYTES
from stratashift.polystrata.index import write_engine_index
from stratashift.quant.pack import expert_blob
from stratashift.stream.reader import SliceReader

PACK_FILES = ("experts.bin", "dense.bin", "embd.bin")


def convert_safetensors(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    with heap_gate():
        return _pack_safetensors(model, out_dir, max_rss_gb)


def _pack_safetensors(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    meta = parse_header(model)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    dst = 0
    aborted = False
    sampler = Sampler(max_rss_gb)  # heap peak below, not VmRSS (p3-03 AC5: page cache inflates VmRSS)

    # engine index is tensor-keyed (WeightTable::find by name) → one row per tensor;
    # the expert map drives slicing/selection, not index rows.
    with open(out / "experts.bin", "wb") as fh:
        pass  # truncate
    # p4-02 root cause (full-suite OOM RC=137, observed 2026-10-17): iter_tensor_payloads
    # materialized the FULL tensor payload in heap (3.35 GB F32 expert tensor), and the
    # `continue` path kept the previous blob alive until the next rebinding — transient
    # 2 blobs ≈ 6.7 GB > available RAM. Gate on entry metadata (shape × dtype) BEFORE
    # reading; read only Q2_0-shaped slices via seek+read (p4-01 streaming hygiene).
    with SliceReader() as reader:
        for entry in meta.tensors:
            if entry.name == PLE_NAME:
                # PLE table is IQ4_NL bytes, not Q2_0 (ngram.hpp PleFormat enum) — never pack it.
                sampler.record()
                continue
            nbytes = numel(entry.shape) * DTYPE_SIZES[entry.dtype]
            if nbytes % Q2_0_BLOCK_BYTES:
                # not a Q2_0-shaped tensor (e.g. F32) — skip, record in manifest
                sampler.record()
                continue
            blob = reader.read(model, meta.data_start + entry.data_offsets[0], nbytes)
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
            sampler.record()
            del blob
            if sampler.over_limit():
                aborted = True
                break

    pool = write_index_txt(out / "index.txt", GGUF_ALIGN_DEFAULT, rows)
    manifest = {
        "model": model,
        "out_dir": str(out),
        "packed_rows": len(rows),
        "skipped_slices": len(sampler.rss) - len(rows),
        "pool_bytes": pool,
        "experts_bin_sha256": sha256_file(out / "experts.bin"),
        "rss_samples_bytes": sampler.rss,
        "rss_peak_bytes": sampler.rss_peak(),
        "heap_peak_bytes": heap_peak_bytes(),
        "aborted_max_rss": aborted,
    }
    write_manifest(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 3 if aborted else 0


# --- GGUF input -------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertLayout:
    """Fused expert tensors of a GGUF: geometry + per-layer role offsets."""
    layers: dict[int, dict[str, int]]
    layer_ids: list[int]
    H: int
    FF: int
    NE: int
    role_bytes: int  # contiguous run per expert per role
    blob_bytes: int


def read_expert_layout(meta, model: str) -> ExpertLayout:
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
    role_bytes = H * FF * Q2_0_BLOCK_BYTES // 64
    return ExpertLayout(layers, layer_ids, H, FF, NE, role_bytes, 3 * role_bytes)


def convert_gguf(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    with heap_gate():
        return _pack_gguf(model, out_dir, max_rss_gb)


def _write_expert_blobs(reader: SliceReader, model: str, meta, lay: ExpertLayout,
                        out: pathlib.Path, sampler: Sampler) -> list[dict] | None:
    """experts.bin: one blob per expert, streamed (C1/C2: one slice alive at a time).
    Returns the per-layer manifest entries, or None when the RSS ceiling aborted."""
    by_name = {t.name: t for t in meta.tensors}
    expert_layers: list[dict] = []
    for layer in lay.layer_ids:
        layer_off = (out / "experts.bin").stat().st_size
        for e in range(lay.NE):
            slices = []
            for role in ("ffn_gate", "ffn_up", "ffn_down"):
                off = meta.data_start + lay.layers[layer][role] + e * lay.role_bytes
                slices.append(reader.read(model, off, lay.role_bytes))
            blob = expert_blob(slices[0], slices[1], slices[2], lay.H, lay.FF)
            del slices
            with open(out / "experts.bin", "ab") as fh:
                fh.write(blob)
            del blob
            sampler.record()
            if sampler.over_limit():
                return None
        role_offsets = {role: by_name[f"blk.{layer}.{role}_exps.weight"].offset
                        for role in ("ffn_gate", "ffn_up", "ffn_down")}
        expert_layers.append({"layer": layer, "offset": layer_off,
                              "role_offsets": role_offsets})
    return expert_layers


def _write_dense_groups(reader: SliceReader, model: str, meta, out: pathlib.Path,
                        sampler: Sampler) -> dict | None:
    """dense.bin / embd.bin: stream one tensor slice at a time, pack planes at ALIGN=64.
    Engine insertion order is dense group first, then embd (tools/strata_pack.py:268-270
    `for fname, group in (("dense.bin", dense), ("embd.bin", embd))`), and pack_index.py
    emits manifest insertion order — so index.txt rows must follow the same grouping.
    Returns the tensor entries, or None when the RSS ceiling aborted."""
    tensor_entries: dict = {}
    seg = {"dense.bin": 0, "embd.bin": 0}
    groups = [("dense.bin", [t for t in meta.tensors
                             if not (t.name.startswith("blk.")
                                     and t.name.endswith("_exps.weight"))
                             and t.name != EMBD_NAME]),
              ("embd.bin", [t for t in meta.tensors if t.name == EMBD_NAME])]
    for fname, group in groups:
        for t in group:
            raw = reader.read(model, meta.data_start + t.offset, t.expected_bytes())
            body, ent = tensor_entry(t.name, t, raw, seg[fname], fname)
            with open(out / fname, "ab") as fh:
                fh.write(body)
            seg[fname] += len(body)
            tensor_entries[t.name] = ent
            sampler.record()
            del raw, body
            if sampler.over_limit():
                return None
    return tensor_entries


def _pack_gguf(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    meta = parse_gguf_header(model)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sampler = Sampler(max_rss_gb)
    lay = read_expert_layout(meta, model)
    for name in PACK_FILES:
        (out / name).write_bytes(b"")  # truncate

    with SliceReader() as reader:
        expert_layers = _write_expert_blobs(reader, model, meta, lay, out, sampler)
        if expert_layers is None:
            return 3
        tensor_entries = _write_dense_groups(reader, model, meta, out, sampler)
        if tensor_entries is None:
            return 3

    experts = {
        "blob_bytes": lay.blob_bytes,
        "offsets": blob_offsets(lay.H, lay.FF),
        "layers": expert_layers,
        "total_bytes": len(lay.layer_ids) * lay.NE * lay.blob_bytes,
        "scales_fp16": True, "source_type": "Q2_0", "form": "S2",
    }
    files = {name: (out / name).stat().st_size for name in PACK_FILES}
    shard2 = str(model).replace("00001-of-00002", "00002-of-00002")
    man = build_manifest(model, len(lay.layer_ids), lay.NE, tensor_entries, experts, files, shard2)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    pool = write_engine_index(out / "index.txt", man)

    manifest = {
        "model": model,
        "out_dir": str(out),
        "layers": lay.layer_ids,
        "geometry": {"H": lay.H, "FF": lay.FF, "NE": lay.NE},
        "pool_bytes": pool,
        "experts_bin_sha256": sha256_file(out / "experts.bin"),
        "dense_bin_sha256": sha256_file(out / "dense.bin"),
        "embd_bin_sha256": sha256_file(out / "embd.bin"),
        "index_txt_sha256": sha256_file(out / "index.txt"),
        "rss_samples_bytes": sampler.rss,
        "rss_peak_bytes": sampler.rss_peak(),
        "heap_peak_bytes": heap_peak_bytes(),
    }
    write_manifest(out / "conversion.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0
