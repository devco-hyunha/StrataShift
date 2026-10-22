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
import io
import json
import mmap
import os
import pathlib
import struct
import sys
import tracemalloc
from contextlib import contextmanager

import numpy as np

# pinned engine checkout (p0-01, .local/polystrata) — read-only tools, engine never edited.
ENGINE_TOOLS = os.environ.get("STRATASHIFT_ENGINE_TOOLS", "/opt/data/polystrata-src/tools")
if ENGINE_TOOLS not in sys.path:
    sys.path.append(ENGINE_TOOLS)

from stratashift.ingest.gguf import GGUF_MAGIC, parse_gguf_header
from stratashift.ingest.safetensors_header import DTYPE_SIZES, parse_header
from stratashift.ingest.shard_index import abs_offset, load_sharded_model
from stratashift.meta.engine_manifest import blob_offsets, build_manifest, tensor_entry
from stratashift.meta.expert_map import build_expert_map
from stratashift.meta.hf_gguf_map import (GLOBAL_DENSE, GEMMA4_GLOBAL_DENSE, GEMMA4_LAYER_DENSE,
                                           LAYER_DENSE, build_tokenizer_meta, derive_geometry,
                                           gdn_qkv_part, hf_layer_id, is_hf_naming,
                                           load_hf_tokenizer, map_dense)
from stratashift.meta.manifest import build_index_row, write_index_txt, write_manifest
from stratashift.monitor.rss import current_rss_bytes
from stratashift.polystrata.headers import (GGUF_ALIGN_DEFAULT, Q2_0_BLOCK_BYTES,
                                            Q3_K_MIN_PACK_ELEMENTS,
                                            Q4_0_MIN_PACK_ELEMENTS, Q4_K_MIN_PACK_ELEMENTS,
                                            Q5_K_MIN_PACK_ELEMENTS, Q6_K_MIN_PACK_ELEMENTS,
                                            Q8_0_MIN_PACK_ELEMENTS, QK4_0, ENGINE_LOAD_BE)
from stratashift.polystrata.index import engine_rows, write_engine_index
from stratashift.quant.pack import (expert_blob, quantize_q2_0_stream, quantize_q4_0_stream,
                                    quantize_q4_k_stream, quantize_q5_k_stream,
                                    quantize_q6_k_stream, quantize_q3_k_stream,
                                    quantize_q8_0_stream)
from stratashift.quant.pack_imatrix import (broadcast_imatrix, plane_tile,
                                            quantize_q4_0_imatrix_stream,
                                            quantize_q4_k_imatrix_stream)
from stratashift.meta.imatrix import load_imatrix_stats

# engine writer is imported lazily inside cmd_emit_gguf (engine tools are read-only,
# p3-01 harness convention); TYPE_IDS lacks IQ4_NL (id 20, gguf_reader GGML_TYPES) —
# harness-side extension only, engine files untouched.
try:
    from gguf_writer import GGUFWriter, _Tensor  # noqa: E402
    import gguf_writer  # noqa: E402
    gguf_writer.TYPE_IDS["IQ4_NL"] = 20
    gguf_writer.TYPE_IDS["Q4_0"] = 2  # GGML_TYPES id, engine gguf_reader.py:21 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q4_K"] = 12  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q5_K"] = 13  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q6_K"] = 14  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q3_K"] = 11  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q8_0"] = 8   # GGML_TYPES id, engine gguf_reader.py GGML_TYPES
                                       # (our ingest/gguf.py GGML_TYPES:24 agrees)
    gguf_writer.TYPE_IDS["BF16"] = 30  # GGML_TYPES id, engine gguf_reader.py GGML_TYPES
                                       # (pin c566ff4; ffn_gate_inp_shexp Bf16InF32, p4-01)
except ImportError:  # engine checkout absent: emit path unavailable, convert/verify unaffected
    GGUFWriter = _Tensor = gguf_writer = None


def _slice_read(mm, src, lo: int, n: int) -> bytes:
    """Read one slice via seek+read, NOT mmap (p4-01 streaming hygiene).

    The streamed fixture is read once, sequentially, so its pages are never reused.
    mmap keeps every touched page resident: on a 10 GB fixture the page cache grows
    VmRSS toward the fixture size and the process is OOM-killed (RC=137, observed
    2026-10-17) even with fadvise DONTNEED per slice, on both 9p and ext4. A plain
    seek+read leaves no page cache behind, so VmRSS stays at the heap peak.
    """
    src.seek(lo)
    return src.read(n)

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
        return cmd_convert_gguf(model, out_dir, max_rss_gb)
    return cmd_convert_safetensors(model, out_dir, max_rss_gb)


@contextmanager
def heap_gate():
    """p5-02 AC4 root cause fix (2026-10-18): tracemalloc start/stop must be PAIRED on
    every exit path. Observed in the full suite: the BF16 emit `ValueError` raised after
    `tracemalloc.start()` bypassed the trailing `stop()`, leaving the tracer active for
    the rest of the session — `test_no_full_load` header peak budget then measured the
    cumulative session peak (1,123,499 B vs 133,770 B standalone) and failed. Bodies
    below run inside `with heap_gate():` so exceptions, RSS aborts (return 3) and normal
    returns all stop the tracer.
    """
    tracemalloc.start()
    try:
        yield
    finally:
        tracemalloc.stop()


def cmd_convert_safetensors(model: str, out_dir: str, max_rss_gb: float | None) -> int:
    with heap_gate():
        return _cmd_convert_safetensors(model, out_dir, max_rss_gb)


def _cmd_convert_safetensors(model: str, out_dir: str, max_rss_gb: float | None) -> int:
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
    # p4-02 root cause (full-suite OOM RC=137, observed 2026-10-17): iter_tensor_payloads
    # materialized the FULL tensor payload in heap (3.35 GB F32 expert tensor), and the
    # `continue` path kept the previous blob alive until the next rebinding — transient
    # 2 blobs ≈ 6.7 GB > available RAM. Gate on entry metadata (shape × dtype) BEFORE
    # reading; read only Q2_0-shaped slices via seek+read (p4-01 streaming hygiene).
    with open(model, "rb") as src:
        for entry in meta.tensors:
            if entry.name == PLE_NAME:
                # PLE table is IQ4_NL bytes, not Q2_0 (ngram.hpp PleFormat enum) — never pack it.
                rss_samples.append(current_rss_bytes())
                continue
            n_el = 1
            for d in entry.shape:
                n_el *= d
            nbytes = n_el * DTYPE_SIZES[entry.dtype]
            if nbytes % Q2_0_BLOCK_BYTES:
                # not a Q2_0-shaped tensor (e.g. F32) — skip, record in manifest
                rss_samples.append(current_rss_bytes())
                continue
            src.seek(meta.data_start + entry.data_offsets[0])
            blob = src.read(nbytes)
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
    with heap_gate():
        return _cmd_convert_gguf(model, out_dir, max_rss_gb)


def _cmd_convert_gguf(model: str, out_dir: str, max_rss_gb: float | None) -> int:
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
                        slices.append(_slice_read(mm, src, off, role_bytes))
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
                    raw = _slice_read(mm, src, meta.data_start + t.offset,
                                      t.expected_bytes())
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


# --- HF emit (p4-01, Phase 4 입력 일반화) -------------------------------------------------

# HF emit order: per-layer engine want2 order (layout.cpp:60-110, pin c566ff4), then the
# fused experts, then want1 order, then globals. Deterministic; the engine pack builder
# follows GGUF tensor order, so this fixes the golden/ours plane order.
# (HF suffix after `model.layers.<L>.` → engine suffix; LAYER_DENSE is the mapping table.)
_HF_EMIT_2D = [
    ("hc_attn_down.weight", "hc_attn_down.weight"),
    ("hc_attn_up.weight", "hc_attn_up.weight"),
    ("hc_attn_inject.weight", "hc_attn_inject.weight"),
    ("hc_ffn_down.weight", "hc_ffn_down.weight"),
    ("hc_ffn_up.weight", "hc_ffn_up.weight"),
    ("hc_ffn_inject.weight", "hc_ffn_inject.weight"),
    ("mlp.gate.weight", "ffn_gate_inp.weight"),
    ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
    ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
    ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
    ("linear_attn.g_proj.weight", "attn_gate.weight"),
    ("linear_attn.o_proj.weight", "ssm_out.weight"),
    ("linear_attn.conv1d.weight", "ssm_conv1d.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("indexer.q_proj.weight", "indexer.q_proj.weight"),
    ("indexer.k_proj.weight", "indexer.k_proj.weight"),
]
# attn_q is FUSED: q_proj rows then q_gate_proj rows (layer.hpp:363 `q_full` =
# n_head·2·head_dim, "the raw attn_q output, q | gate per head", pin c566ff4).
_QSA_Q_ORDER = ["self_attn.q_proj.weight", "self_attn.q_gate_proj.weight"]
_HF_EMIT_1D = [
    ("hc_attn_norm.weight", "hc_attn_norm.weight"),
    ("hc_ffn_norm.weight", "hc_ffn_norm.weight"),
    ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
    ("linear_attn.A_log", "ssm_a"),
    ("linear_attn.dt_bias", "ssm_dt.bias"),
    ("linear_attn.norm.weight", "ssm_norm.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("indexer.q_norm.weight", "indexer.q_norm.weight"),
    ("indexer.k_norm.weight", "indexer.k_norm.weight"),
]
# GDN fused qkv (want2 attn_qkv = concat(q, k, v) rows, geometry.hpp:35 derived width)
_GDN_QKV_ORDER = ["linear_attn.q_proj.weight", "linear_attn.k_proj.weight",
                  "linear_attn.v_proj.weight"]

# gemma4 emit order (p5-06 AC3): engine want2 matrix keys of gemma4.cpp:197-211,225
# (load_moe: ffn_gate_inp [n_embd, n_expert]), then the fused experts, then the engine
# 1-D f32 keys (attn_norm / post_attention_norm / ffn_norm / post_ffw_norm{,_1,_2} /
# pre_ffw_norm_2 / router scales / layer_scalar, gemma4.cpp:197-203,219-233). Suffixes
# are the fixture-measured HF names (tests/fixtures/gemma4_lang_header.json); rules
# come from GEMMA4_LAYER_DENSE. Full layers have no v_proj (attention_k_eq_v measured)
# — the builder's optional-tensor skip rule covers it, no invented tensor.
_GEMMA4_2D = [
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    ("router.proj.weight", "ffn_gate_inp.weight"),
]
_GEMMA4_1D = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "post_attention_norm.weight"),
    ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
    ("post_feedforward_layernorm_1.weight", "post_ffw_norm_1.weight"),
    ("pre_feedforward_layernorm_2.weight", "pre_ffw_norm_2.weight"),
    ("post_feedforward_layernorm_2.weight", "post_ffw_norm_2.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("router.scale", "ffn_gate_inp.scale"),
    ("router.per_expert_scale", "ffn_down_exps.scale"),
    ("layer_scalar", "layer_output_scale.weight"),
]
# GDN 1-D keys: HF A_log/dt_bias are 1-D [v_heads]; engine wants the 2-D (n_embd, v_heads)
# ssm_alpha/ssm_beta (want2) IN ADDITION to the 1-D ssm_a/ssm_dt.bias (want1, emitted
# verbatim above). The fixture carries the 1-D HF tensors; the emit derives the 2-D
# engine forms by broadcasting each value across n_embd rows (fixture-measured, p4-01).
_GDN_1D = [("linear_attn.A_log", "ssm_alpha.weight", "broadcast"),
           ("linear_attn.dt_bias", "ssm_beta.weight", "broadcast")]


def _gemma4_rope_freqs(guard_keys: dict, cfg_json: dict | None) -> np.ndarray:
    """Derived global tensor `rope_freqs.weight` (p5-08 rope fix).

    Engine gemma4.cpp:207 (pin c566ff4) loads the FULL layers' rotary frequency table
    from the FILE: `upload(gg_->find("rope_freqs.weight"), "rope_freqs.weight",
    B.n_rot / 2, 1, kF32)` — measured TK1 gemma4 RUN FAIL rc=1 "the GGUF has no tensor
    rope_freqs.weight" (docs/work/p5-08 §AC4, 2026-10-19). Engine `upload`
    (engine.cpp:884-890, pin c566ff4) gate: shape[0] == n_rot/2, rest == 1 → the
    tensor is F32 [n_rot/2, 1].

    VALUES are the engine's own float64 reference formula, `theta ** (-2i / n_rot)`
    (`src/kernels/rope_parity.cpp` header, pin c566ff4 — "exactly as ref/qsa.py::
    rope_freqs"; consistent with gemma4.cpp:194 `pow(base, -2/n_rot)`), computed in
    float64 and cast to float32. base = guard `gemma4.rope.freq_base`, n_rot = guard
    `gemma4.rope.dimension_count` (engine default rot_full = dim_full, gemma4.cpp:53).

    NEVER-INVENTED gate (p4-01 rule): when the checkpoint's config.json carries
    rope_parameters, they must MEASURE-equal the preset — full_attention.rope_theta
    == freq_base, partial_rotary_factor · global_head_dim == dimension_count,
    sliding_attention.rope_theta == freq_base_swa. Engine gate gemma4.cpp:77:
    rot even, rot ≤ key_length.
    """
    base = guard_keys.get("gemma4.rope.freq_base")
    if base is None:
        raise ValueError("gemma4 emit requires guard key gemma4.rope.freq_base "
                         "(engine gemma4.cpp:207 loads rope_freqs.weight from the file)")
    base = float(base)
    dim_full = guard_keys.get("gemma4.attention.key_length")
    rot = guard_keys.get("gemma4.rope.dimension_count")
    if rot is None:
        rot = dim_full  # engine default: rot_full = dim_full (gemma4.cpp:53, pin c566ff4)
    if rot is None:
        raise ValueError("gemma4 emit needs rope.dimension_count or attention.key_length")
    rot = int(rot)
    if rot % 2 == 1 or (dim_full is not None and rot > int(dim_full)):
        raise ValueError(f"engine gate gemma4.cpp:77 refuses rope.dimension_count={rot} "
                         f"(even, <= key_length={dim_full})")
    if cfg_json is not None:
        tc = cfg_json.get("text_config")
        tc = tc if isinstance(tc, dict) else cfg_json
        rp = tc.get("rope_parameters")
        if isinstance(rp, dict):
            full = rp.get("full_attention")
            if isinstance(full, dict):
                theta = float(full.get("rope_theta", base))
                if theta != base:
                    raise ValueError(f"config full rope_theta {theta} != guard freq_base {base} "
                                     f"(never invented)")
                prf = full.get("partial_rotary_factor")
                if prf is not None:
                    ghd = tc.get("global_head_dim")
                    if ghd is None:
                        ghd = cfg_json.get("head_dim")
                    if ghd is None:
                        ghd = dim_full
                    if ghd is not None and int(round(float(prf) * int(ghd))) != rot:
                        raise ValueError(f"config partial_rotary_factor {prf} · global_head_dim "
                                         f"{ghd} != guard dimension_count {rot}")
            swa = rp.get("sliding_attention")
            bswa = guard_keys.get("gemma4.rope.freq_base_swa")
            if isinstance(swa, dict) and bswa is not None:
                if float(swa.get("rope_theta", bswa)) != float(bswa):
                    raise ValueError(f"config sliding rope_theta {swa.get('rope_theta')} != "
                                     f"guard freq_base_swa {bswa}")
    i = np.arange(rot // 2, dtype=np.float64)
    freqs = np.power(np.float64(base), -2.0 * i / np.float64(rot))
    return freqs.astype("<f4").reshape(rot // 2, 1)


def _to_f32(raw: bytes, dtype: str) -> bytes:
    """Normalize a tensor payload to F32 bytes (BF16 = top 16 bits of F32, F16 → F32).

    Vectorized numpy (peak = tensor slice, C1/C2); bit-identical to the scalar
    engine_manifest.bf16_to_f32_bytes loop (tests/test_hf_gguf_map.py AC2).
    """
    if dtype == "F32":
        return raw
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view("<f4").tobytes()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype("<f4").tobytes()
    raise ValueError(f"unsupported HF dtype {dtype} for emit")


def _dense_gate(n_el: int, dense_type: str) -> bool:
    """True when the engine pack gate refuses this dense_type for n_el elements
    (tensor stays F32 pass-through) — the p3-03~p3-09 gates, verbatim."""
    mins = {"Q4_0": Q4_0_MIN_PACK_ELEMENTS, "Q4_K": Q4_K_MIN_PACK_ELEMENTS,
            "Q5_K": Q5_K_MIN_PACK_ELEMENTS, "Q6_K": Q6_K_MIN_PACK_ELEMENTS,
            "Q3_K": Q3_K_MIN_PACK_ELEMENTS, "Q8_0": Q8_0_MIN_PACK_ELEMENTS}
    if dense_type == "F32":
        return True
    if dense_type == "Q8_0":
        return n_el % Q8_0_MIN_PACK_ELEMENTS != 0  # ndim gate applied by caller
    return n_el % mins[dense_type] != 0


def _dense_select(n_el: int, dense_type: str, ne0: int, ndim: int,
                  force_f32: bool = False) -> str:
    """p5-08: which quantization type the engine ACCEPTS for this tensor, "F32" =
    pass-through. Engine LOAD gate (p5-07 AC5 root cause): tensor_payload_bytes
    requires shape[0] % be == 0, be from block_geometry (gguf_reader.hpp:505,
    pin c566ff4: Q4_K be=256, Q8_0/Q4_0 be=32, F32 be=1) — `ne0` is the engine
    shape's first dim (HF [out,in] → GGUF (in,out) transpose). Measured refusal:
    ffn_down [2112,2816] Q4_K, 2112 % 256 = 64 → load throw rc=1.
    Ladder (docs/work/p5-08.md §design): requested type if it passes BOTH the
    engine load gate and the pack contiguity gate (p3-03~p3-09 `_dense_gate`);
    else the be=32 quantizations (Q8_0 default, Q4_0); else F32 (be=1 always
    passes). Both gates must hold — load gate is shape[0], pack gate is total
    elements; the p4-03 byte-identical contract keeps both in force.
    """
    if force_f32:  # 1-D engine key want1 contract (layout.cpp:88-98) — never quantized
        return "F32"
    if dense_type == "F32":
        return "F32"

    def accepted(t: str) -> bool:
        if t == "Q8_0" and ndim < 2:  # derived scales need ndim ≥ 2 (pack_index.py:174-186)
            return False
        return ne0 % ENGINE_LOAD_BE[t] == 0 and not _dense_gate(n_el, t)

    if accepted(dense_type):
        return dense_type
    for t in ("Q8_0", "Q4_0"):  # be=32 ladder, measured full-sweep pass (probe_ne0.py)
        if accepted(t):
            return t
    return "F32"


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
        if self._since_fsync >= (1 << 28):
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
            infos += struct.pack("<I", gguf_writer.TYPE_IDS[type_name])
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
            out += struct.pack("<I", gguf_writer._META_IDS["array"])
            out += struct.pack("<I", gguf_writer._META_IDS[elem])
            out += struct.pack("<Q", len(value))
            for v in value:
                out += self._scalar(elem, v)
            return bytes(out)
        out += struct.pack("<I", gguf_writer._META_IDS[type_name])
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


def cmd_emit_gguf_hf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
                     dense_type: str = "F32", imatrix_file: str | None = None) -> int:
    with heap_gate():
        return _cmd_emit_gguf_hf(model, out_path, guard_keys, max_rss_gb, dense_type,
                                 imatrix_file)


def _cmd_emit_gguf_hf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
                     dense_type: str = "F32", imatrix_file: str | None = None) -> int:
    """HF safetensors MoE (HF naming, fused gate_up, BF16/F16/F32) → engine GGUF v3.

    Emit rules (docs/work/p4-01.md §Emit 규칙, layout.cpp:60-110 pin c566ff4):
    - fused gate_up (NE,2FF,H): expert e = contiguous row block; gate = rows
      [0,FF), up = rows [FF,2FF) → Q2_0 stream → `blk.L.ffn_{gate,up}_exps.weight`
      shape (H,FF,NE); down (NE,H,FF) → `blk.L.ffn_down_exps.weight` (FF,H,NE).
    - dense HF [out,in] row-major bytes == engine GGUF (in,out) bytes (transpose
      shape metadata only); 1-D keys keep shape; `ffn_gate_inp_shexp.weight` keeps
      BF16 bytes (Bf16InF32, layout.cpp:88-98).
    - guard keys are DERIVED from the HF header (derive_geometry) and cross-checked
      against the preset; a mismatch is refused (never invented).
    - one tensor slice alive at a time (CONSTRAINTS C1/C2).
    """
    if GGUFWriter is None:
        print("emit-gguf requires the pinned engine tools checkout "
              f"(STRATASHIFT_ENGINE_TOOLS={ENGINE_TOOLS})", file=sys.stderr)
        return 2
    # p4-02: directory = multi-shard HF input (index.json + shards, engine-mirrored
    # gates in load_sharded_model); single file keeps the p1-01 parse_header path.
    if pathlib.Path(model).is_dir():
        meta = load_sharded_model(model)
    else:
        meta = parse_header(model)
    if not is_hf_naming(meta):
        print(f"{model}: --hf-emit needs HF naming (model.layers.* / fused experts)",
              file=sys.stderr)
        return 2
    # p6-05 AC3: imatrix stats loaded ONCE (calibration product, C1-safe); None →
    # plain producers byte-unchanged (golden gate). Plane keys are engine keys
    # (measured 2026-10-21 on the full-sweep product: 295 engine-keyed bases).
    imatrix_stats = load_imatrix_stats(imatrix_file) if imatrix_file else None
    by_name = {t.name: t for t in meta.tensors}
    layers = sorted({l for l in (hf_layer_id(t.name) for t in meta.tensors) if l is not None})
    # p4-03: head_dim from the model directory's config.json when the header has no
    # q_norm (Qwen1.5-MoE): head_dim = hidden_size // num_attention_heads — MEASURED
    # from the checkpoint's own config, never invented (p4-01 HYPOTHESIS rule).
    head_dim = None
    cfg_json = None
    d = pathlib.Path(model)
    cfg = (d / "config.json") if d.is_dir() else (d.parent / "config.json")
    if cfg.is_file():
        c = json.loads(cfg.read_text())
        cfg_json = c
        hs, nh = c.get("hidden_size"), c.get("num_attention_heads")
        if isinstance(hs, int) and isinstance(nh, int) and nh and hs % nh == 0:
            head_dim = c.get("head_dim") if isinstance(c.get("head_dim"), int) else hs // nh
    derived = derive_geometry(meta, head_dim, guard_keys.get("architecture", "qwen4exp"))
    prefix = guard_keys.get("architecture", "qwen4exp")
    for key, val in derived.items():
        preset_key = f"{prefix}.{key}"
        if preset_key not in guard_keys:
            continue
        # the engine reads per-layer ARRAY values via of_layer (gemma4.cpp:20-24,64-68);
        # a derived list must EQUAL the preset list elementwise, ints equal (never invented).
        want = guard_keys[preset_key]
        if isinstance(val, list):
            if not isinstance(want, list) or val != want:
                print(f"guard mismatch: derived {key}={val} vs preset {preset_key}={want}",
                      file=sys.stderr)
                return 2
            continue
        want = int(want)
        # block_count is the MODEL geometry: a partial fixture ships fewer layers than the
        # compiled geometry (p3-01 ships blk.0 of block_count=48), so the derived layer
        # count must be ≤ preset; every other derived key must be EQUAL (never invented).
        ok = val <= want if key == "block_count" else val == want
        if not ok:
            print(f"guard mismatch: derived {key}={val} vs preset {preset_key}={want}",
                  file=sys.stderr)
            return 2
    limit = int(max_rss_gb * GB) if max_rss_gb else None
    rss_samples: list[int] = []
    slice_heap_samples: list[int] = []
    tracemalloc.start()  # p3-03 AC5 convention: heap peak gates the streaming slice
    w = StreamingGGUFWriter(tmp_path=str(out_path) + ".tmp")
    w.add("general.architecture", prefix)
    w.add("general.name", guard_keys.get("name", "stratashift hf emit"))
    for key, value in guard_keys.items():
        if key in ("architecture", "name"):
            continue
        w.add(key, value)
    # p5-09 AC2: tokenizer metadata (tokenizer.ggml.*) from the MEASURED HF tokenizer
    # files. Files absent → no tokenizer emitted (engine API gate: serve/models.py
    # reports the model unavailable without vocab.json; strata-poly --tokens runs
    # need no tokenizer). vocab_size guard key is measure-equal-checked (p5-08 rope
    # gate shape, never-invented); a special text outside the measured reference
    # type table refuses the emit. Keys + types: reference GGUF header measured
    # 2026-10-20 (unsloth Q8_0 head, engine gguf_reader parse) — see hf_gguf_map.
    hf_root = pathlib.Path(model)
    tok_dir = hf_root if hf_root.is_dir() else hf_root.parent
    if (tok_dir / "tokenizer.json").is_file():
        hf_tok = load_hf_tokenizer(tok_dir)
        tok_meta = build_tokenizer_meta(hf_tok,
                                        vocab_guard=guard_keys.get(f"{prefix}.vocab_size"))
        for key, (type_name, value) in tok_meta.items():
            w.add(key, value, type_name)
        tokenizer_meta_count = len(tok_meta)
    else:
        tokenizer_meta_count = 0

    def sample():
        rss_samples.append(current_rss_bytes())
        slice_heap_samples.append(tracemalloc.get_traced_memory()[1])
        if limit is not None and rss_samples[-1] > limit:
            return True
        return False

    def emit_dense(name: str, entry, raw: bytes, engine_name: str, rule: str,
                   force_f32: bool = False):
        """One dense tensor: dtype normalize + engine shape + dense_type gate.

        force_f32: the engine want1 contract — every 1-D engine key is read as an
        F32 tensor of its exact length (p4-01 §dtype rule, layout.cpp:88-98; gemma4
        `f32(...)` calls, gemma4.cpp:197-211). A 1-D key is NEVER quantized: tiny
        1-D tensors pass the Q4_K gate (512 elements) and quantizing them produces
        manifest entries with empty scale planes (observed 2026-10-18: engine
        `blk.5.attn_q_norm.weight` Q4_K → "planes not contiguous" at verify).
        """
        n_el = 1
        for d in entry.shape:
            n_el *= d
        if rule == "bf16_keep":
            w.add_tensor(engine_name, list(entry.shape), "BF16", raw)
            return
        f32 = _to_f32(raw, entry.dtype)
        shape = list(reversed(entry.shape)) if rule == "transpose" else list(entry.shape)
        qtype = _dense_select(n_el, dense_type, shape[0], len(shape), force_f32)
        if qtype == "F32":
            w.add_f32(engine_name, np.frombuffer(f32, dtype="<f4").reshape(shape))
        else:
            # p6-05: engine-key imatrix plane consume (plane row = engine shape[0],
            # the HF input dim; element r*row_len + c uses plane[c]). Plain RTN is
            # byte-unchanged when imatrix_stats is None (golden gate, p6-03 AC3).
            vec = imatrix_stats.get(engine_name) if imatrix_stats else None
            im_quantizer = {"Q4_0": quantize_q4_0_imatrix_stream,
                            "Q4_K": quantize_q4_k_imatrix_stream}.get(qtype)
            if vec is not None and im_quantizer is not None:
                q = im_quantizer(f32, plane_tile(vec.reshape(-1), n_el, shape[0], 0))
            else:
                q = {"Q4_0": quantize_q4_0_stream, "Q4_K": quantize_q4_k_stream,
                     "Q5_K": quantize_q5_k_stream, "Q6_K": quantize_q6_k_stream,
                     "Q3_K": quantize_q3_k_stream, "Q8_0": quantize_q8_0_stream}[qtype](f32)
            w.add_tensor(engine_name, shape, qtype, q)
            del q
        del f32

    def emit_dense_stream(entry, engine_name: str, rule: str, force_f32: bool = False) -> None:
        """Stream a large dense tensor chunk by chunk (p4-03): byte-identical to
        emit_dense — quantizer blocks are block-independent, and the F32/BF16
        pass-through streams raw bytes verbatim. `force_f32` is the want1 1-D engine
        key contract (see emit_dense)."""
        n_el = 1
        for d in entry.shape:
            n_el *= d
        shape = list(reversed(entry.shape)) if rule == "transpose" else list(entry.shape)
        if rule == "bf16_keep":
            w.add_tensor_stream(engine_name, list(entry.shape), "BF16", payload_chunks(entry))
            return
        qtype = _dense_select(n_el, dense_type, shape[0], len(shape), force_f32)
        if qtype == "F32":
            # F32 passthrough: BF16 chunks MUST be widened to F32 bytes per chunk
            # (full path does f32 = _to_f32(raw) before add_f32; streaming raw BF16
            # under an F32 type writes half the bytes — caught 2026-10-17 by the
            # GGUF size delta 10.53 GB → 9.29 GB == embed F32 size).
            def f32_chunks():
                for chunk in payload_chunks(entry):
                    yield _to_f32(chunk, entry.dtype) if entry.dtype != "F32" else chunk
            w.add_tensor_stream(engine_name, shape, "F32", f32_chunks())
            return
        # p6-05: imatrix plane consume (engine-key lookup, plane row = engine shape[0] —
        # plane contract imatrix.cpp pin :971 in_sum2 [nval/nmat, nmat], nmat=1 dense).
        # row_len = engine shape[0] == HF entry.shape[-1] (the input dim the plane is
        # over); the stream element r*row_len + c uses plane[c] (broadcast contract).
        vec = imatrix_stats.get(engine_name) if imatrix_stats else None
        row_len = shape[0]
        quantizer = {"Q4_0": quantize_q4_0_stream, "Q4_K": quantize_q4_k_stream,
                     "Q5_K": quantize_q5_k_stream, "Q6_K": quantize_q6_k_stream,
                     "Q3_K": quantize_q3_k_stream, "Q8_0": quantize_q8_0_stream}[qtype]
        im_quantizer = {"Q4_0": quantize_q4_0_imatrix_stream,
                        "Q4_K": quantize_q4_k_imatrix_stream}.get(qtype)

        def qchunks():
            off = 0
            for chunk in payload_chunks(entry):
                f32 = _to_f32(chunk, entry.dtype) if entry.dtype != "F32" else chunk
                if vec is not None and im_quantizer is not None:
                    n = len(f32) // 4
                    yield im_quantizer(f32, plane_tile(vec.reshape(-1), n, row_len, off))
                    off += n
                else:
                    yield quantizer(f32)

        w.add_tensor_stream(engine_name, shape, qtype, qchunks())

    def emit_dense_gate(entry, engine_name: str, rule: str, force_f32: bool = False) -> None:
        """p4-03: tensors above 64 MB stream, never materialize the full payload."""
        if entry.nbytes() > (1 << 26):
            emit_dense_stream(entry, engine_name, rule, force_f32)
        else:
            raw = payload(entry)
            emit_dense(entry.name, entry, raw, engine_name, rule, force_f32)
            del raw

    # p4-02: shard-aware reader. Files are opened on demand and cached (handles, not
    # pages); reads are seek+read, NOT mmap — mmap keeps every touched page resident,
    # so on a 10.6 GB fixture the page cache grows VmRSS toward the fixture size and
    # the process is OOM-killed (RC=137, observed 2026-10-17). A plain seek+read
    # leaves no page cache behind (p4-01 streaming hygiene, VERIFIED).
    handles: dict[str, "io.BufferedReader"] = {}

    def reader_read(shard: str, lo: int, n: int) -> bytes:
        fh = handles.get(shard)
        if fh is None:
            fh = open(shard, "rb")
            handles[shard] = fh
        fh.seek(lo)
        return fh.read(n)

    def shard_of(entry) -> str:
        # weight_map names shards as basenames — resolve against the model directory.
        return str(pathlib.Path(model) / entry.shard) if entry.shard else model

    def payload(entry) -> bytes:
        lo = abs_offset(meta, entry)
        return reader_read(shard_of(entry), lo, entry.nbytes())

    def payload_chunks(entry, chunk_elems: int = 1 << 19):
        """Stream one tensor's payload in 32-element-aligned chunks. Quantizer blocks
        are 16/32 elements (Q3_K/Q6_K: 16, Q4_0/Q4_K/Q8_0: 32), so concatenating
        chunk-boundary outputs is byte-identical to quantizing the full tensor.
        p4-03 root cause: full payload of embed/lm_head (622 MB BF16 → 1.24 GB F32)
        pushed the heap peak to 2.9 GiB > 2 GiB gate (observed 2026-10-17, second
        emit OOM RC=137)."""
        esz = DTYPE_SIZES[entry.dtype]
        step = chunk_elems * esz
        lo = abs_offset(meta, entry)
        shard = shard_of(entry)
        remaining = entry.nbytes()
        while remaining > 0:
            n = min(step, remaining)
            yield reader_read(shard, lo, n)
            lo += n
            remaining -= n

    # p5-06 AC3: gemma4 layers live under `model.language_model.layers.<L>.` with the
    # engine Gemma4 key tables; every other arch keeps the p4-01 prefix/tables.
    gemma4 = prefix == "gemma4"
    layer_prefix = "model.language_model.layers." if gemma4 else "model.layers."
    emit_2d = _GEMMA4_2D if gemma4 else _HF_EMIT_2D
    emit_1d = _GEMMA4_1D if gemma4 else _HF_EMIT_1D
    layer_dense = GEMMA4_LAYER_DENSE if gemma4 else LAYER_DENSE
    try:
        for L in layers:
            p = f"{layer_prefix}{L}."
            # 2-D keys in engine want2 order
            for hf_suffix, engine_suffix in emit_2d:
                entry = by_name.get(p + hf_suffix)
                if entry is None:
                    continue
                rule = layer_dense[hf_suffix][1]
                emit_dense_gate(entry, f"blk.{L}.{engine_suffix}", rule)
                if sample():
                    return 3
            # GDN fused qkv: engine attn_qkv = concat(q, k, v) rows (bytes verbatim)
            parts = [by_name.get(p + s) for s in _GDN_QKV_ORDER]
            if all(pt is not None for pt in parts):
                if all(pt.dtype == parts[0].dtype for pt in parts):
                    dt = parts[0].dtype

                    def qkv_chunks():
                        for pt in parts:
                            yield _to_f32(payload(pt), dt) if dt != "F32" else payload(pt)

                    ne0 = parts[0].shape[1]
                    ne1 = sum(pt.shape[0] for pt in parts)
                    w.add_tensor_stream(f"blk.{L}.attn_qkv.weight", [ne0, ne1],
                                        "F32" if dt == "F32" else "BF16", qkv_chunks())
                    if sample():
                        return 3
            # fused attn_q: q_proj rows then q_gate_proj rows (engine attn_q
            # ne1 = 2·n_head·head_dim, layout.cpp:83 + layer.hpp:363, pin c566ff4)
            q_parts = [by_name.get(p + s) for s in _QSA_Q_ORDER]
            if all(pt is not None for pt in q_parts):
                dt = q_parts[0].dtype

                def attnq_chunks():
                    for pt in q_parts:
                        yield _to_f32(payload(pt), dt) if dt != "F32" else payload(pt)

                w.add_tensor_stream(f"blk.{L}.attn_q.weight",
                                    [q_parts[0].shape[1],
                                     sum(pt.shape[0] for pt in q_parts)],
                                    "F32" if dt == "F32" else "BF16", attnq_chunks())
                if sample():
                    return 3
            # fused experts: one expert run alive at a time (C1/C2); BF16/F16 runs
            # normalize to F32 before quantization (p4-01 §5 — quantizer contract stays F32)
            # p6-05 (user ruling 2026-10-21: Q2_0는 하지마): fused expert slots consume the
            # calibrated imatrix planes. Plane unit contract (imatrix.cpp pin :744, :971 —
            # in_sum2 [nval/nmat, nmat], unit = matrix = expert): row e of
            # `blk.{L}.ffn_gate_up_exps.weight` tiles across the gate/up row at row_len = h
            # (engine ffn_gate_exps shape [h, ff, ne] — h % 256 == 0 Q4_K pass), and
            # `blk.{L}.ffn_down_exps.weight` row e tiles at row_len = ff (engine shape
            # [ff, h, ne] — ff % 256 = 192 남 → engine refuses Q4_K, be=32 Q4_0 pass,
            # docs/work/p5-08.md §design ladder). Plane absent → plain producer fallback.
            gu = by_name.get(p + "mlp.experts.gate_up_proj") \
                or by_name.get(p + "experts.gate_up_proj")
            if gu is not None:
                ne, two_ff, h = gu.shape
                ff = two_ff // 2
                row_elems = ff * h
                esz = DTYPE_SIZES[gu.dtype]
                base = abs_offset(meta, gu)
                gu_plane = imatrix_stats.get(f"blk.{L}.ffn_gate_up_exps.weight") \
                    if imatrix_stats else None
                if gu_plane is not None and gu_plane.shape != (ne, h):
                    print(f"blk.{L}.ffn_gate_up_exps plane {gu_plane.shape} != "
                          f"(ne={ne}, h={h}) — refusing", file=sys.stderr)
                    return 3

                # engine load gate ladder on shape[0] = h (gguf_reader.hpp:505 pin
                # c566ff4): Q4_K be=256 when h % 256 == 0 (product h=2816 = 256×11);
                # fixture h=64 → be=32 Q4_0. Type and quantizer chosen together so
                # the header never says Q4_0 while bytes are Q4_K (both 144 B for
                # 256 elems — size check alone cannot catch the mismatch).
                gu_type = ("Q4_K" if h % 256 == 0 else "Q4_0") \
                    if gu_plane is not None else "Q2_0"
                gu_im_q = {"Q4_K": quantize_q4_k_imatrix_stream,
                           "Q4_0": quantize_q4_0_imatrix_stream}.get(gu_type)

                def gate_chunks():  # engine ffn_gate_exps = gate rows per expert
                    for e in range(ne):
                        lo = base + e * two_ff * h * esz
                        f32 = _to_f32(reader_read(shard_of(gu), lo, row_elems * esz),
                                      gu.dtype)
                        if gu_im_q is not None:
                            yield gu_im_q(f32, plane_tile(gu_plane[e], row_elems, h, 0))
                        else:
                            yield quantize_q2_0_stream(f32)

                def up_chunks():  # engine ffn_up_exps = up rows per expert
                    for e in range(ne):
                        lo = base + (e * two_ff + ff) * h * esz
                        f32 = _to_f32(reader_read(shard_of(gu), lo, row_elems * esz),
                                      gu.dtype)
                        if gu_im_q is not None:
                            # gate and up share the plane unit = the matrix the expert
                            # reads; imatrix.cpp accumulates one unit per expert matrix.
                            yield gu_im_q(f32, plane_tile(gu_plane[e], row_elems, h, 0))
                        else:
                            yield quantize_q2_0_stream(f32)

                gate_bytes = w.add_tensor_stream(f"blk.{L}.ffn_gate_exps.weight",
                                                 [h, ff, ne], gu_type, gate_chunks())
                up_bytes = w.add_tensor_stream(f"blk.{L}.ffn_up_exps.weight",
                                               [h, ff, ne], gu_type, up_chunks())
                if sample():
                    return 3
            dn = by_name.get(p + "mlp.experts.down_proj") \
                or by_name.get(p + "experts.down_proj")
            if dn is not None:
                ne2, h2, ff2 = dn.shape
                esz2 = DTYPE_SIZES[dn.dtype]
                base2 = abs_offset(meta, dn)
                dn_plane = imatrix_stats.get(f"blk.{L}.ffn_down_exps.weight") \
                    if imatrix_stats else None
                if dn_plane is not None and dn_plane.shape != (ne2, ff2):
                    print(f"blk.{L}.ffn_down_exps plane {dn_plane.shape} != "
                          f"(ne={ne2}, ff={ff2}) — refusing", file=sys.stderr)
                    return 3

                def down_chunks():  # engine ffn_down_exps = down rows per expert
                    for e in range(ne2):
                        lo = base2 + e * h2 * ff2 * esz2
                        f32 = _to_f32(reader_read(shard_of(dn), lo, h2 * ff2 * esz2),
                                      dn.dtype)
                        if dn_plane is not None:
                            # engine row length = ff2 (shape [ff2,h2,ne2], measured plane
                            # vector length 704 = ff2); Q4_K refuses (704 % 256 = 192 남),
                            # Q4_0 be=32 passes → Q4_0 imatrix producer.
                            yield quantize_q4_0_imatrix_stream(
                                f32, plane_tile(dn_plane[e], h2 * ff2, ff2, 0))
                        else:
                            yield quantize_q2_0_stream(f32)

                dn_type = "Q4_0" if dn_plane is not None else "Q2_0"
                w.add_tensor_stream(f"blk.{L}.ffn_down_exps.weight",
                                    [ff2, h2, ne2], dn_type, down_chunks())
                if sample():
                    return 3
            # per-expert HF naming (Qwen2MoE/Mixtral per-expert, expert_map p1-01
            # patterns): gather role tensors in expert order, stream one slice at a
            # time; engine fused shape (in, out, NE) — expert e contiguous run.
            refs = [r for r in build_expert_map(meta) if r.layer_id == L]
            if gu is not None or dn is not None:
                # fused gate_up_proj/down_proj emitted above cover all roles;
                # build_expert_map's _FUSED_PATTERN would re-emit them (duplicate
                # tensor name + full-tensor payload, observed 2026-10-17).
                refs = []
            for role, engine_suffix in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"),
                                        ("down", "ffn_down_exps")):
                role_refs = sorted((r for r in refs if r.role == role),
                                   key=lambda r: r.expert_id)
                if not role_refs:
                    continue
                entry = by_name[role_refs[0].tensor_name]
                out_dim, in_dim = entry.shape[0], entry.shape[1]
                # p6-05: engine-key plane for this role (unit e = expert e); absent
                # → plain Q2_0 fallback. Engine load gate: shape[0] = in_dim.
                role_plane = imatrix_stats.get(f"blk.{L}.{engine_suffix}.weight") \
                    if imatrix_stats else None
                if role_plane is not None and role_plane.shape != (len(role_refs), in_dim):
                    print(f"blk.{L}.{engine_suffix}.weight plane {role_plane.shape} != "
                          f"({len(role_refs)}, {in_dim}) — refusing", file=sys.stderr)
                    return 3
                role_q4k = role_plane is not None and in_dim % 256 == 0 \
                    and (out_dim * in_dim) % 256 == 0
                role_q40 = role_plane is not None and not role_q4k and in_dim % 32 == 0 \
                    and (out_dim * in_dim) % 32 == 0

                def role_chunks():
                    for i, r in enumerate(role_refs):
                        e = by_name[r.tensor_name]
                        raw = payload(e)
                        f32 = _to_f32(raw, e.dtype)
                        del raw
                        if role_q4k:
                            yield quantize_q4_k_imatrix_stream(
                                f32, plane_tile(role_plane[i], out_dim * in_dim, in_dim, 0))
                        elif role_q40:
                            yield quantize_q4_0_imatrix_stream(
                                f32, plane_tile(role_plane[i], out_dim * in_dim, in_dim, 0))
                        else:
                            yield quantize_q2_0_stream(f32)

                role_type = ("Q4_K" if role_q4k else "Q4_0" if role_q40 else "Q2_0")

                w.add_tensor_stream(f"blk.{L}.{engine_suffix}.weight",
                                    [in_dim, out_dim, len(role_refs)],
                                    role_type, role_chunks())
                if sample():
                    return 3
            # 1-D keys in engine want1 order — F32 contract, never quantized
            for hf_suffix, engine_suffix in emit_1d:
                entry = by_name.get(p + hf_suffix)
                if entry is None:
                    continue
                rule = layer_dense[hf_suffix][1]
                emit_dense_gate(entry, f"blk.{L}.{engine_suffix}", rule, force_f32=True)
                if sample():
                    return 3
            # GDN 1-D → engine 2-D broadcast rows (fixture-measured, p4-01)
            for hf_suffix, engine_suffix, mode in _GDN_1D:
                entry = by_name.get(p + hf_suffix)
                if entry is None:
                    continue
                raw = payload(entry)
                f32 = _to_f32(raw, entry.dtype)
                v = np.frombuffer(f32, dtype="<f4")
                n_embd = derived.get("embedding_length", meta.tensors[0].shape[0])
                w.add_f32(f"blk.{L}.{engine_suffix}",
                           np.broadcast_to(v, (n_embd, v.size)).astype("<f4"))
                del raw, f32, v
                if sample():
                    return 3
        # globals: token_embd, output, output_norm, PLE (U8 pass-through, p3-01 contract)
        global_dense = GEMMA4_GLOBAL_DENSE if gemma4 else GLOBAL_DENSE
        for hf_name, (engine_name, rule) in global_dense.items():
            entry = by_name.get(hf_name)
            if entry is None:
                continue
            # 1-D globals (output_norm) are engine want1 F32 keys — same contract as the
            # layer 1-D keys above; 2-D globals keep the dense_type gate.
            emit_dense_gate(entry, engine_name, rule,
                            force_f32=(len(entry.shape) == 1))
            if sample():
                return 3
        ple = by_name.get(PLE_NAME)
        if ple is not None:
            if list(ple.shape) != [PLE_BYTES]:
                raise ValueError(f"{PLE_NAME}: PLE payload must be U8[{PLE_BYTES}], "
                                 f"got U8{ple.shape}")
            raw = payload(ple)
            w.add_tensor(PLE_NAME, [PLE_HEAD_DIM, PLE_ROWS], "IQ4_NL", raw)
            del raw
            if sample():
                return 3
        if gemma4:
            # derived global tensor rope_freqs.weight — engine gemma4.cpp:207 (pin
            # c566ff4) loads the FULL layers' rotary frequency table from the FILE;
            # TK1 gemma4 RUN FAIL rc=1 "the GGUF has no tensor rope_freqs.weight"
            # (docs/work/p5-08 §AC4). Values = engine float64 reference
            # `theta ** (-2i/n_rot)` (rope_parity.cpp header); see _gemma4_rope_freqs.
            freqs = _gemma4_rope_freqs(guard_keys, cfg_json)
            w.add_tensor("rope_freqs.weight", list(freqs.shape), "F32", freqs.tobytes())
            if sample():
                return 3
    finally:
        for fh in handles.values():
            fh.close()
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.write(out)
    # engine convention: data section flush at EOF (p3-02 cmd_emit_gguf comment)
    end_meta = parse_gguf_header(str(out))
    data_end = end_meta.data_start + max(t.offset + t.expected_bytes() for t in end_meta.tensors)
    if out.stat().st_size > data_end:
        os.truncate(str(out), data_end)
    heap_peak = tracemalloc.get_traced_memory()[1]
    slice_heap_peak = max(slice_heap_samples) if slice_heap_samples else 0
    tracemalloc.stop()
    print(json.dumps({
        "model": model,
        "gguf": str(out),
        "tensors": len(w.tensors),
        "tokenizer_meta_keys": tokenizer_meta_count,
        "guard_keys": len(guard_keys),
        "derived_guard": derived,
        "hf_layers": layers,
        "rss_samples_bytes": rss_samples,
        "rss_peak_bytes": max(rss_samples) if rss_samples else 0,
        "heap_peak_bytes": heap_peak,
        "slice_heap_peak_bytes": slice_heap_peak,
    }, indent=2))
    return 0


def _verify_p2(man: dict, out: pathlib.Path) -> int:
    """p2 CLI pack self-check (`convert --out` manifest contract, p5-02).

    Checks: header pool/tensors vs manifest packed_rows, per-row codes+scales == dst
    and sequential dst_off (pool recompute), experts.bin size == pool, sha256
    regression recompute, src slices in_bounds against the model file.
    """
    errors: list[str] = []
    lines = (out / "index.txt").read_text(encoding="utf-8").splitlines()
    data_rows = [ln.split() for ln in lines if not ln.startswith("#")]
    header = lines[0].split()  # "# align A pool P tensors N"
    pool = int(header[4])
    if pool != int(man["pool_bytes"]):
        errors.append(f"header pool {pool} != manifest pool {man['pool_bytes']}")
    if int(header[6]) != len(data_rows) or len(data_rows) != man["packed_rows"]:
        errors.append(f"tensors {header[6]} rows {len(data_rows)} "
                      f"packed_rows {man['packed_rows']}")
    model_size: int | None = None
    try:
        model_size = pathlib.Path(man["model"]).stat().st_size
    except OSError:
        errors.append(f"model {man['model']} unreadable for in_bounds check")
    dst = 0
    for r in data_rows:
        codes, scales, dst_bytes = int(r[14]), int(r[15]), int(r[6])
        if codes + scales != dst_bytes:
            errors.append(f"row {r[0]}: codes {codes} + scales {scales} != dst {dst_bytes}")
        if int(r[5]) != dst:
            errors.append(f"row {r[0]}: dst_off {r[5]} != sequential {dst}")
        if model_size is not None and int(r[3]) + int(r[4]) > model_size:
            errors.append(f"row {r[0]}: src {r[3]}+{r[4]} past model {model_size}")
        dst += dst_bytes
    if dst != pool:
        errors.append(f"row sum {dst} != header pool {pool}")
    exp = out / "experts.bin"
    if exp.stat().st_size != pool:
        errors.append(f"experts.bin {exp.stat().st_size} != pool {pool}")
    if _sha256(exp) != man["experts_bin_sha256"]:
        errors.append("experts.bin sha256 mismatch (regression)")
    print(json.dumps({"out_dir": str(out), "format": "p2", "ok": not errors,
                      "errors": errors, "pool_bytes": dst,
                      "experts_bin_sha256": _sha256(exp)}, indent=2))
    return 0 if not errors else 1


def cmd_verify(out_dir: str) -> int:
    """Local self-check of a pack directory. Two manifest contracts (p5-02):

    - engine-format manifest (has "tensors"/"experts"): index 19-field contract with
      plane sums and pool recomputed from the manifest, experts.bin size == layers x NE x
      blob, every layer blob inside the file, output hashes recomputed for regression
      (ACCEPTANCE Phase 3, p4 harness contract).
    - p2 CLI pack manifest (has "packed_rows", written by `convert --out`): index rows
      recomputed sequentially, experts.bin size == pool, sha256 regression, in_bounds
      against the model file. This closes the p5-02 disconnect: the CLI emit/convert
      path wrote p2-form manifests that the engine-form checker KeyError'd on.
    """
    out = pathlib.Path(out_dir)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if "packed_rows" in man:
        return _verify_p2(man, out)
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
                  dense_type: str = "F32", imatrix_file: str | None = None) -> int:
    with heap_gate():
        return _cmd_emit_gguf(model, out_path, guard_keys, max_rss_gb, dense_type, imatrix_file)


def _cmd_emit_gguf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
                  dense_type: str = "F32", imatrix_file: str | None = None) -> int:
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
    # p6-03 AC3 opt-in: imatrix stats file (GGUF format, loader contract transcribed
    # from llama.cpp common/imatrix-loader.cpp + quantize.cpp:204-215). Loaded ONCE as
    # normalized per-matrix vectors; the stats file is small (calibration product), not
    # a model body — C1-safe. None (default) → plain RTN path is byte-unchanged.
    imatrix_stats = load_imatrix_stats(imatrix_file) if imatrix_file else None
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
                        raw = _slice_read(mm, src, meta.data_start + begin + e * run_bytes,
                                          run_bytes)
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
                    raw = _slice_read(mm, src, meta.data_start + begin, end - begin)
                    w.tensors.append(_Tensor(entry.name, [PLE_HEAD_DIM, PLE_ROWS],
                                             "IQ4_NL", raw))
                    del raw
                    rss_samples.append(current_rss_bytes())
                    slice_heap_samples.append(tracemalloc.get_traced_memory()[1])
                else:
                    if entry.dtype != "F32":
                        raise ValueError(f"{entry.name}: only F32 dense tensors supported "
                                         f"for emit, got {entry.dtype}")
                    raw = _slice_read(mm, src, meta.data_start + begin, end - begin)
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
                    elif dense_type == "Q4_K":
                        n_el = 1
                        for d in entry.shape:
                            n_el *= d
                        # Engine pack gate (p3-04): codes plane 128 B/block (64-mult),
                        # scales 32 B/block, offsets 32 B/block; offsets start = 160n,
                        # contiguity needs 160n % 64 == 0 → n even → n_el % 512 == 0.
                        # Tensors below the Q4_K block (256) or below the pack gate (512)
                        # cannot be packed as Q4_K by the engine itself — keep them F32
                        # pass-through (same policy as Q4_0's 128-element gate).
                        if n_el % Q4_K_MIN_PACK_ELEMENTS:
                            w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                        else:
                            # p6-03 AC3: imatrix opt-in consumes the normalized vector over the
                            # input dimension; row-major tensor [out, in] tiles the vector per row.
                            # Plain RTN path is byte-unchanged when imatrix_stats is None (AC gate).
                            # A dense stats entry has nmat=1 (imatrix.cpp:723 single count), so the
                            # vector is the flat column plane [ne0].
                            vec = imatrix_stats.get(entry.name) if imatrix_stats else None
                            row_len = entry.shape[-1]
                            if vec is not None:
                                if vec.shape[0] != 1:
                                    print(f"{entry.name}: imatrix dense path expects nmat=1, "
                                          f"got {vec.shape[0]} — refusing", file=sys.stderr)
                                    return 3
                                q = quantize_q4_k_imatrix_stream(
                                    raw, broadcast_imatrix(vec.reshape(-1), n_el, row_len))
                            else:
                                q = quantize_q4_k_stream(raw)
                            w.tensors.append(_Tensor(entry.name, list(entry.shape), "Q4_K", q))
                            del q
                    elif dense_type == "Q5_K":
                        n_el = 1
                        for d in entry.shape:
                            n_el *= d
                        # Engine pack gate (p3-05): codes plane 256 B/block (S8, 64-mult),
                        # scales 32 B/block, offsets 32 B/block; offsets start = 288n,
                        # contiguity needs 288n % 64 == 0 → n even → n_el % 512 == 0.
                        # Tensors below the pack gate cannot be packed as Q5_K by the
                        # engine itself — keep them F32 pass-through (same policy as Q4_K).
                        if n_el % Q5_K_MIN_PACK_ELEMENTS:
                            w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                        else:
                            q = quantize_q5_k_stream(raw)
                            w.tensors.append(_Tensor(entry.name, list(entry.shape), "Q5_K", q))
                            del q
                    elif dense_type == "Q6_K":
                        n_el = 1
                        for d in entry.shape:
                            n_el *= d
                        # Engine pack gate (p3-06): codes plane 256 B/block (S8 per=1, 64-mult),
                        # scales 16*4 = 64 B/block, no offsets; scales start = 256n, contiguity
                        # 256n % 64 == 0 always → n_el % 256 == 0 (block multiple).
                        # Tensors below the gate stay F32 pass-through (same policy as Q4_K/Q5_K).
                        if n_el % Q6_K_MIN_PACK_ELEMENTS:
                            w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                        else:
                            q = quantize_q6_k_stream(raw)
                            w.tensors.append(_Tensor(entry.name, list(entry.shape), "Q6_K", q))
                            del q
                    elif dense_type == "Q3_K":
                        n_el = 1
                        for d in entry.shape:
                            n_el *= d
                        # Engine pack gate (p3-07): codes plane 128 B/block (S4 per=2, 64-mult),
                        # scales 16*4 = 64 B/block, no offsets; contiguity 128n % 64 == 0
                        # always → n_el % 256 == 0 (block multiple).
                        # Tensors below the gate stay F32 pass-through (same policy as Q4_K/Q5_K).
                        if n_el % Q3_K_MIN_PACK_ELEMENTS:
                            w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                        else:
                            q = quantize_q3_k_stream(raw)
                            w.tensors.append(_Tensor(entry.name, list(entry.shape), "Q3_K", q))
                            del q
                    elif dense_type == "Q8_0":
                        n_el = 1
                        for d in entry.shape:
                            n_el *= d
                        # Engine pack gate (p3-09): codes plane 32 B/block (S8 per=1), scales
                        # 2 B/block fp16, no offsets; scales start = align64(32n)
                        # (strata_pack.py:43 ALIGN=64), contiguity needs 32n % 64 == 0
                        # (pack_index.py:144-152) → n even → n_el % 64 == 0.
                        # AND ndim ≥ 2: pack_index.py:174-186 computes scales as
                        # ne1·n_groups (ne1 = shape[1], 0 for a 1-D tensor) → a 1-D
                        # quantized tensor derives 0 scale bytes and pack_index refuses
                        # ("derived planes disagree"). 1-D tensors (e.g. attn_norm) stay
                        # F32 pass-through, same policy as below the gate.
                        if n_el % Q8_0_MIN_PACK_ELEMENTS or len(entry.shape) < 2:
                            w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
                        else:
                            q = quantize_q8_0_stream(raw)
                            w.tensors.append(_Tensor(entry.name, list(entry.shape), "Q8_0", q))
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
    p_ver = sub.add_parser("verify", help="local self-check of a GGUF-path pack")
    p_ver.add_argument("out")
    args = ap.parse_args(argv)
    if args.cmd == "inspect":
        return cmd_inspect(args.model, args.json)
    if args.cmd == "convert":
        if pathlib.Path(args.model).is_dir():
            # p4-02: directory = multi-shard HF input (index.json + shards). The Q2_0
            # slice convert pack path is single-file only; the engine GGUF emit path
            # (the Phase 4 acceptance target) takes the shard directory directly.
            if not args.emit_gguf:
                print(f"{args.model}: multi-shard convert is not supported — use "
                      f"--emit-gguf (engine GGUF acceptance is the Phase 4 target)",
                      file=sys.stderr)
                return 2
            guard = json.loads(pathlib.Path(args.guard_keys).read_text()) \
                if args.guard_keys else {}
            # p6-05: the HF fused emit path consumes --imatrix planes (engine-key
            # lookup; expert slots gate/up Q4_K + down Q4_0, dense engine-name lookup).
            return cmd_emit_gguf_hf(args.model, args.emit_gguf, guard, args.max_rss_gb,
                                    args.dense_type, args.imatrix)
        rc = cmd_convert(args.model, args.out, args.max_rss_gb)
        if rc == 0 and args.emit_gguf:
            guard = json.loads(pathlib.Path(args.guard_keys).read_text()) \
                if args.guard_keys else {}
            hf = args.hf_emit
            if not hf:
                # p5-02 auto-detection: safetensors with HF naming routes to the fused
                # HF emit path WITHOUT the flag (p4-01 "auto-detected" claim, connected
                # here — is_hf_naming existed in the map module, dispatch never used it).
                with open(args.model, "rb") as f:
                    gguf = struct.unpack("<I", f.read(4))[0] == GGUF_MAGIC
                if not gguf and is_hf_naming(parse_header(args.model)):
                    hf = True
            if hf:
                rc = cmd_emit_gguf_hf(args.model, args.emit_gguf, guard, args.max_rss_gb,
                                      args.dense_type, args.imatrix)
            else:
                rc = cmd_emit_gguf(args.model, args.emit_gguf, guard, args.max_rss_gb,
                                   args.dense_type, args.imatrix)
        return rc
    return cmd_verify(args.out)


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
