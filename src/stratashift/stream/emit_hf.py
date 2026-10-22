"""HF safetensors MoE (HF naming, fused gate_up, BF16/F16/F32) → engine GGUF v3.

Moved out of cli/main.py and split into steps (p6-08). Emit rules
(docs/work/p4-01.md §Emit 규칙, layout.cpp:60-110 pin c566ff4):

- fused gate_up (NE,2FF,H): expert e = contiguous row block; gate = rows [0,FF), up =
  rows [FF,2FF) → `blk.L.ffn_{gate,up}_exps.weight` shape (H,FF,NE); down (NE,H,FF) →
  `blk.L.ffn_down_exps.weight` (FF,H,NE). Type + producer: `quant.select`.
- dense HF [out,in] row-major bytes == engine GGUF (in,out) bytes (transpose shape
  metadata only); 1-D keys keep shape; `ffn_gate_inp_shexp.weight` keeps BF16 bytes
  (Bf16InF32, layout.cpp:88-98).
- guard keys are DERIVED from the HF header (derive_geometry) and cross-checked against
  the preset; a mismatch is refused (never invented).
- one tensor slice alive at a time (CONSTRAINTS C1/C2).
"""
from __future__ import annotations

import json
import pathlib
import sys

from stratashift.ingest.safetensors_header import parse_header
from stratashift.ingest.shard_index import load_sharded_model
from stratashift.meta.hf_emit_order import GEMMA4_1D, GEMMA4_2D, HF_EMIT_1D, HF_EMIT_2D
from stratashift.meta.hf_gguf_map import (GEMMA4_GLOBAL_DENSE, GEMMA4_LAYER_DENSE,
                                          GLOBAL_DENSE, LAYER_DENSE, build_tokenizer_meta,
                                          derive_geometry, hf_layer_id, is_hf_naming,
                                          load_hf_tokenizer)
from stratashift.meta.imatrix import load_imatrix_stats
from stratashift.monitor.heap import heap_gate, heap_peak_bytes
from stratashift.monitor.sampler import Sampler
from stratashift.polystrata.gguf_stream_writer import StreamingGGUFWriter
from stratashift.stream.emit_common import finalize_gguf, require_engine_tools
from stratashift.stream.emit_hf_ctx import ArchTables, EmitContext
from stratashift.stream.emit_hf_dense import (emit_attn_q_fused, emit_gdn_broadcast,
                                              emit_gdn_qkv, emit_global_dense,
                                              emit_layer_dense_1d, emit_layer_dense_2d,
                                              emit_ple, emit_rope_freqs)
from stratashift.stream.emit_hf_experts import emit_layer_experts
from stratashift.stream.reader import TensorReader


def emit_gguf_hf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
                 dense_type: str = "F32", imatrix_file: str | None = None) -> int:
    with heap_gate():
        return _emit_hf(model, out_path, guard_keys, max_rss_gb, dense_type, imatrix_file)


def read_hf_config(model: str) -> tuple[dict | None, int | None]:
    """config.json next to the model → (config, head_dim). p4-03: head_dim from the model
    directory's config.json when the header has no q_norm (Qwen1.5-MoE):
    head_dim = hidden_size // num_attention_heads — MEASURED from the checkpoint's own
    config, never invented (p4-01 HYPOTHESIS rule)."""
    d = pathlib.Path(model)
    cfg = (d / "config.json") if d.is_dir() else (d.parent / "config.json")
    if not cfg.is_file():
        return None, None
    c = json.loads(cfg.read_text(encoding="utf-8"))
    hs, nh = c.get("hidden_size"), c.get("num_attention_heads")
    head_dim = None
    if isinstance(hs, int) and isinstance(nh, int) and nh and hs % nh == 0:
        head_dim = c.get("head_dim") if isinstance(c.get("head_dim"), int) else hs // nh
    return c, head_dim


def find_guard_mismatch(derived: dict, guard_keys: dict, prefix: str) -> str | None:
    """First derived-vs-preset guard key disagreement as a message, else None."""
    for key, val in derived.items():
        preset_key = f"{prefix}.{key}"
        if preset_key not in guard_keys:
            continue
        # the engine reads per-layer ARRAY values via of_layer (gemma4.cpp:20-24,64-68);
        # a derived list must EQUAL the preset list elementwise, ints equal (never invented).
        want = guard_keys[preset_key]
        if isinstance(val, list):
            if not isinstance(want, list) or val != want:
                return f"guard mismatch: derived {key}={val} vs preset {preset_key}={want}"
            continue
        want = int(want)
        # block_count is the MODEL geometry: a partial fixture ships fewer layers than the
        # compiled geometry (p3-01 ships blk.0 of block_count=48), so the derived layer
        # count must be ≤ preset; every other derived key must be EQUAL (never invented).
        ok = val <= want if key == "block_count" else val == want
        if not ok:
            return f"guard mismatch: derived {key}={val} vs preset {preset_key}={want}"
    return None


def write_guard_metadata(w: StreamingGGUFWriter, guard_keys: dict, prefix: str) -> None:
    w.add("general.architecture", prefix)
    w.add("general.name", guard_keys.get("name", "stratashift hf emit"))
    for key, value in guard_keys.items():
        if key in ("architecture", "name"):
            continue
        w.add(key, value)


def write_tokenizer_meta(w: StreamingGGUFWriter, model: str, guard_keys: dict,
                         prefix: str) -> int:
    """p5-09 AC2: tokenizer metadata (tokenizer.ggml.*) from the MEASURED HF tokenizer
    files. Files absent → no tokenizer emitted (engine API gate: serve/models.py
    reports the model unavailable without vocab.json; strata-poly --tokens runs
    need no tokenizer). vocab_size guard key is measure-equal-checked (p5-08 rope
    gate shape, never-invented); a special text outside the measured reference
    type table refuses the emit. Keys + types: reference GGUF header measured
    2026-10-20 (unsloth Q8_0 head, engine gguf_reader parse) — see hf_gguf_map.
    Returns the number of tokenizer keys written."""
    hf_root = pathlib.Path(model)
    tok_dir = hf_root if hf_root.is_dir() else hf_root.parent
    if not (tok_dir / "tokenizer.json").is_file():
        return 0
    hf_tok = load_hf_tokenizer(tok_dir)
    tok_meta = build_tokenizer_meta(hf_tok, vocab_guard=guard_keys.get(f"{prefix}.vocab_size"))
    for key, (type_name, value) in tok_meta.items():
        w.add(key, value, type_name)
    return len(tok_meta)


def select_arch_tables(gemma4: bool) -> ArchTables:
    if gemma4:
        return ArchTables("model.language_model.layers.", GEMMA4_2D, GEMMA4_1D,
                          GEMMA4_LAYER_DENSE, GEMMA4_GLOBAL_DENSE)
    return ArchTables("model.layers.", HF_EMIT_2D, HF_EMIT_1D, LAYER_DENSE, GLOBAL_DENSE)


def emit_layer(ctx: EmitContext, L: int) -> bool:
    """One layer in engine order: 2-D keys (want2), GDN fused qkv, fused attn_q, experts,
    1-D keys (want1), GDN broadcast rows. True = stop with exit code 3."""
    p = f"{ctx.tables.layer_prefix}{L}."
    for step in (emit_layer_dense_2d, emit_gdn_qkv, emit_attn_q_fused, emit_layer_experts,
                 emit_layer_dense_1d, emit_gdn_broadcast):
        if step(ctx, p, L):
            return True
    return False


def _emit_hf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
             dense_type: str, imatrix_file: str | None) -> int:
    if not require_engine_tools():
        return 2
    # p4-02: directory = multi-shard HF input (index.json + shards, engine-mirrored
    # gates in load_sharded_model); single file keeps the p1-01 parse_header path.
    meta = load_sharded_model(model) if pathlib.Path(model).is_dir() else parse_header(model)
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
    cfg_json, head_dim = read_hf_config(model)
    prefix = guard_keys.get("architecture", "qwen4exp")
    derived = derive_geometry(meta, head_dim, prefix)
    mismatch = find_guard_mismatch(derived, guard_keys, prefix)
    if mismatch:
        print(mismatch, file=sys.stderr)
        return 2
    sampler = Sampler(max_rss_gb)
    gemma4 = prefix == "gemma4"
    # the writer's temp file is removed on every exit path (rc=3, refusal, exception)
    with StreamingGGUFWriter(tmp_path=str(out_path) + ".tmp") as w,             TensorReader(model, meta) as reader:
        write_guard_metadata(w, guard_keys, prefix)
        tokenizer_meta_count = write_tokenizer_meta(w, model, guard_keys, prefix)
        ctx = EmitContext(model, meta, by_name, w, reader, sampler, dense_type,
                          imatrix_stats, select_arch_tables(gemma4), derived,
                          guard_keys, cfg_json)
        for L in layers:
            if emit_layer(ctx, L):
                return 3
        # globals: token_embd, output, output_norm, PLE (U8 pass-through, p3-01 contract)
        if emit_global_dense(ctx) or emit_ple(ctx):
            return 3
        if gemma4 and emit_rope_freqs(ctx):
            return 3
        out = finalize_gguf(w, out_path)
    print(json.dumps({
        "model": model,
        "gguf": str(out),
        "tensors": len(w.tensors),
        "tokenizer_meta_keys": tokenizer_meta_count,
        "guard_keys": len(guard_keys),
        "derived_guard": derived,
        "hf_layers": layers,
        "rss_samples_bytes": sampler.rss,
        "rss_peak_bytes": sampler.rss_peak(),
        "heap_peak_bytes": heap_peak_bytes(),
        "slice_heap_peak_bytes": sampler.slice_heap_peak(),
    }, indent=2))
    return 0
