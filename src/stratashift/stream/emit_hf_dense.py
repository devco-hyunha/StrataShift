"""HF emit: dense tensors (moved out of cli/main.py, p6-08).

Layer/global steps return True when the run must stop with exit code 3 (RSS ceiling),
the same `sample()` contract the emit loop always had.
"""
from __future__ import annotations

import numpy as np

from stratashift.ingest.safetensors_header import numel
from stratashift.meta.gemma4_rope import gemma4_rope_freqs
from stratashift.meta.hf_emit_order import GDN_1D, GDN_QKV_ORDER, QSA_Q_ORDER
from stratashift.meta.mixtral_rope import reorder_rows, rope_head_dim, rope_row_order
from stratashift.polystrata.headers import PLE_BYTES, PLE_HEAD_DIM, PLE_NAME, PLE_ROWS
from stratashift.quant.dtype import to_f32
from stratashift.quant.pack_imatrix import plane_tile
from stratashift.quant.select import quantize_dense, select_dense_quant, supports_imatrix
from stratashift.stream.emit_hf_ctx import EmitContext

# 64 MiB: harness heap tuning, not an engine format value. Tensors above it stream
# chunk-by-chunk instead of materializing the payload — p4-03 root cause: full embed/lm_head
# payload (622 MB BF16 → 1.24 GB F32) pushed the heap peak to 2.9 GiB > 2 GiB gate
# (observed 2026-10-17, RC=137; stream.reader.TensorReader.payload_chunks). Gate: heap
# peak < 2 GiB, measured 116,821,170 B with this threshold (docs/work/p4-03.md:70-71);
# applied as "> 64 MB → stream" in docs/work/p5-06.md:75.
STREAM_THRESHOLD = 1 << 26


def emit_dense(ctx: EmitContext, entry, raw: bytes, engine_name: str, rule: str,
               force_f32: bool = False) -> None:
    """One dense tensor: dtype normalize + engine shape + dense_type gate.

    force_f32: the engine want1 contract — every 1-D engine key is read as an
    F32 tensor of its exact length (p4-01 §dtype rule, layout.cpp:88-98; gemma4
    `f32(...)` calls, gemma4.cpp:197-211). A 1-D key is NEVER quantized: tiny
    1-D tensors pass the Q4_K gate (512 elements) and quantizing them produces
    manifest entries with empty scale planes (observed 2026-10-18: engine
    `blk.5.attn_q_norm.weight` Q4_K → "planes not contiguous" at verify).
    """
    n_el = numel(entry.shape)
    if rule == "bf16_keep":
        ctx.w.add_tensor(engine_name, list(entry.shape), "BF16", raw)
        return
    f32 = to_f32(raw, entry.dtype)
    shape = list(reversed(entry.shape)) if rule == "transpose" else list(entry.shape)
    qtype = select_dense_quant(n_el, ctx.dense_type, shape[0], len(shape), force_f32)
    if qtype == "F32":
        ctx.w.add_f32(engine_name, np.frombuffer(f32, dtype="<f4").reshape(shape))
    else:
        # p6-05: engine-key imatrix plane consume (plane row = engine shape[0],
        # the HF input dim; element r*row_len + c uses plane[c]). Plain RTN is
        # byte-unchanged when imatrix_stats is None (golden gate, p6-03 AC3).
        vec = ctx.imatrix_stats.get(engine_name) if ctx.imatrix_stats else None
        weights = plane_tile(vec.reshape(-1), n_el, shape[0], 0) \
            if vec is not None and supports_imatrix(qtype) else None
        q = quantize_dense(qtype, f32, weights)
        ctx.w.add_tensor(engine_name, shape, qtype, q)
        del q
    del f32


def emit_dense_stream(ctx: EmitContext, entry, engine_name: str, rule: str,
                      force_f32: bool = False) -> None:
    """Stream a large dense tensor chunk by chunk (p4-03): byte-identical to
    emit_dense — quantizer blocks are block-independent, and the F32/BF16
    pass-through streams raw bytes verbatim. `force_f32` is the want1 1-D engine
    key contract (see emit_dense)."""
    n_el = numel(entry.shape)
    shape = list(reversed(entry.shape)) if rule == "transpose" else list(entry.shape)
    if rule == "bf16_keep":
        ctx.w.add_tensor_stream(engine_name, list(entry.shape), "BF16",
                                ctx.reader.payload_chunks(entry))
        return
    qtype = select_dense_quant(n_el, ctx.dense_type, shape[0], len(shape), force_f32)
    if qtype == "F32":
        # F32 passthrough: BF16 chunks MUST be widened to F32 bytes per chunk
        # (full path does f32 = to_f32(raw) before add_f32; streaming raw BF16
        # under an F32 type writes half the bytes — caught 2026-10-17 by the
        # GGUF size delta 10.53 GB → 9.29 GB == embed F32 size).
        def f32_chunks():
            for chunk in ctx.reader.payload_chunks(entry):
                yield to_f32(chunk, entry.dtype) if entry.dtype != "F32" else chunk
        ctx.w.add_tensor_stream(engine_name, shape, "F32", f32_chunks())
        return
    # p6-05: imatrix plane consume (engine-key lookup, plane row = engine shape[0] —
    # plane contract imatrix.cpp pin :971 in_sum2 [nval/nmat, nmat], nmat=1 dense).
    # row_len = engine shape[0] == HF entry.shape[-1] (the input dim the plane is
    # over); the stream element r*row_len + c uses plane[c] (broadcast contract).
    vec = ctx.imatrix_stats.get(engine_name) if ctx.imatrix_stats else None
    row_len = shape[0]
    use_imatrix = vec is not None and supports_imatrix(qtype)

    def qchunks():
        off = 0
        for chunk in ctx.reader.payload_chunks(entry):
            f32 = to_f32(chunk, entry.dtype) if entry.dtype != "F32" else chunk
            if use_imatrix:
                n = len(f32) // 4
                yield quantize_dense(qtype, f32, plane_tile(vec.reshape(-1), n, row_len, off))
                off += n
            else:
                yield quantize_dense(qtype, f32)

    ctx.w.add_tensor_stream(engine_name, shape, qtype, qchunks())


def emit_rope_permuted(ctx: EmitContext, entry, engine_name: str) -> None:
    """p6-01 Mixtral q/k: rows reordered within each head for the engine_ext rotary
    (meta/mixtral_rope.py), then emitted as a transposed dense matrix. One q/k matrix
    (n_embd² at most, 32 MB BF16 for 8x7B) is read whole — under STREAM_THRESHOLD."""
    if entry.nbytes() > STREAM_THRESHOLD:
        raise ValueError(f"{entry.name}: {entry.nbytes()} B is above the {STREAM_THRESHOLD} B a "
                         f"reordered q/k matrix is read in one piece")
    rows, cols = entry.shape
    head_dim = rope_head_dim(ctx.guard_keys, ctx.guard_keys["architecture"])
    raw = reorder_rows(ctx.reader.payload(entry), rope_row_order(rows, head_dim),
                       cols * (entry.nbytes() // (rows * cols)))
    emit_dense(ctx, entry, raw, engine_name, "transpose")
    del raw


def emit_dense_gate(ctx: EmitContext, entry, engine_name: str, rule: str,
                    force_f32: bool = False) -> None:
    """p4-03: tensors above 64 MB stream, never materialize the full payload."""
    if rule == "rope_permute":
        emit_rope_permuted(ctx, entry, engine_name)
        return
    if rule == "transpose_f32":  # p6-01: Mixtral router, never quantized (hf_gguf_map MIXTRAL_LAYER_DENSE)
        rule, force_f32 = "transpose", True
    if entry.nbytes() > STREAM_THRESHOLD:
        emit_dense_stream(ctx, entry, engine_name, rule, force_f32)
        return
    raw = ctx.reader.payload(entry)
    emit_dense(ctx, entry, raw, engine_name, rule, force_f32)
    del raw


def emit_layer_dense_2d(ctx: EmitContext, p: str, L: int) -> bool:
    """2-D keys in engine want2 order."""
    for hf_suffix, engine_suffix in ctx.tables.emit_2d:
        entry = ctx.by_name.get(p + hf_suffix)
        if entry is None:
            continue
        rule = ctx.tables.layer_dense[hf_suffix][1]
        emit_dense_gate(ctx, entry, f"blk.{L}.{engine_suffix}", rule)
        if ctx.sampler.sample():
            return True
    return False


def emit_gdn_qkv(ctx: EmitContext, p: str, L: int) -> bool:
    """GDN fused qkv: engine attn_qkv = concat(q, k, v) rows (bytes verbatim)."""
    parts = [ctx.by_name.get(p + s) for s in GDN_QKV_ORDER]
    if not all(pt is not None for pt in parts):
        return False
    if not all(pt.dtype == parts[0].dtype for pt in parts):
        return False
    dt = parts[0].dtype

    def qkv_chunks():
        for pt in parts:
            yield to_f32(ctx.reader.payload(pt), dt) if dt != "F32" else ctx.reader.payload(pt)

    ne0 = parts[0].shape[1]
    ne1 = sum(pt.shape[0] for pt in parts)
    ctx.w.add_tensor_stream(f"blk.{L}.attn_qkv.weight", [ne0, ne1],
                            "F32" if dt == "F32" else "BF16", qkv_chunks())
    return ctx.sampler.sample()


def emit_attn_q_fused(ctx: EmitContext, p: str, L: int) -> bool:
    """fused attn_q: q_proj rows then q_gate_proj rows (engine attn_q
    ne1 = 2·n_head·head_dim, layout.cpp:83 + layer.hpp:363, pin c566ff4)"""
    q_parts = [ctx.by_name.get(p + s) for s in QSA_Q_ORDER]
    if not all(pt is not None for pt in q_parts):
        return False
    dt = q_parts[0].dtype

    def attnq_chunks():
        for pt in q_parts:
            yield to_f32(ctx.reader.payload(pt), dt) if dt != "F32" else ctx.reader.payload(pt)

    ctx.w.add_tensor_stream(f"blk.{L}.attn_q.weight",
                            [q_parts[0].shape[1], sum(pt.shape[0] for pt in q_parts)],
                            "F32" if dt == "F32" else "BF16", attnq_chunks())
    return ctx.sampler.sample()


def emit_layer_dense_1d(ctx: EmitContext, p: str, L: int) -> bool:
    """1-D keys in engine want1 order — F32 contract, never quantized."""
    for hf_suffix, engine_suffix in ctx.tables.emit_1d:
        entry = ctx.by_name.get(p + hf_suffix)
        if entry is None:
            continue
        rule = ctx.tables.layer_dense[hf_suffix][1]
        emit_dense_gate(ctx, entry, f"blk.{L}.{engine_suffix}", rule, force_f32=True)
        if ctx.sampler.sample():
            return True
    return False


def emit_gdn_broadcast(ctx: EmitContext, p: str, L: int) -> bool:
    """GDN 1-D → engine 2-D broadcast rows (fixture-measured, p4-01)."""
    for hf_suffix, engine_suffix, _mode in GDN_1D:
        entry = ctx.by_name.get(p + hf_suffix)
        if entry is None:
            continue
        raw = ctx.reader.payload(entry)
        f32 = to_f32(raw, entry.dtype)
        v = np.frombuffer(f32, dtype="<f4")
        n_embd = ctx.derived.get("embedding_length", ctx.meta.tensors[0].shape[0])
        ctx.w.add_f32(f"blk.{L}.{engine_suffix}",
                      np.broadcast_to(v, (n_embd, v.size)).astype("<f4"))
        del raw, f32, v
        if ctx.sampler.sample():
            return True
    return False


def emit_global_dense(ctx: EmitContext) -> bool:
    """Globals: token_embd, output, output_norm (1-D globals are engine want1 F32 keys —
    same contract as the layer 1-D keys; 2-D globals keep the dense_type gate)."""
    for hf_name, (engine_name, rule) in ctx.tables.global_dense.items():
        entry = ctx.by_name.get(hf_name)
        if entry is None:
            continue
        emit_dense_gate(ctx, entry, engine_name, rule, force_f32=(len(entry.shape) == 1))
        if ctx.sampler.sample():
            return True
    return False


def emit_ple(ctx: EmitContext) -> bool:
    """PLE table: U8 pass-through (p3-01 contract)."""
    ple = ctx.by_name.get(PLE_NAME)
    if ple is None:
        return False
    if list(ple.shape) != [PLE_BYTES]:
        raise ValueError(f"{PLE_NAME}: PLE payload must be U8[{PLE_BYTES}], "
                         f"got U8{ple.shape}")
    raw = ctx.reader.payload(ple)
    ctx.w.add_tensor(PLE_NAME, [PLE_HEAD_DIM, PLE_ROWS], "IQ4_NL", raw)
    del raw
    return ctx.sampler.sample()


def emit_rope_freqs(ctx: EmitContext) -> bool:
    """gemma4 derived global tensor rope_freqs.weight — engine gemma4.cpp:207 (pin
    c566ff4) loads the FULL layers' rotary frequency table from the FILE;
    TK1 gemma4 RUN FAIL rc=1 "the GGUF has no tensor rope_freqs.weight"
    (docs/work/p5-08 §AC4). Values = engine float64 reference
    `theta ** (-2i/n_rot)` (rope_parity.cpp header); see gemma4_rope_freqs."""
    freqs = gemma4_rope_freqs(ctx.guard_keys, ctx.cfg_json)
    ctx.w.add_tensor("rope_freqs.weight", list(freqs.shape), "F32", freqs.tobytes())
    return ctx.sampler.sample()
