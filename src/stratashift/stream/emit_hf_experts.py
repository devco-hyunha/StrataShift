"""HF emit: expert tensors (moved out of cli/main.py, p6-08).

Fused experts: one expert run alive at a time (C1/C2); BF16/F16 runs normalize to F32
before quantization (p4-01 §5 — quantizer contract stays F32).

p6-05 (user ruling 2026-10-21: Q2_0는 하지마): expert slots consume the calibrated imatrix
planes. Plane unit contract (imatrix.cpp pin :744, :971 — in_sum2 [nval/nmat, nmat],
unit = matrix = expert). Plane absent → Q2_0 plain fallback. Which type and producer an
expert slot gets is decided ONLY by `quant.select.select_expert_quant` (type name and
quantizer always come as one pair).

Every step returns True when the run must stop with exit code 3 (plane shape refusal, no
engine-loadable expert type, or the RSS ceiling).
"""
from __future__ import annotations

import sys

from stratashift.ingest.safetensors_header import DTYPE_SIZES
from stratashift.ingest.shard_index import abs_offset
from stratashift.meta.expert_map import build_expert_map
from stratashift.quant.dtype import to_f32
from stratashift.quant.select import FUSED_DOWN_LADDER, expert_refusal, select_expert_quant
from stratashift.stream.emit_hf_ctx import EmitContext


def find_fused_experts(ctx: EmitContext, p: str):
    """(gate_up, down) fused expert entries of a layer prefix, None when absent."""
    by = ctx.by_name
    gu = by.get(p + "mlp.experts.gate_up_proj") or by.get(p + "experts.gate_up_proj")
    dn = by.get(p + "mlp.experts.down_proj") or by.get(p + "experts.down_proj")
    return gu, dn


def _plane_of(ctx: EmitContext, key: str):
    return ctx.imatrix_stats.get(key) if ctx.imatrix_stats else None


def emit_fused_gate_up(ctx: EmitContext, L: int, gu) -> bool:
    """Row e of `blk.{L}.ffn_gate_up_exps.weight` tiles across the gate/up row at
    row_len = h (engine ffn_gate_exps shape [h, ff, ne]). Engine load gate on shape[0] = h
    (gguf_reader.hpp:505 pin c566ff4): Q4_K be=256 when h % 256 == 0 (product h=2816 =
    256×11); fixture h=64 → be=32 Q4_0."""
    ne, two_ff, h = gu.shape
    ff = two_ff // 2
    row_elems = ff * h
    esz = DTYPE_SIZES[gu.dtype]
    base = abs_offset(ctx.meta, gu)
    plane = _plane_of(ctx, f"blk.{L}.ffn_gate_up_exps.weight")
    if plane is not None and plane.shape != (ne, h):
        print(f"blk.{L}.ffn_gate_up_exps plane {plane.shape} != "
              f"(ne={ne}, h={h}) — refusing", file=sys.stderr)
        return True
    eq = select_expert_quant(h, row_elems, plane is not None)
    if eq is None:
        print(expert_refusal(f"blk.{L}.ffn_gate_up_exps", h, row_elems), file=sys.stderr)
        return True
    shard = ctx.reader.shard_of(gu)

    def half_chunks(first_row: int):  # gate rows [0, ff), up rows [ff, 2ff) per expert
        for e in range(ne):
            lo = base + (e * two_ff + first_row) * h * esz
            f32 = to_f32(ctx.reader.read(shard, lo, row_elems * esz), gu.dtype)
            # gate and up share the plane unit = the matrix the expert reads;
            # imatrix.cpp accumulates one unit per expert matrix.
            yield eq.quantize(f32, plane[e] if plane is not None else None)

    ctx.w.add_tensor_stream(f"blk.{L}.ffn_gate_exps.weight", [h, ff, ne],
                            eq.type_name, half_chunks(0))
    ctx.w.add_tensor_stream(f"blk.{L}.ffn_up_exps.weight", [h, ff, ne],
                            eq.type_name, half_chunks(ff))
    return ctx.sampler.sample()


def emit_fused_down(ctx: EmitContext, L: int, dn) -> bool:
    """`blk.{L}.ffn_down_exps.weight` row e tiles at row_len = ff (engine shape
    [ff, h, ne] — ff % 256 = 192 남 → engine refuses Q4_K, be=32 Q4_0 pass,
    docs/work/p5-08.md §design ladder)."""
    ne, h, ff = dn.shape
    esz = DTYPE_SIZES[dn.dtype]
    base = abs_offset(ctx.meta, dn)
    plane = _plane_of(ctx, f"blk.{L}.ffn_down_exps.weight")
    if plane is not None and plane.shape != (ne, ff):
        print(f"blk.{L}.ffn_down_exps plane {plane.shape} != "
              f"(ne={ne}, ff={ff}) — refusing", file=sys.stderr)
        return True
    eq = select_expert_quant(ff, h * ff, plane is not None, FUSED_DOWN_LADDER)
    if eq is None:
        print(expert_refusal(f"blk.{L}.ffn_down_exps", ff, h * ff), file=sys.stderr)
        return True
    shard = ctx.reader.shard_of(dn)

    def down_chunks():  # engine ffn_down_exps = down rows per expert
        for e in range(ne):
            f32 = to_f32(ctx.reader.read(shard, base + e * h * ff * esz, h * ff * esz), dn.dtype)
            # engine row length = ff (shape [ff,h,ne], measured plane vector length 704 = ff)
            yield eq.quantize(f32, plane[e] if plane is not None else None)

    ctx.w.add_tensor_stream(f"blk.{L}.ffn_down_exps.weight", [ff, h, ne],
                            eq.type_name, down_chunks())
    return ctx.sampler.sample()


def layer_expert_refs(ctx: EmitContext, L: int) -> list:
    """Per-expert refs of layer L. build_expert_map scans every tensor name, so it runs
    once per emit (first per-expert layer) and is grouped by layer — not once per layer."""
    if ctx.expert_refs is None:
        ctx.expert_refs = {}
        for r in build_expert_map(ctx.meta):
            ctx.expert_refs.setdefault(r.layer_id, []).append(r)
    return ctx.expert_refs.get(L, [])


def emit_per_expert(ctx: EmitContext, L: int) -> bool:
    """Per-expert HF naming (Qwen2MoE/Mixtral per-expert, expert_map p1-01 patterns):
    gather role tensors in expert order, stream one slice at a time; engine fused shape
    (in, out, NE) — expert e contiguous run. Engine load gate: shape[0] = in_dim."""
    refs = layer_expert_refs(ctx, L)
    for role, engine_suffix in (("gate", "ffn_gate_exps"), ("up", "ffn_up_exps"),
                                ("down", "ffn_down_exps")):
        role_refs = sorted((r for r in refs if r.role == role), key=lambda r: r.expert_id)
        if not role_refs:
            continue
        entry = ctx.by_name[role_refs[0].tensor_name]
        out_dim, in_dim = entry.shape[0], entry.shape[1]
        # p6-05: engine-key plane for this role (unit e = expert e); absent
        # → plain Q2_0 fallback.
        plane = _plane_of(ctx, f"blk.{L}.{engine_suffix}.weight")
        if plane is not None and plane.shape != (len(role_refs), in_dim):
            print(f"blk.{L}.{engine_suffix}.weight plane {plane.shape} != "
                  f"({len(role_refs)}, {in_dim}) — refusing", file=sys.stderr)
            return True
        n_elems = out_dim * in_dim
        eq = select_expert_quant(in_dim, n_elems, plane is not None)
        if eq is None:
            print(expert_refusal(f"blk.{L}.{engine_suffix}.weight", in_dim, n_elems),
                  file=sys.stderr)
            return True

        def role_chunks():
            for i, r in enumerate(role_refs):
                e = ctx.by_name[r.tensor_name]
                raw = ctx.reader.payload(e)
                f32 = to_f32(raw, e.dtype)
                del raw
                yield eq.quantize(f32, plane[i] if plane is not None else None)

        ctx.w.add_tensor_stream(f"blk.{L}.{engine_suffix}.weight",
                                [in_dim, out_dim, len(role_refs)], eq.type_name, role_chunks())
        if ctx.sampler.sample():
            return True
    return False


def emit_layer_experts(ctx: EmitContext, p: str, L: int) -> bool:
    gu, dn = find_fused_experts(ctx, p)
    if gu is not None and emit_fused_gate_up(ctx, L, gu):
        return True
    if dn is not None and emit_fused_down(ctx, L, dn):
        return True
    if gu is not None or dn is not None:
        # fused gate_up_proj/down_proj emitted above cover all roles;
        # build_expert_map's _FUSED_PATTERN would re-emit them (duplicate
        # tensor name + full-tensor payload, observed 2026-10-17).
        return False
    return emit_per_expert(ctx, L)
