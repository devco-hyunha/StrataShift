"""Quantization type + producer selection (single place, p6-08 AC5).

Before p6-08 the HF emit chose the GGUF type name and the quantizer function in different
places (fused gate/up named the type by `h % 256` but always called the Q4_K producer, so
the header said Q4_0 over Q4_K bytes — p6-05 failure 2). Here the type name and the
quantizer are always returned together; callers cannot pick them apart.

Pure functions only (no file I/O). Gate constants come from `polystrata.headers`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from stratashift.ingest.safetensors_header import numel
from stratashift.polystrata.headers import (ENGINE_LOAD_BE, Q3_K_MIN_PACK_ELEMENTS,
                                            Q4_0_MIN_PACK_ELEMENTS, Q4_K_MIN_PACK_ELEMENTS,
                                            Q5_K_MIN_PACK_ELEMENTS, Q6_K_MIN_PACK_ELEMENTS,
                                            Q8_0_MIN_PACK_ELEMENTS, QK2_0, QK4_0)
from stratashift.quant.pack import (quantize_q2_0_stream, quantize_q3_k_stream,
                                    quantize_q4_0_stream, quantize_q4_k_stream,
                                    quantize_q5_k_stream, quantize_q6_k_stream,
                                    quantize_q8_0_stream)
from stratashift.quant.pack_imatrix import (plane_tile, quantize_q4_0_imatrix_stream,
                                            quantize_q4_k_imatrix_stream)

PLAIN_QUANTIZERS: dict[str, Callable[[bytes], bytes]] = {
    "Q2_0": quantize_q2_0_stream, "Q3_K": quantize_q3_k_stream,
    "Q4_0": quantize_q4_0_stream, "Q4_K": quantize_q4_k_stream,
    "Q5_K": quantize_q5_k_stream, "Q6_K": quantize_q6_k_stream,
    "Q8_0": quantize_q8_0_stream,
}
IMATRIX_QUANTIZERS: dict[str, Callable] = {
    "Q4_0": quantize_q4_0_imatrix_stream, "Q4_K": quantize_q4_k_imatrix_stream,
}

# expert type ladders: first type whose engine load gate passes wins
EXPERT_LADDER = ("Q4_K", "Q4_0")  # docs/work/p5-08.md §design, gguf_reader.hpp:505 pin c566ff4
FUSED_DOWN_LADDER = ("Q4_0",)  # ffn_down_exps row = ff (p6-05: ff % 256 = 192 → Q4_K refused)


# --- dense tensors ---------------------------------------------------------------------

PACK_MIN_ELEMENTS = {"Q4_0": Q4_0_MIN_PACK_ELEMENTS, "Q4_K": Q4_K_MIN_PACK_ELEMENTS,
                     "Q5_K": Q5_K_MIN_PACK_ELEMENTS, "Q6_K": Q6_K_MIN_PACK_ELEMENTS,
                     "Q3_K": Q3_K_MIN_PACK_ELEMENTS, "Q8_0": Q8_0_MIN_PACK_ELEMENTS}


def _pack_accepts(t: str, n_el: int, ndim: int) -> bool:
    """Engine pack gate for a dense type (p3-03~p3-09, verbatim): codes→scales planes
    contiguous at ALIGN=64 (n_el % MIN_PACK_ELEMENTS), and Q8_0 needs ndim ≥ 2
    (pack_index.py:174-186 derives scales from ne1). F32/unknown types are never packed."""
    if t not in PACK_MIN_ELEMENTS:
        return False
    if t == "Q8_0" and ndim < 2:
        return False
    return n_el % PACK_MIN_ELEMENTS[t] == 0


def select_pack_quant(shape, dense_type: str) -> str:
    """Single-file emit (p3-03~p3-09): requested type if the pack gate passes, else F32.
    Q4_0 additionally refuses a tensor that is not whole 32-element blocks (engine
    gguf_reader expected_bytes = elements × 18 / 32) with ValueError."""
    n_el = numel(shape)
    if dense_type == "Q4_0" and n_el % QK4_0:
        raise ValueError(f"Q4_0 emit needs a multiple of {QK4_0} elements, "
                         f"shape {shape} = {n_el}")
    return dense_type if _pack_accepts(dense_type, n_el, len(shape)) else "F32"


def select_dense_quant(n_el: int, dense_type: str, ne0: int, ndim: int,
                       force_f32: bool = False) -> str:
    """p5-08: which quantization type the engine ACCEPTS for this tensor, "F32" =
    pass-through. Engine LOAD gate (p5-07 AC5 root cause): tensor_payload_bytes
    requires shape[0] % be == 0, be from block_geometry (gguf_reader.hpp:505,
    pin c566ff4: Q4_K be=256, Q8_0/Q4_0 be=32, F32 be=1) — `ne0` is the engine
    shape's first dim (HF [out,in] → GGUF (in,out) transpose). Measured refusal:
    ffn_down [2112,2816] Q4_K, 2112 % 256 = 64 → load throw rc=1.
    Ladder (docs/work/p5-08.md §design): requested type if it passes BOTH the
    engine load gate and the pack contiguity gate (`_pack_accepts`); else the be=32
    quantizations (Q8_0 default, Q4_0); else F32 (be=1 always passes). Both gates must
    hold — load gate is shape[0], pack gate is total elements; the p4-03 byte-identical
    contract keeps both in force.
    """
    if force_f32:  # 1-D engine key want1 contract (layout.cpp:88-98) — never quantized
        return "F32"
    if dense_type == "F32":
        return "F32"

    def accepted(t: str) -> bool:
        return ne0 % ENGINE_LOAD_BE[t] == 0 and _pack_accepts(t, n_el, ndim)

    if accepted(dense_type):
        return dense_type
    for t in ("Q8_0", "Q4_0"):  # be=32 ladder, measured full-sweep pass (probe_ne0.py)
        if accepted(t):
            return t
    return "F32"


def supports_imatrix(type_name: str) -> bool:
    return type_name in IMATRIX_QUANTIZERS


def quantize_dense(type_name: str, f32: bytes, weights: np.ndarray | None = None) -> bytes:
    """Quantize an F32 stream: imatrix producer when `weights` is given and the type has
    one (Q4_0/Q4_K), else the plain RTN producer. Callers build `weights` only when
    `supports_imatrix(type_name)`."""
    if weights is not None and type_name in IMATRIX_QUANTIZERS:
        return IMATRIX_QUANTIZERS[type_name](f32, weights)
    return PLAIN_QUANTIZERS[type_name](f32)


# --- expert tensors --------------------------------------------------------------------

@dataclass(frozen=True)
class ExpertQuant:
    """GGUF type name, its producer, and the slice geometry the gate was checked on —
    all fixed together at selection time."""
    type_name: str
    row_len: int   # engine shape[0] = imatrix plane length
    n_elems: int   # elements of one expert slice

    def quantize(self, f32: bytes, plane_row: np.ndarray | None) -> bytes:
        """Quantize one expert slice. `plane_row` (len row_len) is the expert's imatrix
        plane; the Q2_0 fallback ignores it (no plane → plain producer)."""
        if self.type_name == "Q2_0":
            return PLAIN_QUANTIZERS["Q2_0"](f32)
        return IMATRIX_QUANTIZERS[self.type_name](
            f32, plane_tile(plane_row, self.n_elems, self.row_len, 0))


def _expert_load_ok(t: str, row_len: int, n_elems: int) -> bool:
    """Engine load gate (row_len % be, gguf_reader.hpp:505 pin c566ff4) and whole blocks
    per expert slice (n_elems % be — the slice is quantized on its own)."""
    be = ENGINE_LOAD_BE[t]
    return row_len % be == 0 and n_elems % be == 0


def select_expert_quant(row_len: int, n_elems: int, has_plane: bool,
                        ladder: tuple[str, ...] = EXPERT_LADDER) -> ExpertQuant | None:
    """Expert slot type: with an imatrix plane, the first ladder type whose engine load
    gate passes; otherwise (no plane, or no ladder type passes) the Q2_0 plain fallback
    when ITS gate passes (case 42 be=64 = QK2_0). None = no type the engine can load —
    the caller refuses the emit instead of writing an unloadable tensor.

    row_len = engine shape[0] (the plane length), n_elems = elements of one expert slice.
    """
    if has_plane:
        for t in ladder:
            if _expert_load_ok(t, row_len, n_elems):
                return ExpertQuant(t, row_len, n_elems)
    if _expert_load_ok("Q2_0", row_len, n_elems):  # QK2_0 = 64 producer blocks too
        return ExpertQuant("Q2_0", row_len, n_elems)
    return None


def expert_refusal(name: str, row_len: int, n_elems: int) -> str:
    """stderr message when `select_expert_quant` finds no loadable type."""
    return (f"{name}: no engine-loadable expert type (row_len={row_len}, "
            f"slice={n_elems} elements; Q2_0 needs multiples of {QK2_0}) — refusing")
