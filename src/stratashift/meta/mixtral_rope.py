"""Mixtral q/k row order for the engine_ext rotary kernel (p6-01 AC2).

HF Mixtral turns rotary pairs (i, i + d/2) of a head (half-split). The engine_ext `Mixtral` class turns them with
the engine's `dsv4_rope` without a norm, which pairs (i, i + 1) — ggml's normal rope (pin c566ff4
`include/strata/poly/kernels.hpp:219-232`, `src/poly/kernels_dsv4.cu:111-137`); the engine's NeoX kernel
`head_norm_rope` always RMS-norms the head (`kernels.cu:933-948`), which Mixtral does not. So the emit reorders the
rows of `q_proj` and `k_proj` within each head: new row 2j is old row j, new row 2j+1 is old row j + d/2 — the
order llama.cpp's converter gives its llama-family GGUFs. The same order on q and k leaves every q·k, and so the
attention, as it is; v and o are untouched.
"""
from __future__ import annotations

import numpy as np


def rope_head_dim(guard_keys: dict, prefix: str) -> int:
    """The head width the rows are reordered by: the guard's attention.key_length, which the class reads
    (engine_ext mixtral.cpp load_geometry). Refuses a guard that rotates only part of a head — the class and
    this order assume rope.dimension_count == key_length (never invented)."""
    head_dim = guard_keys.get(f"{prefix}.attention.key_length")
    if not isinstance(head_dim, int) or head_dim <= 0 or head_dim % 2:
        raise ValueError(f"guard {prefix}.attention.key_length must be an even positive int, got {head_dim!r}")
    rot = guard_keys.get(f"{prefix}.rope.dimension_count", head_dim)
    if rot != head_dim:
        raise ValueError(f"guard {prefix}.rope.dimension_count {rot} != attention.key_length {head_dim}: "
                         f"a partial rotary is not what the Mixtral emit reorders for")
    return head_dim


def rope_row_order(rows: int, head_dim: int) -> np.ndarray:
    """Old row index for each new row of a [rows, in] q or k matrix (rows = heads · head_dim)."""
    if rows % head_dim:
        raise ValueError(f"{rows} rows are not whole heads of {head_dim}")
    half = head_dim // 2
    j = np.arange(half)
    one = np.empty(head_dim, dtype=np.int64)
    one[0::2] = j
    one[1::2] = j + half
    return (np.arange(rows // head_dim)[:, None] * head_dim + one[None, :]).reshape(-1)


def reorder_rows(raw: bytes, order: np.ndarray, row_bytes: int) -> bytes:
    """`raw` (rows of row_bytes each) with its rows in `order`."""
    m = np.frombuffer(raw, dtype=np.uint8).reshape(len(order), row_bytes)
    return m[order].tobytes()
