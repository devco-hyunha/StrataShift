"""imatrix-consuming Q4_K producer (p6-03 AC3) — HYPOTHESIS, dequant round-trip gated.

Contract sources (verbatim, never invented):
- llama.cpp `ggml-quants.c` master pin (fetched 2026-10-21, 5,638 lines):
  `quantize_q4_K` (:1627) — `quant_weights` present → `quantize_row_q4_K_impl` per row;
  `quantize_row_q4_K_impl` (:1554-1627) — per 256-element block:
    `sum_x2 = Σ x[l]²` over QK_K; `sigma2 = 2*sum_x2/QK_K`;
    `weights[l] = qw[QK_K*i + 32*j + l] * sqrt(sigma2 + x[32*j+l]²)` (the imatrix branch,
    :1577), `sw[j] = Σ weights` (:1583), `scales[j], mins[j] =
    make_qkx3_quants(32, 15, x, weights, -0.9, 0.05, 36, false)` (:1584),
    `d_block = make_qp_quants(QK_K/32, 63, scales, sw)`, `m_block =
    make_qp_quants(QK_K/32, 63, mins, sw)` (:1588-1589), scale-byte packing identical to
    the `get_scale_min_k4` contract (docs/rea/bitpack_rules.md §6 — the same bytes as the
    plain RTN producer in pack.py), code `l = clip(nearest_int((x + dm)/d), 0, 15)`
    (:1610-1613), qs low/high nibble pair packing (:1616-1618).
  `nearest_int` (:621-626) — round-to-nearest ties-to-even (the fp trick is rint).
  `make_qkx3_quants` (:993-1075) — weighted MAD grid search over `iscale` sweep
  `rmin + rdelta*is`, closed-form `(this_scale, this_min)` from the weighted sums.
  `make_qp_quants` (:1077-1148) — weighted-MSE `iscale` sweep `-4..4` then 5 weighted
  sumlx/suml2 improvement passes; returns `sumlx/suml2` (scale, min).
- engine consumer contract (pin c566ff4, native_expert_parity.cpp:407-409): the C++ path
  hands `imatrix.data()` to `ggml_quantize_chunk` when `ggml_quantize_requires_imatrix`
  — the imatrix vector is per-column over `cols`, one float per element of the row.
- imatrix vector: `meta/imatrix.py` `normalized_imatrix` (mean of accumulated column
  squares, imatrix.cpp :655-664 contract).

The producer rule is HYPOTHESIS (p3-04 precedent): the engine ships no Python quantizer,
so validity gate = engine dequant round-trip (`q4_k_unpack` == canonical_xcheck
transcription) + quality gate in AC5/AC6. Plain RTN golden bytes in pack.py are untouched.
Vectorized per 2^20-element segment (peak = segment, C1/C2).
"""
from __future__ import annotations

import numpy as np

from stratashift.polystrata.headers import Q4_K_BLOCK_BYTES, QK_K
from stratashift.quant.pack import _FP16, q4_k_unpack

RMIN, RDELTA, NSTEP = -0.9, 0.05, 36
QP_SWEEP = range(-4, 5)
QP_ITRY = 5
GROUP_MAX_EPS = 1e-15


def _rint(v: np.ndarray) -> np.ndarray:
    """nearest_int (ggml-quants.c:621-626): round to nearest, ties-to-even."""
    return np.rint(v)


def make_qkx3_quants(x: np.ndarray, weights: np.ndarray, nmax: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Transcribed make_qkx3_quants (ggml-quants.c:993-1075), vectorized over groups.

    x, weights: leading dims + group axis n. Returns (scale, the_min) with the
    leading dims. `use_mad=false` (the Q4_K call site) → MAD term is the squared diff.
    """
    lead, n = x.shape[:-1], x.shape[-1]
    x = x.reshape(-1, n)
    weights = weights.reshape(-1, n)
    mn = x.min(axis=1)
    mx = x.max(axis=1)
    mn = np.minimum(mn, 0.0)  # `if (min > 0) min = 0`
    sum_w = weights.sum(axis=1)
    sum_x = (weights * x).sum(axis=1)
    deg = mx <= mn  # all-equal: L=0, the_min = -min, scale 0
    rng = np.where(deg, 1.0, mx - mn)
    iscale = np.where(deg, 0.0, nmax / rng)
    scale = np.where(deg, 0.0, 1.0 / np.where(deg, 1.0, iscale))
    L = np.clip(_rint(iscale[:, None] * (x - mn[:, None])), 0, nmax)
    diff = scale[:, None] * L + mn[:, None] - x
    best_mad = (weights * diff * diff).sum(axis=1)
    L = np.where(deg[:, None], 0, L)
    best_mad = np.where(deg, 0.0, best_mad)
    for is_idx in range(NSTEP + 1):
        is_is = (RMIN + RDELTA * is_idx + nmax) / rng
        v = is_is[:, None] * (x - mn[:, None])
        laux = np.where(v > 0, np.clip(_rint(np.minimum(v, nmax)), 0, nmax), 0)
        w = weights
        sum_l = (w * laux).sum(axis=1)
        sum_l2 = (w * laux * laux).sum(axis=1)
        sum_xl = (w * laux * x).sum(axis=1)
        D = sum_w * sum_l2 - sum_l * sum_l
        ok = D > 0
        this_scale = np.where(ok, (sum_w * sum_xl - sum_x * sum_l) / np.where(ok, D, 1.0), scale)
        this_min = np.where(ok, (sum_l2 * sum_x - sum_l * sum_xl) / np.where(ok, D, 1.0), mn)
        flip = ok & (this_min > 0)  # `if (this_min > 0) { min=0; scale=sum_xl/sum_l2 }`
        this_scale = np.where(flip, np.where(sum_l2 > 0, sum_xl / np.where(sum_l2 > 0, sum_l2, 1.0), 0.0), this_scale)
        this_min = np.where(flip, 0.0, this_min)
        diff = this_scale[:, None] * laux + this_min[:, None] - x
        mad = (weights * diff * diff).sum(axis=1)
        win = ok & (mad < best_mad)
        L = np.where(win[:, None], laux, L)
        best_mad = np.where(win, mad, best_mad)
        scale = np.where(win, this_scale, scale)
        mn = np.where(win, this_min, mn)
    return scale.reshape(lead), (-mn).reshape(lead)


def make_qp_quants(x: np.ndarray, weights: np.ndarray, nmax: int = 63) -> tuple[np.ndarray, np.ndarray]:
    """Transcribed make_qp_quants (ggml-quants.c:1077-1148), vectorized over groups.

    Returns (L, scale) per group. Sweeps `is = -4..4` on the weighted MSE, then 5
    sumlx/suml2 improvement passes, returns `sumlx/suml2`.
    """
    g, n = x.shape
    mx = x.max(axis=1)
    allzero = mx < GROUP_MAX_EPS
    rng = np.where(allzero, 1.0, mx)
    iscale = np.where(allzero, 0.0, nmax / rng)
    L = np.where(allzero[:, None], 0, np.clip(_rint(iscale[:, None] * x), 0, nmax).astype(np.int32))
    scale = np.where(allzero, 0.0, 1.0 / np.where(allzero, 1.0, iscale))
    best_mse = (weights * (x - scale[:, None] * L) ** 2).sum(axis=1)
    for is_idx in QP_SWEEP:
        if is_idx == 0:
            continue
        is_is = (0.1 * is_idx + nmax) / rng
        scale_is = 1.0 / is_is
        l_is = np.minimum(_rint(is_is[:, None] * x), nmax).astype(np.int32)
        mse = (weights * (x - scale_is[:, None] * l_is) ** 2).sum(axis=1)
        win = mse < best_mse
        L = np.where(win[:, None], l_is, L)
        best_mse = np.where(win, mse, best_mse)
        iscale = np.where(win, is_is, iscale)
    sumlx = (weights * x * L).sum(axis=1)
    suml2 = (weights * L * L).sum(axis=1)
    for _ in range(QP_ITRY):
        n_changed = np.zeros(g, dtype=bool)
        for i in range(n):
            w = weights[:, i]
            xi = x[:, i]
            li = L[:, i]
            slx = sumlx - w * xi * li
            sl2 = suml2 - w * li * li
            ok = (slx > 0) & (sl2 > 0)
            new_l = np.clip(_rint(xi * sl2 / np.where(sl2 > 0, sl2, 1.0)), 0, nmax).astype(np.int32)
            cand = ok & (new_l != li)
            slx2 = slx + w * xi * new_l
            sl22 = sl2 + w * new_l * new_l
            gain = (slx2 * slx2 * suml2) > (sumlx * sumlx * sl22)
            take = cand & gain
            L[:, i] = np.where(take, new_l, li)
            sumlx = np.where(take, slx2, sumlx)
            suml2 = np.where(take, sl22, suml2)
            n_changed |= take
        if not n_changed.any():
            break
    out_scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1.0), 0.0)
    return L, out_scale


def quantize_q4_k_imatrix_stream(values: bytes, imatrix: np.ndarray) -> bytes:
    """float32 stream + per-element imatrix (len == values count) → Q4_K blocks.

    `quantize_row_q4_K_impl` transcription (imatrix branch), rows = contiguous
    QK_K-element rows of the tensor. imatrix is the normalized column importance
    vector (`meta/imatrix.py`), broadcast per row element position.
    """
    w = np.frombuffer(values, dtype="<f4")
    qw = np.asarray(imatrix, dtype=np.float32)
    if w.size % QK_K:
        raise ValueError(f"Q4_K needs a multiple of {QK_K} elements, got {w.size}")
    if qw.shape[0] != w.size:
        raise ValueError(f"imatrix length {qw.shape[0]} != element count {w.size}")
    out = bytearray()
    seg = 1 << 20
    for start in range(0, w.size, seg):
        b = w[start:start + seg].reshape(-1, QK_K)
        qw_b = qw[start:start + seg].reshape(-1, QK_K)
        sum_x2 = (b * b).sum(axis=1)
        sigma2 = 2.0 * sum_x2 / QK_K
        groups = b.reshape(-1, 8, 32)
        qw_g = qw_b.reshape(-1, 8, 32)
        weights = qw_g * np.sqrt(sigma2[:, None, None] + groups * groups)
        sw = weights.sum(axis=2)
        scales, mins = make_qkx3_quants(groups, weights, nmax=15)
        Ls, d_block = make_qp_quants(scales, sw, nmax=63)
        Lm, m_block = make_qp_quants(mins, sw, nmax=63)
        d16 = d_block.astype(np.float16)
        dm16 = m_block.astype(np.float16)
        d32 = d16.astype(np.float32)
        dm32 = dm16.astype(np.float32)
        # scale-byte packing: the same get_scale_min_k4 contract as the RTN producer
        scb = np.zeros((b.shape[0], 12), dtype=np.uint8)
        for j in range(4):
            scb[:, j] = (Ls[:, j] | ((Ls[:, j + 4] >> 4) << 6)).astype(np.uint8)
            scb[:, j + 4] = Lm[:, j].astype(np.uint8)
            scb[:, j + 8] = ((Ls[:, j + 4] & 0xF) | (Lm[:, j + 4] << 4)).astype(np.uint8)
        code = np.zeros((b.shape[0], QK_K), dtype=np.uint8)
        for j in range(8):
            sc = get_scale_min_q4k(scb, j)
            d = d32[:, None] * sc[:, None]
            m = get_min_q4k(scb, j)
            dm = dm32[:, None] * m[:, None]
            seg32 = groups[:, j, :]
            l = np.clip(_rint((seg32 + dm) / np.where(d > 0, d, 1.0)), 0, 15)
            code[:, j * 32:(j + 1) * 32] = np.where(d > 0, l, 0).astype(np.uint8)
        qs = np.zeros((b.shape[0], 128), dtype=np.uint8)
        for pair in range(4):
            lo = code[:, pair * 64:pair * 64 + 32]
            hi = code[:, pair * 64 + 32:pair * 64 + 64]
            qs[:, pair * 32:pair * 32 + 32] = lo | (hi << 4)
        dd = d16.view(np.uint16)
        dm = dm16.view(np.uint16)
        inter = np.empty((b.shape[0], Q4_K_BLOCK_BYTES), dtype=np.uint8)
        inter[:, 0] = (dd & 0xFF).astype(np.uint8)
        inter[:, 1] = ((dd >> 8) & 0xFF).astype(np.uint8)
        inter[:, 2] = (dm & 0xFF).astype(np.uint8)
        inter[:, 3] = ((dm >> 8) & 0xFF).astype(np.uint8)
        inter[:, 4:16] = scb
        inter[:, 16:] = qs
        out += inter.tobytes()
    return bytes(out)


def get_scale_min_q4k(scb: np.ndarray, j: int) -> np.ndarray:
    """Vectorized get_scale_min_k4 (bitpack_rules §6), scale part, j in 0..7."""
    if j < 4:
        return (scb[:, j] & 63).astype(np.float32)
    return (((scb[:, j + 4] & 0xF) | ((scb[:, j - 4] >> 6) << 4))).astype(np.float32)


def get_min_q4k(scb: np.ndarray, j: int) -> np.ndarray:
    """Vectorized get_scale_min_k4, min part (transcription parity with pack.get_scale_min_k4)."""
    if j < 4:
        return (scb[:, j + 4] & 63).astype(np.float32)
    return (((scb[:, j + 4] >> 4) | ((scb[:, j] >> 6) << 4))).astype(np.float32)


def dequant_roundtrip_gate(block_bytes: bytes) -> np.ndarray:
    """Engine dequant transcription consumer (pack.q4_k_unpack) — the AC3 gate."""
    return q4_k_unpack(block_bytes)


def plane_tile(plane_row: np.ndarray, n_el: int, row_len: int, offset: int) -> np.ndarray:
    """Slice of the broadcast plane vector over stream window [offset, offset+n_el).

    Element g of the stream uses plane_row[g % row_len] (the broadcast contract).
    payload_chunks uses 2^19-element chunks; the measured plane row_lens (2816,
    704, 2112, 4096) do NOT divide 2^19 (measured: 2^19 % 2816 = 512 남), so a
    chunk can start mid-row. The window is the tile rolled to `offset % row_len`.
    """
    v = np.asarray(plane_row, dtype=np.float32)
    if v.shape[0] != row_len:
        raise ValueError(f"plane row len {v.shape[0]} != row_len {row_len}")
    start = offset % row_len
    reps = (start + n_el + row_len - 1) // row_len
    return np.tile(v, reps)[start:start + n_el]


def broadcast_imatrix(imatrix_vector: np.ndarray, n_el: int, row_len: int) -> np.ndarray:
    """Broadcast a per-input-column imatrix vector over a row-major stream.

    A dense HF tensor is stored [out, in] row-major; the imatrix vector is over the
    input dimension `in` (the quantizer's `cols`, engine native_expert_parity.cpp:407
    — one float per element position within a row). Element r*row_len + c uses
    imatrix[c], so the vector tiles across the full stream. (Fused-expert path is
    per-expert unit; this is the dense path. HYPOTHESIS AC3.)
    """
    v = np.asarray(imatrix_vector, dtype=np.float32)
    if v.shape[0] != row_len:
        raise ValueError(f"imatrix vector len {v.shape[0]} != row_len {row_len}")
    reps = n_el // row_len
    if reps * row_len != n_el:
        raise ValueError(f"n_el {n_el} not a multiple of row_len {row_len}")
    return np.tile(v, reps)
