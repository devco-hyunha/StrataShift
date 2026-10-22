"""imatrix streaming accumulation contract (p6-03 AC2).

Contract source (verbatim transcriptions, never invented):
- llama.cpp `tools/imatrix/imatrix.cpp` (master pin, fetched 2026-10-21) :655-664 —
  per-matrix accumulation `acc[j] += x[j] * x[j]` (values), `act[j] += x[j]`
  (activations), `counts[ex]++` per matrix unit `ex`; :966-988 — emit tensors
  `<name>.in_sum2` (values, shape [ne0, nmat]), `<name>.in_sum` (activations),
  `<name>.counts` (shape [1, nmat]) into the GGUF.
- engine consumer contract (pin c566ff4): `ggml_quantize_requires_imatrix(type)`
  passes `imatrix.data()` to the C++ quantize path
  (`src/kernels/native_expert_parity.cpp:407-409`).
- QuantMoE-Bench arxiv 2406.08155 §3.3: expert usage heuristic
  `normalize(sum_i G(W_l X_i))` — top-k routing counts from calibration data
  give the per-expert usage; low-utilized experts are under-calibrated, so the
  accumulator keeps per-expert `counts` and exposes balance for sampling.

Additivity (imatrix.cpp accumulation loop) is the C1 guarantee: each streamed
forward chunk adds `x*x` rows into the per-tensor planes; nothing accumulates a
full model. The accumulation half is pure arithmetic; the loader half
(`load_imatrix_stats`) is a single-tensor-slice mmap read of the calibration
product GGUF — a stats file, not a model body (C1-safe, p6-03 AC4).
"""
from __future__ import annotations

import numpy as np

# per-tensor stat planes, field names mirror the engine-consumed GGUF tensors
# (`in_sum2`, `in_sum`, `counts` — imatrix.cpp :966-988)
TensorStats = dict[str, np.ndarray]


def new_stats(ne0: int, nmat: int) -> TensorStats:
    """Zeroed stat planes for a tensor with input width ne0 and nmat matrices
    (1 for dense, NE for fused experts)."""
    return {
        "in_sum2": np.zeros((nmat, ne0), dtype=np.float64),
        "in_sum": np.zeros((nmat, ne0), dtype=np.float64),
        "counts": np.zeros(nmat, dtype=np.float64),
    }


def add_activation_rows(stats: TensorStats, mat_idx: int, x: np.ndarray) -> None:
    """Accumulate activation rows x (shape [rows, ne0]) into matrix unit `mat_idx`.

    Verbatim contract (imatrix.cpp:658-663): `counts[ex]++` once per touched
    batch, `acc[j] += x[j]*x[j]`, `act[j] += x[j]` per row.
    """
    rows = np.asarray(x, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] != stats["in_sum2"].shape[1]:
        raise ValueError(f"x width {rows.shape[1]} != ne0 {stats['in_sum2'].shape[1]}")
    stats["counts"][mat_idx] += 1.0
    stats["in_sum2"][mat_idx] += (rows * rows).sum(axis=0)
    stats["in_sum"][mat_idx] += rows.sum(axis=0)


def normalized_imatrix(stats: TensorStats, mat_idx: int) -> np.ndarray:
    """Per-element importance = mean of the accumulated squares for matrix unit
    `mat_idx` (sum2 / count) — the imatrix vector the quantize path consumes.

    Zero-count units return ones, verbatim from the consumer contract
    (`quantize.cpp:210-214`: `count == 0` → `e[j*ne0+i] = 1` — an uncalibrated
    expert is fed uniform weights, never a phantom 0/0)."""
    n = stats["counts"][mat_idx]
    if n == 0:
        return np.ones(stats["in_sum2"].shape[1], dtype=np.float64)
    return stats["in_sum2"][mat_idx] / n


def load_imatrix_stats(path: str) -> dict[str, np.ndarray]:
    """Load a GGUF-format imatrix stats file into normalized per-matrix vectors.

    Transcribed consumer contract (llama.cpp master pin, fetched 2026-10-21):
    - `common/imatrix-loader.cpp:121-142` — stats keyed by tensor name from the
      `<name>.in_sum2` / `<name>.counts` (and optional `<name>.in_sum`) triple.
    - `tools/quantize/quantize.cpp:204-215` — per matrix unit j with count c:
      `count > 0` → `e[j*ne0 + i] = sums[j*ne0 + i] / count`; `count == 0` → `e = 1`
      (uncalibrated expert is fed uniform weights).
    The GGUF imatrix tensor is shape [ne0, nmat] (imatrix.cpp:972); ggml stores
    element (i, j) at `data[i + j*ne0]` (matrix-major). Returned planes are
    shape (nmat, ne0) with row j = unit j vector.
    C1-safe: reads a stats file header + per-tensor f32 planes (the file is the
    calibration product, not a model body).
    """
    from stratashift.ingest.gguf import parse_gguf_header

    m = parse_gguf_header(path)
    stats: dict[str, np.ndarray] = {}
    by_base: dict[str, dict[str, list]] = {}
    for t in m.tensors:
        if t.name.endswith(".in_sum2"):
            base = t.name[: -len(".in_sum2")]
            by_base.setdefault(base, {})["sum2"] = t
        elif t.name.endswith(".counts"):
            base = t.name[: -len(".counts")]
            by_base.setdefault(base, {})["counts"] = t
    for base, parts in sorted(by_base.items()):
        ts, tc = parts.get("sum2"), parts.get("counts")
        if ts is None or tc is None:
            continue
        # producer shapes (imatrix.cpp:972-973): in_sum2 = 2-D [nval/nmat, nmat],
        # counts = 2-D [1, nmat]. GGUF v3 strips trailing dims of size 1, so a
        # dense tensor (nmat = 1) lands in the header as in_sum2 ndim=1 (ne0,),
        # counts ndim=1 (1,) — VERIFIED on the producer product 2026-10-21
        # (`blk.0.attn_k.weight.in_sum2` header shape (2816,)).
        nmat = int(tc.shape[-1])
        ne0 = ts.shape[0]
        if len(ts.shape) == 1:
            if nmat != 1:
                raise ValueError(f"{base}: counts unit {nmat} > 1 with 1-D in_sum2 "
                                 f"shape {ts.shape} — fused plane must be 2-D")
        elif ts.shape[1] != nmat:
            raise ValueError(f"{base}: in_sum2 shape {ts.shape} != counts unit {nmat}")
        nbytes = ne0 * nmat * 4
        raw = ts.expected_bytes()
        if raw != nbytes:
            raise ValueError(f"{base}: in_sum2 bytes {raw} != ne0*nmat*4 {nbytes}")
        # producer layout (imatrix.cpp pin 3cf03257 :325, :616/:621-622, :498;
        # quantize.cpp:208): header shape [ne0, nmat], ggml ne0-fastest → element
        # (i, j) at flat i + j*ne0 (matrix-major). Unit j vector = contiguous
        # [j*ne0, j*ne0+ne0) → reshape(nmat, ne0) with no transpose. Dense nmat==1
        # is identical under either reshape.
        data = np.frombuffer(_tensor_bytes(path, ts, nbytes, m.data_start),
                             dtype="<f4").reshape(nmat, ne0)
        counts = np.frombuffer(_tensor_bytes(path, tc, nmat * 4, m.data_start), dtype="<f4")
        vecs = np.where(counts[:, None] > 0, data / np.where(counts[:, None] > 0, counts[:, None], 1.0), 1.0)
        stats[base] = vecs
    return stats


def _tensor_bytes(path: str, tensor, nbytes: int, data_start: int) -> bytes:
    """Slice one tensor payload from the stats file via mmap (single-slice read of one
    tensor, C1-safe)."""
    import mmap

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            start = data_start + tensor.offset
            return mm[start:start + nbytes]
        finally:
            mm.close()


def expert_usage(counts: np.ndarray) -> np.ndarray:
    """QuantMoE-Bench §3.3 heuristic transcribed: normalize routing/touch counts
    over experts — low-usage experts get a low weight and need balanced sampling
    to be calibrated at all."""
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    return counts / total
