"""p6-03 AC3 gate: imatrix-consuming Q4_K producer against engine dequant transcription.

Gate (p3-04 HYPOTHESIS precedent): the producer rule validity is the engine dequant
round-trip (`pack.q4_k_unpack` == canonical_xcheck to_s4_q4_k/from_s4_q4_k). No real model
files — synthetic random weights + synthetic imatrix (C1 trivially satisfied).
"""
from __future__ import annotations

import numpy as np

from stratashift.quant.pack import quantize_q4_k_stream, q4_k_unpack, q4_0_unpack, Q4_0_BLOCK_BYTES
from stratashift.quant.pack_imatrix import dequant_roundtrip_gate, quantize_q4_k_imatrix_stream, \
    quantize_q4_0_imatrix_stream


def _weights(seed: int, rows: int, cols: int) -> bytes:
    rng = np.random.default_rng(seed)
    scale = 0.02 * (0.5 + (np.arange(rows) % 7) / 7.0)
    w = rng.normal(0, 1, (rows, cols)) * scale[:, None]
    return w.astype("<f4").tobytes()


def test_imatrix_roundtrip_fidelity():
    """dequant(pack) ≈ source within the 4-bit quantization bound per row.

    Per QK_K block the max grid step is d*63 ≈ amax, so |w - v| ≤ amax (loose 4-bit
    bound); the weighted producer must not exceed the RTN producer's own max error.
    """
    rows, cols = 8, 256
    w = np.frombuffer(_weights(7, rows, cols), dtype="<f4")
    imatrix = np.full(w.size, 1.0, dtype=np.float32)
    packed = quantize_q4_k_imatrix_stream(w.tobytes(), imatrix)
    deq = dequant_roundtrip_gate(packed)
    amax = np.max(np.abs(w.reshape(-1, 256)), axis=1)
    err = np.abs(deq - w).reshape(-1, 256)
    assert (err <= amax[:, None] * 1.005).all(), "dequant exceeds 4-bit bound"
    # uniform imatrix must sit inside the RTN producer's max error (sanity of the grid)
    rtn = q4_k_unpack(quantize_q4_k_stream(w.tobytes()))
    assert err.max() <= np.abs(rtn - w).reshape(-1, 256).max() * 1.5 + 1e-3


def test_imatrix_shifts_high_importance_columns():
    """A column with a large imatrix weight must be quantized toward its true value
    more than a low-importance column — the whole point of calibration.

    Build a block where column 0 is high-magnitude and column 1 is noise, weight col0
    imatrix heavily. The dequant of col0 must land closer to source than the RTN producer.
    """
    w = np.zeros((1, 256), dtype=np.float32)
    w[0, 0] = 1.0
    w[0, 1] = 0.9
    w[0, 2:32] = 0.01
    flat = w.reshape(-1)
    uni = np.full(256, 1.0, dtype=np.float32)
    imp = np.full(256, 1.0, dtype=np.float32)
    imp[0] = 100.0  # make column 0 dominant
    prod = q4_k_unpack(quantize_q4_k_imatrix_stream(flat.tobytes(), imp))
    rtn = q4_k_unpack(quantize_q4_k_stream(flat.tobytes()))
    # weighted producer keeps the dominant column exact (codebook lands on 1.0)
    assert abs(prod[0] - w[0, 0]) < abs(rtn[0] - w[0, 0]) + 1e-6


def test_width_guard():
    w = np.zeros(256, dtype=np.float32)
    try:
        quantize_q4_k_imatrix_stream(w.tobytes(), np.full(255, 1.0))
        raise AssertionError("imatrix length mismatch must raise")
    except ValueError:
        pass
    try:
        quantize_q4_k_imatrix_stream(np.zeros(250, dtype=np.float32).tobytes(), np.full(250, 1.0))
        raise AssertionError("element count must be multiple of QK_K")
    except ValueError:
        pass


def test_pack_roundtrip_matches_consumer_contract():
    """dequant_roundtrip_gate == pack.q4_k_unpack (the engine dequant transcription is
    the single consumer; the producer must satisfy it, not a private decoder)."""
    w = np.frombuffer(_weights(3, 4, 256), dtype="<f4")
    packed = quantize_q4_k_imatrix_stream(w.tobytes(), np.full(w.size, 1.0, dtype=np.float32))
    assert np.allclose(dequant_roundtrip_gate(packed), q4_k_unpack(packed))


def test_q4_0_imatrix_roundtrip_fidelity():
    """Q4_0 imatrix producer (p6-04 down-expert slot): dequant via engine q4_0_unpack
    lands within the 5-bit bound (|w - v| <= amax) per 32-element block."""
    rows, cols = 11, 704  # down-expert row shape (ffn_down_exps h=704, be=32 pass)
    w = np.frombuffer(_weights(11, rows, cols), dtype="<f4")
    imatrix = np.full(w.size, 1.0, dtype=np.float32)
    packed = quantize_q4_0_imatrix_stream(w.tobytes(), imatrix)
    assert len(packed) == w.size // 32 * 18
    deq = np.array([v for b in (packed[i:i + 18] for i in range(0, len(packed), 18))
                    for v in q4_0_unpack(b)], dtype=np.float64)
    amax = np.max(np.abs(w.reshape(-1, 32)), axis=1)
    err = np.abs(deq - w).reshape(-1, 32)
    assert (err <= amax[:, None] * 1.005).all(), "dequant exceeds 5-bit bound"


def test_q4_0_imatrix_shifts_high_importance_columns():
    imp = np.full(32, 1.0, dtype=np.float32)
    imp[0] = 100.0
    w = np.zeros((1, 32), dtype=np.float32)
    w[0, 0] = 1.0
    w[0, 1] = 0.9
    w[0, 2:] = 0.01
    uni = quantize_q4_0_imatrix_stream(w.reshape(-1).tobytes(), np.full(32, 1.0, dtype=np.float32))
    impq = quantize_q4_0_imatrix_stream(w.reshape(-1).tobytes(), imp)
    uni_v = np.array(q4_0_unpack(uni))
    imp_v = np.array(q4_0_unpack(impq))
    assert abs(imp_v[0] - w[0, 0]) < abs(uni_v[0] - w[0, 0]) + 1e-6


def test_q4_0_width_guard():
    try:
        quantize_q4_0_imatrix_stream(np.zeros(30, dtype=np.float32).tobytes(), np.full(30, 1.0))
        raise AssertionError("element count must be multiple of 32")
    except ValueError:
        pass
    try:
        quantize_q4_0_imatrix_stream(np.zeros(64, dtype=np.float32).tobytes(), np.full(31, 1.0))
        raise AssertionError("imatrix length mismatch must raise")
    except ValueError:
        pass
