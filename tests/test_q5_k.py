"""p3-05 AC1: quantize_q5_k_stream — engine dequant round-trip contract.

The engine ships NO Q5_K quantizer (gguf_writer.py is Q2_0-only, dequant.hpp has dequant
only), so the contract is the engine DEQUANT round-trip, not a quantizer byte-match:

    d = fp16(amax/63), dmin = fp16(−min/63), sc_g = clip(ceil((gmax−gmin)/(31*d)),1,63),
    m_g = clip(round(−gmin/dmin),0,63)
    (groups 4..7: m < 16), code = clip(round((w+dmin*m)/(d*sc)),0,31), qh 5th-bit plane

Invariants (docs/work/p3-05.md AC1, evidence-locked 2026-10-16):
  1. q5_k_unpack (to_s8_q5_k+from_s8_q5_k transcription) == engine to_s8_q5_k+from_s8_q5_k
     bit-exact on our blocks;
  2. quantization error |w - v| <= (d*sc_g)/2 per group for groups 0..3 (free 6-bit mins);
  3. q5_k_planes == engine to_s8_q5_k + pack_codes byte-for-byte (codes plane), and the
     scales/offsets planes equal the engine's f32 planes.
The producer rule is a HYPOTHESIS (no engine quantizer); the acceptance gate is the
engine dequant round-trip + strata_pack verify (engine chain harness).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path("/opt/data/polystrata-src/tools")))

from canonical_xcheck import to_s8_q5_k, from_s8_q5_k, get_scale_min_k4  # noqa: E402
from strata_pack import pack_codes  # noqa: E402

from stratashift.meta.engine_manifest import q5_k_planes  # noqa: E402
from stratashift.quant.pack import q5_k_unpack, quantize_q5_k_stream  # noqa: E402

TOL = 1.005  # fp16 scale rounding tolerance on the (d*sc)/2 bound
QK_K = 256


def _rows(n_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n_blocks * QK_K).astype("<f4")


def test_q5_k_roundtrip_engine_bit_exact():
    """AC1-1: q5_k_unpack == engine to_s8_q5_k + from_s8_q5_k, bit-for-bit."""
    for seed in (1, 2, 3, 4):
        w = _rows(16, seed)
        raw = quantize_q5_k_stream(w.tobytes())
        ours = np.array(q5_k_unpack(raw), dtype=np.float32)
        eng = from_s8_q5_k(to_s8_q5_k(raw)).astype(np.float32)
        assert np.array_equal(ours, eng), f"seed {seed}: round-trip diverges"


def test_q5_k_edge_blocks():
    """AC1-1 edges: zero block, tiny amax, amax upper, group-heavy."""
    w = np.zeros(QK_K, dtype="<f4")
    raw = quantize_q5_k_stream(w.tobytes())
    assert np.array_equal(np.array(q5_k_unpack(raw), dtype=np.float32),
                          from_s8_q5_k(to_s8_q5_k(raw)).astype(np.float32))
    w = np.full(QK_K, 1e-6, dtype="<f4")
    raw = quantize_q5_k_stream(w.tobytes())
    assert np.array_equal(np.array(q5_k_unpack(raw), dtype=np.float32),
                          from_s8_q5_k(to_s8_q5_k(raw)).astype(np.float32))
    w = np.full(QK_K, 63.0, dtype="<f4")
    raw = quantize_q5_k_stream(w.tobytes())
    assert np.array_equal(np.array(q5_k_unpack(raw), dtype=np.float32),
                          from_s8_q5_k(to_s8_q5_k(raw)).astype(np.float32))
    w = np.zeros(QK_K, dtype="<f4")
    w[0:32] = 10.0
    raw = quantize_q5_k_stream(w.tobytes())
    assert np.array_equal(np.array(q5_k_unpack(raw), dtype=np.float32),
                          from_s8_q5_k(to_s8_q5_k(raw)).astype(np.float32))


def test_q5_k_error_bound():
    """AC1-2 (producer quality, HYPOTHESIS): |w - v| <= (d*sc_g)/2 per group for
    groups 0..3, where the 6-bit min m is free (m < 64). Groups 4..7 mins live in
    scb[8..11] HIGH 4 bits (m < 16), so a group whose range exceeds 16*31*d cannot
    be covered by the 0..31 codebook with m < 16 — the ggml make_qkx2 joint search
    is NOT reproduced here (engine ships no quantizer), so the bound is asserted
    only for the unconstrained groups. The engine acceptance gate (strata_pack
    verify) checks the round-trip, not this bound.
    """
    w = _rows(16, 7)
    raw = quantize_q5_k_stream(w.tobytes())
    v = np.array(q5_k_unpack(raw), dtype=np.float32)
    b = w.reshape(-1, QK_K)
    d16 = (np.max(np.abs(b), axis=1) / 63.0).astype(np.float16).astype(np.float32)
    groups = b.reshape(-1, 8, 32)
    gmax = np.max(np.abs(groups), axis=2)
    gmin = np.min(groups, axis=2)
    rng = gmax - gmin
    dmin16 = (-np.min(b, axis=1) / 63.0).astype(np.float16).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.clip(np.ceil(rng / (31.0 * d16[:, None])), 1, 63)
        m = np.clip(np.rint(-gmin / dmin16[:, None]), 0, 63)
    m[:, 4:8] = np.clip(m[:, 4:8], 0, 15)
    bound = (d16[:, None] * sc)[:, :, None] * TOL / 2
    err = np.abs(groups - v.reshape(-1, 8, 32))
    assert np.all(err[:, 0:4] <= np.broadcast_to(bound, (16, 8, 32))[:, 0:4] + 1e-6)


def test_q5_k_planes_engine_byte_identical():
    """AC1-3: q5_k_planes codes == pack_codes(to_s8_q5_k codes, 5); scales/offsets == engine f32."""
    w = _rows(8, 11)
    raw = quantize_q5_k_stream(w.tobytes())
    codes_b, scales_b, offsets_b = q5_k_planes(raw)
    eng_codes, eng_scales, eng_offsets = to_s8_q5_k(raw)
    assert codes_b == pack_codes(eng_codes, 5)
    assert scales_b == np.ascontiguousarray(eng_scales.reshape(-1), dtype=np.float32).tobytes()
    assert offsets_b == np.ascontiguousarray(eng_offsets.reshape(-1), dtype=np.float32).tobytes()


def test_q5_k_get_scale_min_k4_matches_engine():
    """AC1-1 support: our get_scale_min_k4 == engine get_scale_min_k4 on random scale bytes."""
    from stratashift.quant.pack import get_scale_min_k4 as ours
    rng = np.random.default_rng(99)
    for _ in range(200):
        scb = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        for j in range(8):
            assert ours(scb, j) == get_scale_min_k4(scb, j), f"j={j}"


def test_q5_k_stream_needs_multiple_of_256():
    w = np.zeros(300, dtype="<f4")
    with pytest.raises(ValueError):
        quantize_q5_k_stream(w.tobytes())
