"""p3-06 AC1: quantize_q6_k_stream — engine dequant round-trip contract.

The engine ships NO Q6_K quantizer (gguf_writer.py is Q2_0-only, dequant.hpp has dequant
only), so the contract is the engine DEQUANT round-trip, not a quantizer byte-match:

    d = fp16(amax/63), sc_g = clip(round(gmax_g/d), -63, 63) (signed),
    code = clip(round(w/(d*sc_g)) + 32, 0, 63), sc_g == 0 -> code = 32

Invariants (docs/work/p3-06.md AC1, evidence-locked 2026-10-16):
  1. q6_k_unpack (to_s8_q6_K+from_s8_q6_K transcription) == engine to_s8_q6_K+from_s8_q6_k
     bit-exact on our blocks;
  2. quantization error |w - v| <= (d*|sc_g|)/2 per group for groups with sc_g != 0
     (signed sc, no offset plane, no min-bit constraint — all 16 groups free);
  3. q6_k_planes == engine to_s8_q6_K + pack_codes byte-for-byte (codes plane 1 B/elem),
     and the scales plane equals the engine's f32 plane (no offsets plane).
The producer rule is a HYPOTHESIS (no engine quantizer); the acceptance gate is the
engine dequant round-trip + strata_pack verify (engine chain harness).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path("/opt/data/polystrata-src/tools")))

from canonical_xcheck import to_s8_q6_K, from_s8_q6_K  # noqa: E402
from strata_pack import pack_codes  # noqa: E402

from stratashift.meta.engine_manifest import q6_k_planes  # noqa: E402
from stratashift.quant.pack import q6_k_unpack, quantize_q6_k_stream  # noqa: E402

TOL = 1.005  # fp16 scale rounding tolerance on the (d*|sc|)/2 bound
QK_K = 256


def _rows(n_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n_blocks * QK_K).astype("<f4")


def test_q6_k_roundtrip_engine_bit_exact():
    """AC1-1: q6_k_unpack == engine to_s8_q6_K + from_s8_q6_K, bit-for-bit."""
    for seed in (1, 2, 3, 4):
        w = _rows(16, seed)
        raw = quantize_q6_k_stream(w.tobytes())
        ours = np.array(q6_k_unpack(raw), dtype=np.float32)
        eng = from_s8_q6_K(to_s8_q6_K(raw)).astype(np.float32)
        assert np.array_equal(ours, eng), f"seed {seed}: round-trip diverges"


def test_q6_k_edge_blocks():
    """AC1-1 edges: zero block, tiny amax, amax upper, negative-heavy (signed sc)."""
    cases = [np.zeros(QK_K, dtype="<f4"),
            np.full(QK_K, 1e-6, dtype="<f4"),
            np.full(QK_K, 63.0, dtype="<f4"),
            np.full(QK_K, -63.0, dtype="<f4"),
            np.full(QK_K, -1e-6, dtype="<f4")]
    w = np.zeros(QK_K, dtype="<f4")
    w[0:16] = 10.0
    cases.append(w)
    for w in cases:
        raw = quantize_q6_k_stream(w.tobytes())
        assert np.array_equal(np.array(q6_k_unpack(raw), dtype=np.float32),
                              from_s8_q6_K(to_s8_q6_K(raw)).astype(np.float32))


def test_q6_k_signed_scale_preserved():
    """AC1-1 support: all-negative groups keep a negative int8 sc (sign flip)."""
    w = np.full(QK_K, -31.5, dtype="<f4")
    raw = quantize_q6_k_stream(w.tobytes())
    sc = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)[:, 192:208].view(np.int8)
    assert np.all(sc < 0)
    v = np.array(q6_k_unpack(raw), dtype=np.float32)
    assert np.all(v < 0)


def test_q6_k_error_bound():
    """AC1-2 (producer quality, HYPOTHESIS): |w - v| <= (d*|sc_g|)/2 per group for
    groups whose codebook fully covers the range (no code clipping: 0 < min code and
    max code < 63). Signed sc, no offset plane, no min-bit packing — all 16 groups
    are free (unlike Q4_K/Q5_K groups 4..7), but a group whose range exceeds the
    0..63 codebook coverage clips at 0/63 and cannot meet the bound; the ggml
    make_qxq5_k-style joint search is NOT reproduced here (engine ships no quantizer),
    so the bound is asserted only for the covered groups. Groups with sc_g == 0
    quantize to v = 0. The engine acceptance gate (strata_pack verify) checks the
    round-trip, not this bound.
    """
    w = _rows(16, 7)
    raw = quantize_q6_k_stream(w.tobytes())
    v = np.array(q6_k_unpack(raw), dtype=np.float32)
    b = w.reshape(-1, QK_K)
    d16 = (np.max(np.abs(b), axis=1) / 63.0).astype(np.float16).astype(np.float32)
    groups = b.reshape(-1, 16, 16)
    gmax = groups.max(axis=2)
    sc = np.clip(np.rint(gmax / d16[:, None]), -63, 63)
    scale = d16[:, None] * sc.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        code = np.rint(groups / scale[:, :, None]) + 32
    code = np.where(scale[:, :, None] == 0, 32.0, code)
    code = np.clip(code, 0, 63)
    covered = ((code.min(axis=2) > 0) & (code.max(axis=2) < 63) & (sc != 0))[:, :, None]
    bound = np.abs(d16[:, None] * sc)[:, :, None] * TOL / 2
    err = np.abs(groups - v.reshape(-1, 16, 16))
    assert np.all(np.where(covered, err, 0) <= np.broadcast_to(bound, (16, 16, 16)) + 1e-6)


def test_q6_k_planes_engine_byte_identical():
    """AC1-3: q6_k_planes codes == pack_codes(to_s8_q6_K codes, 8); scales == engine f32."""
    w = _rows(8, 11)
    raw = quantize_q6_k_stream(w.tobytes())
    codes_b, scales_b = q6_k_planes(raw)
    eng_codes, eng_scales = to_s8_q6_K(raw)
    assert codes_b == pack_codes(eng_codes, 8)
    assert scales_b == np.ascontiguousarray(eng_scales.reshape(-1), dtype=np.float32).tobytes()


def test_q6_k_stream_needs_multiple_of_256():
    w = np.zeros(300, dtype="<f4")
    with pytest.raises(ValueError):
        quantize_q6_k_stream(w.tobytes())
