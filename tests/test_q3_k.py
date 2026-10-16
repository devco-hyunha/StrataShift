"""p3-07 AC1: quantize_q3_k_stream — engine dequant round-trip contract.

The engine ships NO Q3_K quantizer, so the contract is the engine DEQUANT round-trip:

    d = fp16(amax/63), sc6_g = clip(round(gmax_g/d) + 32, 0, 63) (6-bit, bias -32),
    dl_g = d*(sc6_g - 32), code = clip(round(w/dl_g) + 4, 0, 7), dl_g == 0 -> code = 4

Invariants (docs/work/p3-07.md AC1, evidence-locked 2026-10-16):
  1. q3_k_unpack (to_s4_q3_k+from_s4_q3_k transcription) == engine to_s4_q3_k+from_s4_q3_k
     bit-exact on our blocks;
  2. quantization error |w - v| <= |dl_g|/2 per group for groups with dl_g != 0 and full
     codebook coverage (no code clipping at 0/7);
  3. q3_k_planes == engine to_s4_q3_k + pack_codes byte-for-byte (codes plane 4-bit,
     per=2 LSB-first), and the scales plane equals the engine's f32 plane (no offsets).
The producer rule is a HYPOTHESIS (no engine quantizer); the acceptance gate is the
engine dequant round-trip + strata_pack verify (engine chain harness).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path("/opt/data/polystrata-src/tools")))

from canonical_xcheck import to_s4_q3_k, from_s4_q3_k  # noqa: E402
from strata_pack import pack_codes  # noqa: E402

from stratashift.meta.engine_manifest import q3_k_planes  # noqa: E402
from stratashift.quant.pack import q3_k_unpack, quantize_q3_k_stream  # noqa: E402

TOL = 1.005  # fp16 scale rounding tolerance on the |dl_g|/2 bound
QK_K = 256


def _rows(n_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n_blocks * QK_K).astype("<f4")


def test_q3_k_roundtrip_engine_bit_exact():
    """AC1-1: q3_k_unpack == engine to_s4_q3_k + from_s4_q3_k, bit-for-bit."""
    for seed in (1, 2, 3, 4):
        w = _rows(16, seed)
        raw = quantize_q3_k_stream(w.tobytes())
        ours = np.array(q3_k_unpack(raw), dtype=np.float32)
        eng = from_s4_q3_k(to_s4_q3_k(raw)).astype(np.float32)
        assert np.array_equal(ours, eng), f"seed {seed}: round-trip diverges"


def test_q3_k_edge_blocks():
    """AC1-1 edges: zero block, tiny amax, amax upper, negative-heavy (signed sc6-32)."""
    cases = [np.zeros(QK_K, dtype="<f4"),
            np.full(QK_K, 1e-6, dtype="<f4"),
            np.full(QK_K, 63.0, dtype="<f4"),
            np.full(QK_K, -63.0, dtype="<f4"),
            np.full(QK_K, -1e-6, dtype="<f4")]
    w = np.zeros(QK_K, dtype="<f4")
    w[0:16] = 10.0
    cases.append(w)
    for w in cases:
        raw = quantize_q3_k_stream(w.tobytes())
        assert np.array_equal(np.array(q3_k_unpack(raw), dtype=np.float32),
                              from_s4_q3_k(to_s4_q3_k(raw)).astype(np.float32))


def test_q3_k_signed_scale_preserved():
    """AC1-1 support: all-negative groups keep a negative (sc6-32) scale (sign flip)."""
    w = np.full(QK_K, -31.5, dtype="<f4")
    raw = quantize_q3_k_stream(w.tobytes())
    sb = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 110)[:, 96:108]
    sc6 = np.empty((1, 16), dtype=np.int32)
    for j in range(4):
        t = sb[:, 8 + j].astype(np.int32)
        lo, hi = sb[:, j].astype(np.int32), sb[:, 4 + j].astype(np.int32)
        sc6[:, j] = (lo & 0x0F) | ((t & 0x03) << 4)
        sc6[:, 4 + j] = (hi & 0x0F) | (((t >> 2) & 0x03) << 4)
        sc6[:, 8 + j] = ((lo >> 4) & 0x0F) | (((t >> 4) & 0x03) << 4)
        sc6[:, 12 + j] = ((hi >> 4) & 0x0F) | (((t >> 6) & 0x03) << 4)
    assert np.all(sc6 - 32 < 0)
    v = np.array(q3_k_unpack(raw), dtype=np.float32)
    assert np.all(v < 0)


def test_q3_k_error_bound():
    """AC1-2 (producer quality, HYPOTHESIS): |w - v| <= |dl_g|/2 per group for groups
    whose codebook fully covers the range (no code clipping: min code > 0 and max code
    < 7). A group whose range exceeds the -4..3 coverage clips and cannot meet the
    bound; the ggml joint search is NOT reproduced (engine ships no quantizer), so the
    bound is asserted only for covered groups. Groups with dl_g == 0 quantize to v = 0.
    """
    w = _rows(16, 7)
    raw = quantize_q3_k_stream(w.tobytes())
    v = np.array(q3_k_unpack(raw), dtype=np.float32)
    b = w.reshape(-1, QK_K)
    d16 = (np.max(np.abs(b), axis=1) / 63.0).astype(np.float16).astype(np.float32)
    groups = b.reshape(-1, 16, 16)
    gmax = groups.max(axis=2)
    sc6 = np.clip(np.rint(gmax / d16[:, None]) + 32, 0, 63)
    dl = d16[:, None] * (sc6 - 32).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        code = np.rint(groups / dl[:, :, None]) + 4
    code = np.where(dl[:, :, None] == 0, 4.0, code)
    code = np.clip(code, 0, 7)
    covered = ((code.min(axis=2) > 0) & (code.max(axis=2) < 7) & (dl != 0))[:, :, None]
    bound = np.abs(dl)[:, :, None] * TOL / 2
    err = np.abs(groups - v.reshape(-1, 16, 16))
    assert np.all(np.where(covered, err, 0) <= np.broadcast_to(bound, (16, 16, 16)) + 1e-6)


def test_q3_k_planes_engine_byte_identical():
    """AC1-3: q3_k_planes codes == pack_codes(to_s4_q3_k codes, 4); scales == engine f32."""
    w = _rows(8, 11)
    raw = quantize_q3_k_stream(w.tobytes())
    codes_b, scales_b = q3_k_planes(raw)
    eng_codes, eng_scales = to_s4_q3_k(raw)
    assert codes_b == pack_codes(eng_codes, 4)
    assert scales_b == np.ascontiguousarray(eng_scales.reshape(-1), dtype=np.float32).tobytes()


def test_q3_k_stream_needs_multiple_of_256():
    w = np.zeros(300, dtype="<f4")
    with pytest.raises(ValueError):
        quantize_q3_k_stream(w.tobytes())
