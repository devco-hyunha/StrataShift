"""p3-03 AC1: quantize_q4_0_stream — engine dequant round-trip contract.

The engine ships NO Q4_0 quantizer (gguf_writer.py is Q2_0-only, dequant.hpp has dequant
only), so the contract is the engine DEQUANT round-trip, not a quantizer byte-match:

    d = fp16(amax/7), q = clip(trunc(w*(1/d) + 8.5), 0, 15), qs[j] = q[j] | q[j+16] << 4

Invariants (docs/work/p3-03.md AC1, evidence-locked 2026-10-16):
  1. q4_0_unpack (dequant.hpp:88-95 transcription) == (q-8)*d bit-exact;
  2. gguf-py dequantize agrees bit-for-bit with q4_0_unpack on our blocks — this is the
     reference strata_pack verify compares against (ref/load.py _dequant_flat dispatches
     Q4_0 through gguf-py);
  3. quantization error |w - v| <= d/2 per row, fp16 tolerance factor 1.005;
  4. q4_0_planes == engine to_s4_q4_0 + pack_codes byte-for-byte.
gguf-py quantize (signed-max, quants.py:220-237) is NOT byte-identical to this rule —
the divergence is measured here as EVIDENCE and recorded as HYPOTHESIS in
docs/polystrata_spec_draft.md; it is not the contract.
"""
import sys

import numpy as np
import pytest

from stratashift.polystrata.engine_paths import resolve_engine_paths

# p6-11: the reference is the PINNED gguf-py (llama.cpp 3cf03257, the one ref/load.py
# dispatches through), not the pip `gguf`; drop a cached pip gguf first (test_q8_0 pattern).
sys.path.insert(0, str(resolve_engine_paths().gguf_py))
for _m in [k for k in sys.modules if k == "gguf" or k.startswith("gguf.")]:
    sys.modules.pop(_m, None)

pytest.importorskip("canonical_xcheck")  # pinned PolyStrata tools (.local); absent in CI -> skip module
from canonical_xcheck import to_s4_q4_0  # noqa: E402
from strata_pack import pack_codes  # noqa: E402
from gguf import GGMLQuantizationType as Q, quants  # noqa: E402

from stratashift.meta.engine_manifest import q4_0_planes  # noqa: E402
from stratashift.quant.pack import q4_0_unpack, quantize_q4_0_stream  # noqa: E402

TOL = 1.005  # fp16 scale rounding tolerance on the d/2 bound


def _codes_and_d(w: np.ndarray):
    """Recompute the contract's q and fp16 d from w (independent of the stream code)."""
    b = w.reshape(-1, 32)
    d = np.max(np.abs(b), axis=1) / 7.0
    with np.errstate(divide="ignore"):
        rid = np.where(d > 0, 1.0 / d, 0.0)
    q = np.clip(np.trunc(b * rid[:, None] + 8.5).astype(np.int32), 0, 15)
    return q, d


def test_q4_0_round_trip_bit_exact_random():
    rng = np.random.default_rng(21)
    for _ in range(16):
        w = rng.standard_normal(32 * 64).astype(np.float32)
        blocks = np.frombuffer(quantize_q4_0_stream(w.tobytes()), dtype=np.uint8).reshape(-1, 18)
        q, d = _codes_and_d(w)
        want = (q - 8) * d.astype(np.float16).astype(np.float32)[:, None]
        got = np.array([q4_0_unpack(bytes(blk)) for blk in blocks])
        assert np.array_equal(got, want)


def test_q4_0_round_trip_bit_exact_edges():
    w0 = np.zeros(32, dtype=np.float32)          # d = 0 → q 8 → v 0
    w1 = (np.random.default_rng(7).standard_normal(32) * 1e-3).astype(np.float32)
    w2 = (np.random.default_rng(8).standard_normal(32) * 4e4).astype(np.float32)
    assert np.abs(w2).max() / 7.0 < 65504        # amax/7 must fit fp16
    w3 = np.full(32, 30000.0, dtype=np.float32)  # amax at the positive extreme
    w4 = np.full(32, -30000.0, dtype=np.float32)  # amax at the negative extreme
    for w in (w0, w1, w2, w3, w4):
        blocks = np.frombuffer(quantize_q4_0_stream(w.tobytes()), dtype=np.uint8).reshape(-1, 18)
        q, d = _codes_and_d(w)
        want = (q - 8) * d.astype(np.float16).astype(np.float32)[:, None]
        got = np.array([q4_0_unpack(bytes(blk)) for blk in blocks])
        assert np.array_equal(got, want)


def test_q4_0_gguf_py_dequant_agrees_bit_exact():
    """The reference strata_pack verify compares against: gguf-py dequantize == q4_0_unpack."""
    rng = np.random.default_rng(22)
    for _ in range(16):
        w = rng.standard_normal(32 * 32).astype(np.float32)
        raw = np.frombuffer(quantize_q4_0_stream(w.tobytes()), dtype=np.uint8)
        ref = np.asarray(quants.dequantize(np.ascontiguousarray(raw), Q["Q4_0"]), dtype=np.float32)
        got = np.array([q4_0_unpack(bytes(b)) for b in raw.reshape(-1, 18)]).reshape(-1)
        assert np.array_equal(ref, got)


def test_q4_0_error_bound_half_scale():
    """|w - v| <= d/2 per row (fp16 tolerance factor TOL)."""
    rng = np.random.default_rng(23)
    for _ in range(200):
        w = rng.standard_normal(32 * 8).astype(np.float32)
        blocks = np.frombuffer(quantize_q4_0_stream(w.tobytes()), dtype=np.uint8).reshape(-1, 18)
        v = np.array([q4_0_unpack(bytes(b)) for b in blocks])
        d = np.max(np.abs(w.reshape(-1, 32)), axis=1) / 7.0
        ratio = np.abs(w.reshape(-1, 32) - v).max(axis=1) / (d / 2)
        assert ratio.max() <= TOL


def test_q4_0_planes_match_engine_canonical_byte_identical():
    """q4_0_planes == to_s4_q4_0 + pack_codes byte-for-byte (engine canonical form)."""
    rng = np.random.default_rng(24)
    for _ in range(8):
        w = rng.standard_normal(32 * 40).astype(np.float32)
        raw = quantize_q4_0_stream(w.tobytes())
        codes, scales = to_s4_q4_0(raw)
        want_codes = pack_codes(np.ascontiguousarray(codes), 4)
        want_scales = np.ascontiguousarray(scales.reshape(-1), dtype=np.float16).tobytes()
        got_codes, got_scales = q4_0_planes(raw)
        assert got_codes == want_codes
        assert got_scales == want_scales


def test_q4_0_gguf_py_quantize_divergence_is_evidence_not_contract():
    """Measured divergence: gguf-py quantize (signed-max) is NOT byte-identical to our rule.

    Recorded as HYPOTHESIS in docs/polystrata_spec_draft.md — the engine acceptance
    (strata_pack verify vs gguf-py dequantize) is the real gate, and both encodings
    decode fine under the engine dequant.
    """
    w = np.array([3.0, -1.0] + [0.5] * 30, dtype=np.float32)  # amax at the POSITIVE extreme
    ours = quantize_q4_0_stream(w.tobytes())
    gp = np.asarray(quants.quantize(np.ascontiguousarray(w), Q["Q4_0"]))
    assert bytes(ours) != bytes(gp.tobytes())
    # both encodings keep the amax element within the d/2 bound (valid grids, not equal grids)
    d_half = (np.abs(w).max() / 7.0) / 2
    v_ours = q4_0_unpack(ours[:18])
    v_gp = np.asarray(quants.dequantize(np.ascontiguousarray(gp), Q["Q4_0"]), dtype=np.float32)
    assert abs(v_ours[0] - 3.0) <= d_half and abs(v_gp[0] - 3.0) <= d_half


def test_q4_0_stream_requires_block_multiple():
    with pytest.raises(ValueError):
        quantize_q4_0_stream(b"\x00" * 31)
