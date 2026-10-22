"""p3-02 AC1: quantize_q2_0_stream must be byte-identical to the engine reference.

Engine reference: pinned PolyStrata tools/gguf_writer.py quantize_q2_0 (pin c566ff4).
Engine domain: per-block amax must fit fp16 (<= 65504); the engine itself raises
OverflowError above that, so the contract is defined only inside this domain.
"""
import numpy as np
import pytest

from gguf_writer import quantize_q2_0 as engine_q2_0  # noqa: E402

from stratashift.quant.pack import quantize_q2_0_stream  # noqa: E402


def test_q2_0_stream_byte_identical_random():
    rng = np.random.default_rng(99)
    for _ in range(16):
        w = rng.standard_normal(64 * 64).astype(np.float32)
        assert quantize_q2_0_stream(w.tobytes()) == engine_q2_0(w)


def test_q2_0_stream_byte_identical_edges():
    w0 = np.zeros(64, dtype=np.float32)
    w1 = (np.random.default_rng(7).standard_normal(64) * 1e-3).astype(np.float32)
    w2 = (np.random.default_rng(8).standard_normal(64) * 1e4).astype(np.float32)
    assert np.abs(w2).max() < 65504
    w3 = np.full(64, 30000.0, dtype=np.float32)
    for w in (w0, w1, w2, w3):
        assert quantize_q2_0_stream(w.tobytes()) == engine_q2_0(w)


def test_q2_0_stream_requires_block_multiple():
    with pytest.raises(ValueError):
        quantize_q2_0_stream(b"\x00" * 63)
