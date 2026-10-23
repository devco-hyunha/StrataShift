"""p3-09 AC1: quantize_q8_0_stream — engine quantizer byte-identity + dequant round-trip.

The engine SHIPS a Q8_0 quantizer: `tools/mtp_pack.py:100 q8_0` (pin c566ff4) —
d = amax/127 (f32), inv = 1/d (d==0 → 0), q = np.rint(x·inv).astype(int8), stored
d = fp16(d), qs = q as uint8 (the +128 is the int8→uint8 view). The contract is
mtp_pack.py (np.rint = C rint, ties-to-even); `mtp_rt.py:34` is a round-half-away
variant that diverges at ties — recorded, not the contract.

Invariants (docs/work/p3-09.md AC1, evidence-locked 2026-10-17):
  1. quantize_q8_0_stream == engine mtp_pack.q8_0 byte-for-byte (random + edges);
  2. q8_0_unpack (to_s8_q8_0+from_s8_q8_0 transcription, canonical_xcheck.py:58-84)
     == engine to_s8_q8_0+from_s8_q8_0 bit-exact on our blocks;
  3. q8_0_planes == engine to_s8_q8_0 + pack_codes(bits=8, per=1) byte-for-byte
     (codes plane = qs verbatim), scales fp16 bytes widen to the engine f32 plane;
  4. quantization error |w − v| ≤ d16·(1/2 + 127·2^-10) per block (rint half +
     fp16 d rounding at the extreme code ±127; d16 = fp16-rounded stored scale).
Contract domain: d = amax/127 must be fp16-representable — amax ≤ 65504·127 =
8,319,008 (mtp_pack.py:102 d.astype(np.float16) overflows to inf beyond it).
Edges are tested AT the boundary, never beyond it.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("canonical_xcheck")  # pinned PolyStrata tools (.local); absent in CI -> skip module
from canonical_xcheck import to_s8_q8_0, from_s8_q8_0  # noqa: E402

from stratashift.polystrata.engine_paths import resolve_engine_paths  # noqa: E402

# mtp_pack.py:35 add_gguf_py() needs a gguf-py dir; the pinned llama.cpp checkout in
# the engine build tree (_deps/strata_llamacpp-src, Q2_0=42, Q8_0=8) satisfies it —
# harness-side env, engine untouched. The pip `gguf` lacks Q2_0, so it is NOT the contract.
# Earlier tests (test_engine_smoke) cache the pip gguf in sys.modules; add_gguf_py's
# sys.path.insert alone cannot displace a cached module, so the harness pins the path
# and drops the pip gguf cache before importing mtp_pack.
GGUF_PY = str(resolve_engine_paths().gguf_py)  # STRATASHIFT_GGUF_PY / engine build _deps
os.environ["STRATA_GGUF_PY"] = GGUF_PY
sys.path.insert(0, GGUF_PY)
# pop the pip gguf AND its cached submodules (gguf.constants carries GGMLQuantizationType;
# a top-level-only pop lets the pinned __init__ re-absorb pip submodules from sys.modules).
for _m in [k for k in sys.modules if k == "gguf" or k.startswith("gguf.")]:
    sys.modules.pop(_m, None)
from mtp_pack import q8_0 as engine_q8_0  # noqa: E402
from strata_pack import pack_codes  # noqa: E402

from stratashift.meta.engine_manifest import q8_0_planes  # noqa: E402
from stratashift.quant.pack import q8_0_unpack, quantize_q8_0_stream  # noqa: E402

QK8_0 = 32
BOUND = 0.5 + 127 * 2**-10  # rint half + fp16 d rounding at code ±127


def _rows(n_blocks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n_blocks * QK8_0).astype("<f4")


def test_q8_0_engine_quantizer_byte_identical():
    """AC1-1: quantize_q8_0_stream == engine mtp_pack.q8_0, byte-for-byte."""
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        w = _rows(16, seed)
        assert quantize_q8_0_stream(w.tobytes()) == engine_q8_0(w).tobytes(), \
            f"seed {seed}: quantizer bytes diverge"


def test_q8_0_edge_blocks_byte_identical():
    """AC1-1 edges: all-zero block, tiny amax, large amax, fp16-boundary amax.

    The boundary is amax = 65504·127 = 8,319,008 (largest fp16 d = amax/127);
    beyond it mtp_pack.py:102 d.astype(np.float16) overflows to inf — the contract
    is defined only inside the engine's success domain, so edges sit AT the boundary.
    """
    cases = [np.zeros(QK8_0, dtype="<f4"),
             np.full(QK8_0, 1e-6, dtype="<f4"),
             np.full(QK8_0, -1e-6, dtype="<f4"),
             np.full(QK8_0, 1e6, dtype="<f4"),
             np.full(QK8_0, 65504.0 * 127.0, dtype="<f4"),
             np.full(QK8_0, -65504.0 * 127.0, dtype="<f4")]
    w = np.zeros(QK8_0, dtype="<f4")
    w[0] = 65504.0 * 127.0
    w[1] = -1.0
    cases.append(w)
    for w in cases:
        assert quantize_q8_0_stream(w.tobytes()) == engine_q8_0(w).tobytes()


def test_q8_0_roundtrip_engine_bit_exact():
    """AC1-2: q8_0_unpack == engine to_s8_q8_0 + from_s8_q8_0, bit-for-bit."""
    for seed in (1, 2, 3, 4):
        w = _rows(16, seed)
        raw = quantize_q8_0_stream(w.tobytes())
        ours = np.array(q8_0_unpack(raw), dtype=np.float32)
        eng = from_s8_q8_0(to_s8_q8_0(raw)).astype(np.float32)
        assert np.array_equal(ours, eng), f"seed {seed}: round-trip diverges"


def test_q8_0_planes_engine_byte_identical():
    """AC1-3: codes plane == pack_codes(to_s8_q8_0 codes, 8) (qs verbatim),
    scales fp16 bytes widen to the engine f32 plane (scales_fp16 exact)."""
    w = _rows(8, 11)
    raw = quantize_q8_0_stream(w.tobytes())
    codes_b, scales_b = q8_0_planes(raw)
    eng_codes, eng_scales = to_s8_q8_0(raw)
    assert codes_b == pack_codes(eng_codes, 8)
    assert np.frombuffer(scales_b, dtype="<f2").astype(np.float32).tobytes() \
        == np.ascontiguousarray(eng_scales.reshape(-1), dtype=np.float32).tobytes()


def test_q8_0_error_bound():
    """AC1-4 (producer quality): |w − v| ≤ d16·(1/2 + 127·2^-10) per block.

    d16 = the stored fp16 scale; the bound is the rint half-step (d/2 in the f32 d
    domain) plus the fp16 d rounding amplified at the extreme code ±127. The engine
    quantizer is byte-identical, so this bound holds for the engine's own output too.
    """
    w = _rows(16, 7)
    raw = quantize_q8_0_stream(w.tobytes())
    v = np.array(q8_0_unpack(raw), dtype=np.float32)
    b = w.reshape(-1, QK8_0)
    d16 = (np.abs(b).max(axis=1) / 127.0).astype(np.float16).astype(np.float32)
    err = np.abs(b - v.reshape(-1, QK8_0))
    assert np.all(err <= d16[:, None] * BOUND + 1e-6)


def test_q8_0_stream_requires_block_multiple():
    w = np.zeros(100, dtype="<f4")
    with pytest.raises(ValueError):
        quantize_q8_0_stream(w.tobytes())


def test_q8_0_cli_emit_gate_passthrough(tmp_path: Path):
    """AC2: --dense-type Q8_0 quantizes gate-multiple 2-D dense tensors (GGUF type id 8);
    tensors below the 64-element pack gate AND 1-D tensors stay F32 pass-through
    (pack_index.py:174-186 computes scales as ne1·n_groups — a 1-D tensor has ne1=0
    and pack_index refuses the quantized form).
    """
    from stratashift.cli.main import cmd_emit_gguf
    from stratashift.ingest.gguf import parse_gguf_header
    from tests.test_safetensors_gguf import GUARD, build_small

    st = tmp_path / "small.safetensors"
    gguf = tmp_path / "small.gguf"
    build_small(st)
    rc = cmd_emit_gguf(str(st), str(gguf), GUARD, None, dense_type="Q8_0")
    assert rc == 0
    meta = parse_gguf_header(str(gguf))
    by_name = {t.name: t for t in meta.tensors}
    for name, shape, seed in (
        ("token_embd.weight", (64, 64), 14),
        ("output.weight", (64, 64), 16),
    ):
        t = by_name[name]
        assert t.type_name == "Q8_0" and tuple(t.shape) == tuple(shape)
        w = np.random.default_rng(seed).standard_normal(shape).astype("<f4")
        want = engine_q8_0(w).tobytes()
        with open(gguf, "rb") as fh:
            fh.seek(meta.data_start + t.offset)
            assert fh.read(len(want)) == want, f"{name}: emitted Q8_0 bytes diverge"
    # 1-D tensor (64 elems, gate-multiple) stays F32 — ne1=0 pack refusal
    assert by_name["blk.0.attn_norm"].type_name == "F32"
    # experts stay Q2_0 (engine arena is Q2_0-only), PLE stays IQ4_NL
    assert by_name["blk.0.ffn_gate_exps.weight"].type_name == "Q2_0"
    assert by_name["per_layer_token_embd.weight"].type_name == "IQ4_NL"
