"""p6-08 AC5: expert type name and quantizer are chosen together, in one place.

Pure functions only — no engine tools, no model files. The regression these guard is
p6-05 failure 2: a header that says Q4_0 over Q4_K bytes (both 144 B per 256 elements,
so a size check alone cannot catch it).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.quant.pack import (quantize_q2_0_stream, quantize_q4_0_stream,  # noqa: E402
                                    quantize_q4_k_stream)
from stratashift.quant.pack_imatrix import (quantize_q4_0_imatrix_stream,  # noqa: E402
                                            quantize_q4_k_imatrix_stream)
from stratashift.quant.select import (FUSED_DOWN_LADDER, select_dense_quant,  # noqa: E402
                                      select_expert_quant, select_pack_quant)


@pytest.mark.parametrize("row_len,n_elems,want", [
    (256, 256 * 4, "Q4_K"),      # 256-multiple row and slice → Q4_K (be=256)
    (2816, 2816 * 704, "Q4_K"),  # product gate/up h=2816 = 256×11
    (64, 64 * 64, "Q4_0"),       # fixture h=64 → be=32
    (704, 704 * 2816, "Q4_0"),   # ff=704: 704 % 256 = 192 → Q4_K refused, be=32 passes
    (96, 96 * 8, "Q4_0"),        # 32-multiple but not 256
    (64, 64 * 64 + 16, None),    # slice not whole blocks: no ladder type, Q2_0 (be=64) neither
    (48, 48 * 8, None),          # row not a 32-multiple: ladder fails, Q2_0 row 48 % 64 too
    (128, 128 * 8 + 32, "Q4_0"),  # Q4_K slice gate fails (1056 % 256), be=32 passes
])
def test_ladder_boundaries(row_len: int, n_elems: int, want: str | None) -> None:
    eq = select_expert_quant(row_len, n_elems, has_plane=True)
    assert (eq.type_name if eq else None) == want


def test_no_plane_is_q2_0_plain() -> None:
    eq = select_expert_quant(256, 256 * 4, has_plane=False)
    assert eq.type_name == "Q2_0"
    raw = np.linspace(-1, 1, 256, dtype="<f4").tobytes()
    assert eq.quantize(raw, None) == quantize_q2_0_stream(raw)


def test_fused_down_ladder_is_q4_0_only() -> None:
    # ff=256 would take Q4_K on the default ladder; fused down stays Q4_0 (p6-05 byte contract)
    assert select_expert_quant(256, 256 * 8, True).type_name == "Q4_K"
    assert select_expert_quant(256, 256 * 8, True, FUSED_DOWN_LADDER).type_name == "Q4_0"
    # AC1 allowed change: a row that is not a 32-multiple used to be labeled Q4_0 anyway;
    # Q2_0 (be=64) cannot load it either → refused
    assert select_expert_quant(48, 48 * 8, True, FUSED_DOWN_LADDER) is None


def test_row_not_loadable_is_refused_not_q2_0() -> None:
    """p6-08 review: the Q2_0 fallback is gated by the engine load gate too (case 42
    be=64, gguf_reader.hpp:232-235 pin c566ff4). Before, h=48 got an unloadable Q2_0
    tensor and n_elems % 64 != 0 raised mid-stream inside the Q2_0 producer."""
    assert select_expert_quant(48, 8 * 48, has_plane=True) is None
    assert select_expert_quant(40, 8 * 40, has_plane=False) is None
    assert select_expert_quant(96, 96 * 8, has_plane=False) is None  # 96 % 64 = 32
    assert select_expert_quant(128, 128 * 8, has_plane=False).type_name == "Q2_0"


def test_expert_quant_keeps_selection_geometry() -> None:
    eq = select_expert_quant(64, 64 * 64, has_plane=True)
    assert (eq.type_name, eq.row_len, eq.n_elems) == ("Q4_0", 64, 64 * 64)


@pytest.mark.parametrize("type_name,imatrix_fn,plain_fn,n", [
    ("Q4_K", quantize_q4_k_imatrix_stream, quantize_q4_k_stream, 256),
    ("Q4_0", quantize_q4_0_imatrix_stream, quantize_q4_0_stream, 64),
])
def test_name_and_producer_are_a_pair(type_name, imatrix_fn, plain_fn, n) -> None:
    """The bytes come from the producer that matches the declared type (never Q4_K bytes
    under a Q4_0 name): compare against calling that producer directly."""
    rng = np.random.default_rng(3)
    raw = rng.normal(0, 0.05, n).astype("<f4").tobytes()
    plane = np.ones(n, dtype=np.float32)
    plane[5] = 40.0
    eq = select_expert_quant(n, n, has_plane=True)
    assert eq.type_name == type_name
    got = eq.quantize(raw, plane)
    assert got == imatrix_fn(raw, plane)
    assert got != plain_fn(raw)  # the imatrix producer engaged


@pytest.mark.parametrize("shape,dense_type,want", [
    ((64, 64), "Q4_K", "Q4_K"),   # 4096 % 512 == 0
    ((16, 16), "Q4_K", "F32"),    # below the Q4_K pack gate (512)
    ((4096,), "Q8_0", "F32"),     # Q8_0 needs ndim >= 2
    ((64, 64), "Q8_0", "Q8_0"),
    ((64, 64), "F32", "F32"),
    ((64,), "Q4_0", "F32"),       # whole Q4_0 blocks but below the 128-element pack gate
])
def test_pack_only_selection(shape: tuple, dense_type: str, want: str) -> None:
    assert select_pack_quant(shape, dense_type) == want


def test_pack_q4_0_partial_block_is_refused() -> None:
    """The Q4_0 whole-block refusal moved from emit_gguf into select (p6-08 review);
    emit_gguf prefixes the tensor name, keeping the old message byte-identical."""
    with pytest.raises(ValueError, match=r"Q4_0 emit needs a multiple of 32 elements, "
                                         r"shape \(48,\) = 48"):
        select_pack_quant((48,), "Q4_0")


def test_pack_and_dense_selection_share_the_pack_gate() -> None:
    """select_dense_quant with a load gate that always passes (ne0 = 256·k) agrees with
    select_pack_quant on the requested type — one pack gate, not two copies."""
    for shape in [(256, 2), (256, 1), (256,), (512, 4)]:
        for t in ("Q4_0", "Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q8_0"):
            n = 1
            for d in shape:
                n *= d
            pack = select_pack_quant(shape, t)
            dense = select_dense_quant(n, t, 256, len(shape))
            assert (dense == t) == (pack == t), (shape, t, pack, dense)
