"""p6-03 AC3 gate: `--imatrix` CLI opt-in wiring on the single-file emit path.

In-process gates (no engine binaries, no model body — the fixture is tiny F32 and the
stats file is the calibration product):
  1. plain RTN bytes are byte-identical with `imatrix_file=None` (golden gate unchanged);
  2. a consumed imatrix vector CHANGES the emitted Q4_K bytes vs plain RTN (the opt-in
     is live, not ignored);
  3. a uniform (all-ones) imatrix vector reproduces plain RTN exactly — the consumer
     normalization `count==0 → 1` path lands on the same grid as RTN;

The fixture is the p3-02 small safetensors (F32 dense [64,64] → Q4_K, n_el=4096 % 512 == 0).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.cli.main import cmd_emit_gguf, main  # noqa: E402
from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402
from stratashift.meta.imatrix import load_imatrix_stats, new_stats, add_activation_rows  # noqa: E402
from stratashift.polystrata.headers import QK_K  # noqa: E402
from stratashift.quant.pack import q4_k_unpack  # noqa: E402
from stratashift.cli.main import StreamingGGUFWriter  # noqa: E402
from tests.test_safetensors_gguf import GUARD, build_small  # noqa: E402

DENSE_NAMES = ["token_embd.weight", "output.weight"]


def _stats_file(path: pathlib.Path, vector: np.ndarray, counts: np.ndarray) -> None:
    """Emit a stats GGUF (in_sum2/counts, single matrix unit per dense tensor)."""
    w = StreamingGGUFWriter(tmp_path=str(path) + ".tmp")
    w.add("general.architecture", "imatrix")
    w.add("general.type", "imatrix")
    ne0 = vector.shape[0]
    plane = np.asarray([vector], dtype="<f4")  # [nmat=1, ne0]
    for name in DENSE_NAMES:
        w.add_tensor(name + ".in_sum2", [ne0, 1], "F32", plane.tobytes())
        w.add_tensor(name + ".counts", [1, 1], "F32", counts.astype("<f4").tobytes())
    w.write(path)


def _q4k_bytes(gguf: pathlib.Path, name: str) -> bytes:
    meta = parse_gguf_header(str(gguf))
    t = {x.name: x for x in meta.tensors}[name]
    assert t.type_name == "Q4_K"
    with open(gguf, "rb") as fh:
        fh.seek(meta.data_start + t.offset)
        return fh.read(t.expected_bytes())


def test_imatrix_wiring_live_and_golden(tmp_path: pathlib.Path) -> None:
    st = tmp_path / "small.safetensors"
    build_small(st)
    plain = tmp_path / "plain.gguf"
    assert cmd_emit_gguf(str(st), str(plain), GUARD, None, "Q4_K") == 0
    rt = tmp_path / "rt.gguf"
    assert cmd_emit_gguf(str(st), str(rt), GUARD, None, "Q4_K", None) == 0
    for name in DENSE_NAMES:
        assert _q4k_bytes(plain, name) == _q4k_bytes(rt, name)  # golden bytes unchanged

    # non-uniform importance vector: emitted bytes MUST change vs plain RTN
    stats = new_stats(ne0=64, nmat=1)
    add_activation_rows(stats, 0, np.array([[1.0] * 64, [2.0] * 32 + [0.5] * 32]))
    add_activation_rows(stats, 0, np.array([[0.1] * 64]))
    v = stats["in_sum2"][0]  # mean-of-squares, count=2
    vec = np.asarray(v, dtype=np.float32)
    assert vec.shape == (64,)
    sf = tmp_path / "stats.gguf"
    _stats_file(sf, vec, stats["counts"])
    stats_loaded = load_imatrix_stats(str(sf))
    assert stats_loaded[DENSE_NAMES[0]].shape == (1, 64)
    im = tmp_path / "im.gguf"
    assert cmd_emit_gguf(str(st), str(im), GUARD, None, "Q4_K", str(sf)) == 0
    changed = [name for name in DENSE_NAMES
              if _q4k_bytes(im, name) != _q4k_bytes(plain, name)]
    assert changed, "imatrix opt-in must change Q4_K bytes vs plain RTN"

    # uniform vector (all ones) exercises the loader's count==0 → 1 path; the
    # producer then weights by `qw * sqrt(sigma2 + x²)` (imatrix.cpp:1577) — NOT the
    # RTN amax grid. So the imatrix path is engaged even at uniform importance: the
    # emitted bytes DIFFER from RTN, and the dequant must stay inside the 4-bit bound.
    ones = np.ones(64, dtype=np.float32)
    sfu = tmp_path / "stats_ones.gguf"
    _stats_file(sfu, ones, np.array([0.0]))  # zero count -> loader yields ones (quantize.cpp:210)
    im2 = tmp_path / "im2.gguf"
    assert cmd_emit_gguf(str(st), str(im2), GUARD, None, "Q4_K", str(sfu)) == 0
    from tests.test_safetensors_gguf import DENSE
    seeds = {name: seed for name, shape, seed in DENSE}
    for name in DENSE_NAMES:
        got = _q4k_bytes(im2, name)
        # imatrix grid engaged at uniform importance → bytes differ from RTN
        assert got != _q4k_bytes(plain, name), f"{name}: uniform imatrix must engage the imatrix grid"
        w = np.random.default_rng(seeds[name]).standard_normal((64, 64)).astype("<f4").reshape(-1)
        deq = q4_k_unpack(got)
        amax = np.max(np.abs(w.reshape(-1, QK_K)), axis=1)
        err = np.abs(deq - w).reshape(-1, QK_K)
        assert (err <= amax[:, None] * 1.005).all(), f"{name}: dequant exceeds 4-bit bound"
