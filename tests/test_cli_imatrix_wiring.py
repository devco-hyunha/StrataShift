"""p6-03 AC3 gate: `--imatrix` CLI opt-in wiring on the single-file emit path.

In-process gates (no engine binaries, no model body — the fixture is tiny F32 and the
stats file is the calibration product):
  1. plain RTN bytes are byte-identical with `imatrix_file=None` (golden gate unchanged);
  2. a consumed imatrix vector CHANGES the emitted Q4_K bytes vs plain RTN (the opt-in
     is live, not ignored);
  3. a uniform (all-ones) imatrix vector reproduces plain RTN exactly — the consumer
     normalization `count==0 → 1` path lands on the same grid as RTN;
  4. `--imatrix` on the HF fused path consumes engine-keyed planes and emits
     Q4_K/Q4_0 (never Q2_0; never silently ignored). Declared type must match
     the quantizer (byte-size alone cannot catch Q4_K bytes labeled Q4_0 —
     both land on 144 B for 256 elements).
  5. HF dense engine-key lookup (p6-05 AC4): shared_expert down →
     `blk.L.ffn_down_shexp.weight` plane consume changes bytes vs plain path.
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
from stratashift.polystrata.gguf_stream_writer import StreamingGGUFWriter  # noqa: E402
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


def _hf_fixture(tmp_path: pathlib.Path, *, with_dense_shexp: bool = False) -> pathlib.Path:
    """HF-naming single-file safetensors: fused gate_up (4,8,64) + down (4,64,64).

    Optional shared_expert down (64,64)=[out,in] → engine `ffn_down_shexp` (64,64)
    after transpose (p6-05 AC4 dense engine-key consume).
    """
    import json as _json
    tensors = {}
    parts = []
    off = 0
    rng = np.random.default_rng(11)
    specs = [("model.layers.0.mlp.experts.gate_up_proj", (4, 8, 64)),
             ("model.layers.0.mlp.experts.down_proj", (4, 64, 64))]
    if with_dense_shexp:
        specs.append(("model.layers.0.mlp.shared_expert.down_proj.weight", (64, 64)))
    for nm, shape in specs:
        a = (rng.normal(0, 0.02, shape)).astype("<f4")
        tensors[nm] = {"dtype": "F32", "shape": list(shape), "data_offsets": [off, off + a.nbytes]}
        parts.append(a.tobytes())
        off += a.nbytes
    header = _json.dumps(tensors, separators=(",", ":"))
    header += " " * ((-len(header)) % 8)
    hb = header.encode("utf-8")
    st = tmp_path / "hf.safetensors"
    with open(st, "wb") as d:
        d.write(len(hb).to_bytes(8, "little"))
        d.write(hb)
        for p in parts:
            d.write(p)
    return st


def _fused_stats_file(path: pathlib.Path, plane: np.ndarray) -> None:
    """Engine-keyed fused planes (imatrix.cpp pin 3cf03257 matrix-major):
    in_sum2 header shape [ne0, nmat], bytes flat `j*ne0 + i` (unit j contiguous).
    `plane` is (nmat, ne0) loader orientation — `plane.tobytes()` is the producer
    layout (no transpose; transpose would pair with a broken loader)."""
    w = StreamingGGUFWriter(tmp_path=str(path) + ".tmp")
    w.add("general.architecture", "imatrix")
    w.add("general.type", "imatrix")
    nmat, ne0 = plane.shape
    plane_b = np.ascontiguousarray(plane, dtype="<f4")
    w.add_tensor("blk.0.ffn_gate_up_exps.weight.in_sum2", [ne0, nmat], "F32",
                 plane_b.tobytes())
    w.add_tensor("blk.0.ffn_gate_up_exps.weight.counts", [1, nmat], "F32",
                 np.full(nmat, 1.0, dtype="<f4").tobytes())
    w.add_tensor("blk.0.ffn_down_exps.weight.in_sum2", [ne0, nmat], "F32",
                 plane_b.tobytes())
    w.add_tensor("blk.0.ffn_down_exps.weight.counts", [1, nmat], "F32",
                 np.full(nmat, 1.0, dtype="<f4").tobytes())
    w.write(path)


def _tensor_bytes(gguf: pathlib.Path, name: str) -> tuple[bytes, str]:
    meta = parse_gguf_header(str(gguf))
    t = {x.name: x for x in meta.tensors}[name]
    with open(gguf, "rb") as fh:
        fh.seek(meta.data_start + t.offset)
        return fh.read(t.expected_bytes()), t.type_name


def test_hf_fused_path_consumes_planes(tmp_path: pathlib.Path) -> None:
    """p6-05 AC: `--imatrix` on the HF fused path emits imatrix-typed expert tensors
    consuming the engine-keyed planes (unit = expert e), never Q2_0 (user ruling
    2026-10-21: Q2_0는 하지마).

    Fixture: gate_up (ne=4, 2ff=8, h=64), down (ne=4, h=64, ff=64). Engine load gate
    (_dense_select ladder, p5-08): h=64 → 64 % 256 = 64 남 → Q4_K refused → be=32
    ladder lands on Q4_0 for this fixture; the delivery product's h=2816 passes Q4_K.
    Plane length 64 = engine row length h (measured product: 2816/704 = h/ff).
    """
    st = _hf_fixture(tmp_path)
    plane = np.full((4, 64), 1.0, dtype=np.float32)
    plane[1, 32] = 100.0  # expert 1 column 32 high-importance
    sf = tmp_path / "fused_stats.gguf"
    _fused_stats_file(sf, plane)
    gguf = tmp_path / "hf.gguf"
    rc = main(["convert", str(st), "--out", str(tmp_path / "pack"),
               "--emit-gguf", str(gguf), "--hf-emit", "--dense-type", "Q4_K",
               "--imatrix", str(sf)])
    assert rc == 0
    gate_b, gate_t = _tensor_bytes(gguf, "blk.0.ffn_gate_exps.weight")
    up_b, up_t = _tensor_bytes(gguf, "blk.0.ffn_up_exps.weight")
    down_b, down_t = _tensor_bytes(gguf, "blk.0.ffn_down_exps.weight")
    assert up_t == gate_t and gate_t in ("Q4_K", "Q4_0"), (gate_t, up_t)
    assert down_t in ("Q4_K", "Q4_0"), down_t
    assert "Q2_0" not in (gate_t, up_t, down_t), "Q2_0 must never be emitted with imatrix"
    # dequant fidelity within the 4/5-bit bound per expert row stream.
    # gate/up: ff*h = 256 elems; down: h*ff = 4096 elems — slice sizes differ.
    from stratashift.quant.pack import q4_0_unpack

    def _blk_be(type_name: str) -> tuple[int, int]:
        return (144, 256) if type_name == "Q4_K" else (18, 32)

    def deq(payload: bytes, type_name: str) -> np.ndarray:
        blk, _ = _blk_be(type_name)
        unpack = q4_k_unpack if type_name == "Q4_K" else q4_0_unpack
        return np.array([x for i in range(0, len(payload), blk)
                         for x in unpack(payload[i:i + blk])], dtype=np.float64)

    with open(st, "rb") as d:
        n = int.from_bytes(d.read(8), "little")
        d.read(n)  # padded header (the fixture writes len(header)+pad bytes)
        data = d.read()
    gu = np.frombuffer(data[: 4 * 8 * 64 * 4], dtype="<f4").reshape(4, 8, 64)
    dn = np.frombuffer(data[4 * 8 * 64 * 4:], dtype="<f4").reshape(4, 64, 64)
    g_blk, g_be = _blk_be(gate_t)
    d_blk, d_be = _blk_be(down_t)
    gate_per = 256 // g_be * g_blk   # 256 elems → 144 B (Q4_K or Q4_0)
    down_per = 4096 // d_be * d_blk  # 4096 elems → 2304 B
    for e in range(4):
        w = gu[e, 0:4, :].reshape(-1)  # gate rows, ff*h = 256 elements
        seg = gate_b[e * gate_per:(e + 1) * gate_per]
        amax = np.max(np.abs(w.reshape(-1, g_be)), axis=1)
        assert (np.abs(deq(seg, gate_t) - w).reshape(-1, g_be)
                <= amax[:, None] * 1.005).all(), f"expert {e} gate dequant bound"
        w = dn[e].reshape(-1)
        seg = down_b[e * down_per:(e + 1) * down_per]
        amax = np.max(np.abs(w.reshape(-1, d_be)), axis=1)
        assert (np.abs(deq(seg, down_t) - w).reshape(-1, d_be)
                <= amax[:, None] * 1.005).all(), f"expert {e} down dequant bound"
    # plane engagement proof: WITHOUT --imatrix the same fixture emits Q2_0 (the plain
    # fallback, golden path) — the Q4-typed bytes can only come from the plane consumer.
    plain_gguf = tmp_path / "hf_plain.gguf"
    assert main(["convert", str(st), "--out", str(tmp_path / "pack2"),
                 "--emit-gguf", str(plain_gguf), "--hf-emit", "--dense-type", "Q4_K"]) == 0
    assert _tensor_bytes(plain_gguf, "blk.0.ffn_gate_exps.weight")[1] == "Q2_0"
    assert _tensor_bytes(plain_gguf, "blk.0.ffn_down_exps.weight")[1] == "Q2_0"


def test_hf_dense_path_consumes_engine_key_plane(tmp_path: pathlib.Path) -> None:
    """p6-05 AC4: dense engine-key lookup consumes imatrix plane (fixture, no engine).

    shared_expert.down_proj [64,64] → engine `blk.0.ffn_down_shexp.weight` shape (64,64)
    (transpose). Request `--dense-type Q4_0` so the ladder keeps Q4_0 (imatrix branch
    exists only for Q4_0/Q4_K — Q4_K→Q8_0 ladder would skip imatrix). ne0=64 % 32 = 0,
    n_el=4096 pack-ok. Non-uniform plane must change Q4_0 bytes vs plain RTN.
    """
    st = _hf_fixture(tmp_path, with_dense_shexp=True)
    # fused planes (required so expert path stays typed) + dense shexp plane
    nmat, ne0 = 4, 64
    fused = np.full((nmat, ne0), 1.0, dtype=np.float32)
    dense_vec = np.ones(ne0, dtype=np.float32)
    dense_vec[16] = 80.0  # high-importance column
    sf = tmp_path / "dense_stats.gguf"
    w = StreamingGGUFWriter(tmp_path=str(sf) + ".tmp")
    w.add("general.architecture", "imatrix")
    w.add("general.type", "imatrix")
    plane_b = np.ascontiguousarray(fused, dtype="<f4")
    w.add_tensor("blk.0.ffn_gate_up_exps.weight.in_sum2", [ne0, nmat], "F32",
                 plane_b.tobytes())
    w.add_tensor("blk.0.ffn_gate_up_exps.weight.counts", [1, nmat], "F32",
                 np.full(nmat, 1.0, dtype="<f4").tobytes())
    w.add_tensor("blk.0.ffn_down_exps.weight.in_sum2", [ne0, nmat], "F32",
                 plane_b.tobytes())
    w.add_tensor("blk.0.ffn_down_exps.weight.counts", [1, nmat], "F32",
                 np.full(nmat, 1.0, dtype="<f4").tobytes())
    # dense nmat=1: header [ne0, 1], bytes matrix-major (= flat vector)
    w.add_tensor("blk.0.ffn_down_shexp.weight.in_sum2", [ne0, 1], "F32",
                 dense_vec.astype("<f4").tobytes())
    w.add_tensor("blk.0.ffn_down_shexp.weight.counts", [1, 1], "F32",
                 np.array([1.0], dtype="<f4").tobytes())
    w.write(sf)

    im_gguf = tmp_path / "hf_dense_im.gguf"
    plain_gguf = tmp_path / "hf_dense_plain.gguf"
    assert main(["convert", str(st), "--out", str(tmp_path / "pack_im"),
                 "--emit-gguf", str(im_gguf), "--hf-emit", "--dense-type", "Q4_0",
                 "--imatrix", str(sf)]) == 0
    assert main(["convert", str(st), "--out", str(tmp_path / "pack_plain"),
                 "--emit-gguf", str(plain_gguf), "--hf-emit", "--dense-type", "Q4_0"]) == 0

    im_b, im_t = _tensor_bytes(im_gguf, "blk.0.ffn_down_shexp.weight")
    plain_b, plain_t = _tensor_bytes(plain_gguf, "blk.0.ffn_down_shexp.weight")
    assert im_t == "Q4_0", im_t
    assert plain_t == "Q4_0", plain_t
    assert im_b != plain_b, "dense engine-key imatrix must change Q4_0 bytes vs plain RTN"

    from stratashift.quant.pack import q4_0_unpack
    # dequant within 5-bit-ish bound for Q4_0 (amax * 1.005 per 32-el block)
    with open(st, "rb") as d:
        n = int.from_bytes(d.read(8), "little")
        d.read(n)
        data = d.read()
    # tensor order in fixture: gate_up, down_exps, then shexp down
    shexp_off = 4 * 8 * 64 * 4 + 4 * 64 * 64 * 4
    w_f32 = np.frombuffer(data[shexp_off:shexp_off + 64 * 64 * 4], dtype="<f4")
    # emit stores engine (in,out)=(64,64) bytes == HF [out,in] row-major bytes (transpose
    # shape only) — same float stream as HF file for this square case.
    deq = np.array([x for i in range(0, len(im_b), 18)
                    for x in q4_0_unpack(im_b[i:i + 18])], dtype=np.float64)
    amax = np.max(np.abs(w_f32.reshape(-1, 32)), axis=1)
    assert (np.abs(deq - w_f32).reshape(-1, 32)
            <= amax[:, None] * 1.005).all(), "dense shexp dequant bound"
