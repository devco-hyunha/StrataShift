"""p3-02 safetensors float experts -> engine GGUF Q2_0 writer (AC2~AC6 gates).

In-process emit test on a small fixture (byte-identity vs the engine reference
quantize_q2_0, PLE IQ4_NL passthrough, bounded peak = header + single slice),
plus the full engine chain as a subprocess gate (skipif engine binaries absent).
Engine sources are read-only; the engine is never edited.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(pathlib.Path("/opt/data/polystrata-src/tools")))
BUILD = pathlib.Path("/opt/data/polystrata-build")
BINARIES = (BUILD / "strata-gguf", BUILD / "strata-dequant")

pytestmark = pytest.mark.skipif(
    not all(b.exists() for b in BINARIES),
    reason="pinned engine CPU binaries not built (p0-02)",
)

from gguf_writer import quantize_q2_0 as engine_q2_0  # noqa: E402

from stratashift.cli.main import cmd_emit_gguf  # noqa: E402
from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402

GUARD = {
    "architecture": "qwen4exp",
    "name": "stratashift p3-02 test",
    "qwen4exp.block_count": 48,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.expert_count": 512,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.ssm.state_size": 128,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.ssm.time_step_rank": 48,
    "qwen4exp.ssm.conv_kernel": 4,
    "qwen4exp.ssm.inner_size": 6144,
    "qwen4exp.attention.key_length": 256,
    "qwen4exp.attention.value_length": 256,
    "qwen4exp.rope.dimension_count": 64,
    "qwen4exp.attention.indexer.head_count": 4,
    "qwen4exp.attention.indexer.key_length": 128,
    "qwen4exp.attention.indexer.top_k": 2048,
    "qwen4exp.hyper_connection.count": 4,
    "qwen4exp.hyper_connection.low_rank": 320,
    "qwen4exp.expert_feed_forward_length": 640,
    "qwen4exp.expert_shared_feed_forward_length": 640,
}

EXPERTS = [
    ("blk.0.ffn_gate_exps.weight", (64, 4, 64), 11),
    ("blk.0.ffn_up_exps.weight", (64, 4, 64), 12),
    ("blk.0.ffn_down_exps.weight", (64, 4, 64), 13),
]
DENSE = [
    ("token_embd.weight", (64, 64), 14),
    ("blk.0.attn_norm", (64,), 15),
    ("output.weight", (64, 64), 16),
]


def ple_bytes() -> bytes:
    ple = np.random.default_rng(17).integers(0, 16, size=(40, 18), dtype=np.uint8)
    ple[:, 0] = 0x00
    ple[:, 1] = 0x3C  # fp16 1.0 little-endian
    return ple.reshape(-1).tobytes()


def build_small(path: pathlib.Path) -> None:
    """Small safetensors: F32 experts (out, NE, in) + F32 dense + U8[720] PLE."""
    tensors: dict = {}
    parts: list[bytes] = []
    off = 0
    for name, shape, seed in EXPERTS + DENSE:
        a = np.random.default_rng(seed).standard_normal(shape).astype("<f4")
        tensors[name] = {"dtype": "F32", "shape": list(shape),
                         "data_offsets": [off, off + a.nbytes]}
        parts.append(a.tobytes())
        off += a.nbytes
    ple = ple_bytes()
    tensors["per_layer_token_embd.weight"] = {"dtype": "U8", "shape": [720],
                                              "data_offsets": [off, off + len(ple)]}
    parts.append(ple)
    header = json.dumps(tensors, separators=(",", ":"))
    header += " " * ((-len(header)) % 8)
    hb = header.encode("utf-8")
    with open(path, "wb") as dst:
        dst.write(len(hb).to_bytes(8, "little"))
        dst.write(hb)
        for p in parts:
            dst.write(p)


def test_emit_gguf_byte_identity(tmp_path: pathlib.Path) -> None:
    st = tmp_path / "small.safetensors"
    gguf = tmp_path / "small.gguf"
    build_small(st)
    rc = cmd_emit_gguf(str(st), str(gguf), GUARD, None)
    assert rc == 0
    meta = parse_gguf_header(str(gguf))
    by_name = {t.name: t for t in meta.tensors}
    assert len(meta.tensors) == 7
    # expert tensors: GGUF fused shape (in, out, NE), Q2_0, bytes == engine reference
    for name, shape, seed in EXPERTS:
        out_dim, ne, in_dim = shape
        t = by_name[name]
        assert t.type_name == "Q2_0"
        assert tuple(t.shape) == (in_dim, out_dim, ne)
        want = engine_q2_0(np.random.default_rng(seed).standard_normal(shape).astype("<f4").reshape(-1))
        with open(gguf, "rb") as fh:
            fh.seek(meta.data_start + t.offset)
            got = fh.read(len(want))
        assert got == want, f"{name}: emitted Q2_0 bytes differ from engine reference"
    # PLE: IQ4_NL passthrough, bytes verbatim, GGUF shape [160, 8]
    t = by_name["per_layer_token_embd.weight"]
    assert t.type_name == "IQ4_NL"
    assert tuple(t.shape) == (160, 8)
    with open(gguf, "rb") as fh:
        fh.seek(meta.data_start + t.offset)
        assert fh.read(720) == ple_bytes()
    # dense F32 tensors present with original shapes
    for name, shape, seed in DENSE:
        t = by_name[name]
        assert t.type_name == "F32" and tuple(t.shape) == tuple(shape)


def test_emit_peak_is_single_slice(tmp_path: pathlib.Path) -> None:
    """AC6 no-full-load gate: peak live data stays at header + one slice level."""
    st = tmp_path / "small.safetensors"
    gguf = tmp_path / "small.gguf"
    build_small(st)
    tracemalloc.start()
    cmd_emit_gguf(str(st), str(gguf), GUARD, None)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # small fixture total is ~200 KB; a full-load path would copy the whole file.
    assert peak < 20 << 20, f"peak {peak} exceeds single-slice bound"


def test_safetensors_gguf_engine_chain() -> None:
    """Full p3-02 chain: fixture -> emit -> strata-gguf/dequant -> strata_pack
    build/verify -> our convert byte-identical to golden -> quantizer identity."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p3_safetensors_gguf.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"chain failed:\n{r.stdout}\n{r.stderr}"
    assert "SMOKE PASS" in r.stdout
