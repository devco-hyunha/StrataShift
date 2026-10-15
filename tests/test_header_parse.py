"""Header parsing unit tests: synthetic fixture + real model (if present)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from stratashift.ingest.safetensors_header import ModelMeta, parse_header
from stratashift.meta.expert_map import build_expert_map, classify

MODEL = Path(__file__).resolve().parent.parent / ".local" / "models" / "mixtral-tiny.safetensors"


def make_safetensors(path: Path, header: dict, blob: bytes) -> None:
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    prefix_len = 8 + len(header_bytes)
    pad = (-prefix_len) % 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(b" " * pad)
        f.write(blob)


def test_fixture_header(tmp_path: Path) -> None:
    header = {
        "model.layers.0.mlp.experts.0.gate_proj.weight":
            {"dtype": "F32", "shape": [4, 2], "data_offsets": [0, 32]},
        "model.layers.0.mlp.experts.0.down_proj.weight":
            {"dtype": "F32", "shape": [2, 4], "data_offsets": [32, 64]},
        "__metadata__": {"format": "pt"},
    }
    path = tmp_path / "tiny.safetensors"
    make_safetensors(path, header, bytes(64))

    meta = parse_header(str(path))
    assert meta.header_size == len(json.dumps(header, separators=(",", ":")).encode())
    assert meta.data_start == 8 + meta.header_size + (-8 - meta.header_size) % 8
    assert len(meta.tensors) == 2
    t0 = meta.tensors[0]
    assert t0.dtype == "F32" and t0.shape == (4, 2)
    assert t0.data_offsets == (0, 32) and t0.nbytes() == 32
    assert meta.metadata == {"format": "pt"}

    refs = build_expert_map(meta)
    assert len(refs) == 2
    assert refs[0].layer_id == 0 and refs[0].expert_id == 0
    assert {r.role for r in refs} == {"gate", "down"}


def test_classify_patterns() -> None:
    cases = {
        "model.layers.3.experts.7.gate_proj.weight": (3, 7, "gate"),
        "model.layers.1.mlp.experts.2.up_proj.weight": (1, 2, "up"),
        "model.layers.0.moe.layers.5.mlp.down_proj.weight": (0, 5, "down"),
        "model.layers.2.experts.1.w2": (2, 1, "down"),
        "model.layers.0.attention.q_proj.weight": None,
    }
    for name, expected in cases.items():
        assert classify(name) == expected, name


@pytest.mark.skipif(not MODEL.exists(), reason="test model not downloaded")
def test_real_model_header() -> None:
    meta = parse_header(str(MODEL))
    assert meta.file_size == MODEL.stat().st_size
    assert meta.tensors, "no tensors"
    for t in meta.tensors:
        assert t.data_offsets[0] < t.data_offsets[1]
    refs = build_expert_map(meta)
    layers = {r.layer_id for r in refs}
    assert layers, "mixtral-tiny must yield expert candidates"
    for layer in layers:
        per_expert = {r.role for r in refs if r.layer_id == layer}
        # fused fixture: gate_up + down; per-tensor: gate + up + down
        assert {"down", "gate_up"} <= per_expert or {"gate", "up", "down"} <= per_expert
