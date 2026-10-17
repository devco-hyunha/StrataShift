"""p4-02 AC1: shard_index unit gates (each engine-mirrored gate has one FAIL case).

Gates (docs/work/p4-02.md §근거, pin c566ff4 mirrors):
- missing shard named by index.json (gguf_split.hpp:44-46);
- bidirectional weight_map ↔ shard-header transcription (validate_split mirror,
  gguf_reader.hpp:565-590);
- one tensor in two shards refused (GgufModel ctor, gguf_reader.hpp:536-540);
- payload outside its own shard file (in_bounds mirror, gguf_reader.hpp:557-562);
- total_size == Σ tensor bytes (HF convention gate).
"""
from __future__ import annotations

import json
import pathlib

import pytest

from stratashift.ingest.shard_index import abs_offset, load_sharded_model, locate_index


def write_safetensors(path: pathlib.Path, entries: dict[str, tuple[str, list[int], bytes]],
                      extra_header: dict | None = None) -> None:
    """Tiny safetensors: [u64 header_len][JSON][blob]; entries: name → (dtype, shape, bytes)."""
    off = 0
    header = {}
    blob = b""
    for name, (dtype, shape, data) in entries.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        blob += data
        off += len(data)
    if extra_header:
        header.update(extra_header)
    text = json.dumps(header, separators=(",", ":"))
    text += " " * (-len(text) % 8)
    hb = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(hb).to_bytes(8, "little") + hb + blob)


def write_index(d: pathlib.Path, weight_map: dict[str, str], total_size: int) -> None:
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map}))


T1: tuple[str, list[int], bytes] = ("F32", [1, 2], b"\x01\x02\x03\x04\x05\x06\x07\x08")  # 8 B
T2: tuple[str, list[int], bytes] = ("U8", [3, 4],
                                    b"\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10")  # 12 B


def happy_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "model"
    write_safetensors(d / "model-00001-of-00002.safetensors", {"a.weight": T1})
    write_safetensors(d / "model-00002-of-00002.safetensors", {"b.weight": T2})
    write_index(d, {"a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors"}, 20)
    return d


def test_happy_path_shard_registry(tmp_path):
    d = happy_dir(tmp_path)
    meta = load_sharded_model(str(d))
    assert len(meta.tensors) == 2
    by = {t.name: t for t in meta.tensors}
    assert by["a.weight"].shard == "model-00001-of-00002.safetensors"
    assert by["b.weight"].shard == "model-00002-of-00002.safetensors"
    for t in meta.tensors:
        m = meta.shard_data_start[t.shard]
        assert abs_offset(meta, t) == m + t.data_offsets[0]
        assert abs_offset(meta, t) + t.nbytes() <= meta.file_size  # in own shard
    # shard payload bytes preserved
    fh = open(d / by["b.weight"].shard, "rb")
    fh.seek(abs_offset(meta, by["b.weight"]))
    assert fh.read(by["b.weight"].nbytes()) == T2[2]
    fh.close()


def test_gate_missing_shard_names_the_shard(tmp_path):
    d = happy_dir(tmp_path)
    (d / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="missing model shard model-00002-of-00002.safetensors"):
        load_sharded_model(str(d))


def test_gate_weight_map_tensor_not_in_shard_header(tmp_path):
    d = happy_dir(tmp_path)
    write_index(d, {"a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00001-of-00002.safetensors",
                    "ghost.weight": "model-00001-of-00002.safetensors"}, 20)
    # engine order: the count gate (validate_split) fires at construction, naming
    # tensors from both sides — ghost.weight is named in the weight_map list.
    with pytest.raises(ValueError, match="ghost.weight"):
        load_sharded_model(str(d))


def test_gate_shard_tensor_not_in_weight_map(tmp_path):
    d = happy_dir(tmp_path)
    # weight_map references BOTH shards (index.json is authoritative for the shard
    # set) but declares only a.weight → Σ shard headers 2 vs weight_map 1: the count
    # gate fires and names b.weight (validate_split mirror, gguf_reader.hpp:585-590).
    write_index(d, {"a.weight": "model-00002-of-00002.safetensors"}, 8)
    with pytest.raises(ValueError, match="b.weight"):
        load_sharded_model(str(d))


def test_gate_tensor_in_two_shards(tmp_path):
    d = tmp_path / "model"
    write_safetensors(d / "model-00001-of-00002.safetensors", {"a.weight": T1})
    write_safetensors(d / "model-00002-of-00002.safetensors", {"a.weight": T1})
    # weight_map references both shards (2 entries, count gate passes), a.weight is
    # in both headers → the duplicate gate fires (engine GgufModel ctor,
    # gguf_reader.hpp:536-540: "tensor ... is in two shards").
    write_index(d, {"a.weight": "model-00001-of-00002.safetensors",
                    "a2.weight": "model-00002-of-00002.safetensors"}, 16)
    with pytest.raises(ValueError, match="two shards"):
        load_sharded_model(str(d))


def test_gate_total_size_mismatch(tmp_path):
    d = happy_dir(tmp_path)
    write_index(d, {"a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors"}, 19)
    with pytest.raises(ValueError, match="total_size"):
        load_sharded_model(str(d))


def test_gate_payload_outside_shard_file(tmp_path):
    d = tmp_path / "model"
    # header claims 12 B payload, file ships 8 B → the in_bounds mirror gate fires
    # (parse_header's own data-section check refuses first, same rule as the engine
    # in_bounds, gguf_reader.hpp:557-562)
    write_safetensors(d / "model-00001-of-00002.safetensors", {"a.weight": T1})
    write_safetensors(d / "model-00002-of-00002.safetensors",
                      {"b.weight": ("U8", [3, 4], T2[2][:8])},
                      extra_header={"b.weight": {"dtype": "U8", "shape": [3, 4],
                                                 "data_offsets": [0, 12]}})
    write_index(d, {"a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors"}, 20)
    with pytest.raises(ValueError, match="exceeds file size"):
        load_sharded_model(str(d))


def test_locate_index_plain_index_json(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "index.json").write_text(json.dumps(
        {"metadata": {"total_size": 8}, "weight_map": {"a.weight": "x.safetensors"}}))
    assert locate_index(str(d)).name == "index.json"
