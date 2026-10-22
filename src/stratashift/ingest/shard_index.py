"""HF multi-shard safetensors input: model-00001-of-NNNNN.safetensors + index.json.

HF transformers checkpoint convention (docs/work/p4-02.md §근거 1):
    {"metadata": {"total_size": N}, "weight_map": {tensor_name: "<shard>.safetensors"}}
Tensor name → shard file; payload offsets are per-shard `data_offsets` relative to
that shard's data_start (safetensors spec, p1-01).

Validation gates mirror the pinned engine's split-model rules (pin c566ff4, read-only):
- missing shard → error naming the shard (gguf_split.hpp:44-46, VERIFIED 2026-10-17);
- shard tensor directories must add up (validate_split, gguf_reader.hpp:565-590 —
  "the shards hold X tensors, but split.tensors.count is Y"); mirrored here as
  bidirectional weight_map ↔ shard-header transcription equality;
- a tensor present in two shards is refused (GgufModel ctor, gguf_reader.hpp:536-540);
- payload must lie inside its own shard file (in_bounds, gguf_reader.hpp:557-562);
- total_size == Σ shard payload bytes (HF convention gate, recorded as convention).

CONSTRAINTS C1/C2: only the 8-byte length + header JSON of each shard is read
(parse_header); the data blob is never copied here.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from stratashift.ingest.safetensors_header import ModelMeta, TensorMeta, parse_header

INDEX_BASENAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class ShardIndex:
    index_path: str
    total_size: int
    weight_map: dict[str, str]  # tensor name → shard file basename


def locate_index(model: str) -> pathlib.Path:
    """Resolve the index.json for a directory model (dir/index.json,
    dir/model.safetensors.index.json, dir/model.safetensors.index.json)."""
    d = pathlib.Path(model)
    if not d.is_dir():
        raise ValueError(f"{model}: not a directory (single-file models take the file path)")
    for cand in (d / INDEX_BASENAME, d / "index.json"):
        if cand.is_file():
            return cand
    raise ValueError(f"{model}: no {INDEX_BASENAME} (or index.json) in the directory")


def parse_index(index_path: pathlib.Path) -> ShardIndex:
    doc = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = doc.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: missing/empty weight_map")
    total = doc.get("metadata", {}).get("total_size")
    if not isinstance(total, int):
        raise ValueError(f"{index_path}: metadata.total_size missing or not an int")
    return ShardIndex(index_path=str(index_path), total_size=total, weight_map=weight_map)


def load_sharded_model(model: str) -> ModelMeta:
    """Parse index.json + every shard header (headers only — C1/C2) and run the
    engine-mirrored gates; returns a ModelMeta with per-tensor `shard` tags and a
    `shard_data_start` registry (shard basename → absolute data_start)."""
    d = pathlib.Path(model)
    index = parse_index(locate_index(model))

    # Gate 0 (engine gguf_split.hpp:44-46): every shard named in weight_map must exist.
    shard_files: dict[str, pathlib.Path] = {}
    for shard in sorted(set(index.weight_map.values())):
        p = d / shard
        if not p.is_file():
            raise ValueError(f"missing model shard {shard} (named by {INDEX_BASENAME}; "
                             f"is the download complete?)")
        shard_files[shard] = p

    # Shard headers (C1/C2: 8 + header_len bytes each, never the blob).
    shard_meta = {s: parse_header(str(p)) for s, p in shard_files.items()}

    # Gate 1 (validate_split mirror, gguf_reader.hpp:565-590): the shard tensor
    # directories must add up to the declared total — mirrored as Σ shard-header
    # tensor counts vs the weight_map count (HF index.json is the tensor directory).
    # The engine fires this at construction, before the index is built, so it runs
    # here before the duplicate gate; the message names tensors from both sides.
    header_names = {t.name for m in shard_meta.values() for t in m.tensors}
    map_names = set(index.weight_map)
    total_header = sum(len(m.tensors) for m in shard_meta.values())
    if total_header != len(index.weight_map):
        raise ValueError(f"shard tensor directories do not add up: Σ shard headers "
                         f"{total_header} tensors {sorted(header_names)[:3]}, weight_map "
                         f"declares {len(index.weight_map)} tensors {sorted(map_names)[:3]}")

    # Gate 2 (engine GgufModel ctor, gguf_reader.hpp:536-540): a tensor present in
    # two shards is refused — GGUF has no index to say which one is meant.
    seen: dict[str, str] = {}
    for s, m in shard_meta.items():
        for t in m.tensors:
            if t.name in seen:
                raise ValueError(f"tensor {t.name!r} is in two shards "
                                 f"({seen[t.name]} and {s})")
            seen[t.name] = s

    # Gate 3 (bidirectional transcription, mirror of the engine index build): every
    # weight_map tensor must be in a shard header and vice versa (the shard header is
    # authoritative for dtype/shape/offsets; weight_map carries name → shard only).
    if header_names != map_names:
        missing = sorted(map_names - header_names)
        extra = sorted(header_names - map_names)
        raise ValueError(f"weight_map/shard transcription mismatch: "
                         f"{len(missing)} in weight_map not in shard headers {missing[:3]}, "
                         f"{len(extra)} in shard headers not in weight_map {extra[:3]}")

    # Gate 4 (in_bounds mirror, gguf_reader.hpp:557-562): payload inside its own shard.
    for s, m in shard_meta.items():
        for t in m.tensors:
            b, e = t.data_offsets
            if m.data_start + e > m.file_size:
                raise ValueError(f"tensor {t.name!r} payload exceeds shard {s} "
                                 f"({m.data_start + e} > {m.file_size})")

    # Gate 5 (HF convention): total_size == Σ tensor payload bytes.
    tensor_sum = sum(t.nbytes() for m in shard_meta.values() for t in m.tensors)
    if tensor_sum != index.total_size:
        raise ValueError(f"total_size {index.total_size} != Σ tensor bytes {tensor_sum}")

    tensors: list[TensorMeta] = []
    for s, m in shard_meta.items():
        for t in m.tensors:
            tensors.append(TensorMeta(name=t.name, dtype=t.dtype, shape=t.shape,
                                      data_offsets=t.data_offsets, shard=s))
    tensors.sort(key=lambda t: (t.name, t.shard))
    merged_metadata: dict = {}
    for m in shard_meta.values():
        merged_metadata.update(m.metadata)
    return ModelMeta(path=str(d), tensors=tensors,
                     header_size=sum(m.header_size for m in shard_meta.values()),
                     data_start=0, file_size=sum(m.file_size for m in shard_meta.values()),
                     metadata=merged_metadata,
                     shard_data_start={s: m.data_start for s, m in shard_meta.items()})


def abs_offset(meta: ModelMeta, t: TensorMeta) -> int:
    """Absolute file offset of tensor t in its shard file (shard registry via p4-02)."""
    shard = t.shard or meta.path
    return meta.shard_data_start.get(shard, meta.data_start) + t.data_offsets[0]
