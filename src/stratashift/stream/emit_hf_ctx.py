"""State shared by the HF → engine GGUF emit steps (p6-08).

Before p6-08 these were local variables of one 525-line function that ten nested
closures read; the steps now take this context explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

from stratashift.monitor.sampler import Sampler
from stratashift.polystrata.gguf_stream_writer import StreamingGGUFWriter
from stratashift.stream.reader import TensorReader


@dataclass(frozen=True)
class ArchTables:
    """Per-architecture emit tables (p5-06 AC3: gemma4 layers live under
    `model.language_model.layers.<L>.` with the engine Gemma4 key tables; every other
    arch keeps the p4-01 prefix/tables)."""
    layer_prefix: str
    emit_2d: list
    emit_1d: list
    layer_dense: dict
    global_dense: dict


@dataclass
class EmitContext:
    model: str
    meta: object            # ModelMeta (single file or multi-shard)
    by_name: dict
    w: StreamingGGUFWriter
    reader: TensorReader
    sampler: Sampler
    dense_type: str
    imatrix_stats: dict | None
    tables: ArchTables
    derived: dict
    guard_keys: dict
    cfg_json: dict | None
    expert_refs: dict | None = None  # layer → per-expert refs, built on first use
