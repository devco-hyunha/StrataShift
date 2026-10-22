"""Pinned engine tools checkout (read-only) and its GGUF writer, failure-tolerant import.

Moved out of cli/main.py (p6-08). The engine tools are read-only (p0-01,
`.local/polystrata`), the engine is never edited. When the checkout is absent the emit
paths are unavailable (`available()` is False) while convert/verify keep working.

p6-09: the import is deferred to the first `load()` / `available()` call so the CLI
`--engine-src` option can pick the checkout first; the path itself comes from
`engine_paths.resolve_engine_paths` (option → environment → in-repo `.local/` default).
The first load wins: a later call with another `engine_src` does not re-import.
"""
from __future__ import annotations

import pathlib
import sys

from stratashift.polystrata.engine_paths import resolve_engine_paths

GGUFWriter = _Tensor = gguf_writer = None
ENGINE_TOOLS: pathlib.Path | None = None  # the tools dir the import was attempted from


def load(engine_src: str | None = None) -> bool:
    """Import the engine `gguf_writer` once from the resolved tools dir; True if available."""
    global GGUFWriter, _Tensor, gguf_writer, ENGINE_TOOLS
    if ENGINE_TOOLS is not None:
        return GGUFWriter is not None
    ENGINE_TOOLS = resolve_engine_paths(engine_src).tools
    if str(ENGINE_TOOLS) not in sys.path:
        sys.path.append(str(ENGINE_TOOLS))
    # TYPE_IDS lacks IQ4_NL (id 20, gguf_reader GGML_TYPES) — harness-side extension
    # only, engine files untouched (p3-01 harness convention).
    try:
        from gguf_writer import GGUFWriter as writer_cls, _Tensor as tensor_cls
        import gguf_writer as writer_mod
    except ImportError:  # engine checkout absent: emit path unavailable, convert/verify unaffected
        return False
    writer_mod.TYPE_IDS["IQ4_NL"] = 20
    writer_mod.TYPE_IDS["Q4_0"] = 2  # GGML_TYPES id, engine gguf_reader.py:21 (pin c566ff4)
    writer_mod.TYPE_IDS["Q4_K"] = 12  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    writer_mod.TYPE_IDS["Q5_K"] = 13  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    writer_mod.TYPE_IDS["Q6_K"] = 14  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    writer_mod.TYPE_IDS["Q3_K"] = 11  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    writer_mod.TYPE_IDS["Q8_0"] = 8   # GGML_TYPES id, engine gguf_reader.py GGML_TYPES
                                      # (our ingest/gguf.py GGML_TYPES:24 agrees)
    writer_mod.TYPE_IDS["BF16"] = 30  # GGML_TYPES id, engine gguf_reader.py GGML_TYPES
                                      # (pin c566ff4; ffn_gate_inp_shexp Bf16InF32, p4-01)
    GGUFWriter, _Tensor, gguf_writer = writer_cls, tensor_cls, writer_mod
    return True


def available() -> bool:
    return load()
