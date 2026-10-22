"""Pinned engine tools checkout (read-only) and its GGUF writer, failure-tolerant import.

Moved out of cli/main.py (p6-08). The engine tools are read-only (p0-01,
`.local/polystrata`), the engine is never edited. When the checkout is absent the emit
paths are unavailable (`available()` is False) while convert/verify keep working.
"""
from __future__ import annotations

import os
import sys

# pinned engine checkout (p0-01, .local/polystrata) — read-only tools, engine never edited.
ENGINE_TOOLS = os.environ.get("STRATASHIFT_ENGINE_TOOLS", "/opt/data/polystrata-src/tools")
if ENGINE_TOOLS not in sys.path:
    sys.path.append(ENGINE_TOOLS)

# engine writer is imported at module load, failure tolerated (engine tools are read-only,
# p3-01 harness convention); TYPE_IDS lacks IQ4_NL (id 20, gguf_reader GGML_TYPES) —
# harness-side extension only, engine files untouched.
try:
    from gguf_writer import GGUFWriter, _Tensor  # noqa: E402
    import gguf_writer  # noqa: E402
    gguf_writer.TYPE_IDS["IQ4_NL"] = 20
    gguf_writer.TYPE_IDS["Q4_0"] = 2  # GGML_TYPES id, engine gguf_reader.py:21 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q4_K"] = 12  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q5_K"] = 13  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q6_K"] = 14  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q3_K"] = 11  # GGML_TYPES id, engine gguf_reader.py:22 (pin c566ff4)
    gguf_writer.TYPE_IDS["Q8_0"] = 8   # GGML_TYPES id, engine gguf_reader.py GGML_TYPES
                                       # (our ingest/gguf.py GGML_TYPES:24 agrees)
    gguf_writer.TYPE_IDS["BF16"] = 30  # GGML_TYPES id, engine gguf_reader.py GGML_TYPES
                                       # (pin c566ff4; ffn_gate_inp_shexp Bf16InF32, p4-01)
except ImportError:  # engine checkout absent: emit path unavailable, convert/verify unaffected
    GGUFWriter = _Tensor = gguf_writer = None


def available() -> bool:
    return GGUFWriter is not None
