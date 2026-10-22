"""Pieces shared by the GGUF emit paths (moved out of cli/main.py, p6-08)."""
from __future__ import annotations

import os
import pathlib
import sys

from stratashift.ingest.gguf import parse_gguf_header
from stratashift.polystrata import engine_tools


def require_engine_tools() -> bool:
    """False (with a stderr message) when the pinned engine tools checkout is absent."""
    if engine_tools.available():
        return True
    print("emit-gguf requires the pinned engine tools checkout "
          f"(STRATASHIFT_ENGINE_TOOLS={engine_tools.ENGINE_TOOLS})", file=sys.stderr)
    return False


def finalize_gguf(w, out_path: str) -> pathlib.Path:
    """Write the GGUF and cut the file flush at the end of the data section.

    Engine convention: the data section must end flush at EOF. The engine gguf_writer pads
    EVERY tensor to 64 B (tools/gguf_writer.py:177-179, pin c566ff4), so a last tensor that
    is not a 64-multiple (e.g. PLE IQ4_NL 720 B) leaves trailing padding. strata_pack.py
    verify computes data_section_offset = filelen - max(offset + expected_bytes)
    (tools/canonical_xcheck.py:619-625), which then reads every tensor shifted by the
    trailing pad -> all expert checks MISMATCH. p3-01 fixture GGUF was flush (last tensor
    262144 B = 64-multiple). Truncate to data_start + max(offset + expected_bytes).
    """
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    w.write(out)
    end_meta = parse_gguf_header(str(out))
    data_end = end_meta.data_start + max(t.offset + t.expected_bytes() for t in end_meta.tensors)
    if out.stat().st_size > data_end:
        os.truncate(str(out), data_end)
    return out
