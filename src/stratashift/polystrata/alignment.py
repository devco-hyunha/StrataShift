"""Alignment helpers.

- safetensors data section: 8-byte alignment (HF spec, p1-01 ingest).
- GGUF data section: `general.alignment`, default 32
  (docs/rea/struct_layouts.md §1; DECOMPILED 0x20, loader_entrypoints.md F4).
"""
from __future__ import annotations

from stratashift.polystrata.headers import GGUF_ALIGN_DEFAULT, SAFETENSORS_ALIGN


def ceil_align(pos: int, align: int) -> int:
    """Round pos up to the next multiple of align (GGUF data_start rule)."""
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    return (pos + align - 1) // align * align


def gguf_data_start(header_end: int, alignment: int = GGUF_ALIGN_DEFAULT) -> int:
    return ceil_align(header_end, alignment)


def safetensors_data_start(header_end: int) -> int:
    return ceil_align(header_end, SAFETENSORS_ALIGN)
