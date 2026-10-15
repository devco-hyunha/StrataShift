"""Gate: polystrata constants must match docs/rea recorded values.

The docs are the source of truth (AGENTS: no magic without docs/rea or pin).
This test extracts the documented values from docs/rea/*.md and asserts the
Python constants agree. It also gates the index schema against the pinned
source description (19 fields) and fixture rows.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from stratashift.polystrata import headers as H
from stratashift.polystrata.alignment import ceil_align, gguf_data_start
from stratashift.polystrata.index import INDEX_FIELDS, parse_index_header, parse_index_row

REA = Path(__file__).resolve().parent.parent / "docs" / "rea"


def read_rea() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in REA.glob("*.md"))


def test_constants_match_docs_rea() -> None:
    text = read_rea()
    assert "0x46554747" in text and H.GGUF_MAGIC == 0x46554747
    assert "0x50525453" in text and H.STRP_MAGIC == b"STRP"
    assert "1,382,400" in text and H.BLOB_BYTES == 1_382_400
    assert "0x151800" in text and H.BLOB_BYTES == 0x151800
    assert "512" in text and H.BLOBS_PER_LAYER == 512
    assert "18 B" in text and H.Q2_0_BLOCK_BYTES == 18
    assert "QK 64" in text and H.QK2_0 == 64
    assert "19" in text and H.INDEX_FIELDS == 19


def test_alignment_rules() -> None:
    assert gguf_data_start(100, 32) == 128          # ceil(pos, 32) — GGUF rule
    assert gguf_data_start(32, 32) == 32
    assert ceil_align(8, 8) == 8
    with pytest.raises(ValueError):
        ceil_align(1, 0)


def test_index_header_and_rows() -> None:
    hdr = parse_index_header("# align 32 pool 100 tensors 2")
    assert (hdr.align, hdr.pool, hdr.tensors) == (32, 100, 2)

    row = " ".join(["w.name", "0", "0", "0", "16", "0", "16",
                    "4", "2", "2", "1", "64", "0",
                    "0", "16", "2", "0", "1", "0"])
    parsed = parse_index_row(row)
    assert parsed.values["name"] == "w.name"
    assert len(parsed.values) == INDEX_FIELDS

    with pytest.raises(ValueError):  # 18 fields — refuse
        parse_index_row(" ".join(["n"] + ["0"] * 17))
    with pytest.raises(ValueError):  # non-numeric field
        parse_index_row(" ".join(["n", "x"] + ["0"] * 17))
