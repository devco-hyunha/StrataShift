"""Q2_0 contract tests — unpack/invariant/split/encode against REA rules.

Contract source: docs/rea/bitpack_rules.md §1 (SOURCE dequant.hpp) + §6
(DECOMPILED F5 strata-dequant loop). Synthetic vectors, no model files.
"""
from __future__ import annotations

import struct

import pytest

from stratashift.polystrata.headers import Q2_0_BLOCK_BYTES, QK2_0
from stratashift.quant.pack import (
    fp16_to_fp32,
    q2_0_check_invariant,
    q2_0_encode_row,
    q2_0_split,
    q2_0_unpack,
)

D = 2.0


def make_block(codes: list[int], d: float = D) -> bytes:
    qs = bytearray(QK2_0 // 4)
    for j, c in enumerate(codes):
        qs[j // 4] |= c << ((j % 4) * 2)
    return struct.pack("<e", d) + bytes(qs)


def test_unpack_matches_rea_contract() -> None:
    codes = [0, 1, 2, 3] * (QK2_0 // 4)  # -d, 0, +d, +2d repeating
    block = make_block(codes)
    assert len(block) == Q2_0_BLOCK_BYTES
    vals = q2_0_unpack(block)
    assert vals[0] == -D and vals[1] == 0.0 and vals[2] == D and vals[3] == 2 * D
    assert q2_0_check_invariant(vals, D)
    assert not q2_0_check_invariant([D + 0.5], D)


def test_unpack_lsb_first_bit_layout() -> None:
    # code j=0 is the LOW 2 bits of qs[0]; code j=3 is the HIGH 2 bits
    qs0 = 0b00_10_01_00  # codes: j0=0, j1=1, j2=2, j3=0
    block = struct.pack("<e", 1.0) + bytes([qs0]) + bytes(15)
    vals = q2_0_unpack(block)
    assert vals[0] == -1.0 and vals[1] == 0.0 and vals[2] == 1.0 and vals[3] == -1.0


def test_split_is_bit_exact_relayout() -> None:
    blocks = [make_block([0, 1, 2, 3] * 16, d) for d in (1.0, 2.0, 4.0)]
    tensor = b"".join(blocks)
    codes, scales = q2_0_split(tensor)
    assert len(codes) == 16 * len(blocks)
    assert len(scales) == 2 * len(blocks)
    # rejoin → bit-exact original
    rebuilt = b"".join(scales[2 * i:2 * i + 2] + codes[16 * i:16 * i + 16]
                       for i in range(len(blocks)))
    assert rebuilt == tensor


def test_encode_roundtrip() -> None:
    codes = [3, 2, 1, 0] * (QK2_0 // 4)
    block = make_block(codes)
    vals = q2_0_unpack(block)
    assert q2_0_encode_row(vals, D) == block


def test_fp16_scale_exact() -> None:
    assert fp16_to_fp32(struct.pack("<e", 2.0)) == 2.0
    assert fp16_to_fp32(struct.pack("<e", 0.5)) == 0.5


def test_split_rejects_partial_block() -> None:
    with pytest.raises(ValueError):
        q2_0_split(b"\x00" * 17)
