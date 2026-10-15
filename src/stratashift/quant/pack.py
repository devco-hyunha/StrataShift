"""Q2_0 quantization/packing — pure functions, no file I/O.

Contract (docs/rea/bitpack_rules.md §1 SOURCE + §6 DECOMPILED F5, pin c566ff4…):
    block = { fp16 d; uint8 qs[16] } = 64 elements, 18 bytes
    code(j) = (qs[j//4] >> ((j%4)*2)) & 3   (LSB-first, 4 codes/byte)
    v(j) = (code - 1) * d                    → {-d, 0, +d, +2d}
Pack conversion is a PURE RELAYOUT (pack_layer.py:88-93): codes and scales are
copied verbatim into planes, so dequant(pack) == dequant(ggml) bit-exact.
"""
from __future__ import annotations

import struct

from stratashift.polystrata.headers import Q2_0_BLOCK_BYTES, Q2_0_CODES_PER_BYTE, QK2_0

_FP16 = struct.Struct("<e")  # IEEE 754 half (handles subnormals/inf exactly)


def fp16_to_fp32(scale_bytes: bytes) -> float:
    return _FP16.unpack(scale_bytes)[0]


def q2_0_unpack(block: bytes) -> list[float]:
    """Unpack one 18-byte Q2_0 block into 64 f32 values (REA contract)."""
    if len(block) != Q2_0_BLOCK_BYTES:
        raise ValueError(f"Q2_0 block must be {Q2_0_BLOCK_BYTES} B, got {len(block)}")
    d = fp16_to_fp32(block[0:2])
    qs = block[2:Q2_0_BLOCK_BYTES]
    return [(((qs[j // 4] >> ((j % 4) * 2)) & 3) - 1) * d for j in range(QK2_0)]


def q2_0_check_invariant(values: list[float], d: float) -> bool:
    """Structural invariant: every value is exactly one of {-d, 0, d, 2d}."""
    allowed = {-d, 0.0, d, 2.0 * d}
    return all(v in allowed for v in values)


def q2_0_split(tensor_bytes: bytes) -> tuple[bytes, bytes]:
    """Pure relayout: codes plane (16 B/block) + scales plane (2 B/block, fp16)."""
    if len(tensor_bytes) % Q2_0_BLOCK_BYTES:
        raise ValueError(
            f"tensor bytes {len(tensor_bytes)} not a multiple of block {Q2_0_BLOCK_BYTES}"
        )
    codes = bytearray()
    scales = bytearray()
    for i in range(0, len(tensor_bytes), Q2_0_BLOCK_BYTES):
        block = tensor_bytes[i:i + Q2_0_BLOCK_BYTES]
        scales += block[0:2]
        codes += block[2:Q2_0_BLOCK_BYTES]
    return bytes(codes), bytes(scales)


def q2_0_encode_row(values: list[float], d: float) -> bytes:
    """Encode one QK2_0 row to an 18-byte Q2_0 block (inverse of q2_0_unpack).

    code = v/d + 1, clamped to {0,1,2,3}; LSB-first packing.
    """
    codes = [0] * QK2_0
    for j, v in enumerate(values):
        code = int(round(v / d)) + 1 if d != 0 else 1
        codes[j] = min(3, max(0, code))
    qs = bytearray(QK2_0 // Q2_0_CODES_PER_BYTE)
    for j, c in enumerate(codes):
        qs[j // 4] |= c << ((j % 4) * 2)
    return _FP16.pack(d) + bytes(qs)
