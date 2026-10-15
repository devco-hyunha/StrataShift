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


def role_planes(role_slice: bytes, rows: int, blocks_per_row: int) -> tuple[bytes, bytes]:
    """Split one role slice (rows × blocks_per_row × 18 B) into codes + fp16 scales planes.

    Row-major blocks; expert e is one contiguous run inside the role tensor
    (docs/rea/struct_layouts.md §3, SOURCE pack_layer.py:24-26).
    """
    row_bytes = blocks_per_row * Q2_0_BLOCK_BYTES
    if len(role_slice) != rows * row_bytes:
        raise ValueError(f"role slice {len(role_slice)} != rows {rows} x {row_bytes}")
    codes = bytearray()
    scales = bytearray()
    for r in range(rows):
        row = role_slice[r * row_bytes:(r + 1) * row_bytes]
        for b in range(blocks_per_row):
            blk = row[b * Q2_0_BLOCK_BYTES:(b + 1) * Q2_0_BLOCK_BYTES]
            scales += blk[0:2]
            codes += blk[2:Q2_0_BLOCK_BYTES]
    return bytes(codes), bytes(scales)


def expert_blob(gate: bytes, up: bytes, down: bytes, H: int, FF: int) -> bytes:
    """Relayout one expert's (gate, up, down) slices into the engine blob.

    Layout (docs/rea/struct_layouts.md §3, SOURCE expert.hpp:44-58 + pack_layer.py:44-64,
    VERIFIED end == 1,382,400): gate/up codes rows interleaved g0,u0,g1,u1,...,
    then down codes, then gate/up scales (interleaved rows), then down scales.
    Pure relayout — codes and fp16 scales are copied verbatim, so
    dequant(pack) == dequant(ggml) bit-exact (pack_layer.py:88-93).
    """
    sc_gu = H // QK2_0   # weight blocks per gate/up row (struct_layouts §3 SC_GU=40 @ H=2560)
    sc_d = FF // QK2_0    # SC_D=10 @ FF=640
    gu_codes, gu_scales = bytearray(), bytearray()
    gate_codes, gate_scales = role_planes(gate, FF, sc_gu)
    up_codes, up_scales = role_planes(up, FF, sc_gu)
    row_c, row_s = sc_gu * 16, sc_gu * 2
    for r in range(FF):  # rows interleaved g0,u0,g1,u1,... (share input x)
        gu_codes += gate_codes[r * row_c:(r + 1) * row_c] + up_codes[r * row_c:(r + 1) * row_c]
        gu_scales += gate_scales[r * row_s:(r + 1) * row_s] + up_scales[r * row_s:(r + 1) * row_s]
    d_codes, d_scales = role_planes(down, H, sc_d)
    blob = bytes(gu_codes) + d_codes + bytes(gu_scales) + d_scales
    if len(blob) != 3 * (H * FF * Q2_0_BLOCK_BYTES // QK2_0):
        raise ValueError(f"blob {len(blob)} B does not match geometry H={H} FF={FF}")
    return blob


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


def quantize_q2_0_stream(values: bytes) -> bytes:
    """float32 stream (multiple of QK2_0 elements) → concatenated 18 B Q2_0 blocks.

    Byte-identical to the engine's `tools/gguf_writer.py quantize_q2_0` (pin c566ff4):
    d = amax of the 64-row (NOT amax/2 — the grid is asymmetric, symbols land in
    {-1,0,+1,+2}), code = clip(rint(w/d)+1, 0, 3), 4 codes/byte LSB-first.
    Vectorized per 2^20-element segment (peak = segment), cross-checked byte-for-byte
    in tests/test_safetensors_gguf.py.
    """
    import numpy as np

    w = np.frombuffer(values, dtype="<f4")
    if w.size % QK2_0:
        raise ValueError(f"Q2_0 needs a multiple of {QK2_0} elements, got {w.size}")
    out = bytearray()
    seg = 1 << 20  # 1 Mi elements = 262,144 blocks per segment
    for start in range(0, w.size, seg):
        b = w[start:start + seg].reshape(-1, QK2_0)
        d = np.max(np.abs(b), axis=1)
        code = np.zeros(b.shape, dtype=np.uint8)
        nz = d > 0
        if nz.any():
            code[nz] = np.clip(np.rint(b[nz] / d[nz, None]).astype(np.int32) + 1, 0, 3).astype(np.uint8)
        packed = np.zeros((b.shape[0], QK2_0 // Q2_0_CODES_PER_BYTE), dtype=np.uint8)
        for j in range(QK2_0):
            packed[:, j // 4] |= code[:, j] << ((j % 4) * 2)
        d16 = d.astype(np.float16)
        dd = d16.view(np.uint16)
        inter = np.empty((b.shape[0], 18), dtype=np.uint8)
        inter[:, 0] = (dd & 0xFF).astype(np.uint8)
        inter[:, 1] = ((dd >> 8) & 0xFF).astype(np.uint8)
        inter[:, 2:] = packed
        out += inter.tobytes()
    return bytes(out)
