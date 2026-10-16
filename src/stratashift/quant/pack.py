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

from stratashift.polystrata.headers import (
    Q2_0_BLOCK_BYTES,
    Q2_0_CODES_PER_BYTE,
    Q4_0_BLOCK_BYTES,
    Q4_K_BLOCK_BYTES,
    Q4_K_GROUPS,
    Q5_K_BLOCK_BYTES,
    QK2_0,
    QK4_0,
    QK_K,
)

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


def q4_0_unpack(block: bytes) -> list[float]:
    """Unpack one 18-byte Q4_0 block into 32 f32 values (engine contract).

    Transcription of the pinned engine's `dequantize_q4_0` (include/strata/artifact/
    dequant.hpp:88-95, pin c566ff4): low nibble of qs[j] is element j, high nibble is
    element j+16, and the `- 8` lands on the INTEGER before the multiply — the codebook
    is -8..7, not 0..15.
    """
    if len(block) != Q4_0_BLOCK_BYTES:
        raise ValueError(f"Q4_0 block must be {Q4_0_BLOCK_BYTES} B, got {len(block)}")
    d = fp16_to_fp32(block[0:2])
    qs = block[2:Q4_0_BLOCK_BYTES]
    return [((qs[j] & 0x0F) - 8) * d for j in range(16)] \
        + [((qs[j] >> 4) - 8) * d for j in range(16)]


def quantize_q4_0_stream(values: bytes) -> bytes:
    """float32 stream (multiple of QK4_0 elements) → concatenated 18 B Q4_0 blocks.

    Contract (docs/work/p3-03.md AC1, evidence-locked 2026-10-16): the engine ships NO
    quantizer (gguf_writer.py is Q2_0-only; dequant.hpp has dequant only), so the
    producer rule is the standard llama.cpp `quantize_row_q4_0_ref` (amax/7 + trunc+8.5),
    recorded as HYPOTHESIS in docs/polystrata_spec_draft.md — the PACK FORMAT constants
    (18 B, 32 elems, nibble layout, bias −8) are pin-verified, the quantizer rule is not.

        d    = fp16(amax / 7)                      (grid lands in codebook −7..+7)
        q    = clip(trunc(w * (1/d) + 8.5), 0, 15) (d == 0 → q 8 → v 0)
        qs[j] = q[j] | q[j+16] << 4                (low nibble = element j, high = j+16,
                                                    SOURCE dequant.hpp:92-93)

    Invariants verified in tests/test_q4_0.py:
      - `q4_0_unpack` (engine dequant.hpp:88-95 transcription) == (q−8)*d bit-exactly;
      - gguf-py `dequantize` agrees bit-for-bit with `q4_0_unpack` on our blocks
        (the reference strata_pack verify compares against);
      - quantization error |w − v| ≤ d/2 per row (fp16 tolerance factor 1.005).
    gguf-py `quantize(Q4_0)` is NOT byte-identical to this rule (signed-max d = max/−8,
    quants.py:220-237) — divergence measured, recorded as HYPOTHESIS, not a contract.
    Vectorized per 2^20-element segment (peak = segment, C1/C2).
    """
    import numpy as np

    w = np.frombuffer(values, dtype="<f4")
    if w.size % QK4_0:
        raise ValueError(f"Q4_0 needs a multiple of {QK4_0} elements, got {w.size}")
    out = bytearray()
    seg = 1 << 20  # 1 Mi elements = 32,768 blocks per segment
    for start in range(0, w.size, seg):
        b = w[start:start + seg].reshape(-1, QK4_0)
        amax = np.max(np.abs(b), axis=1)
        d = amax / 7.0
        with np.errstate(divide="ignore"):  # d == 0 rows: rid is masked to 0 by np.where
            rid = np.where(d > 0, 1.0 / d, 0.0)
        q = np.clip(np.trunc(b * rid[:, None] + 8.5).astype(np.int32), 0, 15)
        qs = q[:, 0:16].astype(np.uint8) | (q[:, 16:32].astype(np.uint8) << 4)
        d16 = d.astype(np.float16)
        dd = d16.view(np.uint16)
        inter = np.empty((b.shape[0], Q4_0_BLOCK_BYTES), dtype=np.uint8)
        inter[:, 0] = (dd & 0xFF).astype(np.uint8)
        inter[:, 1] = ((dd >> 8) & 0xFF).astype(np.uint8)
        inter[:, 2:] = qs
        out += inter.tobytes()
    return bytes(out)


def get_scale_min_k4(scb: bytes, j: int) -> tuple[int, int]:
    """Transcription of `get_scale_min_k4` (ggml-quants.c L880, canonical_xcheck.py:111,
    pin c566ff4): the 6-bit scale and min of group j (0..7) packed in the 12 scale bytes.
    Low four groups keep their 6 bits in a byte; high four borrow the top two bits of the
    LOW groups' bytes. Reading only q[j] is right for the first half and silently wrong for
    the second — which is why this is transcribed, not inferred."""
    if j < 4:
        return int(scb[j]) & 63, int(scb[j + 4]) & 63
    return (((int(scb[j + 4]) & 0xF) | ((int(scb[j - 4]) >> 6) << 4)),
            ((int(scb[j + 4]) >> 4) | ((int(scb[j]) >> 6) << 4)))


def q4_k_unpack(block: bytes) -> list[float]:
    """Unpack Q4_K bytes (one 144 B block, or a stream of blocks) into f32 values.

    Transcription of `to_s4_q4_k` + `from_s4_q4_k` (canonical_xcheck.py:128-160, 567,
    pin c566ff4): d = fp16(block[0:2]), dmin = fp16(block[2:4]), 12 packed scale bytes
    (block[4:16]) → 8 groups of (sc, m) via get_scale_min_k4; scales = d*sc,
    offsets = -(dmin*m); qs (block[16:144]): low nibbles of the j-th 32-byte step are
    group is+0, high nibbles group is+1 — 8 groups of 32, NOT interleaved.

    ALL arithmetic is float32 (numpy): the engine dequant multiplies in f32
    (`codes.astype(np.float32) * scales`), so a Python-float (f64) product here
    diverges in the last bits and breaks the bit-exact round-trip.
    """
    import numpy as np

    if len(block) % Q4_K_BLOCK_BYTES:
        raise ValueError(f"Q4_K stream must be a multiple of {Q4_K_BLOCK_BYTES} B, "
                         f"got {len(block)}")
    out: list[float] = []
    for i in range(0, len(block), Q4_K_BLOCK_BYTES):
        b = block[i:i + Q4_K_BLOCK_BYTES]
        d = np.float32(fp16_to_fp32(b[0:2]))
        dmin = np.float32(fp16_to_fp32(b[2:4]))
        scb = b[4:16]
        qs = np.frombuffer(b[16:Q4_K_BLOCK_BYTES], dtype=np.uint8)
        for g in range(Q4_K_GROUPS):
            sc, m = get_scale_min_k4(scb, g)
            d_g = np.float32(d * np.float32(sc))
            m_g = np.float32(dmin * np.float32(m))
            base = (g // 2) * 32
            if g % 2 == 0:  # group is+0: low nibbles
                codes = (qs[base:base + 32] & 0x0F).astype(np.float32)
            else:          # group is+1: high nibbles
                codes = (qs[base:base + 32] >> 4).astype(np.float32)
            out.extend((codes * d_g - m_g).tolist())
    return out


def quantize_q4_k_stream(values: bytes) -> bytes:
    """float32 stream (multiple of QK_K elements) → concatenated 144 B Q4_K blocks.

    Contract (docs/work/p3-04.md AC1, evidence-locked 2026-10-16): the engine ships NO
    quantizer (gguf_writer.py is Q2_0-only; dequant.hpp has dequant only), so the producer
    rule is a HYPOTHESIS validated ONLY by the engine dequant round-trip
    (`q4_k_unpack` == to_s4_q4_k+from_s4_q4_k transcription):

        d    = fp16(amax / 63)                       (6-bit scale range, ggml convention)
        sc_g = clip(round(group_amax_g / d), 1, 63)  (8 groups of 32; sc >= 1)
        m_g  = 0                                     (offsets plane = 0 — the simple case)
        code = clip(round(w / (d*sc_g)), 0, 15)       (4-bit, codebook 0..15, bias 0)
        qs[(g//2)*32+p] = code_g_even | code_g_odd << 4   (low nibble = group is+0,
                                                          high = group is+1)
        scale bytes: scb[j] = sc_j | (sc_{j+4}>>4)<<6 (j<4), scb[4..7] = 0 (m=0),
        scb[8..11] = sc_{j+4} & 0xF (j<4) — the exact get_scale_min_k4 packing with
        mins zeroed. Groups 4..7 sc must be multiples of 16 for the round-trip to be
        bit-exact under this packing (sc_j = (scb[j+4]&0xF) | (scb[j-4]>>6)<<4).

    Vectorized per 2^20-element segment (peak = segment, C1/C2).
    """
    import numpy as np

    w = np.frombuffer(values, dtype="<f4")
    if w.size % QK_K:
        raise ValueError(f"Q4_K needs a multiple of {QK_K} elements, got {w.size}")
    out = bytearray()
    seg = 1 << 20  # 1 Mi elements = 4096 blocks per segment
    for start in range(0, w.size, seg):
        b = w[start:start + seg].reshape(-1, QK_K)
        amax = np.max(np.abs(b), axis=1)
        d = amax / 63.0
        d16 = d.astype(np.float16)
        d32 = d16.astype(np.float32)  # round-trip uses the fp16-rounded d
        groups = b.reshape(-1, Q4_K_GROUPS, 32)
        gmax = np.max(np.abs(groups), axis=2)
        with np.errstate(divide="ignore"):
            sc = np.where(d32[:, None] > 0, np.rint(gmax / d32[:, None]), 0).astype(np.int32)
        sc = np.clip(sc, 1, 63)
        # ggml Q4_K uses a per-group MIN m to shift the 0..15 codebook so negative
        # weights are representable: v = d*sc*code - dmin*m, code = clip(round((w+dmin*m)/(d*sc)),0,15).
        # With dmin = fp16(amax/63) = d, the min is m = round(-min_w/d) clipped 0..63, and the
        # group scale sc = clip(round((gmax - min_w)/d), 1, 63) (the group range over d).
        gmin = np.min(groups, axis=2)  # (n, 8) most-negative per group
        rng = gmax - gmin
        with np.errstate(divide="ignore"):
            sc = np.where(d32[:, None] > 0, np.rint(rng / d32[:, None]), 0).astype(np.int32)
        sc = np.clip(sc, 1, 63)
        m = np.where(d32[:, None] > 0, np.rint(-gmin / d32[:, None]), 0).astype(np.int32)
        m = np.clip(m, 0, 63)
        # get_scale_min_k4 packing: groups 4..7 mins live in scb[8..11] HIGH 4 bits
        # (scb[8+j] = (sc & 0xF) | (m << 4)), so m must be < 16 for groups 4..7.
        m[:, 4:8] = np.clip(m[:, 4:8], 0, 15)
        scale = d32[:, None, None] * sc[:, :, None]
        shift = d32[:, None, None] * m[:, :, None]  # dmin*m, dmin == d here
        with np.errstate(divide="ignore"):
            code = np.where(scale > 0, np.rint((groups + shift) / scale), 0).astype(np.int32)
        code = np.clip(code, 0, 15)
        n = b.shape[0]
        qs = np.zeros((n, 128), dtype=np.uint8)
        for g in range(Q4_K_GROUPS):
            base = (g // 2) * 32
            if g % 2 == 0:
                qs[:, base:base + 32] |= code[:, g, :].astype(np.uint8)
            else:
                qs[:, base:base + 32] |= code[:, g, :].astype(np.uint8) << 4
        scb = np.zeros((n, 12), dtype=np.uint8)
        # Exact get_scale_min_k4 packing (canonical_xcheck.py:111, pin c566ff4):
        #   j<4:  sc_j = scb[j] & 63,  m_j = scb[j+4] & 63
        #   j>=4: sc_j = (scb[j+4] & 0xF) | ((scb[j-4] >> 6) << 4)
        #         m_j = (scb[j+4] >> 4) | ((scb[j] >> 6) << 4)
        # So scb[0..3] = sc_j | (sc_{j+4}>>4)<<6, scb[4..7] = m_j,
        # scb[8..11] = (sc_{j+4} & 0xF) | (m_{j+4} << 4).  The coupling forces
        # sc_j (j>=4) low 4 bits == scb[j+4] low 4 bits, so groups 4..7 sc must be
        # multiples of 16 (low 4 bits 0) and scb[8..11] = m_{j+4} << 4.
        for j in range(4):
            scb[:, j] = (sc[:, j] | ((sc[:, j + 4] >> 4) << 6)).astype(np.uint8)
            scb[:, j + 4] = m[:, j].astype(np.uint8)
            scb[:, j + 8] = ((sc[:, j + 4] & 0xF) | (m[:, j + 4] << 4)).astype(np.uint8)
        dd = d16.view(np.uint16)
        dm = d16.view(np.uint16)  # dmin = d (the producer's min-shift uses dmin==d)
        inter = np.empty((n, Q4_K_BLOCK_BYTES), dtype=np.uint8)
        inter[:, 0] = (dd & 0xFF).astype(np.uint8)
        inter[:, 1] = ((dd >> 8) & 0xFF).astype(np.uint8)
        inter[:, 2] = (dm & 0xFF).astype(np.uint8)
        inter[:, 3] = ((dm >> 8) & 0xFF).astype(np.uint8)
        inter[:, 4:16] = scb
        inter[:, 16:] = qs
        out += inter.tobytes()
    return bytes(out)


def q5_k_unpack(block: bytes) -> list[float]:
    """Unpack Q5_K bytes (one 176 B block, or a stream of blocks) into f32 values.

    Transcription of `to_s8_q5_k` + `from_s8_q5_k` (canonical_xcheck.py:370-416, 568,
    pin c566ff4): d = fp16(block[0:2]), dmin = fp16(block[2:4]), 12 packed scale bytes
    (block[4:16]) → 8 groups of (sc, m) via get_scale_min_k4; scales = d*sc,
    offsets = -(dmin*m). qh (block[16:48]) is the 5th-bit plane: the j-th 64-element
    step uses u1 = 1<<(2j), u2 = 2<<(2j) — the SAME 32 qh bytes are re-read by all four
    steps and the walking bits select the 5th bit per group. codes: lo = (qs[l]&0xF) +
    (qh[l]&u1 ? 16 : 0) (group is+0), hi = (qs[l]>>4) + (qh[l]&u2 ? 16 : 0) (group is+1).

    ALL arithmetic is float32 (numpy): the engine dequant multiplies in f32, so a
    Python-float (f64) product diverges in the last bits.
    """
    import numpy as np

    if len(block) % Q5_K_BLOCK_BYTES:
        raise ValueError(f"Q5_K stream must be a multiple of {Q5_K_BLOCK_BYTES} B, "
                         f"got {len(block)}")
    out: list[float] = []
    for i in range(0, len(block), Q5_K_BLOCK_BYTES):
        b = block[i:i + Q5_K_BLOCK_BYTES]
        d = np.float32(fp16_to_fp32(b[0:2]))
        dmin = np.float32(fp16_to_fp32(b[2:4]))
        scb = b[4:16]
        qh = np.frombuffer(b[16:48], dtype=np.uint8)
        qs = np.frombuffer(b[48:176], dtype=np.uint8)
        for g in range(8):
            sc, m = get_scale_min_k4(scb, g)
            d_g = np.float32(d * np.float32(sc))
            m_g = np.float32(dmin * np.float32(m))
            j = g // 2
            u = 1 << (2 * j) if g % 2 == 0 else 2 << (2 * j)
            base = j * 32
            lo = (qs[base:base + 32] & 0x0F).astype(np.float32)
            hi = (qs[base:base + 32] >> 4).astype(np.float32)
            if g % 2 == 0:
                codes = lo + np.where((qh & u) != 0, 16.0, 0.0)
            else:
                codes = hi + np.where((qh & u) != 0, 16.0, 0.0)
            out.extend((codes * d_g - m_g).tolist())
    return out


def quantize_q5_k_stream(values: bytes) -> bytes:
    """float32 stream (multiple of QK_K elements) → concatenated 176 B Q5_K blocks.

    Contract (docs/work/p3-05.md AC1, evidence-locked 2026-10-16): the engine ships NO
    quantizer, so the producer rule is a HYPOTHESIS validated ONLY by the engine dequant
    round-trip (`q5_k_unpack` == to_s8_q5_k+from_s8_q5_k transcription):

        d    = fp16(amax / 63)                       (6-bit scale range, ggml convention)
        dmin = fp16(−min/63)                          (block-min offset scale; dmin·63 = −min, so m ≤ 63 never clips)
        sc_g = clip(ceil((gmax_g − gmin_g) / (31*d)), 1, 63)  (8 groups of 32; ceil keeps 31*d*sc ≥ rng so the codebook never clips)
        m_g  = clip(round(−gmin_g / dmin), 0, 63)    (groups 4..7: m < 16, scb[8..11] high 4 bits)
        code = clip(round((w + dmin*m) / (d*sc)), 0, 31)  (5-bit, codebook 0..31, bias −16)
        qs[l] = code_low & 0xF | code_high & 0xF << 4   (low nibble = group is+0, high = is+1)
        qh[l] bit (2j) = code_low ≥ 16, bit (2j+1) = code_high ≥ 16   (the 5th-bit plane adds +16)
        scale bytes: the exact get_scale_min_k4 packing (see quantize_q4_k_stream).

    Vectorized per 2^20-element segment (peak = segment, C1/C2).
    """
    import numpy as np

    w = np.frombuffer(values, dtype="<f4")
    if w.size % QK_K:
        raise ValueError(f"Q5_K needs a multiple of {QK_K} elements, got {w.size}")
    out = bytearray()
    seg = 1 << 20
    for start in range(0, w.size, seg):
        b = w[start:start + seg].reshape(-1, QK_K)
        amax = np.max(np.abs(b), axis=1)
        bmin = np.min(b, axis=1)
        d = amax / 63.0
        d16 = d.astype(np.float16)
        d32 = d16.astype(np.float32)
        groups = b.reshape(-1, Q4_K_GROUPS, 32)
        gmax = np.max(np.abs(groups), axis=2)
        gmin = np.min(groups, axis=2)
        rng = gmax - gmin
        # dmin = −min/63 (block min): dmin·63 = −min exactly, so m = rint(−gmin/dmin) ≤ 63
        # and dmin·m ≈ −gmin for every group (gmin ≥ min ⇒ code ≥ 0).
        dmin16 = (-bmin / 63.0).astype(np.float16)
        dmin32 = dmin16.astype(np.float32)
        with np.errstate(divide="ignore"):
            sc = np.where(d32[:, None] > 0, np.ceil(rng / (31.0 * d32[:, None])), 0).astype(np.int32)
        sc = np.clip(sc, 1, 63)
        with np.errstate(divide="ignore"):
            m = np.where(dmin32[:, None] > 0, np.rint(-gmin / dmin32[:, None]), 0).astype(np.int32)
        m = np.clip(m, 0, 63)
        m[:, 4:8] = np.clip(m[:, 4:8], 0, 15)
        scale = d32[:, None, None] * sc[:, :, None]
        shift = dmin32[:, None, None] * m[:, :, None]
        with np.errstate(divide="ignore"):
            code = np.where(scale > 0, np.rint((groups + shift) / scale), 0).astype(np.int32)
        code = np.clip(code, 0, 31)
        n = b.shape[0]
        qs = np.zeros((n, 128), dtype=np.uint8)
        qh = np.zeros((n, 32), dtype=np.uint8)
        for g in range(Q4_K_GROUPS):
            j = g // 2
            base = j * 32
            c = code[:, g, :]
            lo = (c & 0xF).astype(np.uint8)
            hi = (c & 0xF).astype(np.uint8)
            bit = 1 << (2 * j) if g % 2 == 0 else 2 << (2 * j)
            if g % 2 == 0:
                qs[:, base:base + 32] |= lo
                qh[:, 0:32] |= np.where(c >= 16, bit, 0).astype(np.uint8)
            else:
                qs[:, base:base + 32] |= hi << 4
                qh[:, 0:32] |= np.where(c >= 16, bit, 0).astype(np.uint8)
        scb = np.zeros((n, 12), dtype=np.uint8)
        for j in range(4):
            scb[:, j] = (sc[:, j] | ((sc[:, j + 4] >> 4) << 6)).astype(np.uint8)
            scb[:, j + 4] = m[:, j].astype(np.uint8)
            scb[:, j + 8] = ((sc[:, j + 4] & 0xF) | (m[:, j + 4] << 4)).astype(np.uint8)
        dd = d16.view(np.uint16)
        dm = dmin16.view(np.uint16)
        inter = np.empty((n, Q5_K_BLOCK_BYTES), dtype=np.uint8)
        inter[:, 0] = (dd & 0xFF).astype(np.uint8)
        inter[:, 1] = ((dd >> 8) & 0xFF).astype(np.uint8)
        inter[:, 2] = (dm & 0xFF).astype(np.uint8)
        inter[:, 3] = ((dm >> 8) & 0xFF).astype(np.uint8)
        inter[:, 4:16] = scb
        inter[:, 16:48] = qh
        inter[:, 48:] = qs
        out += inter.tobytes()
    return bytes(out)
