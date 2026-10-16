"""Engine-format pack writer: manifest.json (strata-pack v1) + dense/embd plane layout.

Field schema is the pinned engine's own pack builder (tools/strata_pack.py build,
pin c566ff4) — the engine's `verify` (THE ONE RULE: decode(pack) == dequant_ggml(source)
bit-for-bit) is the acceptance gate, so our writer must agree with it field-for-field.

Plane placement (tools/strata_pack.py:117-123 `put`): every plane starts on a
64-byte boundary inside its file; offsets are absolute in the file.

Q2_0 rows are VERBATIM planes (codes 16 B/block + fp16 scales 2 B/block copied
verbatim — docs/rea/bitpack_rules.md §1, pack_layer.py:88-93).
F32 tensors are P32 pass-through. BF16 is promoted to F32 (to_p32_bf16).
"""
from __future__ import annotations

import struct

from stratashift.polystrata.headers import Q4_K_BLOCK_BYTES

# engine pack align — tools/strata_pack.py:44 ALIGN = 64 ("every plane starts on a
# 64-byte boundary (P1.S7)"), NOT the GGUF file alignment (32).
PACK_ALIGN = 64

# canonical mapping table fields for the forms we emit (tools/canonical_xcheck.py
# MAPPINGS:563-575, pin c566ff4)
FORM_INFO: dict[str, dict] = {
    "Q2_0": {"form": "S2", "codebook": None, "code_bias": -1, "has_offset": False,
            "block_elems": 64, "block_bytes": 18, "group_elems": 64, "code_bits": 2},
    "F32": {"form": "P32", "codebook": None, "code_bias": 0, "has_offset": False,
            "block_elems": 1, "block_bytes": 4, "group_elems": 1, "code_bits": 0},
    "BF16": {"form": "P32", "codebook": None, "code_bias": 0, "has_offset": False,
             "block_elems": 1, "block_bytes": 2, "group_elems": 1, "code_bits": 0},
    "IQ4_NL": {"form": "S4", "codebook": "IQ4NL", "code_bias": 0, "has_offset": False,
               "block_elems": 32, "block_bytes": 18, "group_elems": 32, "code_bits": 4},
    "Q4_0": {"form": "S4", "codebook": None, "code_bias": -8, "has_offset": False,
             "block_elems": 32, "block_bytes": 18, "group_elems": 32, "code_bits": 4},
    "Q4_K": {"form": "S4", "codebook": None, "code_bias": 0, "has_offset": True,
             "block_elems": 256, "block_bytes": 144, "group_elems": 32, "code_bits": 4},
}

# IQ4NL codebook — tools/canonical_xcheck.py:260 KV_IQ4NL (pin c566ff4): the ggml
# kvalues_iq4nl table as float32. The pack manifest carries this exact table
# (tools/strata_pack.py:246), so our manifest must carry the identical 16 values.
IQ4NL_CODEBOOK: tuple[float, ...] = (
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
    1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0,
)


def align_up(n: int, a: int = PACK_ALIGN) -> int:
    return (n + a - 1) // a * a


def bf16_to_f32_bytes(raw: bytes) -> bytes:
    """BF16 → promoted F32 (to_p32_bf16: a bf16 value IS the top half of its f32 image)."""
    out = bytearray()
    for i in range(0, len(raw), 2):
        out += struct.pack("<I", struct.unpack("<H", raw[i:i + 2])[0] << 16)
    return bytes(out)


def q2_0_planes(raw: bytes) -> tuple[bytes, bytes]:
    """Q2_0 canonical planes: codes (16 B/block) + fp16 scales (2 B/block), verbatim.

    scales_fp16 exactness: the pack stores fp16 scales when the round-trip is exact
    (tools/strata_pack.py:24-27 SCALE PRECISION); Q2_0 scales ARE fp16 by construction
    (docs/rea/bitpack_rules.md §1), so scales_fp16=True and bytes = n_blocks*2.
    """
    if len(raw) % 18:
        raise ValueError(f"Q2_0 raw {len(raw)} not a multiple of 18")
    codes = bytearray()
    scales = bytearray()
    for i in range(0, len(raw), 18):
        scales += raw[i:i + 2]
        codes += raw[i + 2:i + 18]
    return bytes(codes), bytes(scales)


def _nibble_s4_planes(raw: bytes) -> tuple[bytes, bytes]:
    """Shared S4 planes for the nibble-packed 18 B blocks (Q4_0, IQ4_NL).

    Code order is the canonical nibble layout — `to_s4_q4_0`
    (tools/canonical_xcheck.py:86-105) and `to_s4_iq4_nl` (:264, pin c566ff4) agree:
        codes[0:16] = qs[j] & 0xF, codes[16:32] = qs[j] >> 4   (j = 0..15)
    then pack_codes (tools/strata_pack.py:60) packs 2 codes/byte LSB-first, so the
    packed byte k = codes[2k] | codes[2k+1] << 4 — NOT the raw qs bytes.
    """
    if len(raw) % 18:
        raise ValueError(f"raw {len(raw)} not a multiple of 18")
    codes = bytearray()
    scales = bytearray()
    for i in range(0, len(raw), 18):
        scales += raw[i:i + 2]
        qs = raw[i + 2:i + 18]
        c = [qs[j] & 0x0F for j in range(16)] + [qs[j] >> 4 for j in range(16)]
        codes += bytes(c[2 * k] | (c[2 * k + 1] << 4) for k in range(16))
    return bytes(codes), bytes(scales)


def iq4_nl_planes(raw: bytes) -> tuple[bytes, bytes]:
    """IQ4_NL canonical planes: S4 codes (16 B/block) + fp16 scales (2 B/block).

    Same nibble layout as Q4_0 (see `_nibble_s4_planes`); the manifest carries
    codebook IQ4NL and code_bias 0, so the planes are byte-identical and only the
    decode table differs.
    """
    return _nibble_s4_planes(raw)


def q4_0_planes(raw: bytes) -> tuple[bytes, bytes]:
    """Q4_0 canonical planes: S4 codes (16 B/block) + fp16 scales (2 B/block).

    `to_s4_q4_0` (tools/canonical_xcheck.py:86-105, pin c566ff4): low nibbles fill the
    FIRST half of the block, high nibbles the SECOND (not interleaved), and `from_s4_q4_0`
    applies the `- 8` to the integer before the multiply — codebook -8..7. scales_fp16
    is exact by construction: d IS fp16 (quantize_q4_0_stream), so
    choose_scale_dtype (tools/strata_pack.py:90-95) returns fp16.
    """
    return _nibble_s4_planes(raw)


def q4_k_planes(raw: bytes) -> tuple[bytes, bytes, bytes]:
    """Q4_K canonical planes: S4 codes (128 B/block) + f32 scales (32 B/block) +
    f32 offsets (32 B/block) — the 3-plane form (has_offset True).

    `to_s4_q4_k` (tools/canonical_xcheck.py:128-160, pin c566ff4): d = fp16(block[0:2]),
    dmin = fp16(block[2:4]), 12 packed scale bytes (block[4:16]) → 8 groups of (sc, m)
    via get_scale_min_k4; scales = d*sc, offsets = -(dmin*m). qs (block[16:144]): the
    j-th 32-byte step's LOW nibbles are group is+0, HIGH nibbles group is+1 — 8 groups
    of 32 in order g0..g7, NOT interleaved. Then pack_codes (tools/strata_pack.py:60)
    packs 2 codes/byte LSB-first. scales/offsets are f32 (d*sc is not fp16-exact, so
    choose_scale_dtype returns f32; our producer's offsets are zeros → fp16-exact).
    """
    import numpy as np

    if len(raw) % Q4_K_BLOCK_BYTES:
        raise ValueError(f"Q4_K raw {len(raw)} not a multiple of {Q4_K_BLOCK_BYTES}")
    n = len(raw) // Q4_K_BLOCK_BYTES
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n, Q4_K_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    scb = blocks[:, 4:16]
    qs = blocks[:, 16:]
    codes = np.zeros((n, 256), dtype=np.uint8)
    scales = np.zeros((n, 8), dtype=np.float32)
    offsets = np.zeros((n, 8), dtype=np.float32)
    for g in range(8):
        base = (g // 2) * 32
        if g < 4:
            sc = scb[:, g] & 63
            m = scb[:, g + 4] & 63
        else:
            sc = (scb[:, g + 4] & 0xF) | ((scb[:, g - 4] >> 6) << 4)
            m = (scb[:, g + 4] >> 4) | ((scb[:, g] >> 6) << 4)
        scales[:, g] = d * sc
        offsets[:, g] = -(dmin * m)
        if g % 2 == 0:
            codes[:, g * 32:(g + 1) * 32] = qs[:, base:base + 32] & 0x0F
        else:
            codes[:, g * 32:(g + 1) * 32] = qs[:, base:base + 32] >> 4
    codes_b = np.zeros((n, 128), dtype=np.uint8)
    for k in range(128):
        codes_b[:, k] = codes[:, 2 * k] | (codes[:, 2 * k + 1] << 4)
    return codes_b.tobytes(), scales.astype("<f4").tobytes(), offsets.astype("<f4").tobytes()


def tensor_entry(name: str, t, raw: bytes, seg_base: int, file_name: str) -> tuple[bytes, dict]:
    """Encode one dense tensor into pack planes; returns (file body bytes, manifest entry).

    Mirrors tools/strata_pack.py:111-152 `tensor_entry` for the forms in FORM_INFO:
    planes are placed in order codes→scales→offsets, each padded to PACK_ALIGN.
    """
    info = FORM_INFO[t.type_name]
    body = bytearray()
    base = seg_base
    ent: dict = {
        "name": name, "source_type": t.type_name, "form": info["form"],
        "codebook": info["codebook"], "code_bias": info["code_bias"],
        "has_offset": info["has_offset"], "block_elems": info["block_elems"],
        "block_bytes": info["block_bytes"], "group_elems": info["group_elems"],
        "shape": list(t.shape), "source_offset": t.offset, "file": file_name,
    }

    def put(b: bytes) -> dict:
        nonlocal body
        pad = (-len(body)) % PACK_ALIGN
        body += b"\0" * pad
        off = base + len(body)
        body += b
        return {"offset": off, "bytes": len(b)}

    elements = t.elements()
    if info["form"] == "P32":
        vb = raw if t.type_name == "F32" else bf16_to_f32_bytes(raw)
        ent["elements"] = elements
        ent["values_fp16"] = False
        ent["values"] = put(vb)
        return bytes(body), ent

    if t.type_name == "Q2_0":
        codes, scales = q2_0_planes(raw)
    elif t.type_name == "IQ4_NL":
        codes, scales = iq4_nl_planes(raw)
    elif t.type_name == "Q4_0":
        codes, scales = q4_0_planes(raw)
    elif t.type_name == "Q4_K":
        codes, scales, offsets = q4_k_planes(raw)
        ent["elements"] = elements
        ent["code_bits"] = info["code_bits"]
        ent["codes"] = put(codes)
        # choose_scale_dtype (tools/strata_pack.py:90-95): fp16 when the round-trip is
        # exact, else f32. d*sc is f32 (not fp16-exact); our producer's offsets are
        # zeros → fp16-exact. The plane bytes must match the fp16 flag the manifest
        # carries, so widen/narrow here exactly as the engine pack builder does.
        import numpy as np
        sc16 = np.array(np.frombuffer(scales, dtype="<f4")).astype(np.float16)
        scales_fp16 = bool(np.array_equal(sc16.astype(np.float32),
                                          np.frombuffer(scales, dtype="<f4")))
        ent["scales"] = put(sc16.astype("<f2").tobytes() if scales_fp16 else scales)
        ent["scales_fp16"] = scales_fp16
        off16 = np.array(np.frombuffer(offsets, dtype="<f4")).astype(np.float16)
        offs_fp16 = bool(np.array_equal(off16.astype(np.float32),
                                        np.frombuffer(offsets, dtype="<f4")))
        ent["offsets"] = put(off16.astype("<f2").tobytes() if offs_fp16 else offsets)
        ent["offsets_fp16"] = offs_fp16
        return bytes(body), ent
    else:
        raise ValueError(f"no plane builder for source type {t.type_name}")
    ent["elements"] = elements
    ent["code_bits"] = info["code_bits"]
    ent["codes"] = put(codes)
    ent["scales"] = put(scales)
    ent["scales_fp16"] = True
    return bytes(body), ent


def build_manifest(gguf_path: str, n_layers: int, n_experts_per_layer: int,
                  tensors: dict, experts: dict, files: dict, shard2: str) -> dict:
    """Engine-format manifest.json (tools/strata_pack.py build:236-243 field set)."""
    return {
        "format": "strata-pack",
        "format_version": 1,
        "source": {"shard1": str(gguf_path), "shard2": str(shard2)},
        "n_layers": n_layers,
        "align": PACK_ALIGN,
        "codebooks": {"IQ4NL": [float(x) for x in IQ4NL_CODEBOOK]},
        "tensors": tensors,
        "experts": experts,
        "n_experts_per_layer": n_experts_per_layer,
        "files": files,
    }


def blob_offsets(H: int, FF: int) -> dict[str, int]:
    """Expert blob plane offsets (tools/pack_layer.py:60-66 blob_offsets, VERIFIED
    end == 1,382,400 at H=2560/FF=640 — docs/rea/struct_layouts.md §3)."""
    gu_codes = 0
    d_codes = gu_codes + 2 * FF * (H // 4)          # ROW_GU = H*2/8 = H//4 bytes/row
    gu_scales = d_codes + H * (FF // 4)             # ROW_D = FF//4
    d_scales = gu_scales + 2 * FF * (H // 64) * 2   # SC_GU = H//64 scales/row, fp16
    end = d_scales + H * (FF // 64) * 2             # SC_D = FF//64
    if end != 3 * (H * FF * 18 // 64):
        raise ValueError(f"blob offsets end {end} != geometry total")
    return {"gate_up_codes": gu_codes, "down_codes": d_codes,
            "gate_up_scales": gu_scales, "down_scales": d_scales}
