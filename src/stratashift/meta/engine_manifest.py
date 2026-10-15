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
}

# IQ4NL codebook — tools/strata_pack.py:240 man["codebooks"] (KV_IQ4NL); the pack
# carries it, so our manifest must carry the identical table.
IQ4NL_CODEBOOK: tuple[float, ...] = (
    -0.12024, -0.11591, -0.11035, -0.10601, -0.10166, -0.09732, -0.09308, -0.08944,
    -0.08500, -0.08115, -0.07681, -0.07246, -0.06812, -0.06377, -0.05942, -0.05508,
    -0.04952, -0.04396, -0.03840, -0.03284, -0.02728, -0.02172, -0.01616, -0.01060,
    -0.00504, 0.00000, 0.00504, 0.01060, 0.01616, 0.02172, 0.02728, 0.03284,
    0.03840, 0.04396, 0.04952, 0.05508, 0.05942, 0.06377, 0.06812, 0.07246,
    0.07681, 0.08115, 0.08500, 0.08944, 0.09308, 0.09732, 0.10166, 0.10601,
    0.11035, 0.11591, 0.12024, 0.12580, 0.13136, 0.13692, 0.14248, 0.14804,
    0.15360, 0.15916, 0.16472, 0.17028, 0.17584, 0.18140, 0.18696, 0.19252,
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

    codes, scales = q2_0_planes(raw)
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
