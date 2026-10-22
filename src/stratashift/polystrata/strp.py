"""STRP expert-profile writer/parser — docs/rea/struct_layouts.md §5 (VERIFIED).

Layout (pin c566ff4, SOURCE expert_cache.hpp:42-46 + expert_cache.cpp:170-206,
VERIFIED against engine data/expert-profile.bin, DECOMPILED read_expert_profile F6):

    0            magic "STRP" (u32 0x50525453 LE)
    4            version, n_layers, n_expert, slots, n_ranked   (5 × u32 LE)
    24           n_ranked × (u16 layer, u16 expert)             — ranked[i] = slot i
    24+4·n_ranked  int32[n_layers × n_expert] — rank[l·ne + e] = slot index of
                   expert (l,e); −1 = non-resident

Size formula: 24 + 4·n_ranked + 4·nl·ne. Rejection gates mirror the engine's
read_expert_profile (DECOMPILED F6): magic, version≠1, nl/ne model mismatch,
n_ranked > slots.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from stratashift.polystrata.headers import STRP_HEADER_U32_COUNT, STRP_MAGIC, STRP_VERSION

_HEADER = struct.Struct("<5I")  # version, n_layers, n_expert, slots, n_ranked
_PAIR = struct.Struct("<HH")    # (layer, expert) of slot i
_RANK = struct.Struct("<i")     # slot index of expert (l,e); -1 non-resident


@dataclass(frozen=True)
class StrpProfile:
    n_layers: int
    n_expert: int
    slots: int
    ranked: tuple[tuple[int, int], ...]  # ranked[i] = (layer, expert) resident at slot i
    rank: tuple[int, ...]                # rank[l·ne + e] = slot index, -1 non-resident


def build_strp(n_layers: int, n_expert: int, slots: int,
              ranked: list[tuple[int, int]]) -> bytes:
    """Emit the STRP byte stream for one profile. n_ranked = len(ranked) ≤ slots."""
    n_ranked = len(ranked)
    if n_ranked > slots:
        raise ValueError(f"n_ranked {n_ranked} > slots {slots} (engine read_expert_profile refuses)")
    for layer, expert in ranked:
        if not (0 <= layer < n_layers and 0 <= expert < n_expert):
            raise ValueError(f"ranked pair ({layer},{expert}) out of model range "
                             f"({n_layers}×{n_expert})")
    rank = [-1] * (n_layers * n_expert)
    for slot, (layer, expert) in enumerate(ranked):
        rank[layer * n_expert + expert] = slot
    out = bytearray(STRP_MAGIC)
    out += _HEADER.pack(STRP_VERSION, n_layers, n_expert, slots, n_ranked)
    for layer, expert in ranked:
        out += _PAIR.pack(layer, expert)
    out += struct.pack("<" + "i" * len(rank), *rank)
    return bytes(out)


def parse_strp(data: bytes, expect_layers: int | None = None,
              expect_experts: int | None = None) -> StrpProfile:
    """Parse and validate one STRP stream; rejects mirror read_expert_profile (F6)."""
    if data[:4] != STRP_MAGIC:
        raise ValueError(f"bad magic {data[:4]!r}, expected {STRP_MAGIC!r}")
    version, n_layers, n_expert, slots, n_ranked = _HEADER.unpack_from(data, 4)
    if version != STRP_VERSION:
        raise ValueError(f"STRP version {version}, engine accepts {STRP_VERSION} only")
    if expect_layers is not None and n_layers != expect_layers:
        raise ValueError(f"STRP n_layers {n_layers} != model {expect_layers}")
    if expect_experts is not None and n_expert != expect_experts:
        raise ValueError(f"STRP n_expert {n_expert} != model {expect_experts}")
    if n_ranked > slots:
        raise ValueError(f"STRP n_ranked {n_ranked} > slots {slots}")
    size = 4 + 4 * STRP_HEADER_U32_COUNT + 4 * n_ranked + 4 * n_layers * n_expert
    if len(data) != size:
        raise ValueError(f"STRP size {len(data)} != formula {size} "
                         f"(24 + 4·n_ranked + 4·nl·ne)")
    pairs = struct.unpack_from("<" + "H" * (2 * n_ranked), data, 24)
    ranked = tuple((pairs[2 * i], pairs[2 * i + 1]) for i in range(n_ranked))
    rank = struct.unpack_from("<" + "i" * (n_layers * n_expert), data, 24 + 4 * n_ranked)
    return StrpProfile(n_layers, n_expert, slots, ranked, rank)
