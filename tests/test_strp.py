"""p0-04 STRP writer/parser gates (docs/rea/struct_layouts.md §5, VERIFIED).

AC1: engine data/expert-profile.bin parse → writer reproduce → sha256 byte-identical.
AC2: round-trip parse(build(profile)) == profile.
AC3: rejection gates mirror read_expert_profile (DECOMPILED F6): magic, version≠1,
     nl/ne model mismatch, n_ranked > slots.
AC4: fixture-scale profile round-trip + size formula.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

from stratashift.polystrata.engine_paths import ENV_SRC, describe_missing, resolve_engine_paths
from stratashift.polystrata.strp import build_strp, parse_strp

ENGINE_PROFILE = resolve_engine_paths().src / "data" / "expert-profile.bin"

pytestmark = pytest.mark.skipif(
    not ENGINE_PROFILE.exists(), reason="engine data/expert-profile.bin not present: "
    + describe_missing(ENGINE_PROFILE, ENV_SRC)
)


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_engine_profile_byte_identical() -> None:
    raw = ENGINE_PROFILE.read_bytes()
    prof = parse_strp(raw)
    rebuilt = build_strp(prof.n_layers, prof.n_expert, prof.slots, list(prof.ranked))
    assert _sha256(rebuilt) == _sha256(raw), "writer must reproduce the engine artifact byte-for-byte"


def test_roundtrip_engine_params() -> None:
    raw = ENGINE_PROFILE.read_bytes()
    prof = parse_strp(raw)
    assert prof.n_layers == 48 and prof.n_expert == 512 and prof.slots == 24576
    rt = parse_strp(build_strp(prof.n_layers, prof.n_expert, prof.slots, list(prof.ranked)))
    assert rt.ranked == prof.ranked
    assert rt.rank == prof.rank
    # rank[ranked[i]] == i for every ranked pair (VERIFIED invariant)
    assert all(prof.rank[l * prof.n_expert + e] == i for i, (l, e) in enumerate(prof.ranked))


def test_rejections() -> None:
    raw = ENGINE_PROFILE.read_bytes()
    prof = parse_strp(raw)
    base = build_strp(prof.n_layers, prof.n_expert, prof.slots, list(prof.ranked))

    bad_magic = b"XXXX" + base[4:]
    with pytest.raises(ValueError, match="magic"):
        parse_strp(bad_magic)

    bad_version = bytearray(base)
    bad_version[4] = 2
    with pytest.raises(ValueError, match="version"):
        parse_strp(bytes(bad_version))

    with pytest.raises(ValueError, match="n_layers"):
        parse_strp(base, expect_layers=47)
    with pytest.raises(ValueError, match="n_expert"):
        parse_strp(base, expect_experts=256)

    with pytest.raises(ValueError, match="n_ranked"):
        build_strp(48, 512, 4, [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])

    with pytest.raises(ValueError, match="out of model range"):
        build_strp(48, 512, 16, [(48, 0)])


def test_fixture_scale_roundtrip() -> None:
    ranked = [(0, e) for e in range(16)]
    data = build_strp(48, 512, 16, ranked)
    assert len(data) == 24 + 4 * 16 + 4 * 48 * 512  # size formula
    prof = parse_strp(data, expect_layers=48, expect_experts=512)
    assert prof.ranked == tuple(ranked)
    assert prof.rank[0 * 512 + 5] == 5
    assert prof.rank[1 * 512 + 0] == -1  # non-resident
    # regression: writer is deterministic
    assert _sha256(data) == _sha256(build_strp(48, 512, 16, ranked))
