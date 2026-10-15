"""CONSTRAINTS C1/C2 guards: no full-load APIs, bounded peak memory.

Static gate: forbidden APIs must not appear in src/ or scripts/.
Runtime gate: header parse + single mmap slice must keep peak allocation
at "header + one slice" level, not the whole file.
"""
from __future__ import annotations

import pathlib
import tracemalloc

import pytest

from stratashift.ingest.safetensors_header import parse_header
from stratashift.meta.expert_map import build_expert_map

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = ROOT / ".local" / "models" / "mixtral-tiny.safetensors"

FORBIDDEN = [
    "safetensors.torch.load_file",
    "torch.load",
    "load_state_dict",
    "torch.save",
    "safetensors.load(",
]


def test_no_forbidden_apis() -> None:
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    assert files, "no source files to guard"
    for path in files:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN:
            assert needle not in text, f"{path.relative_to(ROOT)}: forbidden API {needle!r}"


@pytest.mark.skipif(not MODEL.exists(), reason="test model not downloaded")
def test_header_parse_peak_memory() -> None:
    meta = parse_header(str(MODEL))  # warm-up outside the measurement
    tracemalloc.start()
    parsed = parse_header(str(MODEL))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    # header JSON + parsed objects only; must stay far below the whole file
    assert parsed.file_size > meta.header_size  # sanity: body exists
    budget = meta.header_size * 4 + 1_000_000
    assert peak < budget, f"peak {peak} B exceeds header+margin budget {budget} B"


@pytest.mark.skipif(not MODEL.exists(), reason="test model not downloaded")
def test_single_slice_peak_memory() -> None:
    import mmap

    meta = parse_header(str(MODEL))
    refs = build_expert_map(meta)
    assert refs, "no expert candidates"
    first = refs[0]
    begin, end = first.data_offsets
    with open(MODEL, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        tracemalloc.start()
        blob = mm[meta.data_start + begin: meta.data_start + end]
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        mm.close()
    assert len(blob) == end - begin
    # one slice only — peak must not grow with the file size
    assert peak <= (end - begin) + 100_000, f"slice peak {peak} B"


GGUF_FIXTURE = ROOT / ".local" / "models" / "p3-01-fixture" / "fixture-1l.gguf"


@pytest.mark.skipif(not GGUF_FIXTURE.exists(), reason="p3-01 fixture not built")
def test_gguf_header_parse_peak_memory() -> None:
    """GGUF path (AC8): parse keeps only tensor records + objects, not the file."""
    from stratashift.ingest.gguf import parse_gguf_header

    parse_gguf_header(str(GGUF_FIXTURE))  # warm-up outside the measurement
    tracemalloc.start()
    meta = parse_gguf_header(str(GGUF_FIXTURE))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert meta.tensors
    # fixture is ~708 MB; parsed records must stay far below it
    budget = 2_000_000
    assert peak < budget, f"peak {peak} B exceeds parsed-objects budget {budget} B"


@pytest.mark.skipif(not GGUF_FIXTURE.exists(), reason="p3-01 fixture not built")
def test_gguf_single_slice_peak_memory() -> None:
    """GGUF path (AC8): one expert-role contiguous run, peak = header + one slice."""
    import mmap

    from stratashift.ingest.gguf import parse_gguf_header

    meta = parse_gguf_header(str(GGUF_FIXTURE))
    gate = next(t for t in meta.tensors if t.name == "blk.0.ffn_gate_exps.weight")
    h, ff, ne = gate.shape
    role_bytes = h * ff * 18 // 64  # Q2_0 contiguous run per expert (pack.py contract)
    with open(GGUF_FIXTURE, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        tracemalloc.start()
        blob = mm[meta.data_start + gate.offset: meta.data_start + gate.offset + role_bytes]
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        mm.close()
    assert len(blob) == role_bytes
    assert peak <= role_bytes + 100_000, f"slice peak {peak} B"
