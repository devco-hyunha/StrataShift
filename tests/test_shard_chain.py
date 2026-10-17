"""p4-02 engine chain gate: scripts/p4_shard_split.py as a permanent pytest gate.

Runs the multi-shard chain end-to-end: qwen4exp HF fixture split into 2 shards +
index.json → CLI `convert --emit-gguf --hf-emit` on the directory → GGUF sha256
identical to the single-file input's GGUF, two emits identical, engine chain
acceptance on the shard-emitted GGUF (strata-gguf guard, strata-dequant + CHECKSUM,
strata_pack build + verify 0 bad, our convert byte-identical to golden, engine
verify OUR pack), no-full-load gate. Skips when the pinned engine CPU binaries are
not built (p0-02). Never run in parallel with the full suite (memory contention —
recorded OOM pitfall, p4-01 convention).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = pathlib.Path("/opt/data/polystrata-build")
BINARIES = (BUILD / "strata-gguf", BUILD / "strata-dequant")

pytestmark = pytest.mark.skipif(
    not all(b.exists() for b in BINARIES),
    reason="pinned engine CPU binaries not built (p0-02)",
)


def test_shard_chain() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p4_shard_split.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"shard chain failed:\n{r.stdout}\n{r.stderr}"
    assert "SHARD SMOKE PASS" in r.stdout
    assert "GGUF sha256 identical" in r.stdout
