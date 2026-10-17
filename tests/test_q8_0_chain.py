"""p3-09 engine chain gate: scripts/p3_q8_0.py as a permanent pytest gate.

Runs the Q8_0 chain end-to-end: fixture, CLI emit --dense-type Q8_0, engine
strata-gguf guard PASS, strata-dequant PASS + CHECKSUM cross-check, engine
strata_pack build + verify 0 bad, our convert byte-identical to golden, engine
verify on OUR pack, two converts identical, quantizer byte-identity + round-trip
vs engine to_s8_q8_0+from_s8_q8_0. Skips when the pinned engine CPU binaries
are not built (p0-02). Never run in parallel with the full suite (memory
contention — recorded OOM pitfall).
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


def test_q8_0_chain() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p3_q8_0.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"Q8_0 chain failed:\n{r.stdout}\n{r.stderr}"
    assert "SMOKE PASS" in r.stdout
