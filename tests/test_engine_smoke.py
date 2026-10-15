"""p3-01 engine load smoke chain (AC2~AC6) as a pytest gate.

Runs scripts/p3_engine_smoke.py end-to-end: fixture build, engine strata-gguf
--check + strata-dequant PASS, engine golden build + verify PASS, our convert
byte-identical to golden (experts/dense/embd/index), engine verify on OUR pack,
reproducibility over two converts. Skips when the pinned engine CPU binaries
are not built (p0-02).
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


def test_engine_smoke_chain() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p3_engine_smoke.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"smoke chain failed:\n{r.stdout}\n{r.stderr}"
    assert "SMOKE PASS" in r.stdout
