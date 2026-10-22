"""p4-01 engine chain gate: scripts/p4_hf_emit.py as a permanent pytest gate.

Runs the HF emit chain end-to-end: qwen4exp HF fixture (fused gate_up + GDN/QSA/
indexer/hc HF naming) → CLI --hf-emit → engine strata-gguf guard PASS, strata-dequant
PASS + CHECKSUM cross-check, engine strata_pack build + verify 0 bad, our convert
byte-identical to golden, engine verify on OUR pack, two converts identical, no-full-
load gate; then mixtral-tiny (BF16, fused) → engine gguf_reader parse match +
canonical_xcheck dequant round-trip + our verify PASS. Skips when the pinned engine
CPU binaries are not built (p0-02). Never run in parallel with the full suite
(memory contention — recorded OOM pitfall).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from stratashift.polystrata.engine_paths import (
    ENV_BUILD, describe_missing, engine_binary, resolve_engine_paths)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = resolve_engine_paths()
BINARIES = (engine_binary(ENGINE, "strata-gguf"), engine_binary(ENGINE, "strata-dequant"))

pytestmark = pytest.mark.skipif(
    not all(b.exists() for b in BINARIES),
    reason="pinned engine CPU binaries not built (p0-02): "
           + describe_missing(BINARIES[0], ENV_BUILD),
)


def test_hf_emit_chain() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p4_hf_emit.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"HF emit chain failed:\n{r.stdout}\n{r.stderr}"
    assert "SMOKE PASS" in r.stdout
