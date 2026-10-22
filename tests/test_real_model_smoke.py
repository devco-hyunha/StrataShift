"""p4-03 real-model smoke gate: scripts/p4_real_model_smoke.py as a permanent pytest gate.

Qwen1.5-MoE-A2.7B (8 shards, BF16, per-expert naming) → CLI convert --emit-gguf
--hf-emit on the directory → reference verifier acceptance (engine guard NOT
applicable — geometry ≠ qwen4exp, p4-01 근거 §2). Skips when the model is not
downloaded (network gate) or the pinned engine tools checkout is missing.
Never run in parallel with the full suite (memory contention — recorded OOM pitfall).
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from stratashift.polystrata.engine_paths import ENV_SRC, describe_missing, resolve_engine_paths

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".local" / "models" / "qwen1.5-moe-a2.7b"
ENGINE = resolve_engine_paths().src

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model-00008-of-00008.safetensors").is_file()
    or not (ENGINE / "tools" / "gguf_reader.py").is_file(),
    reason="real model not downloaded (network gate) or engine tools checkout missing: "
           + describe_missing(ENGINE / "tools" / "gguf_reader.py", ENV_SRC),
)


def test_real_model_smoke() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p4_real_model_smoke.py")],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"real-model smoke failed:\n{r.stdout}\n{r.stderr}"
    assert "REAL MODEL SMOKE PASS" in r.stdout
    assert "parse match PASS" in r.stdout
