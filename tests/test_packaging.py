"""p5-01 packaging gate: pyproject contract + real install + C1 no-full-load import graph.

AC1: version 0.4.0, console script entry, no runtime deps.
AC2: `pip install . --target` then the installed console script runs `--help` exit 0.
AC3: importing the installed package pulls in no safetensors/torch (CONSTRAINTS C1).
"""
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"


def test_pyproject_contract():
    cfg = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    proj = cfg["project"]
    assert proj["version"] == "0.4.0"
    assert proj["scripts"]["stratashift"] == "stratashift.cli.main:cli_entry"
    # C1: the only allowed third-party dep is numpy; safetensors/torch must never appear.
    assert proj.get("dependencies", []) == ["numpy"]
    banned = {"safetensors", "torch"}
    assert not (banned & {d.split(">")[0].split("=")[0].strip() for d in proj.get("dependencies", [])})


def test_cli_entry_propagates_exit_code(monkeypatch):
    # The wrapper contract: cli_entry raises SystemExit(main()). Console scripts ignore
    # return values, so this is the exit-code propagation gate (p5-01 AC1).
    sys.path.insert(0, str(REPO / "src"))
    import stratashift.cli.main as m

    monkeypatch.setattr(m, "main", lambda argv=None: 2)
    try:
        m.cli_entry()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        pytest.fail("cli_entry did not propagate an exit code")


@pytest.mark.skipif(subprocess.call([sys.executable, "-m", "pip", "--version"],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL) != 0,
                   reason="pip unavailable")
def test_install_gate(tmp_path):
    target = tmp_path / "inst"
    r = subprocess.call([sys.executable, "-m", "pip", "install", str(REPO),
                         "--target", str(target), "--no-deps", "-q"])
    assert r == 0, "pip install . --target failed"

    script = target / "bin" / "stratashift"
    assert script.exists(), "console script not installed"

    env = {**os.environ, "PYTHONPATH": str(target)}
    r = subprocess.run([str(script), "--help"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"--help exit {r.returncode}: {r.stderr[:200]}"
    assert "inspect" in r.stdout and "convert" in r.stdout, r.stdout[:200]

    # AC3: import graph of the installed package has no full-load APIs.
    check = subprocess.run(
        [sys.executable, "-c",
         "import sys; import stratashift.cli.main; "
         "banned = {'safetensors', 'torch'} & {m.split('.')[0] for m in sys.modules}; "
         "print(sorted(banned))"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(target)},
    )
    assert check.returncode == 0, check.stderr[:200]
    assert check.stdout.strip() == "[]", f"C1 violation: {check.stdout.strip()}"
