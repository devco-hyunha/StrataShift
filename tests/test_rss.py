"""p6-10 AC1/AC2: process RSS measurement, None-sample handling, --max-rss-gb refusal."""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from stratashift.cli import main as cli_main
from stratashift.monitor import rss
from stratashift.monitor import sampler as sampler_mod
from stratashift.monitor.rss import current_rss_bytes, process_rss_bytes

MiB = 1 << 20

measurable = pytest.mark.skipif(
    not (sys.platform == "win32" or sys.platform.startswith("linux")),
    reason=f"RSS not measurable on {sys.platform} (monitor/rss.py)")


@measurable
def test_current_rss_positive_int() -> None:
    value = current_rss_bytes()
    assert isinstance(value, int) and value > 0


def test_unsupported_platform_is_none(monkeypatch) -> None:
    monkeypatch.setattr(rss.sys, "platform", "darwin")
    assert process_rss_bytes() is None


@measurable
def test_dead_pid_is_none() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    pid = child.pid
    del child  # Windows: the Popen handle keeps the exited process queryable until closed
    assert process_rss_bytes(pid) is None


@measurable
def test_huge_pid_is_none() -> None:
    assert process_rss_bytes(2**31 - 2) is None


@measurable
def test_child_rss_sees_allocation() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; b = bytearray(64 << 20); time.sleep(5)"])
    try:
        deadline = time.monotonic() + 3.0
        value = None
        while time.monotonic() < deadline:
            value = process_rss_bytes(child.pid)
            if value is not None and value >= 64 * MiB:
                break
            time.sleep(0.05)
        assert value is not None and value >= 64 * MiB
    finally:
        child.kill()
        child.wait()


def test_sampler_none_sample(monkeypatch) -> None:
    monkeypatch.setattr(sampler_mod, "current_rss_bytes", lambda: None)
    s = sampler_mod.Sampler(1.0)
    s.record()
    assert s.rss == [None]
    assert s.rss_peak() is None
    assert s.over_limit() is False


def test_cli_refuses_max_rss_when_unmeasurable(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(cli_main, "current_rss_bytes", lambda: None)
    model = tmp_path / "any.safetensors"
    model.write_bytes(b"")
    rc = cli_main.main(["convert", str(model), "--out", str(tmp_path / "out"),
                        "--max-rss-gb", "1"])
    assert rc == 2
    assert "--max-rss-gb" in capsys.readouterr().err
