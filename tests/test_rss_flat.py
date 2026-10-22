"""RSS flatness + --max-rss-gb safety gate. AC3/AC5."""
from __future__ import annotations

import json
import sys

import pytest

from stratashift.cli.main import cmd_convert
from stratashift.monitor.rss import current_rss_bytes

from tests.test_stream_roundtrip import make_synthetic_model

GB = 1 << 30

pytestmark = pytest.mark.skipif(
    current_rss_bytes() is None,
    reason=f"RSS not measurable on {sys.platform} (monitor/rss.py)")


def test_rss_flat_across_blobs(tmp_path) -> None:
    model = tmp_path / "synthetic.safetensors"
    make_synthetic_model(model)
    out_dir = tmp_path / "out"
    assert cmd_convert(str(model), str(out_dir), None) == 0
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["rss_samples_bytes"]
    assert len(samples) == 4  # one sample per tensor slice
    # measured, not 0/None (a 0-vs-0 comparison must not pass the flatness check)
    assert all(isinstance(s, int) and s > 0 for s in samples)
    # streaming: peak must stay within a small margin of the first sample
    margin = 4 * 1024 * 1024
    assert manifest["rss_peak_bytes"] - samples[0] <= margin, \
        f"RSS grew: first={samples[0]}, peak={manifest['rss_peak_bytes']}"
    # no monotonic growth over the last samples
    tail = samples[-4:]
    assert max(tail) - min(tail) <= margin


def test_max_rss_gate_aborts(tmp_path) -> None:
    model = tmp_path / "synthetic.safetensors"
    make_synthetic_model(model)
    out_dir = tmp_path / "out"
    limit = current_rss_bytes() - 1024 * 1024  # below current → abort on first sample
    rc = cmd_convert(str(model), str(out_dir), limit / GB)
    assert rc == 3
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["aborted_max_rss"] is True
    assert manifest["packed_rows"] < 16
