"""Shared engine paths for the test suite (p6-09).

pytest imports this before any test module, so the pinned engine `tools/` and `ref/`
dirs are on sys.path once here instead of a fixed absolute-path insert in every file.
Paths come from `polystrata.engine_paths` (environment → in-repo `.local/` default).
Skip conditions evaluated at collection time call `resolve_engine_paths()` directly.
"""
from __future__ import annotations

import sys

import pytest

from stratashift.polystrata.engine_paths import EnginePaths, resolve_engine_paths

ENGINE = resolve_engine_paths()
for _dir in (ENGINE.tools, ENGINE.src / "ref"):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))


@pytest.fixture
def engine_paths() -> EnginePaths:
    return ENGINE
