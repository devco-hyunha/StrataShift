"""p6-11 AC2: text file reads/writes must pass encoding="utf-8" explicitly.

Korean Windows defaults to cp949, so Path.read_text / Path.write_text and text-mode
open() without an explicit encoding silently depend on the OS locale. This guard
scans src/, scripts/ and tests/ with ast and fails on any such call.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "scripts", "tests")


def _iter_py_files() -> list[Path]:
    files: list[Path] = []
    for name in SCAN_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return files


def _has_encoding(call: ast.Call) -> bool:
    return any(kw.arg == "encoding" for kw in call.keywords)


def _open_is_text_mode(call: ast.Call) -> bool:
    """True if open() is provably text mode; False for binary or undecidable."""
    mode_node: ast.expr | None = None
    if len(call.args) >= 2:
        mode_node = call.args[1]
    for kw in call.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
    if mode_node is None:
        return True  # default mode is "r" (text)
    if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
        return "b" not in mode_node.value
    return False  # non-constant mode: cannot decide, skip


def _find_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(ROOT).as_posix()
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            if not _has_encoding(node):
                found.append(f"{rel}:{node.lineno}")
        elif isinstance(func, ast.Name) and func.id == "open":
            if _open_is_text_mode(node) and not _has_encoding(node):
                found.append(f"{rel}:{node.lineno}")
    return found


def test_text_io_has_explicit_utf8_encoding() -> None:
    violations: list[str] = []
    for path in _iter_py_files():
        violations.extend(_find_violations(path))
    assert not violations, "missing encoding=\"utf-8\":\n" + "\n".join(violations)
