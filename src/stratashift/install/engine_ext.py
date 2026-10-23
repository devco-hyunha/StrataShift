"""Put `engine_ext/` into a copy of the pinned engine tree (p6-01, `build-engine --ext`).

The pinned tree is never changed: it is copied, StrataShift's files (`engine_ext/overlay.json` `copy`) go into the
copy, and the anchor edits (`edits`) are made there. Before that, the engine files the classes depend on
(`engine_ext/pin_files.json`) are hashed; a pin whose copy of one differs is refused, and `diff_pin_files` shows
what changed (scripts/engine_ext_diff.py).
"""
from __future__ import annotations

import difflib
import hashlib
import json
import pathlib
import shutil

from stratashift.polystrata.engine_paths import ENGINE_EXT_DIR


class OverlayError(RuntimeError):
    """The pinned tree does not take the overlay (changed file, anchor missing or repeated)."""


def load_spec(name: str, ext_dir: pathlib.Path = ENGINE_EXT_DIR) -> dict:
    return json.loads((ext_dir / name).read_text(encoding="utf-8"))


def sha256_of(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(tree: pathlib.Path, names) -> dict[str, str]:
    """sha256 of each listed engine file; a missing file hashes to "missing"."""
    return {n: sha256_of(tree / n) if (tree / n).is_file() else "missing" for n in names}


def changed_pin_files(src: pathlib.Path, ext_dir: pathlib.Path = ENGINE_EXT_DIR) -> list[str]:
    """The depended-on engine files whose hash differs from `pin_files.json`."""
    want = load_spec("pin_files.json", ext_dir)["files"]
    got = file_hashes(src, want)
    return [n for n in want if got[n] != want[n]]


def apply_edit(text: str, anchor: str, where: str, lines: list[str]) -> str:
    """Insert `lines` before or after the one line equal to `anchor`, in the file's own newline."""
    newline = "\r\n" if "\r\n" in text else "\n"
    rows = text.split(newline)
    hits = [i for i, row in enumerate(rows) if row == anchor]
    if len(hits) != 1:
        raise OverlayError(f"anchor found {len(hits)} times (want 1): {anchor!r}")
    at = hits[0] if where == "before" else hits[0] + 1
    return newline.join(rows[:at] + lines + rows[at:])


def stage_overlay(src: pathlib.Path, dst: pathlib.Path,
                  ext_dir: pathlib.Path = ENGINE_EXT_DIR) -> list[str]:
    """Fresh copy of `src` at `dst` with engine_ext put in. Returns what was done, a line each."""
    changed = changed_pin_files(src, ext_dir)
    if changed:
        raise OverlayError("the engine tree differs from the pin engine_ext was written for: "
                           + ", ".join(changed) + " (see scripts/engine_ext_diff.py)")
    spec = load_spec("overlay.json", ext_dir)
    # edits are checked on the pinned files before anything is copied
    edited: dict[str, str] = {}
    for e in spec["edits"]:
        base = edited.get(e["file"])
        if base is None:
            base = (src / e["file"]).read_bytes().decode("utf-8")
        edited[e["file"]] = apply_edit(base, e["anchor"], e["where"], e["lines"])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    done = [f"copied {src} -> {dst}"]
    for ours, theirs in spec["copy"].items():
        target = dst / theirs
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ext_dir / ours, target)
        done.append(f"put engine_ext/{ours} at {theirs}")
    for name, text in edited.items():
        (dst / name).write_bytes(text.encode("utf-8"))
        done.append(f"edited {name}")
    return done


def diff_pin_files(old: pathlib.Path, new: pathlib.Path,
                   ext_dir: pathlib.Path = ENGINE_EXT_DIR) -> str:
    """Unified diff of the depended-on engine files between two engine trees."""
    out: list[str] = []
    for name in load_spec("pin_files.json", ext_dir)["files"]:
        a, b = old / name, new / name
        ta = a.read_text(encoding="utf-8").splitlines(keepends=True) if a.is_file() else []
        tb = b.read_text(encoding="utf-8").splitlines(keepends=True) if b.is_file() else []
        out.extend(difflib.unified_diff(ta, tb, f"old/{name}", f"new/{name}"))
    return "".join(out)
