#!/usr/bin/env python3
"""p6-01: when the PolyStrata pin moves, compare only the engine files engine_ext depends on.

Usage:
  python scripts/engine_ext_diff.py OLD_TREE NEW_TREE   unified diff of the files in engine_ext/pin_files.json
  python scripts/engine_ext_diff.py --hashes TREE       the `files` table for pin_files.json, from TREE

Exit 0 when the files are the same (or after --hashes), 1 when they differ.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.install.engine_ext import diff_pin_files, file_hashes, load_spec  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--hashes":
        names = load_spec("pin_files.json")["files"]
        print(json.dumps(file_hashes(pathlib.Path(argv[1]), names), indent=4))
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    diff = diff_pin_files(pathlib.Path(argv[0]), pathlib.Path(argv[1]))
    sys.stdout.write(diff)
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
