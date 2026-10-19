"""p5-06 AC3 fixture generator: language-model header inventory (headers only, C1).

Reads the 8-byte header length + header JSON of every shard in
.local/models/gemma4-26b-a4b-it (AC2 measured fixture, .local — gitignored),
keeps only `model.language_model.*` tensors (the engine Gemma4 execution family
is the text model; vision tensors are NOT engine Gemma4 keys), and writes
tests/fixtures/gemma4_lang_header.json: name -> {dtype, shape, shard}.

Run: python3 scripts/p5_gemma4_fixture.py
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys

FIXTURE_DIR = pathlib.Path("/workspace/.local/models/gemma4-26b-a4b-it")
OUT = pathlib.Path("/workspace/tests/fixtures/gemma4_lang_header.json")


def main() -> int:
    inv: dict[str, dict] = {}
    for p in sorted(FIXTURE_DIR.glob("*.safetensors")):
        with open(p, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))  # header JSON only — C1 gate
        for k, v in header.items():
            if k == "__metadata__" or not k.startswith("model.language_model."):
                continue
            inv[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": p.name}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(sorted(inv.items())), indent=1) + "\n")
    layers = sorted({int(k.split(".")[3]) for k in inv if k.split(".")[3].isdigit()})
    langs = sorted({k.split(".")[3] for k in inv if not k.split(".")[3].isdigit()})
    print(f"fixture rc=0 tensors={len(inv)} layers={len(layers)} "
          f"(min {layers[0]} max {layers[-1]}) global={langs}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
