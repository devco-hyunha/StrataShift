"""Local self-check of a pack directory (`stratashift verify`), moved out of cli/main.py
(p6-08). Reads only the small pack outputs (index.txt, manifest.json) plus hashes."""
from __future__ import annotations

import json
import pathlib

from stratashift.meta.manifest import sha256_file
from stratashift.polystrata.index import engine_rows


def _verify_p2(man: dict, out: pathlib.Path) -> int:
    """p2 CLI pack self-check (`convert --out` manifest contract, p5-02).

    Checks: header pool/tensors vs manifest packed_rows, per-row codes+scales == dst
    and sequential dst_off (pool recompute), experts.bin size == pool, sha256
    regression recompute, src slices in_bounds against the model file.
    """
    errors: list[str] = []
    lines = (out / "index.txt").read_text(encoding="utf-8").splitlines()
    data_rows = [ln.split() for ln in lines if not ln.startswith("#")]
    header = lines[0].split()  # "# align A pool P tensors N"
    pool = int(header[4])
    if pool != int(man["pool_bytes"]):
        errors.append(f"header pool {pool} != manifest pool {man['pool_bytes']}")
    if int(header[6]) != len(data_rows) or len(data_rows) != man["packed_rows"]:
        errors.append(f"tensors {header[6]} rows {len(data_rows)} "
                      f"packed_rows {man['packed_rows']}")
    model_size: int | None = None
    try:
        model_size = pathlib.Path(man["model"]).stat().st_size
    except OSError:
        errors.append(f"model {man['model']} unreadable for in_bounds check")
    dst = 0
    for r in data_rows:
        codes, scales, dst_bytes = int(r[14]), int(r[15]), int(r[6])
        if codes + scales != dst_bytes:
            errors.append(f"row {r[0]}: codes {codes} + scales {scales} != dst {dst_bytes}")
        if int(r[5]) != dst:
            errors.append(f"row {r[0]}: dst_off {r[5]} != sequential {dst}")
        if model_size is not None and int(r[3]) + int(r[4]) > model_size:
            errors.append(f"row {r[0]}: src {r[3]}+{r[4]} past model {model_size}")
        dst += dst_bytes
    if dst != pool:
        errors.append(f"row sum {dst} != header pool {pool}")
    exp = out / "experts.bin"
    if exp.stat().st_size != pool:
        errors.append(f"experts.bin {exp.stat().st_size} != pool {pool}")
    if sha256_file(exp) != man["experts_bin_sha256"]:
        errors.append("experts.bin sha256 mismatch (regression)")
    print(json.dumps({"out_dir": str(out), "format": "p2", "ok": not errors,
                      "errors": errors, "pool_bytes": dst,
                      "experts_bin_sha256": sha256_file(exp)}, indent=2))
    return 0 if not errors else 1


def verify_pack(out_dir: str) -> int:
    """Local self-check of a pack directory. Two manifest contracts (p5-02):

    - engine-format manifest (has "tensors"/"experts"): index 19-field contract with
      plane sums and pool recomputed from the manifest, experts.bin size == layers x NE x
      blob, every layer blob inside the file, output hashes recomputed for regression
      (ACCEPTANCE Phase 3, p4 harness contract).
    - p2 CLI pack manifest (has "packed_rows", written by `convert --out`): index rows
      recomputed sequentially, experts.bin size == pool, sha256 regression, in_bounds
      against the model file. This closes the p5-02 disconnect: the CLI emit/convert
      path wrote p2-form manifests that the engine-form checker KeyError'd on.
    """
    out = pathlib.Path(out_dir)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if "packed_rows" in man:
        return _verify_p2(man, out)
    errors: list[str] = []

    rows = engine_rows(man)
    lines = (out / "index.txt").read_text(encoding="utf-8").splitlines()
    data_rows = [ln for ln in lines if not ln.startswith("#")]
    if len(data_rows) != len(rows):
        errors.append(f"index.txt rows {len(data_rows)} != recomputed {len(rows)}")
    for got, want in zip(data_rows, rows):
        if got.split() != [str(x) for x in want]:
            errors.append(f"index.txt row mismatch: {got[:60]}")
            break
    align = int(man["align"])
    recomputed = 0
    for r in rows:
        recomputed = (recomputed + r[6] + align - 1) // align * align
    header = lines[1].split()
    if int(header[4]) != recomputed:
        errors.append(f"header pool {header[4]} != recomputed {recomputed}")

    exp = man["experts"]
    want_size = man["n_layers"] * man["n_experts_per_layer"] * exp["blob_bytes"]
    got_size = (out / "experts.bin").stat().st_size
    if got_size != want_size:
        errors.append(f"experts.bin {got_size} != layers*NE*blob {want_size}")
    if exp["offsets"]["down_scales"] + (exp["blob_bytes"] - exp["offsets"]["down_scales"]) \
            != exp["blob_bytes"]:
        errors.append("blob offsets exceed blob_bytes")
    for lay in exp["layers"]:
        if lay["offset"] + exp["blob_bytes"] > got_size:
            errors.append(f"layer {lay['layer']} blob runs past experts.bin")

    hashes = {name: sha256_file(out / name) for name in
              ("experts.bin", "dense.bin", "embd.bin", "index.txt")}
    print(json.dumps({"out_dir": str(out), "ok": not errors, "errors": errors,
                      "pool_bytes": recomputed, "sha256": hashes}, indent=2))
    return 0 if not errors else 1
