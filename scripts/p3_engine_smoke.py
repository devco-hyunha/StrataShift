"""p3-01 engine load smoke harness (Goal p3-01, AC2~AC6).

Injects OUR conversion outputs into the pinned PolyStrata engine's CPU-verifiable
loader path. No GPU here: `WeightTable::load` needs cudaHostAlloc, so the engine's
own spec-of-record verifier (tools/strata_pack.py verify — THE ONE RULE:
decode(pack) == dequant_ggml(source) bit-for-bit) plus the CPU binaries
(strata-gguf, strata-dequant) are the acceptance gate (ACCEPTANCE Phase 3 minimum:
"로더 self-check 통과").

Chain:
  1. fixture GGUF (1 layer, H=2560 FF=640 NE=512, Q2_0) built with the engine's
     own gguf_writer._Tensor (raw Q2_0 blocks; gguf-py cannot write type 42 — REA P0.S6)
  2. strata-gguf --check + strata-dequant PASS; CHECKSUM cross-checked by our Python
     q2_0_unpack on identical bytes (two implementations, one answer)
  3. engine strata_pack.py build → golden pack; engine verify PASS
  4. our CLI convert on the fixture → byte-identical experts.bin/dense.bin/embd.bin,
     byte-identical index.txt (vs engine pack_index.py), manifest planes agree
  5. engine strata_pack.py verify --out ours → PASS (engine acceptance of OUR artifact)
  6. regression: convert twice → identical sha256

Engine sources are read-only (sys.path import + subprocess); the engine is never edited.
Run: python scripts/p3_engine_smoke.py [--fixture DIR] [--keep]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ENGINE = pathlib.Path("/opt/data/polystrata-src")
BUILD = pathlib.Path("/opt/data/polystrata-build")
sys.path.insert(0, str(ENGINE / "tools"))

from gguf_writer import GGUFWriter, _Tensor  # noqa: E402  (engine tool, read-only)
from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402
from stratashift.quant.pack import q2_0_unpack  # noqa: E402

H, FF, NE = 2560, 640, 512
BLOCKS = 18
QK = 64
ROLE_BYTES = H * FF * BLOCKS // QK  # 460,800 contiguous run per expert per role
N_LAYERS = 1


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def make_q2_0_tensor(name: str, shape: tuple[int, int, int], seed: int) -> _Tensor:
    """Raw Q2_0 blocks: fp16 scale + 16 B codes, deterministic RNG, no full-model float array."""
    rng = np.random.default_rng(seed)
    blocks = shape[0] * shape[1] * shape[2] // QK
    scales = rng.standard_normal(blocks).astype(np.float32)
    scales = np.maximum(scales, 1e-3).astype(np.float16)
    codes = rng.integers(0, 4, size=blocks * 16, dtype=np.uint8)
    data = np.empty((blocks, BLOCKS), dtype=np.uint8)
    data[:, :2] = scales.view(np.uint16).reshape(blocks, 1).repeat(2, axis=1)
    data[:, 2:] = codes.reshape(blocks, 16)
    return _Tensor(name, list(shape), "Q2_0", data.tobytes())


def build_fixture(path: pathlib.Path) -> None:
    w = GGUFWriter(alignment=32)
    w.add("general.architecture", "qwen4exp")
    w.add("qwen4exp.block_count", N_LAYERS, "u32")
    w.add("general.name", "stratashift p3-01 fixture")
    w.tensors.append(make_q2_0_tensor("blk.0.ffn_gate_exps.weight", (H, FF, NE), 1))
    w.tensors.append(make_q2_0_tensor("blk.0.ffn_up_exps.weight", (H, FF, NE), 2))
    w.tensors.append(make_q2_0_tensor("blk.0.ffn_down_exps.weight", (FF, H, NE), 3))
    embd = np.random.default_rng(4).standard_normal((1024, 64)).astype("<f4")
    w.add_f32("token_embd.weight", embd)
    w.add_f32("blk.0.attn_norm", np.random.default_rng(5).standard_normal(64).astype("<f4"))
    w.add_f32("output.weight", np.random.default_rng(6).standard_normal((64, 1024)).astype("<f4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    w.write(path)


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(REPO / ".local" / "models" / "p3-01-fixture"))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    fx_dir = pathlib.Path(args.fixture)
    fx = fx_dir / "fixture-1l.gguf"
    golden = fx_dir / "golden"
    ours = fx_dir / "ours"
    ours2 = fx_dir / "ours2"

    if not fx.exists():
        print(f"[1] building fixture {fx}")
        build_fixture(fx)
    else:
        print(f"[1] fixture exists: {fx}")

    print("[2a] strata-gguf --check")
    out = run([str(BUILD / "strata-gguf"), str(fx), "--check"])
    print(out)
    print("[2b] strata-dequant")
    out = run([str(BUILD / "strata-dequant"), str(fx)])
    print(out)
    if "PASS" not in out.splitlines()[-1]:
        raise SystemExit("FAIL: strata-dequant did not print PASS")

    # CHECKSUM cross-check: engine C++ vs our Python q2_0_unpack on identical bytes
    meta = parse_gguf_header(str(fx))
    first_q2 = next(t for t in meta.tensors if t.type_name == "Q2_0")
    with open(fx, "rb") as fh:
        fh.seek(meta.data_start + first_q2.offset)
        raw = fh.read(64 * BLOCKS)
    s = sa = 0.0
    for b in range(64):
        for v in q2_0_unpack(raw[b * BLOCKS:(b + 1) * BLOCKS]):
            s += v
            sa += abs(v)
    engine_line = [ln for ln in out.splitlines() if ln.strip().startswith("CHECKSUM")]
    if not engine_line:
        raise SystemExit("FAIL: no CHECKSUM line from strata-dequant")
    parts = engine_line[0].split()
    e_sum, e_sumabs = float(parts[3].split("=", 1)[1]), float(parts[4].split("=", 1)[1])
    if abs(e_sum - s) > 1e-6 * max(1.0, abs(e_sum)) or abs(e_sumabs - sa) > 1e-6 * max(1.0, abs(e_sumabs)):
        raise SystemExit(f"FAIL: CHECKSUM mismatch engine({e_sum},{e_sumabs}) vs python({s},{sa})")
    print(f"[2c] CHECKSUM cross-check PASS: sum={s:.10g} sumabs={sa:.10g}")

    print("[3] engine strata_pack.py build (golden) + verify")
    run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "build",
         "--gguf", str(fx), "--out", str(golden), "--layers", str(N_LAYERS), "--skip-hash"])
    run([sys.executable, str(ENGINE / "tools" / "pack_index.py"), "--pack", str(golden)])
    out = run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "verify",
               "--gguf", str(fx), "--out", str(golden)])
    print(out)

    print("[4] our CLI convert on the fixture")
    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    out = run([sys.executable, str(cli), "convert", str(fx), "--out", str(ours)])
    print(out)

    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        g, o = golden / name, ours / name
        if not o.exists():
            raise SystemExit(f"FAIL: ours missing {name}")
        if _sha256(g) != _sha256(o):
            raise SystemExit(f"FAIL: {name} differs from golden "
                             f"(golden {_sha256(g)[:12]} vs ours {_sha256(o)[:12]})")
        print(f"[4] {name} byte-identical to engine golden ({o.stat().st_size} B)")

    # manifest plane fields agree (codes/scales offsets+bytes per dense tensor)
    gm = json.loads((golden / "manifest.json").read_text())
    om = json.loads((ours / "manifest.json").read_text())
    for name, ent in gm["tensors"].items():
        oent = om["tensors"][name]
        for key in ("codes", "scales", "values"):
            if key in ent and ent[key] != oent.get(key):
                raise SystemExit(f"FAIL: manifest plane mismatch for {name}/{key}: "
                                 f"{ent[key]} vs {oent.get(key)}")
    print("[4] manifest planes agree")

    print("[5] engine strata_pack.py verify on OUR pack")
    out = run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "verify",
               "--gguf", str(fx), "--out", str(ours)])
    print(out)

    print("[6] regression: convert twice → identical hashes")
    run([sys.executable, str(cli), "convert", str(fx), "--out", str(ours2)])
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        if _sha256(ours / name) != _sha256(ours2 / name):
            raise SystemExit(f"FAIL: {name} not reproducible")
    print("[6] regression PASS (sha256 identical over two converts)")

    print("[7] stratashift verify (local self-check)")
    out = run([sys.executable, str(cli), "verify", str(ours)])
    print(out)

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
