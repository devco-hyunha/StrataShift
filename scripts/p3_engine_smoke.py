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
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ENGINE = pathlib.Path("/opt/data/polystrata-src")
BUILD = pathlib.Path("/opt/data/polystrata-build")
sys.path.insert(0, str(ENGINE / "tools"))

from gguf_writer import GGUFWriter, _Tensor  # noqa: E402  (engine tool, read-only)
import gguf_writer  # noqa: E402

# the engine writer writes F32/F16/Q2_0 only; IQ4_NL is ggml type id 20
# (gguf_reader.py GGML_TYPES, pin c566ff4). Harness-side extension, engine untouched.
gguf_writer.TYPE_IDS["IQ4_NL"] = 20

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
    # fp16 scale as little-endian 2 bytes (engine read_u16 = p[0] | p[1]<<8), then 16 code bytes
    su = scales.view(np.uint16)
    data = np.empty((blocks, BLOCKS), dtype=np.uint8)
    data[:, 0] = (su & 0xFF).astype(np.uint8)
    data[:, 1] = ((su >> 8) & 0xFF).astype(np.uint8)
    data[:, 2:] = codes.reshape(blocks, 16)
    return _Tensor(name, list(shape), "Q2_0", data.tobytes())


def build_fixture(path: pathlib.Path) -> None:
    w = GGUFWriter(alignment=32)
    w.add("general.architecture", "qwen4exp")
    w.add("general.name", "stratashift p3-01 fixture")
    # architecture guard (gguf_reader.hpp read_geometry + geometry_unsupported): the six required
    # keys plus every pinned key must equal kCompiledGeometry. block_count is pinned to 48, so the
    # fixture carries the full compiled geometry and ships only blk.0 tensors.
    w.add("qwen4exp.block_count", 48, "u32")
    w.add("qwen4exp.embedding_length", 2560, "u32")  # compiled geometry.hpp value
    w.add("qwen4exp.expert_count", NE, "u32")
    w.add("qwen4exp.expert_used_count", 10, "u32")
    w.add("qwen4exp.attention.head_count", 24, "u32")
    w.add("qwen4exp.attention.head_count_kv", 2, "u32")
    w.add("qwen4exp.full_attention_interval", 4, "u32")
    w.add("qwen4exp.ssm.state_size", 128, "u32")
    w.add("qwen4exp.ssm.group_count", 16, "u32")
    w.add("qwen4exp.ssm.time_step_rank", 48, "u32")
    w.add("qwen4exp.ssm.conv_kernel", 4, "u32")
    w.add("qwen4exp.ssm.inner_size", 6144, "u32")
    w.add("qwen4exp.attention.key_length", 256, "u32")
    w.add("qwen4exp.attention.value_length", 256, "u32")
    w.add("qwen4exp.rope.dimension_count", 64, "u32")
    w.add("qwen4exp.attention.indexer.head_count", 4, "u32")
    w.add("qwen4exp.attention.indexer.key_length", 128, "u32")
    w.add("qwen4exp.attention.indexer.top_k", 2048, "u32")
    w.add("qwen4exp.hyper_connection.count", 4, "u32")
    w.add("qwen4exp.hyper_connection.low_rank", 320, "u32")
    w.add("qwen4exp.expert_feed_forward_length", FF, "u32")
    w.add("qwen4exp.expert_shared_feed_forward_length", FF, "u32")
    w.tensors.append(make_q2_0_tensor("blk.0.ffn_gate_exps.weight", (H, FF, NE), 1))
    w.tensors.append(make_q2_0_tensor("blk.0.ffn_up_exps.weight", (H, FF, NE), 2))
    w.tensors.append(make_q2_0_tensor("blk.0.ffn_down_exps.weight", (FF, H, NE), 3))
    embd = np.random.default_rng(4).standard_normal((1024, 64)).astype("<f4")
    w.add_f32("token_embd.weight", embd)
    # PLE table: strata_pack.py build:301 requires per_layer_token_embd.weight in the
    # shard2 path (the fixture IS shard2 — the 00001→00002 replace is a no-op here).
    # ngram.cpp:246 requires shape[0] == PLE_HEAD_DIM (160) and a ple_formats type;
    # IQ4_NL is in the list. 160*8 = 1280 elements = 40 blocks * 18 B; fp16 scale 1.0.
    ple = np.random.default_rng(7).integers(0, 16, size=(40, 18), dtype=np.uint8)
    ple[:, 0] = 0x00
    ple[:, 1] = 0x3C  # fp16 1.0 little-endian
    w.tensors.append(_Tensor("per_layer_token_embd.weight", [160, 8], "IQ4_NL", ple.tobytes()))
    w.add_f32("blk.0.attn_norm", np.random.default_rng(5).standard_normal(64).astype("<f4"))
    w.add_f32("output.weight", np.random.default_rng(6).standard_normal((64, 1024)).astype("<f4"))
    path.parent.mkdir(parents=True, exist_ok=True)
    w.write(path)


def run(cmd: list[str], env: dict | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
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

    # always rebuild: the fixture is cheap and must carry the current guard values
    print(f"[1] building fixture {fx}")
    build_fixture(fx)
    # harness artifacts are disposable; strata_pack refuses to build into a dir that
    # already holds a pack, so clear them before every run
    for d in (golden, ours, ours2):
        if d.exists():
            shutil.rmtree(d)

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
    # CHECKSUM <tensor> sum=<x> sumabs=<y> -> parts[2], parts[3]
    e_sum, e_sumabs = float(parts[2].split("=", 1)[1]), float(parts[3].split("=", 1)[1])
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
    cli_env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = run([sys.executable, str(cli), "convert", str(fx), "--out", str(ours)], cli_env)
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
    run([sys.executable, str(cli), "convert", str(fx), "--out", str(ours2)], cli_env)
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        if _sha256(ours / name) != _sha256(ours2 / name):
            raise SystemExit(f"FAIL: {name} not reproducible")
    print("[6] regression PASS (sha256 identical over two converts)")

    print("[7] stratashift verify (local self-check)")
    out = run([sys.executable, str(cli), "verify", str(ours)], cli_env)
    print(out)

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
