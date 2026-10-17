"""p3-02 safetensors float experts → engine GGUF Q2_0 → engine chain acceptance.

Chain (docs/work/p3-02.md):
  1. fixture safetensors: 1 layer, Qwen4exp geometry (H=2560, FF=640, NE=512), float32
     expert tensors `blk.0.ffn_{gate,up,down}_exps.weight` shape (out, NE, in) row-major
     + F32 dense/embd tensors.
  2. our CLI `convert --emit-gguf` → engine GGUF v3 (Q2_0 tensors, guard preset).
  3. engine strata-gguf --check PASS (architecture guard), strata-dequant PASS (0 violations),
     CHECKSUM cross-checked by our Python q2_0_unpack on identical bytes.
  4. engine strata_pack build + pack_index + verify PASS on the emitted GGUF.
  5. our CLI convert on the emitted GGUF → pack byte-identical to engine golden.
  6. quantizer byte-identity: our quantize_q2_0_stream == engine quantize_q2_0 (random rows).

Engine sources are read-only; the engine is never edited.
Run: python scripts/p3_safetensors_gguf.py
"""
from __future__ import annotations

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

from gguf_writer import quantize_q2_0 as engine_quantize  # noqa: E402
from stratashift.quant.pack import q2_0_unpack, quantize_q2_0_stream  # noqa: E402

H, FF, NE = 2560, 640, 512
N_LAYERS = 1
GUARD = {
    "architecture": "qwen4exp",
    "name": "stratashift p3-02 fixture",
    "qwen4exp.block_count": 48,
    "qwen4exp.embedding_length": 2560,
    "qwen4exp.expert_count": NE,
    "qwen4exp.expert_used_count": 10,
    "qwen4exp.attention.head_count": 24,
    "qwen4exp.attention.head_count_kv": 2,
    "qwen4exp.full_attention_interval": 4,
    "qwen4exp.ssm.state_size": 128,
    "qwen4exp.ssm.group_count": 16,
    "qwen4exp.ssm.time_step_rank": 48,
    "qwen4exp.ssm.conv_kernel": 4,
    "qwen4exp.ssm.inner_size": 6144,
    "qwen4exp.attention.key_length": 256,
    "qwen4exp.attention.value_length": 256,
    "qwen4exp.rope.dimension_count": 64,
    "qwen4exp.attention.indexer.head_count": 4,
    "qwen4exp.attention.indexer.key_length": 128,
    "qwen4exp.attention.indexer.top_k": 2048,
    "qwen4exp.hyper_connection.count": 4,
    "qwen4exp.hyper_connection.low_rank": 320,
    "qwen4exp.expert_feed_forward_length": FF,
    "qwen4exp.expert_shared_feed_forward_length": FF,
}


def run(cmd: list[str], env: dict | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def build_fixture(path: pathlib.Path) -> None:
    """safetensors: [u64 header_len][JSON][blob]; experts F32 (out, NE, in) row-major.

    Streamed per expert run (rows × in_features): one (rows, in) float array alive at a
    time (CONSTRAINTS C1/C2 — the 1.34 GB tensor is never materialized). Blobs go to a
    temp file first; the final file is header + streamed copy of the temp file.
    """
    tensors = {}
    off = 0
    blob = path.with_suffix(".blob")
    with open(blob, "wb") as fh:
        for name, shape, seed in (
            ("blk.0.ffn_gate_exps.weight", (FF, NE, H), 11),
            ("blk.0.ffn_up_exps.weight", (FF, NE, H), 12),
            ("blk.0.ffn_down_exps.weight", (H, NE, FF), 13),
        ):
            out_dim, ne, in_dim = shape
            begin = off
            rng = np.random.default_rng(seed)
            for e in range(ne):
                a = rng.standard_normal((out_dim, in_dim)).astype("<f4")
                fh.write(a.tobytes())
                off += a.nbytes
                del a
            tensors[name] = {"dtype": "F32", "shape": list(shape),
                             "data_offsets": [begin, off]}
        for name, shape, seed in (
            ("token_embd.weight", (1024, 64), 14),
            ("blk.0.attn_norm", (64,), 15),
            ("output.weight", (64, 1024), 16),
        ):
            rng = np.random.default_rng(seed)
            a = rng.standard_normal(shape).astype("<f4")
            tensors[name] = {"dtype": "F32", "shape": list(shape),
                             "data_offsets": [off, off + a.nbytes]}
            fh.write(a.tobytes())
            off += a.nbytes
        # PLE table: raw IQ4_NL block bytes stored as U8[720] (engine ple_formats has no
        # F32; strata_pack build:301 requires per_layer_token_embd.weight in shard2;
        # CLI emit rebuilds GGUF shape [160, 8] = 40 blocks × 18 B, fp16 scale 1.0).
        ple = np.random.default_rng(17).integers(0, 16, size=(40, 18), dtype=np.uint8)
        ple[:, 0] = 0x00
        ple[:, 1] = 0x3C  # fp16 1.0 little-endian
        a = ple.reshape(-1)
        tensors["per_layer_token_embd.weight"] = {"dtype": "U8", "shape": [720],
                                                  "data_offsets": [off, off + a.nbytes]}
        fh.write(a.tobytes())
        off += a.nbytes
    header = json.dumps(tensors, separators=(",", ":"))
    pad = (-len(header)) % 8
    header += " " * pad
    hb = header.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as dst, open(blob, "rb") as src:
        dst.write(len(hb).to_bytes(8, "little"))
        dst.write(hb)
        while True:
            chunk = src.read(1 << 24)
            if not chunk:
                break
            dst.write(chunk)
    blob.unlink()


def main() -> int:
    # The 10 GB fixture lives on ext4 scratch, NOT 9p: the CLI emit mmap-reads it
    # sequentially, and on 9p the page cache grows VmRSS toward the fixture size and
    # the process is OOM-killed (RC=137, observed 2026-10-17) even with fadvise DONTNEED
    # per slice. ext4 mmap reads are fast and the page cache is reclaimable (p3-09 convention).
    fx_dir = pathlib.Path("/opt/data/cache/scratch/p3-02-fixture")
    fx_dir.mkdir(parents=True, exist_ok=True)
    st = fx_dir / "fixture-1l.safetensors"
    guard = fx_dir / "guard.json"
    out_dir = pathlib.Path("/opt/data/cache/scratch/p3-02-out")
    gguf = out_dir / "emitted-1l.gguf"
    golden = out_dir / "golden"
    ours = out_dir / "ours"
    for d in (golden, ours):
        if d.exists():
            import shutil
            shutil.rmtree(d)
    for d in (golden, ours):
        d.mkdir(parents=True, exist_ok=True)

    # resume gate: rebuild only when the fixture is absent OR predates the PLE tensor
    # (strata_pack build:301 requires per_layer_token_embd.weight in shard2).
    need_build = not st.exists()
    if st.exists():
        with open(st, "rb") as fh:
            hl = int.from_bytes(fh.read(8), "little")
            if b"per_layer_token_embd.weight" not in fh.read(hl):
                need_build = True
    print("[1] fixture" + (" rebuild" if need_build else " exists with PLE, resume (no rebuild)"))
    if need_build:
        build_fixture(st)
    guard.write_text(json.dumps(GUARD, indent=1))

    print("[2] our CLI convert --emit-gguf")
    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    cli_env = dict(__import__("os").environ, PYTHONPATH=str(REPO / "src"))
    out = run([sys.executable, str(cli), "convert", str(st), "--out", str(ours / "unused"),
               "--emit-gguf", str(gguf), "--guard-keys", str(guard)], cli_env)
    print(out)

    print("[3a] strata-gguf --check on emitted GGUF")
    out = run([str(BUILD / "strata-gguf"), str(gguf), "--check"])
    print(out)
    if "architecture guard: PASS" not in out:
        raise SystemExit("FAIL: architecture guard not PASS on emitted GGUF")

    print("[3b] strata-dequant + CHECKSUM cross-check")
    out = run([str(BUILD / "strata-dequant"), str(gguf)])
    print(out)
    if "PASS" not in out.splitlines()[-1]:
        raise SystemExit("FAIL: strata-dequant did not print PASS")
    from stratashift.ingest.gguf import parse_gguf_header
    meta = parse_gguf_header(str(gguf))
    first_q2 = next(t for t in meta.tensors if t.type_name == "Q2_0")
    # strata-dequant folds CHECKSUM over the FIRST 64 blocks of the first Q2_0 tensor
    # (src/artifact/dequant.cpp:38-50, pin c566ff4) — compare the same window.
    with open(gguf, "rb") as fh:
        fh.seek(meta.data_start + first_q2.offset)
        raw = fh.read(64 * 18)
    s = sa = 0.0
    for b in range(64):
        for v in q2_0_unpack(raw[b * 18:(b + 1) * 18]):
            s += v
            sa += abs(v)
    engine_line = [ln for ln in out.splitlines() if ln.strip().startswith("CHECKSUM")]
    parts = engine_line[0].split()
    e_sum, e_sumabs = float(parts[2].split("=", 1)[1]), float(parts[3].split("=", 1)[1])
    if abs(e_sum - s) > 1e-6 * max(1.0, abs(e_sum)) or abs(e_sumabs - sa) > 1e-6 * max(1.0, abs(e_sumabs)):
        raise SystemExit(f"FAIL: CHECKSUM mismatch engine({e_sum},{e_sumabs}) vs python({s},{sa})")
    print(f"[3b] CHECKSUM cross-check PASS: sum={s:.10g} sumabs={sa:.10g}")

    print("[4] engine strata_pack build + verify on emitted GGUF")
    run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "build",
         "--gguf", str(gguf), "--out", str(golden), "--layers", str(N_LAYERS), "--skip-hash"])
    run([sys.executable, str(ENGINE / "tools" / "pack_index.py"), "--pack", str(golden)])
    out = run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "verify",
               "--gguf", str(gguf), "--out", str(golden)])
    print(out)

    print("[5] our CLI convert on emitted GGUF → byte-identical to golden")
    out = run([sys.executable, str(cli), "convert", str(gguf), "--out", str(ours)], cli_env)
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        g, o = golden / name, ours / name
        if not o.exists():
            raise SystemExit(f"FAIL: ours missing {name}")
        import hashlib
        gh = hashlib.sha256(g.read_bytes()).hexdigest()
        oh = hashlib.sha256(o.read_bytes()).hexdigest()
        if gh != oh:
            raise SystemExit(f"FAIL: {name} differs from golden ({gh[:12]} vs {oh[:12]})")
        print(f"[5] {name} byte-identical to engine golden ({o.stat().st_size} B)")

    print("[6] quantizer byte-identity vs engine quantize_q2_0")
    rng = np.random.default_rng(99)
    for trial in range(8):
        w = rng.standard_normal(64 * 32).astype(np.float32)
        ours_b = quantize_q2_0_stream(w.tobytes())
        eng_b = engine_quantize(w)
        if ours_b != eng_b:
            raise SystemExit(f"FAIL: quantizer byte mismatch at trial {trial}")
    print("[6] quantizer byte-identical to engine over 8×32 blocks")

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
