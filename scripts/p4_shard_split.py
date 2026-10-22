"""p4-02 multi-shard chain: qwen4exp HF fixture split into 2 shards + index.json →
CLI `convert --emit-gguf` on the directory → GGUF byte-identical to the single-file
input's GGUF → engine chain acceptance (chain_a reuse) + no-full-load gate.

Split rule (docs/work/p4-02.md): tensor payloads are byte-identical per tensor, so
any weight_map partition yields the same tensor bytes; the emitted GGUF must be
sha256-identical to the single-file emit. Shard names follow the HF convention
model-00001-of-00002.safetensors (engine gguf_split.hpp:18-50 five-digit pattern).

Run: python3 scripts/p4_shard_split.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from stratashift.polystrata.engine_paths import scratch_dir  # noqa: E402
from p4_hf_emit import GUARD, LAYER_TENSORS, VOCAB, H, chain_a, cli_json, emit_inprocess  # noqa: E402

SCRATCH = scratch_dir()
FX_DIR = SCRATCH / "p4-01-fixture"
ST = FX_DIR / "hf-fixture-1l.safetensors"
SHARD_DIR = SCRATCH / "p4-02-shards"
OUT_DIR = SCRATCH / "p4-02-out"


def tensor_specs() -> list[tuple[str, str, list[int], int]]:
    """(name, dtype, shape, seed) in the exact build_hf_fixture order (p4-01 seeds)."""
    specs = [(f"model.layers.0.{s}", dt, list(sh), seed)
             for s, dt, sh, seed in LAYER_TENSORS]
    for name, shape, seed in (("model.embed_tokens.weight", (VOCAB, H), 61),
                              ("lm_head.weight", (VOCAB, H), 62),
                              ("model.norm.weight", (H,), 63)):
        specs.append((name, "F32", list(shape), seed))
    specs.append(("per_layer_token_embd.weight", "U8", [720], 64))
    return specs


def tensor_chunks(name: str, dtype: str, shape: list[int], seed: int):
    """Generator of one tensor's payload chunks, byte-identical to build_hf_fixture's
    writer branch (p4-01 seeds). Fused tensors stream expert-run by expert-run (C1/C2) —
    accumulating 6.7 GB in bytes += is quadratic (observed 2026-10-17: 98% CPU, no file
    growth at 99 MB of 6.7 GB gate_up blob)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if name.endswith("experts.gate_up_proj"):
        ne, two_ff, h = shape
        for e in range(ne):  # expert e = contiguous row block (NE,2FF,H)
            a = rng.standard_normal((two_ff, h)).astype("<f4")
            yield a.tobytes()
    elif name.endswith("experts.down_proj"):
        ne, h, ff = shape
        for e in range(ne):  # expert e = contiguous row block (NE,H,FF)
            a = rng.standard_normal((h, ff)).astype("<f4")
            yield a.tobytes()
    elif dtype == "BF16":
        a = rng.standard_normal(shape).astype("<f4")
        yield a.astype("<f2").view(np.uint16).tobytes()
    elif dtype == "U8":  # PLE: p4-01 builder — integers(0,16,(40,18)), col0=0, col1=0x3C
        ple = rng.integers(0, 16, size=(40, 18), dtype=np.uint8)
        ple[:, 0] = 0x00
        ple[:, 1] = 0x3C
        yield ple.reshape(-1).tobytes()
    else:
        a = rng.standard_normal(shape).astype("<f4")
        yield a.tobytes()


def write_shard(path: pathlib.Path, specs: list[tuple[str, str, list[int], int]]) -> None:
    """One shard: [u64 header_len][JSON][blob] — payloads streamed chunk by chunk
    to an ext4 blob, then prefix+header+streamed copy (p4-01 builder convention)."""
    header: dict[str, dict] = {}
    off = 0
    blob = path.with_suffix(".blob")
    with open(blob, "wb") as fh:
        for name, dtype, shape, seed in specs:
            begin = off
            for chunk in tensor_chunks(name, dtype, shape, seed):
                fh.write(chunk)
                off += len(chunk)
                del chunk
            header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [begin, off]}
    text = json.dumps(header, separators=(",", ":"))
    text += " " * (-len(text) % 8)
    hb = text.encode("utf-8")
    with open(path, "wb") as dst, open(blob, "rb") as src:
        dst.write(len(hb).to_bytes(8, "little"))
        dst.write(hb)
        while True:
            chunk = src.read(1 << 24)
            if not chunk:
                break
            dst.write(chunk)
    blob.unlink()


def build_shards() -> pathlib.Path:
    specs = tensor_specs()
    assert len(specs) == 41, f"fixture spec count changed: {len(specs)}"
    shard1 = specs[:22]   # layer tensors 0..21 (gate_up lives in shard 2)
    shard2 = specs[22:]   # layer tensors 22..39 + globals + PLE
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    p1 = SHARD_DIR / "model-00001-of-00002.safetensors"
    p2 = SHARD_DIR / "model-00002-of-00002.safetensors"
    print(f"[S1] write shard 1 ({len(shard1)} tensors)")
    write_shard(p1, shard1)
    print(f"[S2] write shard 2 ({len(shard2)} tensors)")
    write_shard(p2, shard2)
    weight_map = {}
    total = 0
    shard1_names = {name for name, _, _, _ in shard1}
    for name, dtype, shape, seed in specs:
        weight_map[name] = p1.name if name in shard1_names else p2.name
        n = 1
        for d in shape:
            n *= d
        total += n * {"F32": 4, "BF16": 2, "U8": 1}[dtype]
    (SHARD_DIR / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
    print(f"[S3] index.json: {len(weight_map)} tensors, total_size={total}")
    # CLI inspect on the directory (header-only, C1/C2 — shards + expert map)
    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(cli), "inspect", str(SHARD_DIR)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or "shards=2" not in r.stdout:
        raise SystemExit(f"FAIL: CLI inspect on shard dir rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    print(f"[S3] CLI inspect: {r.stdout.strip()}")
    return SHARD_DIR


def cli_emit(model_dir: str, gguf: pathlib.Path) -> dict:
    """CLI subprocess emit on the shard directory (the real user path, not in-process)."""
    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    guard_path = FX_DIR / "guard.json"
    r = subprocess.run([sys.executable, str(cli), "convert", model_dir,
                        "--out", str(OUT_DIR / "cli-convert-shard"),
                        "--emit-gguf", str(gguf), "--hf-emit",
                        "--guard-keys", str(guard_path)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: CLI shard emit rc={r.returncode}\n{r.stdout}\n{r.stderr}")
    return cli_json(r.stdout)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[0] single-file fixture" + (" build" if not ST.exists() else " exists, resume"))
    if not ST.exists():
        from p4_hf_emit import build_hf_fixture
        build_hf_fixture(ST)
    (FX_DIR / "guard.json").write_text(json.dumps(GUARD, indent=1))

    print("[G] single-file emit = golden GGUF (in-process, p4-01 path)")
    golden_gguf = OUT_DIR / "hf-emitted.gguf"
    if not golden_gguf.exists():
        emit_inprocess(ST, golden_gguf, GUARD)
    golden_sha = hashlib.sha256(golden_gguf.read_bytes()).hexdigest()

    shard_dir = build_shards()

    print("[S4] CLI convert --emit-gguf --hf-emit on the shard directory")
    emit = cli_emit(str(shard_dir), OUT_DIR / "shard-emitted.gguf")
    shard_sha = hashlib.sha256((OUT_DIR / "shard-emitted.gguf").read_bytes()).hexdigest()
    if shard_sha != golden_sha:
        raise SystemExit(f"FAIL: shard GGUF {shard_sha[:12]} != single-file {golden_sha[:12]}")
    print(f"[S4] GGUF sha256 identical to single-file input ({shard_sha[:12]})")

    print("[S5] regression: CLI shard emit twice → identical sha256")
    emit2 = cli_emit(str(shard_dir), OUT_DIR / "shard-emitted-2.gguf")
    sha2 = hashlib.sha256((OUT_DIR / "shard-emitted-2.gguf").read_bytes()).hexdigest()
    if sha2 != shard_sha:
        raise SystemExit("FAIL: shard emit not reproducible")
    print("[S5] reproducible PASS")

    print("[S6] engine chain acceptance on the shard input (chain_a reuse — emits from "
          "the shard directory, engine chain on that GGUF)")
    chain_a(shard_dir, GUARD, OUT_DIR / "shard-chain")

    limit = 2 << 30
    if emit["slice_heap_peak_bytes"] > limit:
        raise SystemExit(f"FAIL: shard emit slice heap peak {emit['slice_heap_peak_bytes']}")
    print(f"[S7] no-full-load gate: shard emit slice heap peak = "
          f"{emit['slice_heap_peak_bytes']} B (< {limit} B)")

    print("SHARD SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
