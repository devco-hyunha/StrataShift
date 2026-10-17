"""p4-03 real-model smoke: Qwen/Qwen1.5-MoE-A2.7B (8 shards, BF16, per-expert
Qwen2MoE naming, 24 layers, NE=60, H=2048, moe_FF=1408) → CLI
`convert <dir> --emit-gguf --hf-emit` → reference verifier acceptance.

Engine guard NOT applicable: compiled geometry is qwen4exp-only (struct_layouts §1
VERIFIED); Qwen1.5-MoE geometry (2048/24) is outside the strata-gguf guard
(p4-01 근거 §2, mixtral-tiny chain_b precedent). Acceptance is the reference
verifier: engine gguf_reader.py parse == our parser, canonical_xcheck dequant
round-trip bit-exact, our verify PASS.

Download: stdlib urllib from the HF CDN (307 redirect verified 2026-10-17) into
`.local/models/qwen1.5-moe-a2.7b/` — GITIGNORE policy, never committed.
Harness outputs live on ext4 scratch (p3-09/p4-01 convention).

Run: python3 scripts/p4_real_model_smoke.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from p4_hf_emit import GGUFFile, from_s2, to_s2  # noqa: E402
from stratashift.quant.pack import q2_0_unpack  # noqa: E402

MODEL_DIR = REPO / ".local" / "models" / "qwen1.5-moe-a2.7b"
OUT_DIR = pathlib.Path("/opt/data/cache/scratch/p4-03-out")
HF_BASE = "https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/resolve/main/"
N_SHARDS = 8
FILES = ["config.json", "model.safetensors.index.json"] \
        + [f"model-{i:05d}-of-00008.safetensors" for i in range(1, N_SHARDS + 1)]


def hf_size(name: str) -> int:
    """Content-Length from the HF CDN (HEAD) — the authoritative shard size."""
    req = urllib.request.Request(HF_BASE + name, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def download() -> None:
    """Fetch config.json + index.json + 8 shards into .local/models (GITIGNORE).
    Resume-safe: skip only when size == Content-Length. A connection that closes
    early truncates the file (observed 2026-10-17: shard 7 short by 55 MB, the
    in_bounds gate caught it) — verify Content-Length and re-download on mismatch."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = MODEL_DIR / name
        expected = hf_size(name)
        if dst.is_file() and expected and dst.stat().st_size == expected:
            print(f"[D] {name} complete ({expected} B), skip")
            continue
        print(f"[D] download {name} ({expected} B expected) ...")
        got = 0
        with urllib.request.urlopen(HF_BASE + name, timeout=60) as r, \
                open(dst, "wb") as fh:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
        if expected and got != expected:
            raise SystemExit(f"FAIL: {name} truncated ({got} < {expected} B)")
        print(f"[D] {name} done ({dst.stat().st_size} B)")


def cli(args: list[str]) -> str:
    cli_path = REPO / "src" / "stratashift" / "cli" / "main.py"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(cli_path)] + args,
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: CLI {' '.join(args)} rc={r.returncode}\n"
                         f"{r.stdout}\n{r.stderr}")
    return r.stdout


def guard_preset() -> dict:
    """Guard keys from config.json MEASURED values (never invented, p4-01 rule):
    embedding_length=hidden_size, block_count=num_hidden_layers,
    expert_count=num_experts, expert_feed_forward_length=moe_intermediate_size,
    head_count/ head_count_kv from num_attention_heads/num_key_value_heads."""
    c = json.loads((MODEL_DIR / "config.json").read_text())
    return {
        "architecture": "qwen4exp",
        "name": "stratashift p4-03 qwen1.5-moe-a2.7b",
        "qwen4exp.embedding_length": c["hidden_size"],
        "qwen4exp.block_count": c["num_hidden_layers"],
        "qwen4exp.expert_count": c["num_experts"],
        "qwen4exp.expert_feed_forward_length": c["moe_intermediate_size"],
        "qwen4exp.attention.head_count": c["num_attention_heads"],
        "qwen4exp.attention.head_count_kv": c["num_key_value_heads"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[0] real model dir" + (" exists, resume" if (MODEL_DIR / "model-00008-of-00008.safetensors").is_file() else " download"))
    download()

    print("[R1] engine-mirrored shard gates (load_sharded_model, 8 shards)")
    from stratashift.ingest.shard_index import load_sharded_model
    meta = load_sharded_model(str(MODEL_DIR))
    n_shards = len(meta.shard_data_start)
    if n_shards != N_SHARDS:
        raise SystemExit(f"FAIL: {n_shards} shards != {N_SHARDS}")
    print(f"[R1] gates PASS: shards={n_shards}, tensors={len(meta.tensors)}")

    print("[R2] CLI inspect on the directory")
    out = cli(["inspect", str(MODEL_DIR)])
    if f"shards={N_SHARDS}" not in out:
        raise SystemExit(f"FAIL: inspect shards != {N_SHARDS}\n{out}")
    print(f"[R2] {out.strip()}")

    print("[R3] CLI convert --emit-gguf --hf-emit on the shard directory")
    gguf = OUT_DIR / "qwen15-emitted.gguf"
    guard = OUT_DIR / "guard.json"
    guard.write_text(json.dumps(guard_preset(), indent=1))
    out = cli(["convert", str(MODEL_DIR), "--out", str(OUT_DIR / "cli-convert"),
               "--emit-gguf", str(gguf), "--hf-emit", "--guard-keys", str(guard)])
    emit = json.loads(out)
    print(f"[R3] emitted {emit['tensors']} tensors, slice heap peak "
          f"{emit['slice_heap_peak_bytes']} B")

    print("[R4] engine guard NOT applicable (geometry ≠ qwen4exp, p4-01 근거 §2) — "
          "reference verifier: engine gguf_reader.py parse == our parser")
    eng = GGUFFile(gguf)
    from stratashift.ingest.gguf import parse_gguf_header
    ours_meta = parse_gguf_header(str(gguf))
    eng_list = [(t.name, list(t.shape), t.offset) for t in eng.tensors]
    ours_list = [(t.name, list(t.shape), t.offset) for t in ours_meta.tensors]
    if eng_list != ours_list:
        raise SystemExit(f"FAIL: engine/our GGUF parse differ:\n engine {eng_list[:3]} "
                         f"\n ours {ours_list[:3]}")
    print(f"[R4] parse match PASS: {len(eng_list)} tensors, names/shapes/offsets identical")

    print("[R5] canonical_xcheck dequant round-trip on the first Q2_0 tensor")
    first_q2 = next(t for t in eng.tensors if t.type_name == "Q2_0")
    with open(gguf, "rb") as fh:
        fh.seek(eng.data_start + first_q2.offset)
        raw = fh.read(64 * 18)
    ours_v = []
    for b in range(64):
        ours_v.extend(q2_0_unpack(raw[b * 18:(b + 1) * 18]))
    eng_v = from_s2(to_s2(raw)).astype(np.float32)
    if not np.array_equal(np.array(ours_v, dtype=np.float32), eng_v):
        raise SystemExit("FAIL: canonical_xcheck Q2_0 round-trip mismatch")
    print(f"[R5] canonical_xcheck round-trip PASS ({len(ours_v)} values bit-exact)")

    print("[R6] our CLI convert on emitted GGUF + our verify PASS")
    ours = OUT_DIR / "ours"
    if ours.exists():
        import shutil
        shutil.rmtree(ours)
    ours.mkdir(parents=True, exist_ok=True)
    cli(["convert", str(gguf), "--out", str(ours)])
    out = cli(["verify", str(ours)])
    res = json.loads(out)
    if not res["ok"]:
        raise SystemExit(f"FAIL: our verify not ok: {res['errors']}")
    print(f"[R6] our verify PASS (pool {res['pool_bytes']} B, {len(res['sha256'])} hashes)")

    print("[R7] regression: CLI shard emit twice → identical sha256")
    import hashlib

    def sha256_file(p):
        # Stream in 1 MB chunks: read_bytes() on the 10.5 GB GGUF is a full load
        # and was OOM-killed (RC=137) on the 7.7 GB WSL box 2026-10-17.
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    sha1 = sha256_file(gguf)
    gguf2 = OUT_DIR / "qwen15-emitted-2.gguf"
    cli(["convert", str(MODEL_DIR), "--out", str(OUT_DIR / "cli-convert-2"),
         "--emit-gguf", str(gguf2), "--hf-emit", "--guard-keys", str(guard)])
    sha2 = sha256_file(gguf2)
    if sha1 != sha2:
        raise SystemExit("FAIL: real-model emit not reproducible")
    print(f"[R7] reproducible PASS (sha256 {sha1[:12]})")

    limit = 2 << 30
    if emit["slice_heap_peak_bytes"] > limit:
        raise SystemExit(f"FAIL: real-model emit slice heap peak {emit['slice_heap_peak_bytes']}")
    print(f"[R8] no-full-load gate: slice heap peak = {emit['slice_heap_peak_bytes']} B "
          f"(< {limit} B)")

    print("REAL MODEL SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
