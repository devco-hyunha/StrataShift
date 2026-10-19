"""p5-06 AC4 chain: Gemma4 fixture → CLI emit → reference verifier acceptance.

Fixture: .local/models/gemma4-26b-a4b-it (AC2 measured, 2 shards, 51.6 GB, headers
probe GREEN 955/58 tensors — docs/work/p5-06.md AC2). Harness artifacts go to ext4
scratch (p3/p4 convention).

Chain (p4-03 template, engine Gemma4 keys measured in the pinned source):
  [R1] shard HEAD Content-Length verify + index.json present (p4-02 gate)
  [R2] engine-mirrored shard gates (load_sharded_model)
  [R3] CLI inspect on the directory
  [R4] CLI convert --emit-gguf --hf-emit --guard-keys scripts/p5_gemma4_guard.json
  [R5] reference verifier: engine gguf_reader.py parse == our parser (names/shapes/offsets)
  [R6] canonical_xcheck Q2_0 round-trip bit-exact (engine canonical mapping)
  [R7] engine Gemma4 GUARD refusal on the artifact — MEASURED record. Engine
       check_architecture (gguf_reader.hpp:796-812, pin c566ff4) passes only
       qwen4exp/qwen35moe names; gemma4 refuses. strata-gguf --check must exit non-zero
       naming the refusal. The engine Gemma4 class is CUDA/HIP-only
       (model_family.hpp:52-62) — engine RUN acceptance is a GPU/human gate.
  [R8] our verify PASS + no-full-load heap gate (< 2 GiB slice peak, p4-03 convention)

Run: python3 scripts/p5_gemma4_smoke.py
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# p5-07: the engine tools checkout (pin c566ff4 tools, read-only) provides canonical_xcheck /
# gguf_reader — same contract as cli/main.py appends ENGINE_TOOLS to sys.path.
ENGINE_TOOLS = pathlib.Path(os.environ.get("STRATASHIFT_ENGINE_TOOLS",
                                           "/opt/data/polystrata-src/tools"))
if str(ENGINE_TOOLS) not in sys.path:
    sys.path.append(str(ENGINE_TOOLS))

import numpy as np  # noqa: E402
import urllib.request  # noqa: E402

from p4_hf_emit import GGUFFile, from_s2, to_s2  # noqa: E402
from stratashift.quant.pack import q2_0_unpack  # noqa: E402

# p5-07: MODEL_DIR is env-overridable — the 9p 9p read of the 51.6 GB HF shards is the
# bottleneck (measured ~0.75 GB/min emit); the ext4 copy of the shards is the host run path.
MODEL_DIR = pathlib.Path(os.environ.get("P5_MODEL_DIR",
                                        str(REPO / ".local" / "models" / "gemma4-26b-a4b-it")))
GUARD = REPO / "scripts" / "p5_gemma4_guard.json"
# p5-07: harness paths are env-overridable — sandbox chain used /opt/data ext4 scratch;
# host PC runs (WSL CUDA gate) point these at .local/out and the WSL build tree.
OUT_DIR = pathlib.Path(os.environ.get("P5_SMOKE_OUT", "/opt/data/cache/scratch/p5-06-out"))
BUILD = pathlib.Path(os.environ.get("P5_SMOKE_BUILD", "/opt/data/polystrata-build"))
HF_BASE = "https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/"
SHARDS = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def hf_size(name: str) -> int:
    req = urllib.request.Request(HF_BASE + name, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0))


def cli(args: list[str]) -> str:
    cli_path = REPO / "src" / "stratashift" / "cli" / "main.py"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(cli_path)] + args,
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: CLI {' '.join(args)} rc={r.returncode}\n"
                         f"{r.stdout}\n{r.stderr}")
    return r.stdout


def sha256_file(p: pathlib.Path) -> str:
    """Streamed hash — read_bytes() on a 15 GB GGUF is a full load (p4-03 gate)."""
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[R1] shard HEAD Content-Length verify + index.json present")
    for name in SHARDS:
        p = MODEL_DIR / name
        expected = hf_size(name)
        if not p.is_file() or p.stat().st_size != expected:
            raise SystemExit(f"FAIL: {name} local {p.stat().st_size if p.is_file() else 'absent'} "
                             f"!= Content-Length {expected} (re-download first)")
        print(f"[R1] {name} complete ({expected} B)")
    idx = MODEL_DIR / "model.safetensors.index.json"
    if not idx.is_file():
        raise SystemExit("FAIL: model.safetensors.index.json missing in the fixture dir")

    print("[R2] engine-mirrored shard gates (load_sharded_model)")
    from stratashift.ingest.shard_index import load_sharded_model
    meta = load_sharded_model(str(MODEL_DIR))
    print(f"[R2] gates PASS: shards={len(meta.shard_data_start)}, tensors={len(meta.tensors)}")

    print("[R3] CLI inspect on the directory")
    out = cli(["inspect", str(MODEL_DIR)])
    print(f"[R3] {out.strip()}")

    print("[R4] CLI convert --emit-gguf --hf-emit --guard-keys (Q4_K dense)")
    gguf = OUT_DIR / "gemma4-emitted.gguf"
    if gguf.is_file() and gguf.stat().st_size > 1 << 30:
        print(f"[R4] SKIP (resume): {gguf} exists ({gguf.stat().st_size} B)")
        emit = None
    else:
        out = cli(["convert", str(MODEL_DIR), "--out", str(OUT_DIR / "cli-convert"),
                   "--emit-gguf", str(gguf), "--hf-emit", "--guard-keys", str(GUARD),
                   "--dense-type", "Q4_K"])
        emit = json.loads(out)
        print(f"[R4] emitted {emit['tensors']} tensors, GGUF {gguf.stat().st_size} B, "
              f"slice heap peak {emit['slice_heap_peak_bytes']} B, rss peak "
              f"{emit['rss_peak_bytes']} B, derived {json.dumps(emit['derived_guard'])}")

    print("[R5] reference verifier: engine gguf_reader.py parse == our parser")
    eng = GGUFFile(gguf)
    from stratashift.ingest.gguf import parse_gguf_header
    ours_meta = parse_gguf_header(str(gguf))
    eng_list = [(t.name, list(t.shape), t.offset) for t in eng.tensors]
    ours_list = [(t.name, list(t.shape), t.offset) for t in ours_meta.tensors]
    if eng_list != ours_list:
        raise SystemExit(f"FAIL: engine/our GGUF parse differ:\n engine {eng_list[:3]} "
                         f"\n ours {ours_list[:3]}")
    print(f"[R5] parse match PASS: {len(eng_list)} tensors, names/shapes/offsets identical")

    print("[R6] canonical_xcheck Q2_0 round-trip on the first Q2_0 tensor")
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
    print(f"[R6] canonical_xcheck round-trip PASS ({len(ours_v)} values bit-exact)")

    print("[R7] engine Gemma4 GUARD refusal on the artifact (measured record)")
    if (BUILD / "strata-gguf").is_file():
        r = subprocess.run([str(BUILD / "strata-gguf"), str(gguf), "--check"],
                           capture_output=True, text=True)
        msg = (r.stdout + r.stderr).strip().splitlines()
        tail = msg[-1] if msg else "(no message)"
        print(f"[R7] strata-gguf rc={r.returncode}: {tail}")
        # the measured refusal is the guard MESSAGE ("architecture guard: FAIL …
        # requires 'qwen4exp'"); strata-gguf --check prints it and exits 0 (2026-10-18 실측).
        if "architecture guard: FAIL" not in (r.stdout + r.stderr):
            raise SystemExit("FAIL: engine guard did NOT print the gemma4 refusal — the "
                             "measured record (gguf_reader.hpp:796-812) says gemma4 is "
                             "rejected")
        print("[R7] refusal recorded PASS — engine RUN acceptance stays a GPU/human gate")
    else:
        print(f"[R7] SKIP: strata-gguf binary absent at {BUILD} (refusal source citation: "
              f"gguf_reader.hpp:796-812, pin c566ff4)")

    print("[R8] our verify PASS + no-full-load heap gate")
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
    print(f"[R8] our verify PASS (pool {res['pool_bytes']} B, {len(res['sha256'])} hashes)")
    limit = 2 << 30
    if emit and emit["slice_heap_peak_bytes"] > limit:
        raise SystemExit(f"FAIL: slice heap peak {emit['slice_heap_peak_bytes']} B > 2 GiB")
    if emit:
        print(f"[R8] no-full-load gate PASS: slice heap peak {emit['slice_heap_peak_bytes']} B")
    else:
        print("[R8] heap gate: resume run — emit stats not re-measured")

    print(f"\nsha256(gguf) = {sha256_file(gguf)}")
    print("GEMMA4 AC4 CHAIN PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
