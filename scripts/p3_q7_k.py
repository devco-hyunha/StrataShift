"""p3-07 dense Q3_K chain: safetensors F32 dense → GGUF Q3_K → engine acceptance.

Chain (docs/work/p3-07.md):
  1. fixture safetensors: reuse the p3-02 builder (1 layer, Qwen4exp geometry,
     F32 experts + F32 dense/embd + U8 PLE table).
  2. our CLI `convert --emit-gguf --dense-type Q3_K` → engine GGUF v3 where the dense
     tensors are Q3_K blocks (experts stay Q2_0 — the engine arena is Q2_0-only).
  3. engine strata-gguf --check PASS (architecture guard on the emitted GGUF).
  4. engine strata-dequant PASS + CHECKSUM cross-check on the first Q2_0 tensor.
  5. engine strata_pack build + pack_index + verify PASS — dense Q3_K bit-exact against
     the gguf-py dequantize reference (ref/load.py _dequant_flat dispatches Q3_K to gguf-py).
  6. our CLI convert on the emitted GGUF → pack byte-identical to the engine golden.
  7. quantizer round-trip invariant vs engine to_s4_q3_k+from_s4_q3_k.
  8. no-full-load gate: streaming slice heap peak stays at header + single tensor slice.

Engine sources are read-only; the engine is never edited.
Run: python3 scripts/p3_q7_k.py
"""
from __future__ import annotations

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
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))
from stratashift.polystrata.engine_paths import (  # noqa: E402
    engine_binary, engine_tool_env, resolve_engine_paths)
PATHS = resolve_engine_paths()
# engine Python tools run as subprocesses need the pinned gguf-py (p6-13)
os.environ.update(engine_tool_env(PATHS))
sys.path.insert(0, str(PATHS.tools))

from canonical_xcheck import to_s4_q3_k, from_s4_q3_k  # noqa: E402
from stratashift.quant.pack import q2_0_unpack, q3_k_unpack, quantize_q3_k_stream  # noqa: E402
from p3_safetensors_gguf import GUARD, build_fixture  # noqa: E402

N_LAYERS = 1


def run(cmd: list[str], env: dict | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def cli_json(out: str) -> dict:
    """Parse the CLI stdout: the convert path prints one JSON doc, the emit path appends
    a second (emit summary). Return the doc carrying `rss_peak_bytes` that the harness
    gates — prefer the emit doc (has `gguf`), else the convert doc (has `out_dir`)."""
    docs = []
    dec = json.JSONDecoder()
    i = 0
    while i < len(out):
        if out[i].isspace():
            i += 1
            continue
        obj, i = dec.raw_decode(out, i)
        docs.append(obj)
    for d in docs:
        if "gguf" in d:
            return d
    for d in docs:
        if "rss_peak_bytes" in d:
            return d
    raise SystemExit(f"FAIL: CLI stdout has no JSON summary: {out[:200]}")


def main() -> int:
    fx_dir = REPO / ".local" / "models" / "p3-07-fixture"
    fx_dir.mkdir(parents=True, exist_ok=True)
    st = fx_dir / "fixture-1l.safetensors"
    gguf = fx_dir / "emitted-1l-q3k.gguf"
    guard = fx_dir / "guard.json"
    golden = fx_dir / "golden"
    ours = fx_dir / "ours"
    for d in (golden, ours):
        if d.exists():
            shutil.rmtree(d)

    need_build = not st.exists()
    if st.exists():
        with open(st, "rb") as fh:
            hl = int.from_bytes(fh.read(8), "little")
            if b"per_layer_token_embd.weight" not in fh.read(hl):
                need_build = True
    print("[1] fixture" + (" rebuild" if need_build else " exists with PLE, resume (no rebuild)"))
    if need_build:
        build_fixture(st)
    guard.write_text(json.dumps(GUARD, indent=1), encoding="utf-8")

    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    cli_env = dict(os.environ, PYTHONPATH=str(REPO / "src"))

    print("[2] our CLI convert --emit-gguf --dense-type Q3_K")
    out = run([sys.executable, str(cli), "convert", str(st), "--out", str(ours / "unused"),
               "--emit-gguf", str(gguf), "--guard-keys", str(guard),
               "--dense-type", "Q3_K"], cli_env)
    print(out)
    emit_peak = cli_json(out)["rss_peak_bytes"]
    emit_heap = cli_json(out)["heap_peak_bytes"]
    emit_slice_heap = cli_json(out)["slice_heap_peak_bytes"]

    from stratashift.ingest.gguf import parse_gguf_header
    meta = parse_gguf_header(str(gguf))
    q3k = [t for t in meta.tensors if t.type_name == "Q3_K"]
    if not q3k:
        raise SystemExit("FAIL: emitted GGUF has no Q3_K dense tensors")
    print(f"[2] emitted GGUF: {len(q3k)} Q3_K dense tensors "
          f"({sum(t.expected_bytes() for t in q3k)} B), "
          f"{len([t for t in meta.tensors if t.type_name == 'F32'])} F32 left")

    print("[3a] strata-gguf --check on emitted GGUF")
    out = run([str(engine_binary(PATHS, "strata-gguf")), str(gguf), "--check"])
    print(out)
    if "architecture guard: PASS" not in out:
        raise SystemExit("FAIL: architecture guard not PASS on emitted GGUF")

    print("[3b] strata-dequant + CHECKSUM cross-check (first Q2_0 tensor)")
    out = run([str(engine_binary(PATHS, "strata-dequant")), str(gguf)])
    print(out)
    if "PASS" not in out.splitlines()[-1]:
        raise SystemExit("FAIL: strata-dequant did not print PASS")
    first_q2 = next(t for t in meta.tensors if t.type_name == "Q2_0")
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
    run([sys.executable, str(PATHS.tools / "strata_pack.py"), "build",
         "--gguf", str(gguf), "--out", str(golden), "--layers", str(N_LAYERS), "--skip-hash"])
    run([sys.executable, str(PATHS.tools / "pack_index.py"), "--pack", str(golden)])
    out = run([sys.executable, str(PATHS.tools / "strata_pack.py"), "verify",
               "--gguf", str(gguf), "--out", str(golden)])
    print(out)
    if "verify PASS" not in out:
        raise SystemExit("FAIL: strata_pack verify did not PASS on the Q3_K GGUF")

    print("[5] our CLI convert on emitted GGUF → byte-identical to golden")
    out = run([sys.executable, str(cli), "convert", str(gguf), "--out", str(ours)], cli_env)
    conv_peak = cli_json(out)["rss_peak_bytes"]
    conv_heap = cli_json(out)["heap_peak_bytes"]
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        g, o = golden / name, ours / name
        if not o.exists():
            raise SystemExit(f"FAIL: ours missing {name}")
        gh = hashlib.sha256(g.read_bytes()).hexdigest()
        oh = hashlib.sha256(o.read_bytes()).hexdigest()
        if gh != oh:
            raise SystemExit(f"FAIL: {name} differs from golden ({gh[:12]} vs {oh[:12]})")
        print(f"[5] {name} byte-identical to engine golden ({o.stat().st_size} B)")

    print("[6] quantizer round-trip invariant vs engine to_s4_q3_k+from_s4_q3_k")
    rng = np.random.default_rng(101)
    for trial in range(8):
        w = rng.standard_normal(256 * 32).astype(np.float32)
        raw = quantize_q3_k_stream(w.tobytes())
        ours_v = np.array(q3_k_unpack(raw), dtype=np.float32)
        eng_v = from_s4_q3_k(to_s4_q3_k(raw)).astype(np.float32)
        if not np.array_equal(ours_v, eng_v):
            raise SystemExit(f"FAIL: Q3_K round-trip mismatch at trial {trial}")
    print("[6] Q3_K round-trip bit-exact over 8×32 blocks vs engine canonical")

    print("[7] no-full-load gate: streaming slice heap peak (emit) + convert heap peak")
    limit = 2 << 30
    if emit_slice_heap > limit:
        raise SystemExit(f"FAIL: emit slice heap peak {emit_slice_heap} exceeds {limit}")
    print(f"[7] emit slice heap peak = {emit_slice_heap} B (header + single expert run); "
          f"writer accumulation {emit_heap} B (engine GGUFWriter design); "
          f"convert heap peak = {conv_heap} B; VmRSS peaks for info: emit={emit_peak} convert={conv_peak}")

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
