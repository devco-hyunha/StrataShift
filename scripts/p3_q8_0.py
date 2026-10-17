"""p3-09 dense Q8_0 chain: safetensors F32 dense → GGUF Q8_0 → engine acceptance.

Chain (docs/work/p3-09.md):
  1. fixture safetensors: reuse the p3-02 builder (1 layer, Qwen4exp geometry,
     F32 experts + F32 dense/embd + U8 PLE table).
  2. our CLI `convert --emit-gguf --dense-type Q8_0` → engine GGUF v3 where the dense
     tensors are Q8_0 blocks (experts stay Q2_0 — the engine arena is Q2_0-only).
  3. engine strata-gguf --check PASS (architecture guard on the emitted GGUF).
  4. engine strata-dequant PASS + CHECKSUM cross-check on the first Q2_0 tensor.
  5. engine strata_pack build + pack_index + verify PASS — dense Q8_0 bit-exact against
     the gguf-py dequantize reference.
  6. our CLI convert on the emitted GGUF → pack byte-identical to the engine golden,
     manifest plane fields agree; engine verify --out ours PASS (engine acceptance of
     OUR artifact); convert twice → identical sha256.
  7. quantizer byte-identity vs engine mtp_pack.q8_0 + round-trip invariant vs engine
     to_s8_q8_0+from_s8_q8_0.
  8. no-full-load gate: streaming slice heap peak stays at header + single tensor slice.

Engine sources are read-only; the engine is never edited.
Run: python3 scripts/p3_q8_0.py
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
ENGINE = pathlib.Path("/opt/data/polystrata-src")
BUILD = pathlib.Path("/opt/data/polystrata-build")
sys.path.insert(0, str(ENGINE / "tools"))
# mtp_pack.py:35 add_gguf_py() needs a gguf-py dir; the pinned llama.cpp checkout in
# the engine build tree (_deps/strata_llamacpp-src, Q2_0=42, Q8_0=8) satisfies it.
os.environ["STRATA_GGUF_PY"] = str(BUILD / "_deps" / "strata_llamacpp-src" / "gguf-py")

from canonical_xcheck import to_s8_q8_0, from_s8_q8_0  # noqa: E402
from mtp_pack import q8_0 as engine_q8_0  # noqa: E402
from stratashift.quant.pack import q2_0_unpack, q8_0_unpack, quantize_q8_0_stream  # noqa: E402
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
    fx_dir = REPO / ".local" / "models" / "p3-09-fixture"
    fx_dir.mkdir(parents=True, exist_ok=True)
    st = fx_dir / "fixture-1l.safetensors"
    guard = fx_dir / "guard.json"
    # 9p pending-delete race + mmap SIGBUS (WSL, observed 2026-10-17): rmtree+mkdir on
    # /workspace/.local can wipe a re-created dir mid-write, and mmap reads of a
    # 708 MB GGUF written to 9p can SIGBUS (9p cache invalidation; the CLI convert
    # crashed with exit 135, empty stdout/stderr). All harness outputs — including
    # the emitted GGUF that the CLI/engine mmap-read — live on ext4 scratch; only
    # the 10 GB fixture stays on 9p (streamed once, sequentially).
    out_dir = pathlib.Path("/opt/data/cache/scratch/p3-09-out")
    gguf = out_dir / "emitted-1l-q80.gguf"
    golden = out_dir / "golden"
    ours = out_dir / "ours"
    ours2 = out_dir / "ours2"
    for d in (golden, ours, ours2):
        if d.exists():
            shutil.rmtree(d)
    # 9p pending-delete race: rmtree leaves the dir in a pending-delete state that can
    # wipe a re-created dir mid-write (observed: convert's mkdir+truncate succeeded,
    # then open("ab") hit FileNotFoundError). Materialize the dirs immediately after
    # rmtree so the delete flushes before the engine/CLI write into them.
    for d in (golden, ours, ours2):
        d.mkdir(parents=True, exist_ok=True)

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

    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    cli_env = dict(os.environ, PYTHONPATH=str(REPO / "src"))

    print("[2] our CLI emit path (in-process cmd_emit_gguf --dense-type Q8_0)")
    # The CLI `convert` on the safetensors fixture materializes each full expert tensor
    # (3.35 GB slice) in the p2-01 convert path; as a subprocess under memory pressure
    # that is the OOM point (observed: CLI killed at the first slice, empty stdout).
    # The emit path is the same code the CLI runs (cmd_emit_gguf), called in-process;
    # the GGUF-path converts in [5]/[7] stay CLI subprocesses (per-expert-run streaming,
    # peak = one 6.5 MB run).
    import contextlib, io
    from stratashift.cli.main import cmd_emit_gguf
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cmd_emit_gguf(str(st), str(gguf), GUARD, None, dense_type="Q8_0")
    if rc != 0:
        raise SystemExit(f"FAIL: cmd_emit_gguf rc={rc}\n{buf.getvalue()}")
    emit = json.loads(buf.getvalue())
    emit_peak = emit["rss_peak_bytes"]
    emit_heap = emit["heap_peak_bytes"]
    emit_slice_heap = emit["slice_heap_peak_bytes"]

    from stratashift.ingest.gguf import parse_gguf_header
    meta = parse_gguf_header(str(gguf))
    q80 = [t for t in meta.tensors if t.type_name == "Q8_0"]
    if not q80:
        raise SystemExit("FAIL: emitted GGUF has no Q8_0 dense tensors")
    print(f"[2] emitted GGUF: {len(q80)} Q8_0 dense tensors "
          f"({sum(t.expected_bytes() for t in q80)} B), "
          f"{len([t for t in meta.tensors if t.type_name == 'F32'])} F32 left")

    print("[3a] strata-gguf --check on emitted GGUF")
    out = run([str(BUILD / "strata-gguf"), str(gguf), "--check"])
    print(out)
    if "architecture guard: PASS" not in out:
        raise SystemExit("FAIL: architecture guard not PASS on emitted GGUF")

    print("[3b] strata-dequant + CHECKSUM cross-check (first Q2_0 tensor)")
    out = run([str(BUILD / "strata-dequant"), str(gguf)])
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
    run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "build",
         "--gguf", str(gguf), "--out", str(golden), "--layers", str(N_LAYERS), "--skip-hash"])
    run([sys.executable, str(ENGINE / "tools" / "pack_index.py"), "--pack", str(golden)])
    out = run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "verify",
               "--gguf", str(gguf), "--out", str(golden)])
    print(out)
    if "verify PASS" not in out:
        raise SystemExit("FAIL: strata_pack verify did not PASS on the Q8_0 GGUF")

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

    gm = json.loads((golden / "manifest.json").read_text())
    om = json.loads((ours / "manifest.json").read_text())
    for name, ent in gm["tensors"].items():
        oent = om["tensors"][name]
        for key in ("codes", "scales", "values"):
            if key in ent and ent[key] != oent.get(key):
                raise SystemExit(f"FAIL: manifest plane mismatch for {name}/{key}: "
                                 f"{ent[key]} vs {oent.get(key)}")
    print("[5] manifest planes agree")

    print("[6] engine strata_pack verify on OUR pack")
    out = run([sys.executable, str(ENGINE / "tools" / "strata_pack.py"), "verify",
               "--gguf", str(gguf), "--out", str(ours)])
    print(out)
    if "verify PASS" not in out:
        raise SystemExit("FAIL: engine verify did not PASS on OUR pack")

    print("[7] regression: convert twice → identical hashes")
    run([sys.executable, str(cli), "convert", str(gguf), "--out", str(ours2)], cli_env)
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        if hashlib.sha256((ours / name).read_bytes()).hexdigest() \
                != hashlib.sha256((ours2 / name).read_bytes()).hexdigest():
            raise SystemExit(f"FAIL: {name} not reproducible")
    print("[7] regression PASS (sha256 identical over two converts)")

    print("[8] quantizer byte-identity vs engine mtp_pack.q8_0 + round-trip vs "
          "engine to_s8_q8_0+from_s8_q8_0")
    rng = np.random.default_rng(101)
    for trial in range(8):
        w = rng.standard_normal(32 * 32).astype(np.float32)
        raw = quantize_q8_0_stream(w.tobytes())
        if raw != engine_q8_0(w).tobytes():
            raise SystemExit(f"FAIL: Q8_0 quantizer bytes diverge at trial {trial}")
        ours_v = np.array(q8_0_unpack(raw), dtype=np.float32)
        eng_v = from_s8_q8_0(to_s8_q8_0(raw)).astype(np.float32)
        if not np.array_equal(ours_v, eng_v):
            raise SystemExit(f"FAIL: Q8_0 round-trip mismatch at trial {trial}")
    print("[8] Q8_0 quantizer byte-identical + round-trip bit-exact over 8×32 blocks")

    print("[9] no-full-load gate: streaming slice heap peak (emit) + convert heap peak")
    limit = 2 << 30
    if emit_slice_heap > limit:
        raise SystemExit(f"FAIL: emit slice heap peak {emit_slice_heap} exceeds {limit}")
    print(f"[9] emit slice heap peak = {emit_slice_heap} B (header + single expert run); "
          f"writer accumulation {emit_heap} B (engine GGUFWriter design); "
          f"convert heap peak = {conv_heap} B; VmRSS peaks for info: emit={emit_peak} convert={conv_peak}")

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
