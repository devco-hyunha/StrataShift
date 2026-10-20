"""p5-09 AC3/AC4 smoke: our emitted GGUF → PolyStrata serve/server.py → OpenAI API.

Chain (measured contracts, pin c566ff4 read-only — docs/work/p5-09.md, docs/rea/tokenizer_keys.md):
  [R1] mini fixture + HF tokenizer files (measured file structure, mini-scaled V=2048)
  [R2] CLI emit → GGUF with tokenizer.ggml.* meta (tokenizer_meta_keys > 0)
  [R3] engine `tools/strata_tokenizer.py:extract()` → tokenizer dir (vocab.json,
       merges.json, token_type.json, tokenizer.json, chat_template.jinja — the
       server load_tokenizer contract, serve/server.py:6853-6858)
  [R4] run config JSON (exe, args --model --serve, cwd, tokenizer, model_name —
       serve/models.py:96-119 gates, server.py:7018 --engine strata,
       start_model 6862-6930 spawn, StrataEngine GEN-line protocol 750-778)
  [R5] `python -m serve.server --engine strata --config CFG --port P`
  [R6] GET /health + GET /v1/models, POST /v1/chat/completions (non-stream),
       assert HTTP 200 + non-empty content, record the engine DONE figures
  [R7] teardown, rc=0 with the measured report printed

mini run path (AC3) is the harness gate; the real-model path (AC4) uses the same
config builder — pass --model to switch. Run (WSL host, engine tree + CUDA build):
  python3 scripts/p5_api_smoke.py                 # mini chain
  python3 scripts/p5_api_smoke.py --model /path/gemma4-emitted.gguf --model-dir \
      /path/gemma4-26b-a4b-it   # real chain (AC4)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
ENGINE_ROOT = pathlib.Path(os.environ.get("P5_ENGINE_ROOT", "/opt/p5-07/PolyStrata-main"))
BUILD = pathlib.Path(os.environ.get("P5_SMOKE_BUILD", "/opt/p5-07/build-cuda13"))
OUT_DIR = pathlib.Path(os.environ.get("P5_SMOKE_OUT", "/opt/p5-07/scratch/p5-09-out"))
PORT = int(os.environ.get("P5_SMOKE_PORT", "8095"))
sys.path.append(str(ENGINE_ROOT / "tools"))

import strata_tokenizer as ST  # noqa: E402

from test_gemma4_mini_emit import build_mini, GUARD  # noqa: E402
from test_tokenizer_emit import mini_tokenizer_files, write_tokenizer_dir  # noqa: E402


def engine_samples(tag: str):
    """RSS/VRAM/Swap of the engine process (strata-poly child of the server) —
    AC4 measured gate: VmPeak/VmRSS/VmSwap from /proc/<pid>/status, VRAM via
    nvidia-smi memory.used. p5-08 AC4 precedent (RSS 7,076,756 kB, Swaps 0)."""
    out = {}
    r = subprocess.run(["pgrep", "-x", "strata-poly"], capture_output=True, text=True)
    pids = r.stdout.split()
    if pids:
        status = open(f"/proc/{pids[0]}/status").read()
        for line in status.splitlines():
            if line.startswith(("VmPeak:", "VmRSS:", "VmSwap:")):
                k, v = line.split(":", 1)
                out[k] = v.strip()
    nv = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True)
    out["gpu_used_mib"] = nv.stdout.strip()
    print(f"engine samples {tag}: {out}", flush=True)
    return out


def http_json(method: str, path: str, body: dict | None, port: int, timeout: float = 120.0):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def emit_gguf(model_dir: pathlib.Path, guard_path: pathlib.Path, gguf: pathlib.Path) -> int:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    # --guard-keys takes a JSON FILE path (cli/main.py argv contract), the same
    # way tests/test_gemma4_mini_emit.py drives it
    r = subprocess.run([sys.executable, str(REPO / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(model_dir), "--out", str(OUT_DIR / "pack"),
                        "--emit-gguf", str(gguf), "--hf-emit",
                        "--guard-keys", str(guard_path), "--dense-type", "Q4_K"],
                       capture_output=True, text=True, env=env)
    print(r.stdout, r.stderr, flush=True)
    assert r.returncode == 0, "R2 emit rc != 0"
    return json.loads(r.stdout)["tokenizer_meta_keys"]


def make_run_config(gguf: pathlib.Path, tok_dir: pathlib.Path, exe: pathlib.Path) -> pathlib.Path:
    cfg = {
        "exe": str(exe),
        "args": ["--model", str(gguf), "--serve", "--max-context", "256"],
        "cwd": str(OUT_DIR),
        "tokenizer": str(tok_dir),
        "model_name": "stratashift-smoke",
    }
    p = OUT_DIR / "run-config.json"
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="pre-emitted GGUF (real chain, AC4); skips R1/R2")
    ap.add_argument("--model-dir", help="HF dir with tokenizer files (real chain)")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.model:  # AC4 real chain: tokenizer dir from the pre-emitted GGUF
        gguf = pathlib.Path(a.model)
        tokenizer_meta = "pre-emitted"
    else:        # AC3 mini chain: build fixture + tokenizer, emit
        model = OUT_DIR / "mini"
        build_mini(model)
        write_tokenizer_dir(model)
        guard = OUT_DIR / "guard.json"
        guard.write_text(json.dumps(GUARD))
        gguf = OUT_DIR / "mini-tok.gguf"
        tokenizer_meta = emit_gguf(model, guard, gguf)
        if tokenizer_meta <= 0:
            print(f"R2 FAIL: tokenizer_meta_keys={tokenizer_meta}", file=sys.stderr)
            return 1
        print(f"R2 PASS: emit rc=0 tokenizer_meta_keys={tokenizer_meta}", flush=True)

    # R3: engine extract() → tokenizer dir
    tok_dir = OUT_DIR / "tokenizer"
    cfg_out = ST.extract(gguf, OUT_DIR)
    files = sorted(p.name for p in (tok_dir).iterdir())
    print(f"R3 PASS: extract vocab_size={cfg_out['vocab_size']} model={cfg_out['model']} "
          f"files={files}", flush=True)

    # R4: run config
    exe = BUILD / "strata-poly"
    if not exe.is_file():
        print(f"R4 FAIL: engine exe {exe} missing", file=sys.stderr)
        return 1
    cfg_path = make_run_config(gguf, tok_dir, exe)
    print(f"R4 PASS: config {cfg_path} exe={exe}", flush=True)

    # R5: launch the server
    server = subprocess.Popen(
        [sys.executable, "-m", "serve.server", "--engine", "strata",
         "--config", str(cfg_path), "--port", str(PORT)],
        cwd=str(ENGINE_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    deadline = time.time() + 180
    health = None
    while time.time() < deadline:
        if server.poll() is not None:
            out = server.stdout.read() if server.stdout else ""
            print(f"R5 FAIL: server rc={server.returncode} log tail: {out[-800:]}",
                  file=sys.stderr)
            return 1
        try:
            health = http_json("GET", "/health", None, PORT, 5)
            if health[0] == 200:
                break
        except Exception:
            time.sleep(2)
    if health is None or health[0] != 200:
        print("R5 FAIL: /health not 200 within 180 s", file=sys.stderr)
        server.terminate()
        return 1
    print("R5 PASS: server up, /health 200", flush=True)
    samples_load = engine_samples("after-load")

    # R6: the OpenAI round trip
    status, models = http_json("GET", "/v1/models", None, PORT)
    assert status == 200, status
    print(f"R6a PASS: /v1/models 200 model={models.get('data', [{}])[0].get('id')}",
          flush=True)
    body = {"model": "stratashift-smoke",
            "messages": [{"role": "user", "content": "Reply with exactly: P509APIOK"}],
            "max_tokens": 24, "temperature": 0}
    status, resp = http_json("POST", "/v1/chat/completions", body, PORT)
    if status != 200:
        print(f"R6 FAIL: chat completion HTTP {status} {resp}", file=sys.stderr)
        server.terminate()
        return 1
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    print(f"R6 PASS: HTTP 200 content={content!r} usage={usage}", flush=True)
    samples_gen = engine_samples("after-generation")
    server.terminate()
    server.wait(timeout=10)
    print(f"R7 PASS: teardown rc=0 tokenizer_meta={tokenizer_meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
