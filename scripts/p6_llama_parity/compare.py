#!/usr/bin/env python3
"""p6-01 AC4: strata-shift-poly against the pinned llama.cpp on the same mixtral GGUF (engine docs/BACKENDS.md
"Adding an architecture" step 5: tokens and probabilities, short and long prompts, every expert on the CPU, with
and without guessed tokens).

llama_ref (build.py) runs the llama-named copy (to_llama.py) and writes the whole next-token distribution after
the prompt and after each greedy pick; strata-shift-poly takes those picks with --follow, so both see the same
context, and prints its whole distribution with --top-probs. Per step: the most likely token, KL(llama||engine)
and the largest probability difference. The Strata paper's yardstick (docs/paper/Strata-Paper.pdf in the pinned
tree): KL 0.022 against llama.cpp, 0.058 between llama.cpp's own CPU and GPU builds.

Usage: python scripts/p6_llama_parity/compare.py MIXTRAL.gguf [--steps 16]   -> JSON on stdout
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from stratashift.install.probe import find_nvcc  # noqa: E402
from stratashift.polystrata.engine_paths import (EXT_TARGET, LLAMACPP_PIN, engine_binary,  # noqa: E402
                                                 ext_paths, resolve_engine_paths)

FULL = ROOT / ".local" / "llama.cpp" / "full" / f"llama.cpp-{LLAMACPP_PIN}"
REF = ROOT / ".local" / "llama.cpp" / "parity-build" / "llama_ref.exe"
PROMPTS = {"short_3": [1, 2, 3], "long_41": list(range(5, 46)), "batch_200": list(range(100, 300))}
CONFIGS = {"default": [], "all_experts_cpu": ["--no-gpu-experts"], "window_1": ["--window", "1"],
           "no_graphs": ["--no-graphs"], "prompt_batch_1": ["--prompt-batch", "1"]}


def engine_env() -> dict:
    env = dict(os.environ)
    found = find_nvcc()
    if found:  # the CUDA runtime DLLs: bin/x64 beside nvcc's bin on Windows
        bins = [found[0].parent / "x64", found[0].parent]
        env["PATH"] = os.pathsep.join([str(b) for b in bins if b.is_dir()] + [env.get("PATH", "")])
    return env


def run_ref(gguf_llama: pathlib.Path, prompt: list[int], n: int, work: pathlib.Path):
    out = work / "ref"
    subprocess.run([str(REF), str(gguf_llama), ",".join(map(str, prompt)), str(n), str(out)], check=True,
                   capture_output=True)
    toks = [int(x) for x in (work / "ref.tokens").read_text(encoding="utf-8").strip().split(",")]
    probs = np.fromfile(work / "ref.probs", np.float32)
    return toks, probs.reshape(n + 1, -1)[:n].astype(np.float64)


def run_engine(exe: pathlib.Path, gguf: str, prompt: list[int], follow: list[int], args: list[str],
               n_vocab: int) -> np.ndarray:
    r = subprocess.run([str(exe), "--model", gguf, "--tokens", ",".join(map(str, prompt)),
                        "--follow", ",".join(map(str, follow)), "--max-new", str(len(follow)),
                        "--max-context", "1024", "--vram-reserve-mib", "512", "--top-probs", str(n_vocab), *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=engine_env())
    if r.returncode != 0:
        raise RuntimeError(f"{exe.name} rc={r.returncode}: {r.stderr[-500:]}")
    steps = []
    for line in r.stderr.splitlines():
        if line.startswith("top "):
            q = np.zeros(n_vocab)
            for i, p in re.findall(r"(\d+):([0-9.eE+-]+)", line):
                q[int(i)] = float(p)
            steps.append(q)
    return np.array(steps)


def stats(p: np.ndarray, q: np.ndarray) -> dict:
    eps = 1e-12
    kl = (p * (np.log(p + eps) - np.log(q + eps))).sum(1)
    return {"steps": len(p), "same_top1_%": round(100 * float((p.argmax(1) == q.argmax(1)).mean()), 1),
            "kl_mean": float(f"{kl.mean():.3g}"), "kl_max": float(f"{kl.max():.3g}"),
            "max_abs_prob_diff": float(f"{np.abs(p - q).max():.3g}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--steps", type=int, default=16)
    a = ap.parse_args()
    exe = engine_binary(ext_paths(resolve_engine_paths()), EXT_TARGET)
    for need, how in ((REF, "python scripts/p6_llama_parity/build.py"),
                      (exe, "stratashift build-engine --gpu --ext")):
        if not need.is_file():
            print(f"error: {need} not found - {how}", file=sys.stderr)
            return 2
    sys.path.insert(0, str(FULL / "gguf-py"))
    from to_llama import to_llama
    res: dict = {}
    with tempfile.TemporaryDirectory() as d:
        work = pathlib.Path(d)
        gguf_llama = work / "llama.gguf"
        to_llama(a.gguf, str(gguf_llama))
        for pn, prompt in PROMPTS.items():
            toks, p = run_ref(gguf_llama, prompt, a.steps, work)
            res[pn] = {"llama_tokens": toks}
            for cn, args in CONFIGS.items():
                res[pn][cn] = stats(p, run_engine(exe, a.gguf, prompt, toks, args, p.shape[1]))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
