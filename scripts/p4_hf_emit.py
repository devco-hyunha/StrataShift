"""p4-01 HF MoE 입력 일반화 체인: HF safetensors → engine GGUF → engine acceptance.

Chain (docs/work/p4-01.md AC3/AC4):
  A. qwen4exp geometry HF fixture (fused gate_up + GDN/QSA/indexer/hc HF naming, F32,
     BF16 shared_expert_gate) → our CLI `--hf-emit` → engine GGUF v3.
     1. strata-gguf --check PASS (architecture guard on the emitted GGUF)
     2. strata-dequant PASS + CHECKSUM cross-check on the first Q2_0 tensor
     3. engine strata_pack build + pack_index + verify PASS (golden)
     4. our CLI convert on the emitted GGUF → pack byte-identical to golden,
        manifest planes agree; engine verify --out ours PASS
     5. convert twice → identical sha256
     6. no-full-load gate: streaming slice heap peak < 2 GiB
  B. mixtral-tiny (BF16, fused, 2 layers, NE=4) → `--hf-emit` → engine gguf_reader.py
     parsing matches OUR parser (tensor count/shape/offsets), canonical_xcheck dequant
     round-trip on the first Q2_0 tensor, our verify PASS. (The qwen4exp geometry guard
     is NOT applicable to Mixtral geometry — docs/work/p4-01.md 근거 §2 — acceptance is
     the reference verifier, not strata-gguf.)

Engine sources are read-only; the engine is never edited.
Run: python3 scripts/p4_hf_emit.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
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
from stratashift.polystrata.engine_paths import (  # noqa: E402
    engine_binary, resolve_engine_paths, scratch_dir)
PATHS = resolve_engine_paths()
sys.path.insert(0, str(PATHS.tools))
# mtp_pack.py:35 add_gguf_py() needs a gguf-py dir; the pinned llama.cpp checkout in
# the engine build tree (_deps/strata_llamacpp-src) satisfies it (p3-09 convention).
os.environ["STRATA_GGUF_PY"] = str(PATHS.gguf_py)

from canonical_xcheck import to_s2, from_s2  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402
from stratashift.quant.pack import q2_0_unpack  # noqa: E402

H, FF, NE = 2560, 640, 512
VOCAB = 1024
N_LAYERS = 1
# compiled geometry preset (p3-01 GUARD, VERIFIED) — the HF fixture is built so the
# header-DERIVED keys (embedding_length, expert_count, expert_feed_forward_length,
# head_count, head_count_kv) match; block_count stays the model geometry (48).
GUARD = {
    "architecture": "qwen4exp",
    "name": "stratashift p4-01 hf fixture",
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

# per-layer HF tensors: (suffix, dtype, shape, seed). Row-major HF [out, in] convention;
# 1-D tensors keep shape. Naming is fixture-measured (p4-01 risk rule: no HYPOTHESIS).
LAYER_TENSORS = [
    ("input_layernorm.weight", "F32", (H,), 21),
    ("post_attention_layernorm.weight", "F32", (H,), 22),
    ("hc_attn_norm.weight", "F32", (4 * H,), 23),
    ("hc_ffn_norm.weight", "F32", (4 * H,), 24),
    ("hc_attn_down.weight", "F32", (320, 4 * H), 25),
    ("hc_attn_up.weight", "F32", (4 * H, 320), 26),
    ("hc_attn_inject.weight", "F32", (4, 4 * H), 27),
    ("hc_ffn_down.weight", "F32", (320, 4 * H), 28),
    ("hc_ffn_up.weight", "F32", (4 * H, 320), 29),
    ("hc_ffn_inject.weight", "F32", (4, 4 * H), 30),
    ("linear_attn.q_proj.weight", "F32", (16 * 128, H), 31),
    ("linear_attn.k_proj.weight", "F32", (16 * 128, H), 32),
    ("linear_attn.v_proj.weight", "F32", (48 * 128, H), 33),
    ("linear_attn.g_proj.weight", "F32", (48 * 128, H), 34),
    ("linear_attn.o_proj.weight", "F32", (H, 48 * 128), 35),
    ("linear_attn.conv1d.weight", "F32", (2 * 16 * 128 + 48 * 128, 4), 36),
    ("linear_attn.A_log", "F32", (48,), 37),
    ("linear_attn.dt_bias", "F32", (48,), 38),
    ("linear_attn.norm.weight", "F32", (128,), 39),
    ("self_attn.q_proj.weight", "F32", (24 * 256, H), 40),
    ("self_attn.q_gate_proj.weight", "F32", (24 * 256, H), 41),
    ("self_attn.k_proj.weight", "F32", (2 * 256, H), 42),
    ("self_attn.v_proj.weight", "F32", (2 * 256, H), 43),
    ("self_attn.o_proj.weight", "F32", (H, 24 * 256), 44),
    ("self_attn.q_norm.weight", "F32", (256,), 45),
    ("self_attn.k_norm.weight", "F32", (256,), 46),
    ("indexer.q_proj.weight", "F32", (4 * 128, H), 47),
    ("indexer.k_proj.weight", "F32", (128, H), 48),
    ("indexer.q_norm.weight", "F32", (128,), 49),
    ("indexer.k_norm.weight", "F32", (128,), 50),
    ("mlp.gate.weight", "F32", (NE, H), 51),
    ("mlp.shared_expert.gate_proj.weight", "F32", (FF, H), 52),
    ("mlp.shared_expert.up_proj.weight", "F32", (FF, H), 53),
    ("mlp.shared_expert.down_proj.weight", "F32", (H, FF), 54),
    ("mlp.shared_expert_gate.weight", "BF16", (H,), 55),
    ("mlp.experts.gate_up_proj", "F32", (NE, 2 * FF, H), 56),
    ("mlp.experts.down_proj", "F32", (NE, H, FF), 57),
]


def build_hf_fixture(path: pathlib.Path) -> None:
    """safetensors: [u64 header_len][JSON][blob]; streamed per tensor, never full model.

    The fused gate_up blob is written expert-run by expert-run (gate rows then up rows
    per expert, C1/C2); every other tensor is one small array. Blobs go to a temp file
    first; the final file is header + streamed copy (p3-02 builder convention).

    WSL 9p write-cache OOM (observed 2026-10-17: direct 10 GB blob write to /workspace
    9p was SIGKILLed at 7.2 GB under memory pressure, RC=137): the blob lives on ext4
    scratch, and the final copy to 9p is streamed in 16 MB chunks with fsync every
    256 MB so the 9p cache flushes incrementally (p3-09 ext4-scratch convention).
    """
    tensors = {}
    off = 0
    blob = scratch_dir() / "p4-01-fixture.blob"
    with open(blob, "wb") as fh:
        for suffix, dtype, shape, seed in LAYER_TENSORS:
            name = f"model.layers.0.{suffix}"
            rng = np.random.default_rng(seed)
            begin = off
            if name.endswith("experts.gate_up_proj"):
                ne, two_ff, h = shape
                for e in range(ne):  # expert e = contiguous row block (NE,2FF,H)
                    a = rng.standard_normal((two_ff, h)).astype("<f4")
                    fh.write(a.tobytes())
                    off += a.nbytes
                    del a
            elif name.endswith("experts.down_proj"):
                ne, h, ff = shape
                for e in range(ne):  # expert e = contiguous row block (NE,H,FF)
                    a = rng.standard_normal((h, ff)).astype("<f4")
                    fh.write(a.tobytes())
                    off += a.nbytes
                    del a
            elif dtype == "BF16":
                a = rng.standard_normal(shape).astype("<f4")
                # store the BF16 image bytes verbatim (uint16 = top 16 bits of f32)
                u = a.astype("<f2").view(np.uint16)
                fh.write(u.tobytes())
                off += u.nbytes
            else:
                a = rng.standard_normal(shape).astype("<f4")
                fh.write(a.tobytes())
                off += a.nbytes
            tensors[name] = {"dtype": dtype, "shape": list(shape),
                             "data_offsets": [begin, off]}
        # globals + PLE (p3-01 contract: U8[720] IQ4_NL raw block bytes)
        for name, shape, seed in (
            ("model.embed_tokens.weight", (VOCAB, H), 61),
            ("lm_head.weight", (VOCAB, H), 62),
            ("model.norm.weight", (H,), 63),
        ):
            rng = np.random.default_rng(seed)
            a = rng.standard_normal(shape).astype("<f4")
            tensors[name] = {"dtype": "F32", "shape": list(shape),
                             "data_offsets": [off, off + a.nbytes]}
            fh.write(a.tobytes())
            off += a.nbytes
        ple = np.random.default_rng(64).integers(0, 16, size=(40, 18), dtype=np.uint8)
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
        flushed = 0
        while True:
            chunk = src.read(1 << 24)
            if not chunk:
                break
            dst.write(chunk)
            flushed += len(chunk)
            if flushed % (1 << 28) == 0:  # fsync every 256 MB — incremental 9p flush
                dst.flush()
                os.fsync(dst.fileno())
    blob.unlink()


def run(cmd: list[str], env: dict | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"FAIL: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def cli_json(out: str) -> dict:
    """Parse CLI stdout JSON docs; prefer the emit doc (has `gguf`)."""
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


def emit_inprocess(st: pathlib.Path, gguf: pathlib.Path, guard: dict) -> dict:
    """The same code the CLI runs (cmd_emit_gguf_hf), called in-process (p3-09 OOM
    convention: the safetensors convert subprocess is skipped, GGUF converts stay
    subprocesses)."""
    from stratashift.cli.main import cmd_emit_gguf_hf
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cmd_emit_gguf_hf(str(st), str(gguf), guard, None)
    if rc != 0:
        raise SystemExit(f"FAIL: cmd_emit_gguf_hf rc={rc}\n{buf.getvalue()}")
    return json.loads(buf.getvalue())


def chain_a(st: pathlib.Path, guard: dict, out_dir: pathlib.Path) -> None:
    gguf = out_dir / "hf-emitted.gguf"
    golden = out_dir / "golden"
    ours = out_dir / "ours"
    ours2 = out_dir / "ours2"
    for d in (golden, ours, ours2):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    print("[A1] our CLI HF emit path (in-process cmd_emit_gguf_hf)")
    emit = emit_inprocess(st, gguf, guard)
    emit_slice_heap = emit["slice_heap_peak_bytes"]
    emit_heap = emit["heap_peak_bytes"]
    print(f"[A1] emitted {emit['tensors']} tensors, derived guard {emit['derived_guard']}")

    print("[A2] strata-gguf --check on emitted GGUF")
    out = run([str(engine_binary(PATHS, "strata-gguf")), str(gguf), "--check"])
    print(out)
    if "architecture guard: PASS" not in out:
        raise SystemExit("FAIL: architecture guard not PASS on emitted GGUF")

    print("[A3] strata-dequant + CHECKSUM cross-check (first Q2_0 tensor)")
    out = run([str(engine_binary(PATHS, "strata-dequant")), str(gguf)])
    print(out)
    if "PASS" not in out.splitlines()[-1]:
        raise SystemExit("FAIL: strata-dequant did not print PASS")
    from stratashift.ingest.gguf import parse_gguf_header
    meta = parse_gguf_header(str(gguf))
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
    if abs(e_sum - s) > 1e-6 * max(1.0, abs(e_sum)) \
            or abs(e_sumabs - sa) > 1e-6 * max(1.0, abs(e_sumabs)):
        raise SystemExit(f"FAIL: CHECKSUM mismatch engine({e_sum},{e_sumabs}) vs python({s},{sa})")
    print(f"[A3] CHECKSUM cross-check PASS: sum={s:.10g} sumabs={sa:.10g}")

    print("[A4] engine strata_pack build + pack_index + verify (golden)")
    run([sys.executable, str(PATHS.tools / "strata_pack.py"), "build",
         "--gguf", str(gguf), "--out", str(golden), "--layers", str(N_LAYERS), "--skip-hash"])
    run([sys.executable, str(PATHS.tools / "pack_index.py"), "--pack", str(golden)])
    out = run([sys.executable, str(PATHS.tools / "strata_pack.py"), "verify",
               "--gguf", str(gguf), "--out", str(golden)])
    print(out)
    if "verify PASS" not in out:
        raise SystemExit("FAIL: strata_pack verify did not PASS on the HF-emitted GGUF")

    print("[A5] our CLI convert on emitted GGUF → byte-identical to golden")
    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    cli_env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = run([sys.executable, str(cli), "convert", str(gguf), "--out", str(ours)], cli_env)
    conv_heap = cli_json(out)["heap_peak_bytes"]
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        g, o = golden / name, ours / name
        if not o.exists():
            raise SystemExit(f"FAIL: ours missing {name}")
        gh = hashlib.sha256(g.read_bytes()).hexdigest()
        oh = hashlib.sha256(o.read_bytes()).hexdigest()
        if gh != oh:
            raise SystemExit(f"FAIL: {name} differs from golden ({gh[:12]} vs {oh[:12]})")
        print(f"[A5] {name} byte-identical to engine golden ({o.stat().st_size} B)")
    gm = json.loads((golden / "manifest.json").read_text())
    om = json.loads((ours / "manifest.json").read_text())
    for name, ent in gm["tensors"].items():
        oent = om["tensors"][name]
        for key in ("codes", "scales", "values"):
            if key in ent and ent[key] != oent.get(key):
                raise SystemExit(f"FAIL: manifest plane mismatch for {name}/{key}")
    print("[A5] manifest planes agree")

    print("[A6] engine strata_pack verify on OUR pack")
    out = run([sys.executable, str(PATHS.tools / "strata_pack.py"), "verify",
               "--gguf", str(gguf), "--out", str(ours)])
    print(out)
    if "verify PASS" not in out:
        raise SystemExit("FAIL: engine verify did not PASS on OUR pack")

    print("[A7] regression: convert twice → identical hashes")
    run([sys.executable, str(cli), "convert", str(gguf), "--out", str(ours2)], cli_env)
    for name in ("experts.bin", "dense.bin", "embd.bin", "index.txt"):
        if hashlib.sha256((ours / name).read_bytes()).hexdigest() \
                != hashlib.sha256((ours2 / name).read_bytes()).hexdigest():
            raise SystemExit(f"FAIL: {name} not reproducible")
    print("[A7] regression PASS (sha256 identical over two converts)")

    print("[A8] no-full-load gate: streaming slice heap peak < 2 GiB")
    limit = 2 << 30
    if emit_slice_heap > limit:
        raise SystemExit(f"FAIL: emit slice heap peak {emit_slice_heap} exceeds {limit}")
    print(f"[A8] emit slice heap peak = {emit_slice_heap} B; writer accumulation "
          f"{emit_heap} B; convert heap peak = {conv_heap} B")


def chain_b(mixtral: pathlib.Path, out_dir: pathlib.Path) -> None:
    gguf = out_dir / "mixtral-emitted.gguf"
    ours = out_dir / "mixtral-ours"
    if ours.exists():
        shutil.rmtree(ours)
    ours.mkdir(parents=True, exist_ok=True)

    print("[B1] our CLI HF emit on mixtral-tiny (BF16, fused, 2 layers)")
    # mixtral geometry is NOT qwen4exp — the compiled guard is not applicable
    # (p4-01 근거 §2); emit with an empty preset, acceptance is the reference verifier.
    emit = emit_inprocess(mixtral, gguf, {"architecture": "qwen4exp",
                                          "name": "stratashift p4-01 mixtral"})
    print(f"[B1] emitted {emit['tensors']} tensors, slice heap peak {emit['slice_heap_peak_bytes']} B")

    print("[B2] engine gguf_reader.py parse == our parser (tensor count/shape/offsets)")
    eng = GGUFFile(gguf)
    from stratashift.ingest.gguf import parse_gguf_header
    ours_meta = parse_gguf_header(str(gguf))
    eng_list = [(t.name, list(t.shape), t.offset) for t in eng.tensors]
    ours_list = [(t.name, list(t.shape), t.offset) for t in ours_meta.tensors]
    if eng_list != ours_list:
        raise SystemExit(f"FAIL: engine/our GGUF parse differ:\n engine {eng_list[:3]} "
                         f"\n ours {ours_list[:3]}")
    print(f"[B2] parse match PASS: {len(eng_list)} tensors, names/shapes/offsets identical")

    print("[B3] canonical_xcheck dequant round-trip on the first Q2_0 tensor")
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
    print(f"[B3] canonical_xcheck round-trip PASS ({len(ours_v)} values bit-exact)")

    print("[B4] our CLI convert on emitted GGUF + our verify PASS")
    cli = REPO / "src" / "stratashift" / "cli" / "main.py"
    cli_env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = run([sys.executable, str(cli), "convert", str(gguf), "--out", str(ours)], cli_env)
    out = run([sys.executable, str(cli), "verify", str(ours)], cli_env)
    res = json.loads(out)
    if not res["ok"]:
        raise SystemExit(f"FAIL: our verify not ok: {res['errors']}")
    print(f"[B4] our verify PASS (pool {res['pool_bytes']} B, {len(res['sha256'])} hashes)")


def main() -> int:
    # p4 fixture lives on ext4 scratch, NOT 9p: the 10.6 GB fixture mmap-read on 9p
    # grew the page cache to 6.5 GB RSS and the process was SIGKILLed twice (RC=137,
    # observed 2026-10-17). ext4 mmap reads are fast and the page cache is reclaimable.
    fx_dir = scratch_dir() / "p4-01-fixture"
    fx_dir.mkdir(parents=True, exist_ok=True)
    st = fx_dir / "hf-fixture-1l.safetensors"
    guard = fx_dir / "guard.json"
    # WSL 9p pending-delete race + mmap SIGBUS (p3-09 convention): harness outputs live
    # on ext4 scratch; only the fixture stays on 9p (streamed once, sequentially).
    out_dir_a = scratch_dir() / "p4-01-out-a"
    out_dir_b = scratch_dir() / "p4-01-out-b"
    out_dir_a.mkdir(parents=True, exist_ok=True)
    out_dir_b.mkdir(parents=True, exist_ok=True)

    print("[0] HF fixture" + (" build" if not st.exists() else " exists, resume (no rebuild)"))
    if not st.exists():
        build_hf_fixture(st)
    guard.write_text(json.dumps(GUARD, indent=1))

    chain_a(st, GUARD, out_dir_a)
    chain_b(REPO / ".local" / "models" / "mixtral-tiny.safetensors", out_dir_b)

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
