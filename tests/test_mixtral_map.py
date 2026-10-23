"""p6-01 AC2/AC4: Mixtral HF → engine_ext GGUF mapping, emit, and the strata-shift-poly run.

Grounds (never invented): HF keys = tests/fixtures/mixtral_header.json (`key_patterns`, measured
2026-10-20); engine keys = what engine_ext/src/poly/mixtral.cpp loads (docs/rea/mixtral_class_contract.md);
HF→GGUF pairing = pinned llama.cpp gguf-py tensor_mapping.py (hf_gguf_map MIXTRAL_* comment).

The mini fixture keeps the measured names and scales the shapes down (H 1024, 2 layers, 8 heads of 128 with
2 kv heads, 4 experts top-2, FF 1024, vocab 2048). Its weights are small random BF16 values, not zeros, so
the run computes on real numbers. The run tests skip when strata-shift-poly is not built
(`stratashift build-engine --gpu --ext`).
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402
from stratashift.meta.expert_map import classify  # noqa: E402
from stratashift.meta.hf_gguf_map import map_dense  # noqa: E402
from stratashift.meta.mixtral_rope import rope_head_dim, rope_row_order  # noqa: E402
from stratashift.polystrata import engine_tools  # noqa: E402

# convert --hf-emit needs the pinned engine tools checkout (CI has none, p5-04); pure mapping/RoPE tests stay ungated.
ENGINE_GATE = pytest.mark.skipif(not engine_tools.available(), reason="pinned engine tools checkout required (STRATASHIFT_ENGINE_TOOLS)")

FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "mixtral_header.json").read_text(encoding="utf-8"))

H, FF, NE, USED, V = 1024, 1024, 4, 2, 2048
HEADS, KV, HEAD_DIM = 8, 2, 128
LAYERS = (0, 1)

GUARD = {
    "architecture": "mixtral",
    "name": "p6-01 mini mixtral",
    "mixtral.block_count": len(LAYERS),
    "mixtral.embedding_length": H,
    "mixtral.expert_count": NE,
    "mixtral.expert_used_count": USED,
    "mixtral.expert_feed_forward_length": FF,
    "mixtral.attention.head_count": HEADS,
    "mixtral.attention.head_count_kv": KV,
    "mixtral.attention.key_length": HEAD_DIM,
    "mixtral.attention.value_length": HEAD_DIM,
    "mixtral.rope.freq_base": 1000000.0,  # fixture config rope_theta
    "mixtral.attention.layer_norm_rms_epsilon": 1e-05,  # mini value; the real guard reads config.json
    "mixtral.vocab_size": V,
}


def layer_tensors(L: int) -> dict[str, tuple]:
    """Mini shapes for the measured layer key vocabulary (fixture key_patterns)."""
    p = f"model.layers.{L}."
    t = {
        p + "self_attn.q_proj.weight": (HEADS * HEAD_DIM, H),
        p + "self_attn.k_proj.weight": (KV * HEAD_DIM, H),
        p + "self_attn.v_proj.weight": (KV * HEAD_DIM, H),
        p + "self_attn.o_proj.weight": (H, HEADS * HEAD_DIM),
        p + "input_layernorm.weight": (H,),
        p + "post_attention_layernorm.weight": (H,),
        p + "block_sparse_moe.gate.weight": (NE, H),
    }
    for e in range(NE):
        q = f"{p}block_sparse_moe.experts.{e}."
        t[q + "w1.weight"] = (FF, H)
        t[q + "w2.weight"] = (H, FF)
        t[q + "w3.weight"] = (FF, H)
    return t


def _bf16(values: np.ndarray) -> bytes:
    return (values.astype("<f4").view("<u4") >> 16).astype("<u2").tobytes()


def build_mini(dirpath: pathlib.Path) -> None:
    """One BF16 safetensors shard + index.json + config.json. Norm weights are 1, the rest small random."""
    dirpath.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple] = {
        "model.embed_tokens.weight": (V, H),
        "lm_head.weight": (V, H),
        "model.norm.weight": (H,),
    }
    for L in LAYERS:
        tensors.update(layer_tensors(L))
    rng = np.random.default_rng(601)
    off, header, payloads = 0, {}, []
    for name, shape in sorted(tensors.items()):
        n = int(np.prod(shape))
        vals = np.ones(n, np.float32) if len(shape) == 1 else rng.normal(0, 0.02, n).astype(np.float32)
        payload = _bf16(vals)
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [off, off + len(payload)]}
        payloads.append(payload)
        off += len(payload)
    header["__metadata__"] = {"format": "pt"}
    text = json.dumps(header, separators=(",", ":"))
    text += " " * ((-len(text)) % 8)
    blob = text.encode()
    shard = dirpath / "model-00001-of-00001.safetensors"
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for p in payloads:
            fh.write(p)
    (dirpath / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(len(p) for p in payloads)},
         "weight_map": {n: shard.name for n in tensors}}, indent=1), encoding="utf-8")
    cfg = dict(FIXTURE["config"], hidden_size=H, num_hidden_layers=len(LAYERS), num_attention_heads=HEADS,
               num_key_value_heads=KV, head_dim=HEAD_DIM, num_local_experts=NE, num_experts_per_tok=USED,
               intermediate_size=FF, vocab_size=V)
    (dirpath / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def _cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def _emit(tmp: pathlib.Path, guard: dict, dense_type: str = "Q4_K") -> tuple[subprocess.CompletedProcess,
                                                                               pathlib.Path]:
    model = tmp / "mini"
    if not model.is_dir():
        build_mini(model)
    g = tmp / "guard.json"
    g.write_text(json.dumps(guard), encoding="utf-8")
    gguf = tmp / f"mini-{dense_type}.gguf"
    r = _cli("convert", str(model), "--out", str(tmp / "pack"), "--emit-gguf", str(gguf), "--hf-emit",
             "--guard-keys", str(g), "--dense-type", dense_type)
    return r, gguf


# ---------- mapping (fixture vocabulary) ----------

def test_every_measured_key_is_mapped():
    """Each measured HF key is an engine dense key or a classified expert — none dropped, none invented."""
    want_dense = {
        "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
        "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
        "block_sparse_moe.gate.weight": "ffn_gate_inp.weight",
    }
    for name in FIXTURE["key_patterns"]:
        hit = map_dense(name, "mixtral")
        expert = classify(name)
        assert (hit is None) != (expert is None), name
        if hit is not None and hit[0] >= 0:
            suffix = name.split(".", 3)[3]
            assert hit[1] == f"blk.{hit[0]}.{want_dense[suffix]}", name
    assert map_dense("model.embed_tokens.weight", "mixtral") == (-1, "token_embd.weight", "transpose")
    assert map_dense("lm_head.weight", "mixtral") == (-1, "output.weight", "transpose")
    assert map_dense("model.norm.weight", "mixtral") == (-1, "output_norm.weight", "same")


def test_expert_roles_from_measured_names():
    """w1 = gate, w3 = up, w2 = down (llama.cpp tensor_mapping.py :597/:543/:674)."""
    assert classify("model.layers.0.block_sparse_moe.experts.3.w1.weight") == (0, 3, "gate")
    assert classify("model.layers.0.block_sparse_moe.experts.3.w3.weight") == (0, 3, "up")
    assert classify("model.layers.7.block_sparse_moe.experts.0.w2.weight") == (7, 0, "down")
    assert classify("model.layers.0.block_sparse_moe.gate.weight") is None


def test_measured_shapes_give_engine_shapes():
    """The fixture's layer-0 shapes become the engine shapes the class uploads."""
    keys = FIXTURE["measured_keys"]
    cfg = FIXTURE["config"]
    hd = cfg["head_dim"]
    assert keys["model.layers.0.self_attn.q_proj.weight"]["shape"] == [cfg["num_attention_heads"] * hd,
                                                                      cfg["hidden_size"]]
    assert keys["model.layers.0.self_attn.k_proj.weight"]["shape"][0] // hd == cfg["num_key_value_heads"]
    assert keys["model.layers.0.block_sparse_moe.gate.weight"]["shape"] == [cfg["num_local_experts"],
                                                                           cfg["hidden_size"]]
    w1 = keys["model.layers.0.block_sparse_moe.experts.0.w1.weight"]["shape"]
    w2 = keys["model.layers.0.block_sparse_moe.experts.0.w2.weight"]["shape"]
    assert w1 == [cfg["intermediate_size"], cfg["hidden_size"]] and w2 == w1[::-1]


# ---------- rotary row order ----------

def _rope_half(x: np.ndarray, pos: int, base: float) -> np.ndarray:
    """HF rotate_half rotary of one head: pairs (i, i + d/2)."""
    d = x.size
    half = d // 2
    th = pos * base ** (-2.0 * np.arange(half) / d)
    a, b = x[:half], x[half:]
    return np.concatenate([a * np.cos(th) - b * np.sin(th), a * np.sin(th) + b * np.cos(th)])


def _rope_normal(x: np.ndarray, pos: int, base: float) -> np.ndarray:
    """ggml normal rotary (dsv4_rope, norm off): pairs (2j, 2j + 1)."""
    d = x.size
    th = pos * base ** (-2.0 * np.arange(d // 2) / d)
    a, b = x[0::2], x[1::2]
    y = np.empty_like(x)
    y[0::2] = a * np.cos(th) - b * np.sin(th)
    y[1::2] = a * np.sin(th) + b * np.cos(th)
    return y


def test_row_order_turns_half_split_into_normal_pairs():
    """Rows reordered, the normal-pair rotary of the reordered head is the HF rotary reordered, and q·k
    between positions is the same as HF's."""
    rng = np.random.default_rng(1)
    d, base = 128, 1e6
    order = rope_row_order(2 * d, d)
    assert sorted(order.tolist()) == list(range(2 * d))
    assert order[:4].tolist() == [0, 64, 1, 65] and order[d:d + 2].tolist() == [128, 192]
    one = order[:d]
    q, k = rng.normal(size=d), rng.normal(size=d)
    for pos in (0, 1, 17, 4095):
        assert np.allclose(_rope_normal(q[one], pos, base), _rope_half(q, pos, base)[one])
    hf = _rope_half(q, 9, base) @ _rope_half(k, 4, base)
    ours = _rope_normal(q[one], 9, base) @ _rope_normal(k[one], 4, base)
    assert np.isclose(hf, ours)


def test_rope_head_dim_gate():
    assert rope_head_dim(GUARD, "mixtral") == HEAD_DIM
    with pytest.raises(ValueError, match="key_length"):
        rope_head_dim({}, "mixtral")
    with pytest.raises(ValueError, match="dimension_count"):
        rope_head_dim(dict(GUARD, **{"mixtral.rope.dimension_count": 64}), "mixtral")
    with pytest.raises(ValueError, match="whole heads"):
        rope_row_order(100, HEAD_DIM)


# ---------- emit ----------

@ENGINE_GATE
def test_mini_emit_keys_types_shapes(tmp_path):
    """Q4_K dense emit: every key the class loads, at the shape it uploads; 1-D keys F32; experts Q2_0;
    our pack builder and verify accept the file."""
    r, gguf = _emit(tmp_path, GUARD)
    assert r.returncode == 0, r.stdout + r.stderr
    meta = parse_gguf_header(str(gguf))
    assert meta.metadata["general.architecture"] == "mixtral"
    t = {x.name: x for x in meta.tensors}
    for L in LAYERS:
        b = f"blk.{L}."
        assert list(t[b + "attn_q.weight"].shape) == [H, HEADS * HEAD_DIM]
        assert list(t[b + "attn_k.weight"].shape) == [H, KV * HEAD_DIM]
        assert list(t[b + "attn_v.weight"].shape) == [H, KV * HEAD_DIM]
        assert list(t[b + "attn_output.weight"].shape) == [HEADS * HEAD_DIM, H]
        assert list(t[b + "ffn_gate_inp.weight"].shape) == [H, NE]
        assert list(t[b + "ffn_gate_exps.weight"].shape) == [H, FF, NE]
        assert list(t[b + "ffn_up_exps.weight"].shape) == [H, FF, NE]
        assert list(t[b + "ffn_down_exps.weight"].shape) == [FF, H, NE]
        for k in ("attn_norm.weight", "ffn_norm.weight"):
            assert t[b + k].type_name == "F32" and list(t[b + k].shape) == [H]
        assert t[b + "attn_q.weight"].type_name == "Q4_K"
        # the router is never quantized (pinned llama.cpp llama-quant.cpp:306)
        assert t[b + "ffn_gate_inp.weight"].type_name == "F32"
        assert t[b + "ffn_gate_exps.weight"].type_name == "Q2_0"
    assert t["output_norm.weight"].type_name == "F32"
    assert list(t["token_embd.weight"].shape) == [H, V] and list(t["output.weight"].shape) == [H, V]
    assert not any("shexp" in n for n in t)  # no shared expert (contract §3)
    out = tmp_path / "ours"
    r = _cli("convert", str(gguf), "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    r = _cli("verify", str(out))
    assert r.returncode == 0 and json.loads(r.stdout)["ok"], r.stdout + r.stderr


@ENGINE_GATE
def test_mini_emit_q_rows_reordered(tmp_path):
    """F32 dense emit: attn_q/attn_k bytes are the HF rows in rope_row_order; v keeps HF order."""
    r, gguf = _emit(tmp_path, GUARD, dense_type="F32")
    assert r.returncode == 0, r.stdout + r.stderr
    from stratashift.ingest.safetensors_header import parse_header
    src = tmp_path / "mini" / "model-00001-of-00001.safetensors"
    hdr = parse_header(str(src))
    hf = {x.name: x for x in hdr.tensors}
    meta = parse_gguf_header(str(gguf))
    t = {x.name: x for x in meta.tensors}

    def hf_rows(name):
        e = hf[name]
        with open(src, "rb") as fh:
            fh.seek(hdr.data_start + e.data_offsets[0])
            raw = np.frombuffer(fh.read(e.nbytes()), "<u2").astype("<u4") << 16
        return raw.view("<f4").reshape(e.shape)

    def ours(name, rows):
        with open(gguf, "rb") as fh:
            fh.seek(meta.data_start + t[name].offset)
            return np.frombuffer(fh.read(t[name].expected_bytes()), "<f4").reshape(rows, H)

    q = hf_rows("model.layers.1.self_attn.q_proj.weight")
    assert np.array_equal(ours("blk.1.attn_q.weight", HEADS * HEAD_DIM), q[rope_row_order(len(q), HEAD_DIM)])
    k = hf_rows("model.layers.1.self_attn.k_proj.weight")
    assert np.array_equal(ours("blk.1.attn_k.weight", KV * HEAD_DIM), k[rope_row_order(len(k), HEAD_DIM)])
    r = hf_rows("model.layers.1.block_sparse_moe.gate.weight")
    with open(gguf, "rb") as fh:
        fh.seek(meta.data_start + t["blk.1.ffn_gate_inp.weight"].offset)
        got = np.frombuffer(fh.read(t["blk.1.ffn_gate_inp.weight"].expected_bytes()), "<f4").reshape(NE, H)
    assert np.array_equal(got, r)  # BF16 values widened, bytes in HF row order
    v = hf_rows("model.layers.1.self_attn.v_proj.weight")
    assert np.array_equal(ours("blk.1.attn_v.weight", KV * HEAD_DIM), v)


@ENGINE_GATE
def test_no_gemma4_tokenizer_for_mixtral(tmp_path):
    """A checkpoint with tokenizer.json gets no tokenizer metadata (Mixtral's is not measured yet) —
    never the gemma4 tables the tokenizer builder holds."""
    build_mini(tmp_path / "mini")
    (tmp_path / "mini" / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE"}}), encoding="utf-8")
    r, gguf = _emit(tmp_path, GUARD)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no tokenizer metadata for mixtral" in r.stderr
    meta = parse_gguf_header(str(gguf))
    assert not any(k.startswith("tokenizer.") for k in meta.metadata)


@ENGINE_GATE
def test_guard_mismatch_refused(tmp_path):
    """A preset that disagrees with the header-derived geometry is refused (rc 2), no file written."""
    r, gguf = _emit(tmp_path, dict(GUARD, **{"mixtral.attention.head_count_kv": 4}))
    assert r.returncode == 2 and "guard mismatch" in r.stderr, r.stdout + r.stderr
    assert not gguf.exists()


# ---------- run on strata-shift-poly (AC4) ----------

def _ext_exe() -> pathlib.Path:
    from stratashift.polystrata.engine_paths import EXT_TARGET, engine_binary, ext_paths, resolve_engine_paths
    return engine_binary(ext_paths(resolve_engine_paths()), EXT_TARGET)


def _run_env() -> dict:
    """CUDA runtime DLLs on PATH (Windows keeps them in bin/x64 next to nvcc's bin)."""
    from stratashift.install.probe import find_nvcc
    env = dict(os.environ)
    found = find_nvcc()
    if found:
        bins = [found[0].parent / "x64", found[0].parent]
        env["PATH"] = os.pathsep.join([str(b) for b in bins if b.is_dir()] + [env.get("PATH", "")])
    return env


def _run(exe: pathlib.Path, gguf: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), "--model", str(gguf), "--tokens", "1,2,3", "--max-new", "4",
                           "--max-context", "256"], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=_run_env(), timeout=600)


@pytest.fixture(scope="module")
def ext_exe():
    exe = _ext_exe()
    if not exe.is_file():
        pytest.skip(f"strata-shift-poly not built ({exe}): stratashift build-engine --gpu --ext")
    return exe


@ENGINE_GATE
def test_mini_mixtral_runs(tmp_path, ext_exe):
    r, gguf = _emit(tmp_path, GUARD)
    assert r.returncode == 0, r.stdout + r.stderr
    run = _run(ext_exe, gguf)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout + run.stderr)


@ENGINE_GATE
def test_ext_exe_still_runs_gemma4_mini(tmp_path, ext_exe):
    """The dispatch lines engine_ext adds leave the engine's own families running."""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_gemma4_mini_emit import _emit_mini
    gguf = _emit_mini(tmp_path)
    run = _run(ext_exe, gguf)
    assert run.returncode == 0, run.stdout + run.stderr
