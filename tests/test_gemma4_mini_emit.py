"""p5-06 regression: the engine want1 1-D keys stay F32 under --dense-type Q4_K.

Root cause closed in p5-06 (2026-10-18): tiny 1-D keys (q_norm [512], norms [2816],
router.scale, layer_scalar [1]) pass the Q4_K pack gate (elements % 512) and were
quantized, producing manifest entries with empty scale planes — our verify refused
"planes not contiguous", and the engine contract (gemma4.cpp:197-211 f32() gate,
p4-01 §dtype rule layout.cpp:88-98) requires every 1-D engine key as an F32 tensor.
This test builds a synthetic mini gemma4 fixture (every key measured in the AC2
fixture, mini shapes divisible by the 512-element pack gate) and asserts:
- every 1-D engine key in the emitted GGUF is F32,
- the 2-D matrix keys are Q4_K,
- the experts are Q2_0.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import struct
import sys

import numpy as np
import os

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402

H = 1024
NE = 4
FF = 1024
V = 2048
HEAD_DIM = 512
Q_ROWS, K_ROWS, O_COLS = 2048, 1024, 2048   # heads 4·512, kv 2·512 (measured-shape family)
LAYERS = (0, 1)

GUARD = {
    "architecture": "gemma4",
    "name": "p5-06 mini fixture",
    "gemma4.block_count": 2,
    "gemma4.embedding_length": H,
    "gemma4.expert_count": NE,
    # poly 엔진 required key (engine.cpp:696 shape_i("expert_used_count")) — mini synthetic
    # gate: 1 ≤ used ≤ n_expert; mini scaled 4 experts → 2 routed
    "gemma4.expert_used_count": 2,
    "gemma4.expert_feed_forward_length": FF,
    "gemma4.attention.head_count": Q_ROWS // HEAD_DIM,
    "gemma4.attention.head_count_kv": [K_ROWS // HEAD_DIM] * 2,
    "gemma4.attention.key_length": HEAD_DIM,
    "gemma4.attention.key_length_swa": HEAD_DIM,
    "gemma4.attention.value_length": HEAD_DIM,
    "gemma4.attention.value_length_swa": HEAD_DIM,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.sliding_window_pattern": [1, 1],
    "gemma4.rope.freq_base": 1000000.0,
    "gemma4.rope.freq_base_swa": 10000.0,
    "gemma4.final_logit_softcapping": 30.0,
    "gemma4.attention.shared_kv_layers": 0,
    "gemma4.attention.layer_norm_rms_epsilon": 1e-06,
    "gemma4.vocab_size": V,
}

# HF shapes exactly as the AC2 fixture inventory, mini-scaled
def layer_tensors(L: int) -> dict[str, tuple]:
    p = f"model.language_model.layers.{L}."
    t = {
        p + "self_attn.q_proj.weight": (Q_ROWS, H),
        p + "self_attn.k_proj.weight": (K_ROWS, H),
        p + "self_attn.v_proj.weight": (K_ROWS, H),
        p + "self_attn.o_proj.weight": (H, O_COLS),
        p + "self_attn.q_norm.weight": (HEAD_DIM,),
        p + "self_attn.k_norm.weight": (HEAD_DIM,),
        p + "input_layernorm.weight": (H,),
        p + "post_attention_layernorm.weight": (H,),
        p + "pre_feedforward_layernorm.weight": (H,),
        p + "post_feedforward_layernorm.weight": (H,),
        p + "post_feedforward_layernorm_1.weight": (H,),
        p + "pre_feedforward_layernorm_2.weight": (H,),
        p + "post_feedforward_layernorm_2.weight": (H,),
        p + "router.proj.weight": (NE, H),
        p + "router.scale": (H,),
        p + "router.per_expert_scale": (NE,),
        p + "layer_scalar": (1,),
        p + "mlp.gate_proj.weight": (FF // 2, H),
        p + "mlp.up_proj.weight": (FF // 2, H),
        p + "mlp.down_proj.weight": (H, FF // 2),
        p + "experts.gate_up_proj": (NE, 2 * FF, H),
        p + "experts.down_proj": (NE, H, FF),
    }
    return t


def build_mini(dirpath: pathlib.Path) -> None:
    """Write one BF16 safetensors shard + index.json + config.json (zeros — the test
    asserts key/type structure, not weight values)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple] = {
        "model.language_model.embed_tokens.weight": (V, H),
        "model.language_model.norm.weight": (H,),
    }
    for L in LAYERS:
        tensors.update(layer_tensors(L))
    off = 0
    header = {}
    payloads = []
    for name, shape in sorted(tensors.items()):
        n = 1
        for d in shape:
            n *= d
        payload = np.zeros(n, dtype=np.uint16).tobytes()  # BF16 zeros
        header[name] = {"dtype": "BF16", "shape": list(shape),
                        "data_offsets": [off, off + len(payload)]}
        payloads.append(payload)
        off += len(payload)
    header["__metadata__"] = {"format": "PT"}
    text = json.dumps(header, separators=(",", ":"))
    pad = (-len(text)) % 8
    text += " " * pad
    blob = text.encode()
    shard = dirpath / "model-00001-of-00001.safetensors"
    with open(shard, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for p in payloads:
            fh.write(p)
    (dirpath / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(len(p) for p in payloads)},
         "weight_map": {n: shard.name for n in tensors}}, indent=1))
    (dirpath / "config.json").write_text(json.dumps({
        "hidden_size": H, "num_hidden_layers": 2, "num_attention_heads": Q_ROWS // HEAD_DIM,
        "num_key_value_heads": K_ROWS // HEAD_DIM, "head_dim": HEAD_DIM,
        "num_experts": NE, "moe_intermediate_size": FF, "tie_word_embeddings": True}))


def test_mini_gemma4_one_d_keys_stay_f32(tmp_path):
    model = tmp_path / "mini"
    build_mini(model)
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps(GUARD))
    gguf = tmp_path / "mini.gguf"
    out_dir = tmp_path / "pack"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(model), "--out", str(out_dir),
                        "--emit-gguf", str(gguf), "--hf-emit",
                        "--guard-keys", str(guard), "--dense-type", "Q4_K"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    meta = parse_gguf_header(str(gguf))
    types = {t.name: t.type_name for t in meta.tensors}
    one_d_keys = set()
    for L in LAYERS:
        for engine in ("attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                       "post_ffw_norm.weight", "post_ffw_norm_1.weight",
                       "pre_ffw_norm_2.weight", "post_ffw_norm_2.weight",
                       "attn_q_norm.weight", "attn_k_norm.weight",
                       "ffn_gate_inp.scale", "ffn_down_exps.scale",
                       "layer_output_scale.weight"):
            key = f"blk.{L}.{engine}"
            assert key in types, key
            one_d_keys.add(key)
    assert types["output_norm.weight"] == "F32"  # global 1-D key
    for key in one_d_keys:
        assert types[key] == "F32", (key, types[key])
    for L in LAYERS:
        assert types[f"blk.{L}.attn_q.weight"] == "Q4_K", L
        assert types[f"blk.{L}.ffn_gate_inp.weight"] == "Q4_K", L
        assert types[f"blk.{L}.ffn_gate.weight"] == "Q4_K", L
        assert types[f"blk.{L}.ffn_gate_exps.weight"] == "Q2_0", L
    # the pack builder accepts our own GGUF: verify must pass (the plane gate the
    # root cause tripped on)
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(gguf), "--out", str(tmp_path / "ours")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "verify", str(tmp_path / "ours")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["ok"]
