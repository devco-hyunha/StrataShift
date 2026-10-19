"""p5-08 AC1 gate gate: dense quantization selection must satisfy the engine LOAD gate.

Root cause (p5-07 AC5, full sweep probe_ne0.py, 2026-10-19): engine
`tensor_payload_bytes` (gguf_reader.hpp:505, pin c566ff4) requires
`shape[0] % be == 0` (be from block_geometry: Q4_K be=256, Q8_0/Q4_0 be=32,
F32 be=1) — `shape[0]` is the ENGINE shape first dim. Our emit `ffn_down
[2112,2816]` Q4_K: 2112 % 256 = 64 → payload 0 → load throw rc=1. All 30
ffn_down tensors fail; all pass at be=32. The p3-03~p3-09 pack-contiguity
gate (`_dense_gate`, total elements) is kept verbatim; the load gate is
ADDED, and the ladder is: requested type → Q8_0 → Q4_0 → F32.

AC1 table (measured real-measured shapes):
- engine [2112,2816] Q4_K → ladder Q8_0 (2112%256=64 fails, 2112%32=0)
- engine [2816,2112] Q4_K → stays Q4_K (2816%256=0, n_el%512=0)
- engine [2816,128]  Q4_K → stays Q4_K
- engine [704,2816,128] Q4_K → Q8_0 (704%256=192 fails, 704%32=0)
Plus the integration gate: a mini gemma4 fixture with the REAL Gemma4 geometry
(H=2816, FF=2112) emits ffn_up Q4_K, ffn_down Q8_0, attn Q4_K, experts Q2_0,
all 1-D keys F32 — and the GGUF passes our own verify.
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

import stratashift.cli.main as cli_mod  # noqa: E402
from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402

GGUF_GATE = pytest.mark.skipif(
    cli_mod.GGUFWriter is None,
    reason="pinned engine tools checkout required (STRATASHIFT_ENGINE_TOOLS)",
)


def sel(shape: tuple[int, ...], dense_type: str = "Q4_K",
        force_f32: bool = False) -> str:
    """Gate as the emit path applies it: engine shape (rule=transpose), n_el."""
    n_el = 1
    for d in shape:
        n_el *= d
    return cli_mod._dense_select(n_el, dense_type, shape[0], len(shape), force_f32)


def test_gate_table_measured_shapes() -> None:
    """AC1 table — the measured-shape fixtures of docs/work/p5-08.md §테스트 계획."""
    assert sel((2112, 2816)) == "Q8_0"      # the ffn_down refusal: be=32 ladder
    assert sel((2816, 2112)) == "Q4_K"      # ffn_up family: Q4_K kept
    assert sel((2816, 128)) == "Q4_K"       # kept per AC1
    assert sel((704, 2816, 128)) == "Q8_0"  # 3-D gate input: ladder Q8_0
    # F32 pass-through is always accepted (be=1)
    assert sel((2112, 2816), "F32") == "F32"
    # 1-D engine key want1 contract: never quantized (p4-01 dtype rule)
    assert sel((2816,), "Q4_K", force_f32=True) == "F32"
    # Q8_0 ndim gate: 1-D Q8_0 needs ndim ≥ 2 → ladder lands on Q4_0
    assert sel((2816,), "Q8_0") == "Q4_0"
    # requests that already satisfy both gates are kept verbatim
    assert sel((2816, 2112), "Q8_0") == "Q8_0"
    assert sel((2816, 2112), "Q4_0") == "Q4_0"
    # tiny tensor fails every quantization → F32
    assert sel((16, 8)) == "F32"


# --- integration: mini fixture with the REAL gemma4 geometry (H=2816, FF=2112) ---
H, FF, NE, V, HEAD_DIM = 2816, 2112, 4, 2048, 512
Q_ROWS, K_ROWS = 2048, 1024  # heads 4·512, kv 2·512 (measured family, mini)
LAYERS = (0,)

GUARD = {
    "architecture": "gemma4",
    "name": "p5-08 mini fixture (real geometry)",
    "gemma4.block_count": 1,
    "gemma4.embedding_length": H,
    "gemma4.expert_count": NE,
    "gemma4.expert_used_count": 2,
    "gemma4.expert_feed_forward_length": FF,
    "gemma4.attention.head_count": Q_ROWS // HEAD_DIM,
    "gemma4.attention.head_count_kv": [K_ROWS // HEAD_DIM],
    "gemma4.attention.key_length": HEAD_DIM,
    "gemma4.attention.key_length_swa": HEAD_DIM,
    "gemma4.attention.value_length": HEAD_DIM,
    "gemma4.attention.value_length_swa": HEAD_DIM,
    "gemma4.attention.sliding_window": 1024,
    "gemma4.attention.sliding_window_pattern": [1],
    "gemma4.rope.freq_base": 1000000.0,
    "gemma4.rope.freq_base_swa": 10000.0,
    "gemma4.final_logit_softcapping": 30.0,
    "gemma4.attention.shared_kv_layers": 0,
    "gemma4.attention.layer_norm_rms_epsilon": 1e-06,
    "gemma4.vocab_size": V,
}


def build_mini(dirpath: pathlib.Path) -> None:
    """BF16 safetensors shard + index.json with the p5-07 root-cause shapes:
    HF down_proj [2816,2112] → engine [2112,2816] (the measured ffn_down refusal)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    p = "model.language_model.layers.0."
    tensors: dict[str, tuple] = {
        "model.language_model.embed_tokens.weight": (V, H),
        "model.language_model.norm.weight": (H,),
        p + "self_attn.q_proj.weight": (Q_ROWS, H),
        p + "self_attn.k_proj.weight": (K_ROWS, H),
        p + "self_attn.v_proj.weight": (K_ROWS, H),
        p + "self_attn.o_proj.weight": (H, Q_ROWS),
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
        p + "mlp.gate_proj.weight": (FF, H),
        p + "mlp.up_proj.weight": (FF, H),
        p + "mlp.down_proj.weight": (H, FF),
        p + "experts.gate_up_proj": (NE, 2 * FF, H),
        p + "experts.down_proj": (NE, H, FF),
    }
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
        for pl in payloads:
            fh.write(pl)
    (dirpath / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(len(pl) for pl in payloads)},
         "weight_map": {n: shard.name for n in tensors}}, indent=1))


@GGUF_GATE
def test_ffn_down_falls_to_be32_on_real_geometry(tmp_path: pathlib.Path) -> None:
    """AC1 integration: --dense-type Q4_K on the real gemma4 geometry.
    engine ffn_down [2112,2816] must be Q8_0 (not Q4_K), ffn_up/ffn_gate
    [2816,2112] stay Q4_K, experts stay Q2_0, 1-D keys stay F32."""
    model = tmp_path / "mini"
    build_mini(model)
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps(GUARD))
    gguf = tmp_path / "mini.gguf"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(model), "--out", str(tmp_path / "pack"),
                        "--emit-gguf", str(gguf), "--hf-emit",
                        "--guard-keys", str(guard), "--dense-type", "Q4_K"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    meta = parse_gguf_header(str(gguf))
    types = {t.name: t.type_name for t in meta.tensors}
    shapes = {t.name: t.shape for t in meta.tensors}
    # the measured refusal shape is present as ENGINE shape [2112,2816]
    assert shapes["blk.0.ffn_down.weight"] == (2112, 2816)
    assert types["blk.0.ffn_down.weight"] == "Q8_0"
    # engine-passing 2-D keys keep the requested Q4_K verbatim
    assert types["blk.0.ffn_gate.weight"] == "Q4_K"
    assert types["blk.0.ffn_up.weight"] == "Q4_K"
    assert types["blk.0.attn_q.weight"] == "Q4_K"
    assert types["blk.0.ffn_gate_inp.weight"] == "Q4_K"
    # experts keep Q2_0, 1-D want1 keys stay F32
    assert types["blk.0.ffn_gate_exps.weight"] == "Q2_0"
    assert types["blk.0.attn_norm.weight"] == "F32"
    assert types["blk.0.ffn_gate_inp.scale"] == "F32"
    assert types["output_norm.weight"] == "F32"
    # every quantized tensor satisfies the engine load gate (shape[0] % be == 0)
    from stratashift.polystrata.headers import ENGINE_LOAD_BE
    for t in meta.tensors:
        be = ENGINE_LOAD_BE.get(t.type_name)
        assert be is None or t.shape[0] % be == 0, (t.name, t.shape, t.type_name)
    # the GGUF passes our own verify (pack builder contract intact)
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(gguf), "--out", str(tmp_path / "ours")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "verify", str(tmp_path / "ours")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads(r.stdout)["ok"]
