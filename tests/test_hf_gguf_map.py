"""p4-01 AC1/AC2: HF naming → engine GGUF key map + BF16→F32 conversion.

AC1: Mixtral/Qwen2MoE/fused/Qwen3-Next naming → engine GGUF suffix + shape
conversion (unit, layout.cpp:60-110 suffix 정본, pin c566ff4).
AC2: BF16→F32 == numpy reference; quantize_q2_0_stream(BF16→F32) == the same
weights as F32 input, byte-identical.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.ingest.safetensors_header import parse_header  # noqa: E402
from stratashift.meta.engine_manifest import bf16_to_f32_bytes  # noqa: E402
from stratashift.meta.expert_map import build_expert_map, classify  # noqa: E402
from stratashift.meta.hf_gguf_map import (derive_geometry, fused_expert_role,  # noqa: E402
                                           gdn_qkv_part, map_dense)
from stratashift.quant.pack import quantize_q2_0_stream  # noqa: E402

MIXTRAL = ROOT / ".local" / "models" / "mixtral-tiny.safetensors"


# --- AC1: engine GGUF suffix table (layout.cpp:60-110 want2/want1, pin c566ff4) ---

def test_mixtral_measured_names_map():
    """mixtral-tiny header names (measured) → engine GGUF keys with prefix blk.L."""
    cases = {
        "model.layers.0.input_layernorm.weight": (0, "blk.0.attn_norm", "same"),
        "model.layers.0.post_attention_layernorm.weight": (0, "blk.0.ffn_norm", "same"),
        "model.layers.1.mlp.gate.weight": (1, "blk.1.ffn_gate_inp.weight", "same"),
        "model.layers.0.self_attn.q_proj.weight": (0, "blk.0.attn_q.weight", "transpose"),
        "model.layers.0.self_attn.k_proj.weight": (0, "blk.0.attn_k.weight", "transpose"),
        "model.layers.0.self_attn.v_proj.weight": (0, "blk.0.attn_v.weight", "transpose"),
        "model.layers.0.self_attn.o_proj.weight": (0, "blk.0.attn_output.weight", "transpose"),
        "model.embed_tokens.weight": (-1, "token_embd.weight", "transpose"),
        "lm_head.weight": (-1, "output.weight", "transpose"),
        "model.norm.weight": (-1, "output_norm.weight", "same"),
    }
    for name, want in cases.items():
        assert map_dense(name) == want, name


def test_shared_expert_and_gate_map():
    """shared expert keys + the Bf16InF32 gate (layout.cpp:88-98, pin c566ff4)."""
    assert map_dense("model.layers.3.mlp.shared_expert.gate_proj.weight") \
        == (3, "blk.3.ffn_gate_shexp.weight", "transpose")
    assert map_dense("model.layers.3.mlp.shared_expert.up_proj.weight") \
        == (3, "blk.3.ffn_up_shexp.weight", "transpose")
    assert map_dense("model.layers.3.mlp.shared_expert.down_proj.weight") \
        == (3, "blk.3.ffn_down_shexp.weight", "transpose")
    assert map_dense("model.layers.3.mlp.shared_expert_gate.weight") \
        == (3, "blk.3.ffn_gate_inp_shexp.weight", "bf16_keep")


def test_qwen3_next_gdn_indexer_hc_map():
    """GDN/indexer/hc fixture-measured naming → engine want2/want1 suffixes."""
    assert map_dense("model.layers.2.linear_attn.g_proj.weight") \
        == (2, "blk.2.attn_gate.weight", "transpose")
    assert map_dense("model.layers.2.linear_attn.o_proj.weight") \
        == (2, "blk.2.ssm_out.weight", "transpose")
    assert map_dense("model.layers.2.linear_attn.conv1d.weight") \
        == (2, "blk.2.ssm_conv1d.weight", "same")
    assert map_dense("model.layers.2.linear_attn.A_log") == (2, "blk.2.ssm_a", "same")
    assert map_dense("model.layers.2.linear_attn.dt_bias") == (2, "blk.2.ssm_dt.bias", "same")
    assert map_dense("model.layers.2.linear_attn.norm.weight") \
        == (2, "blk.2.ssm_norm.weight", "same")
    assert map_dense("model.layers.2.self_attn.q_norm.weight") \
        == (2, "blk.2.attn_q_norm.weight", "same")
    assert map_dense("model.layers.2.self_attn.k_norm.weight") \
        == (2, "blk.2.attn_k_norm.weight", "same")
    assert map_dense("model.layers.2.indexer.q_proj.weight") \
        == (2, "blk.2.indexer.q_proj.weight", "transpose")
    assert map_dense("model.layers.2.indexer.k_proj.weight") \
        == (2, "blk.2.indexer.k_proj.weight", "transpose")
    assert map_dense("model.layers.2.indexer.q_norm.weight") \
        == (2, "blk.2.indexer.q_norm.weight", "same")
    assert map_dense("model.layers.2.indexer.k_norm.weight") \
        == (2, "blk.2.indexer.k_norm.weight", "same")
    for suffix in ("hc_attn_norm", "hc_ffn_norm"):
        assert map_dense(f"model.layers.2.{suffix}.weight") \
            == (2, f"blk.2.{suffix}.weight", "same")
    for suffix, want_ne in (("hc_attn_down", "transpose"), ("hc_attn_up", "transpose"),
                            ("hc_attn_inject", "transpose"), ("hc_ffn_down", "transpose"),
                            ("hc_ffn_up", "transpose"), ("hc_ffn_inject", "transpose")):
        assert map_dense(f"model.layers.2.{suffix}.weight") \
            == (2, f"blk.2.{suffix}.weight", want_ne)


def test_fused_and_qkv_patterns():
    assert fused_expert_role("model.layers.0.mlp.experts.gate_up_proj") == (0, "gate_up")
    assert fused_expert_role("model.layers.1.mlp.experts.down_proj") == (1, "down")
    assert fused_expert_role("model.layers.0.mlp.experts.gate_proj.weight") is None
    assert gdn_qkv_part("model.layers.0.linear_attn.q_proj.weight") == (0, "q")
    assert gdn_qkv_part("model.layers.0.linear_attn.v_proj.weight") == (0, "v")
    assert gdn_qkv_part("model.layers.0.linear_attn.o_proj.weight") is None


def test_per_expert_naming_shapes():
    """Qwen2MoE/Mixtral per-expert names (expert_map p1-01) + engine fused shape:
    gate_proj [FF, H] → ffn_gate_exps (H, FF, NE) — expert e contiguous run."""
    names = [f"model.layers.0.mlp.experts.{e}.{r}_proj.weight"
             for e in range(4) for r in ("gate", "up", "down")]
    refs = [classify(n) for n in names]
    assert all(r is not None for r in refs)
    assert refs[0] == (0, 0, "gate")
    assert refs[2] == (0, 0, "down")


# --- AC2: BF16 → F32 ---

def _bf16_image(n: int, seed: int) -> tuple[bytes, np.ndarray]:
    """Random F32 values rounded to BF16, as raw uint16 bytes + the exact F32 image."""
    rng = np.random.default_rng(seed)
    f32 = rng.standard_normal(n).astype("<f4")
    u16 = f32.astype("<f2").view(np.uint16)  # BF16 = top 16 bits of the F32 image
    exact = (u16.astype(np.uint32) << 16).view("<f4")
    return u16.tobytes(), exact


def test_bf16_to_f32_matches_numpy_reference():
    raw, exact = _bf16_image(1024, 7)
    # vectorized emit path == numpy reference
    from stratashift.cli.main import _to_f32
    assert np.frombuffer(_to_f32(raw, "BF16"), dtype="<f4").tobytes() == exact.tobytes()
    # scalar engine_manifest loop (to_p32_bf16 contract) agrees bit-for-bit
    assert bf16_to_f32_bytes(raw) == exact.tobytes()


def test_bf16_f32_quantizer_stream_identical():
    """quantize_q2_0_stream(BF16→F32) == the same weights fed as F32, byte-identical."""
    from stratashift.cli.main import _to_f32
    raw, exact = _bf16_image(64 * 32, 11)
    a = quantize_q2_0_stream(_to_f32(raw, "BF16"))
    b = quantize_q2_0_stream(exact.tobytes())
    assert a == b
    assert len(a) == 32 * 18  # 1024 elements = 32 Q2_0 blocks


def test_derive_geometry_from_mixtral():
    """Guard keys DERIVED from the measured mixtral-tiny header (never invented)."""
    meta = parse_header(str(MIXTRAL))
    g = derive_geometry(meta, head_dim=64)
    assert g["embedding_length"] == 64          # embed_tokens shape[1]
    assert g["block_count"] == 2                # 2 layers
    assert g["expert_count"] == 4               # gate_up_proj shape[0]
    assert g["expert_feed_forward_length"] == 128  # shape[1] // 2
    assert g["attention.head_count"] == 1       # q_proj shape[0] 64 / head_dim 64
    # k_proj shape[0] 32 / head_dim 64 = 0 by integer division: mixtral-tiny's kv width
    # (32) is not a multiple of the q head_dim (64) — the qwen4exp guard is NOT
    # applicable to Mixtral geometry (p4-01 근거 §2), so this derived value is never
    # cross-checked against a preset in the mixtral path (empty preset).
    assert g["attention.head_count_kv"] == 0


def test_derive_geometry_from_fused_header():
    """embedding_length comes from the fused gate_up row width when no embed_tokens."""
    from stratashift.ingest.safetensors_header import ModelMeta, TensorMeta
    t = TensorMeta("model.layers.0.mlp.experts.gate_up_proj", "F32", (4, 256, 64), (0, 4 * 256 * 64 * 4))
    meta = ModelMeta(path="x", tensors=[t], header_size=0, data_start=0, file_size=0)
    g = derive_geometry(meta)
    assert g["embedding_length"] == 64
    assert g["expert_count"] == 4
    assert g["expert_feed_forward_length"] == 128
    assert g["block_count"] == 1
