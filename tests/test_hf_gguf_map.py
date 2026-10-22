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
from stratashift.meta.hf_gguf_map import (derive_geometry, fused_expert_role,
                                           gdn_qkv_part, hf_layer_id, map_dense)
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
    from stratashift.quant.dtype import to_f32
    assert np.frombuffer(to_f32(raw, "BF16"), dtype="<f4").tobytes() == exact.tobytes()
    # scalar engine_manifest loop (to_p32_bf16 contract) agrees bit-for-bit
    assert bf16_to_f32_bytes(raw) == exact.tobytes()


def test_bf16_f32_quantizer_stream_identical():
    """quantize_q2_0_stream(BF16→F32) == the same weights fed as F32, byte-identical."""
    from stratashift.quant.dtype import to_f32
    raw, exact = _bf16_image(64 * 32, 11)
    a = quantize_q2_0_stream(to_f32(raw, "BF16"))
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

# --- p5-06 AC3: gemma4 naming (measured fixture, never-invented) ---

import json  # noqa: E402

import stratashift.meta.hf_gguf_map as m  # noqa: E402

from stratashift.ingest.safetensors_header import ModelMeta, TensorMeta  # noqa: E402

GEMMA4_FIXTURE = ROOT / "tests" / "fixtures" / "gemma4_lang_header.json"
GEMMA4_GUARD = ROOT / "scripts" / "p5_gemma4_guard.json"
_FULL_LAYERS = (5, 11, 17, 23, 29)  # config layer_types: 5 sliding + 1 full (AC2 measured)


def _fixture_meta() -> ModelMeta:
    """The measured language-model header inventory as a ModelMeta (shapes + dtypes,
    no payload — header JSON is the fixture, C1 convention)."""
    inv = json.loads(GEMMA4_FIXTURE.read_text(encoding="utf-8"))
    tensors = [TensorMeta(name=k, dtype=v["dtype"], shape=tuple(v["shape"]),
                          data_offsets=(0, 0), shard=v["shard"])
               for k, v in inv.items()]
    return ModelMeta(path="fixture", tensors=tensors, header_size=0,
                     data_start=0, file_size=0)


def test_gemma4_fixture_shape_facts():
    """The fixture IS the AC2 header measurement: 30 layers, fused experts in every
    layer, full layers (5/11/17/23/29) have NO v_proj (attention_k_eq_v measured)."""
    meta = _fixture_meta()
    names = {t.name for t in meta.tensors}
    layers = sorted({hf_layer_id(n) for n in names if hf_layer_id(n) is not None})
    assert layers == list(range(30))
    for L in range(30):
        p = f"model.language_model.layers.{L}."
        assert p + "experts.gate_up_proj" in names, L
        assert p + "experts.down_proj" in names, L
        assert (p + "self_attn.v_proj.weight" in names) == (L not in _FULL_LAYERS), L
        assert len(meta.tensors) == 657


def test_gemma4_every_measured_key_maps():
    """Coverage gate: EVERY fixture language key maps to an engine Gemma4 key — the
    fixture keys are measured, so a mapping gap is a Goal failure, not a skip."""
    meta = _fixture_meta()
    unmapped = [t.name for t in meta.tensors
                if not m.GEMMA4_GLOBAL_DENSE.get(t.name)
                and map_dense(t.name, "gemma4") is None
                and fused_expert_role(t.name) is None]
    assert unmapped == [], unmapped[:6]
    # every mapped engine key is in the engine Gemma4 vocabulary (gemma4.cpp, pin c566ff4)
    engine_vocab = {"attn_norm.weight", "post_attention_norm.weight", "ffn_norm.weight",
                    "post_ffw_norm.weight", "post_ffw_norm_1.weight", "pre_ffw_norm_2.weight",
                    "post_ffw_norm_2.weight", "attn_q_norm.weight", "attn_k_norm.weight",
                    "ffn_gate_inp.scale", "ffn_down_exps.scale", "layer_output_scale.weight",
                    "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                    "ffn_gate_inp.weight", "token_embd.weight", "output_norm.weight",
                    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
    for t in meta.tensors:
        r = map_dense(t.name, "gemma4")
        if r is None:
            continue
        suffix = r[1] if r[0] == -1 else r[1].split("blk.", 1)[1].split(".", 1)[1]
        assert suffix in engine_vocab, (t.name, suffix)


def test_gemma4_layer_dense_map():
    """Fixture HF suffixes → engine Gemma4 keys (gemma4.cpp:197-211,219-234, pin c566ff4)."""
    cases = {
        "model.language_model.layers.0.input_layernorm.weight":
            (0, "blk.0.attn_norm.weight", "same"),
        "model.language_model.layers.0.post_attention_layernorm.weight":
            (0, "blk.0.post_attention_norm.weight", "same"),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight":
            (0, "blk.0.ffn_norm.weight", "same"),
        "model.language_model.layers.0.post_feedforward_layernorm.weight":
            (0, "blk.0.post_ffw_norm.weight", "same"),
        "model.language_model.layers.0.post_feedforward_layernorm_1.weight":
            (0, "blk.0.post_ffw_norm_1.weight", "same"),
        "model.language_model.layers.0.pre_feedforward_layernorm_2.weight":
            (0, "blk.0.pre_ffw_norm_2.weight", "same"),
        "model.language_model.layers.0.post_feedforward_layernorm_2.weight":
            (0, "blk.0.post_ffw_norm_2.weight", "same"),
        "model.language_model.layers.0.router.scale":
            (0, "blk.0.ffn_gate_inp.scale", "same"),
        "model.language_model.layers.0.router.per_expert_scale":
            (0, "blk.0.ffn_down_exps.scale", "same"),
        "model.language_model.layers.0.layer_scalar":
            (0, "blk.0.layer_output_scale.weight", "same"),
        "model.language_model.layers.5.self_attn.q_proj.weight":
            (5, "blk.5.attn_q.weight", "transpose"),
        "model.language_model.layers.0.mlp.gate_proj.weight":
            (0, "blk.0.ffn_gate.weight", "transpose"),
        "model.language_model.layers.0.router.proj.weight":
            (0, "blk.0.ffn_gate_inp.weight", "transpose"),
        "model.language_model.embed_tokens.weight":
            (-1, "token_embd.weight", "transpose"),
        "model.language_model.norm.weight":
            (-1, "output_norm.weight", "same"),
    }
    for name, want in cases.items():
        assert map_dense(name, "gemma4") == want, name


def test_gemma4_unmeasured_names_not_mapped():
    """Never-invented: vision-tower keys (measured in the fixture, NOT engine Gemma4
    keys) and unmeasured HF names stay unmapped; the emit builder skips them."""
    assert map_dense("model.vision_tower.encoder.layers.0.mlp.gate_proj.weight",
                     "gemma4") is None
    assert map_dense("model.language_model.layers.0.self_attn.q_gate_proj.weight",
                     "gemma4") is None
    assert map_dense("model.language_model.layers.0.router.bias", "gemma4") is None
    assert map_dense("lm_head.weight", "gemma4") is None  # tied embeddings, key ABSENT
    assert "lm_head.weight" not in m.GEMMA4_GLOBAL_DENSE


def test_gemma4_fused_expert_role():
    """fused gate_up/down under the language prefix, no `mlp.` segment (fixture-measured)."""
    assert fused_expert_role("model.language_model.layers.0.experts.gate_up_proj") \
        == (0, "gate_up")
    assert fused_expert_role("model.language_model.layers.29.experts.down_proj") \
        == (29, "down")
    # the old `mlp.experts` pattern stays intact for the p4-01 arches
    assert fused_expert_role("model.layers.1.mlp.experts.down_proj") == (1, "down")


def test_gemma4_derive_geometry():
    """Guard keys DERIVED from the fixture header: per-layer head_dim = q_norm length,
    head_count_kv = the per-layer array the engine reads (of_layer, gemma4.cpp:64-68)."""
    g = derive_geometry(_fixture_meta(), arch="gemma4")
    assert g["embedding_length"] == 2816
    assert g["block_count"] == 30
    assert g["expert_count"] == 128
    assert g["expert_feed_forward_length"] == 704
    assert g["attention.head_count"] == 16
    kv = g["attention.head_count_kv"]
    assert kv == [2 if L in _FULL_LAYERS else 8 for L in range(30)]


def test_gemma4_derive_head_count_varying_refused():
    """gemma4.cpp:67 refuses layers with different head counts — the builder must
    refuse BEFORE emit, never write a head_count that the engine will reject."""
    inv = json.loads(GEMMA4_FIXTURE.read_text(encoding="utf-8"))
    tensors = [TensorMeta(name=k, dtype=v["dtype"], shape=tuple(v["shape"]),
                          data_offsets=(0, 0), shard=v["shard"])
               for k, v in inv.items()]
    for t in tensors:
        if t.name == "model.language_model.layers.1.self_attn.q_proj.weight":
            t = TensorMeta(name=t.name, dtype=t.dtype, shape=(4096 + 256, 2816),
                           data_offsets=(0, 0), shard=t.shard)
            tensors[tensors.index(next(x for x in tensors
                                       if x.name == t.name))] = t
    meta = ModelMeta(path="fixture", tensors=tensors, header_size=0,
                     data_start=0, file_size=0)
    try:
        derive_geometry(meta, arch="gemma4")
    except ValueError as e:
        assert "head_count" in str(e)
    else:
        raise AssertionError("varying head_count must be refused")


def test_gemma4_pack_gate_elements_divisible():
    """Every 2-D fixture key must satisfy the engine Q4_K pack gate (elements % 512,
    p3-04 VERIFIED gate) so --dense-type Q4_K never silently falls back to F32."""
    meta = _fixture_meta()
    for t in meta.tensors:
        r = map_dense(t.name, "gemma4")
        if r is None or r[1] == "token_embd.weight" or len(t.shape) < 2:
            continue
        n = t.shape[0] * t.shape[1]
        assert n % 512 == 0, (t.name, t.shape, n)


def test_gemma4_guard_preset_cross_check():
    """The preset's derived keys equal the header-derived values, and every derived
    key exists in the preset (the CLI refuses a preset missing a derived key)."""
    preset = json.loads(GEMMA4_GUARD.read_text(encoding="utf-8"))
    g = derive_geometry(_fixture_meta(), arch="gemma4")
    for key, val in g.items():
        pk = f"gemma4.{key}"
        assert pk in preset, pk
        assert preset[pk] == val, (pk, val, preset[pk])
    # engine array key measured: full layers are 2, sliding layers 8
    assert preset["gemma4.attention.head_count_kv"] == g["attention.head_count_kv"]
    assert preset["gemma4.attention.sliding_window_pattern"][5] == 0
    assert preset["gemma4.attention.sliding_window_pattern"][0] == 1
