"""HF → engine GGUF emit order tables (moved out of cli/main.py, p6-08).

Data only: which HF tensor suffix goes to which engine key, and in which order the HF
emit writes them. Rules per suffix live in `hf_gguf_map` (LAYER_DENSE / GEMMA4_LAYER_DENSE).
"""
from __future__ import annotations

# HF emit order: per-layer engine want2 order (layout.cpp:60-110, pin c566ff4), then the
# fused experts, then want1 order, then globals. Deterministic; the engine pack builder
# follows GGUF tensor order, so this fixes the golden/ours plane order.
# (HF suffix after `model.layers.<L>.` → engine suffix; LAYER_DENSE is the mapping table.)
HF_EMIT_2D = [
    ("hc_attn_down.weight", "hc_attn_down.weight"),
    ("hc_attn_up.weight", "hc_attn_up.weight"),
    ("hc_attn_inject.weight", "hc_attn_inject.weight"),
    ("hc_ffn_down.weight", "hc_ffn_down.weight"),
    ("hc_ffn_up.weight", "hc_ffn_up.weight"),
    ("hc_ffn_inject.weight", "hc_ffn_inject.weight"),
    ("mlp.gate.weight", "ffn_gate_inp.weight"),
    ("mlp.shared_expert.gate_proj.weight", "ffn_gate_shexp.weight"),
    ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
    ("mlp.shared_expert.down_proj.weight", "ffn_down_shexp.weight"),
    ("linear_attn.g_proj.weight", "attn_gate.weight"),
    ("linear_attn.o_proj.weight", "ssm_out.weight"),
    ("linear_attn.conv1d.weight", "ssm_conv1d.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("indexer.q_proj.weight", "indexer.q_proj.weight"),
    ("indexer.k_proj.weight", "indexer.k_proj.weight"),
]
# attn_q is FUSED: q_proj rows then q_gate_proj rows (layer.hpp:363 `q_full` =
# n_head·2·head_dim, "the raw attn_q output, q | gate per head", pin c566ff4).
QSA_Q_ORDER = ["self_attn.q_proj.weight", "self_attn.q_gate_proj.weight"]
HF_EMIT_1D = [
    ("hc_attn_norm.weight", "hc_attn_norm.weight"),
    ("hc_ffn_norm.weight", "hc_ffn_norm.weight"),
    ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
    ("linear_attn.A_log", "ssm_a"),
    ("linear_attn.dt_bias", "ssm_dt.bias"),
    ("linear_attn.norm.weight", "ssm_norm.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("indexer.q_norm.weight", "indexer.q_norm.weight"),
    ("indexer.k_norm.weight", "indexer.k_norm.weight"),
]
# GDN fused qkv (want2 attn_qkv = concat(q, k, v) rows, geometry.hpp:35 derived width)
GDN_QKV_ORDER = ["linear_attn.q_proj.weight", "linear_attn.k_proj.weight",
                  "linear_attn.v_proj.weight"]

# gemma4 emit order (p5-06 AC3): engine want2 matrix keys of gemma4.cpp:197-211,225
# (load_moe: ffn_gate_inp [n_embd, n_expert]), then the fused experts, then the engine
# 1-D f32 keys (attn_norm / post_attention_norm / ffn_norm / post_ffw_norm{,_1,_2} /
# pre_ffw_norm_2 / router scales / layer_scalar, gemma4.cpp:197-203,219-233). Suffixes
# are the fixture-measured HF names (tests/fixtures/gemma4_lang_header.json); rules
# come from GEMMA4_LAYER_DENSE. Full layers have no v_proj (attention_k_eq_v measured)
# — the builder's optional-tensor skip rule covers it, no invented tensor.
GEMMA4_2D = [
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    ("router.proj.weight", "ffn_gate_inp.weight"),
]
GEMMA4_1D = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "post_attention_norm.weight"),
    ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
    ("post_feedforward_layernorm_1.weight", "post_ffw_norm_1.weight"),
    ("pre_feedforward_layernorm_2.weight", "pre_ffw_norm_2.weight"),
    ("post_feedforward_layernorm_2.weight", "post_ffw_norm_2.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("router.scale", "ffn_gate_inp.scale"),
    ("router.per_expert_scale", "ffn_down_exps.scale"),
    ("layer_scalar", "layer_output_scale.weight"),
]
# GDN 1-D keys: HF A_log/dt_bias are 1-D [v_heads]; engine wants the 2-D (n_embd, v_heads)
# ssm_alpha/ssm_beta (want2) IN ADDITION to the 1-D ssm_a/ssm_dt.bias (want1, emitted
# verbatim above). The fixture carries the 1-D HF tensors; the emit derives the 2-D
# engine forms by broadcasting each value across n_embd rows (fixture-measured, p4-01).
GDN_1D = [("linear_attn.A_log", "ssm_alpha.weight", "broadcast"),
           ("linear_attn.dt_bias", "ssm_beta.weight", "broadcast")]
