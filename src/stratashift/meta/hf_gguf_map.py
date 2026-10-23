"""HF naming → engine GGUF key map (p4-01, Phase 4 입력 일반화).

Grounding (docs/work/p4-01.md 근거 계약, pin c566ff4 — engine read-only):
- Engine GGUF key vocabulary: `src/core/layout.cpp:60-110` want2/want1 suffixes,
  prefix `blk.<layer>.` (`LayerView::name`, layout.cpp:137-139).
- Measured HF naming: `.local/models/mixtral-tiny.safetensors` header (21 tensors,
  BF16, NE=4, H=64, FF=128) + the p4 fixture naming (scripts/p4_hf_emit.py).
  Unmeasured HF naming is NOT mapped (p4-01 risk rule: no HYPOTHESIS expansion).

Shape convention: HF stores `[out, in]` row-major; engine GGUF stores
`(ne0=in, ne1=out)` row-major — the SAME bytes, transposed shape metadata
(p3-02 fused expert contract, docs/rea/struct_layouts.md §1 VERIFIED). 1-D
tensors keep their shape. `transpose` in the table marks the shape flip.

dtype rule (layout.cpp:88-98 Want1 comment, pin c566ff4): every 1-D engine key
is F32 EXCEPT `ffn_gate_inp_shexp.weight`, which is Bf16InF32 — the BF16 bytes
are kept verbatim (`bf16_keep`).
"""
from __future__ import annotations

import json
import pathlib
import re

from stratashift.ingest.safetensors_header import ModelMeta

# --- global (non-layer) dense keys, measured in mixtral-tiny -----------------
# HF [vocab, H] row-major == GGUF (H, vocab) row-major bytes (transpose flag).
GLOBAL_DENSE: dict[str, tuple[str, str]] = {
    "model.embed_tokens.weight": ("token_embd.weight", "transpose"),
    "lm_head.weight": ("output.weight", "transpose"),
    "model.norm.weight": ("output_norm.weight", "same"),
}

# --- per-layer dense keys: HF suffix (after `model.layers.<L>.`) → engine suffix
# Engine suffixes are the layout.cpp want2/want1 vocabulary (pin c566ff4).
LAYER_DENSE: dict[str, tuple[str, str]] = {
    # norms and MoE router (measured in mixtral-tiny)
    "input_layernorm.weight": ("attn_norm", "same"),
    "post_attention_layernorm.weight": ("ffn_norm", "same"),
    "mlp.gate.weight": ("ffn_gate_inp.weight", "same"),
    # shared expert (p4 fixture naming; engine want2 shexp keys)
    "mlp.shared_expert.gate_proj.weight": ("ffn_gate_shexp.weight", "transpose"),
    "mlp.shared_expert.up_proj.weight": ("ffn_up_shexp.weight", "transpose"),
    "mlp.shared_expert.down_proj.weight": ("ffn_down_shexp.weight", "transpose"),
    "mlp.shared_expert_gate.weight": ("ffn_gate_inp_shexp.weight", "bf16_keep"),
    # QSA full attention (measured in mixtral-tiny; engine want2 attn_q/k/v/output)
    "self_attn.q_proj.weight": ("attn_q.weight", "transpose"),
    "self_attn.q_gate_proj.weight": ("attn_q_gate", "part"),  # fused into attn_q rows
    "self_attn.k_proj.weight": ("attn_k.weight", "transpose"),
    "self_attn.v_proj.weight": ("attn_v.weight", "transpose"),
    "self_attn.o_proj.weight": ("attn_output.weight", "transpose"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "same"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "same"),
    # GDN (fixture-measured Qwen3-Next naming; engine want2 GDN keys)
    "linear_attn.g_proj.weight": ("attn_gate.weight", "transpose"),
    "linear_attn.o_proj.weight": ("ssm_out.weight", "transpose"),
    "linear_attn.conv1d.weight": ("ssm_conv1d.weight", "same"),
    "linear_attn.A_log": ("ssm_a", "same"),
    "linear_attn.dt_bias": ("ssm_dt.bias", "same"),
    "linear_attn.norm.weight": ("ssm_norm.weight", "same"),
    # indexer (fixture-measured; engine want2/want1 indexer keys)
    "indexer.q_proj.weight": ("indexer.q_proj.weight", "transpose"),
    "indexer.k_proj.weight": ("indexer.k_proj.weight", "transpose"),
    "indexer.q_norm.weight": ("indexer.q_norm.weight", "same"),
    "indexer.k_norm.weight": ("indexer.k_norm.weight", "same"),
    # hyper-connection (fixture-measured; engine want2/want1 hc keys)
    "hc_attn_norm.weight": ("hc_attn_norm.weight", "same"),
    "hc_ffn_norm.weight": ("hc_ffn_norm.weight", "same"),
    "hc_attn_down.weight": ("hc_attn_down.weight", "transpose"),
    "hc_attn_up.weight": ("hc_attn_up.weight", "transpose"),
    "hc_attn_inject.weight": ("hc_attn_inject.weight", "transpose"),
    "hc_ffn_down.weight": ("hc_ffn_down.weight", "transpose"),
    "hc_ffn_up.weight": ("hc_ffn_up.weight", "transpose"),
    "hc_ffn_inject.weight": ("hc_ffn_inject.weight", "transpose"),
}

# fused expert tensors (measured in mixtral-tiny): shape[0] = num experts,
# expert e is the contiguous row block [e·prod(shape[1:]), (e+1)·prod(shape[1:]))
# (expert_map.py fused rule, p1-01).
FUSED_EXPERT = re.compile(
    r"^model\.(?:language_model\.)?layers\.(\d+)\.(?:mlp\.)?experts\.(gate_up_proj|down_proj)$")

# GDN fused qkv: engine `attn_qkv.weight` is ONE tensor = concat(q, k, v) rows
# (layout.cpp:60 ssm_conv_channels = q+k+v width, derived geometry.hpp:35).
GDN_QKV = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.(q_proj|k_proj|v_proj)\.weight$")

_LAYER_RE = re.compile(r"^model\.(?:language_model\.)?layers\.(\d+)\.(.+)$")

# --- gemma4 (p5-06 AC3) — measured 2026-10-18 from the AC2 fixture headers
# (.local/models/gemma4-26b-a4b-it, inventory tests/fixtures/gemma4_lang_header.json).
# Engine key vocabulary: pin c566ff4 `src/poly/gemma4.cpp:197-211,219-234` —
# attn_norm / post_attention_norm / ffn_norm / post_ffw_norm{,_1,_2} / pre_ffw_norm_2 /
# attn_q/k/v/output + attn_q_norm/attn_k_norm / ffn_gate/up/down.weight (the dense MLP,
# gemma4.cpp:85-87) / ffn_gate_inp.weight + ffn_gate_inp.scale / ffn_down_exps.scale /
# layer_output_scale.weight.  HF keys OUTSIDE this vocabulary (vision tower, and any
# unmeasured key) are NOT mapped (p4-01 never-invented rule): the emit builder skips them.
GEMMA4_LAYER_DENSE: dict[str, tuple[str, str]] = {
    # 1-D norms (engine f32() contract: F32 tensor of n_embd — BF16 header widened)
    "input_layernorm.weight": ("attn_norm.weight", "same"),
    "post_attention_layernorm.weight": ("post_attention_norm.weight", "same"),
    "pre_feedforward_layernorm.weight": ("ffn_norm.weight", "same"),
    "post_feedforward_layernorm.weight": ("post_ffw_norm.weight", "same"),
    "post_feedforward_layernorm_1.weight": ("post_ffw_norm_1.weight", "same"),
    "pre_feedforward_layernorm_2.weight": ("pre_ffw_norm_2.weight", "same"),
    "post_feedforward_layernorm_2.weight": ("post_ffw_norm_2.weight", "same"),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", "same"),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", "same"),
    # router: HF `router.*` names, engine `ffn_gate_inp.*` / `ffn_down_exps.scale`
    "router.scale": ("ffn_gate_inp.scale", "same"),            # F32 [n_embd]
    "router.per_expert_scale": ("ffn_down_exps.scale", "same"),  # F32 [n_expert]
    "layer_scalar": ("layer_output_scale.weight", "same"),      # F32 [1]
    # 2-D matrices: HF [out, in] row-major bytes == engine (in, out) bytes
    "self_attn.q_proj.weight": ("attn_q.weight", "transpose"),
    "self_attn.k_proj.weight": ("attn_k.weight", "transpose"),
    "self_attn.v_proj.weight": ("attn_v.weight", "transpose"),
    "self_attn.o_proj.weight": ("attn_output.weight", "transpose"),
    "mlp.gate_proj.weight": ("ffn_gate.weight", "transpose"),
    "mlp.up_proj.weight": ("ffn_up.weight", "transpose"),
    "mlp.down_proj.weight": ("ffn_down.weight", "transpose"),
    "router.proj.weight": ("ffn_gate_inp.weight", "transpose"),
}

# --- mixtral (p6-01 AC2) — HF keys measured in tests/fixtures/mixtral_header.json
# (`key_patterns`, Mixtral-8x7B-v0.1 shard 1 + index, 2026-10-20). Engine key vocabulary: what the engine_ext
# `Mixtral` class loads (engine_ext/src/poly/mixtral.cpp load_layer; load_moe engine.cpp:721-765, pin c566ff4),
# the same names as gemma4.cpp:197-209 / qwen35.cpp:84-106. HF → GGUF pairing: pinned llama.cpp
# gguf-py/gguf/tensor_mapping.py (3cf03257) — input_layernorm→attn_norm :193, post_attention_layernorm→ffn_norm
# :405, block_sparse_moe.gate→ffn_gate_inp :458 ("mixtral"); names constants.py:1434-1469.
# rope_permute: q/k rows reordered for the class's normal-pairing rotary (meta/mixtral_rope.py), then the
# transpose shape flip. transpose_f32: the transpose flip, kept F32 like the 1-D keys. The per-expert w1/w3/w2 go through expert_map (gate/up/down → ffn_*_exps).
MIXTRAL_LAYER_DENSE: dict[str, tuple[str, str]] = {
    "input_layernorm.weight": ("attn_norm.weight", "same"),
    "post_attention_layernorm.weight": ("ffn_norm.weight", "same"),
    "self_attn.q_proj.weight": ("attn_q.weight", "rope_permute"),
    "self_attn.k_proj.weight": ("attn_k.weight", "rope_permute"),
    "self_attn.v_proj.weight": ("attn_v.weight", "transpose"),
    "self_attn.o_proj.weight": ("attn_output.weight", "transpose"),
    # [n_embd, n_expert], load_moe. Never quantized, whatever --dense-type: pinned llama.cpp
    # src/llama-quant.cpp:306 "do not quantize expert gating tensors" (3cf03257); arXiv 2312.17238 keeps
    # Mixtral's MoE gates in 16-bit (docs/work/p6-01.md 수정 후보 1).
    "block_sparse_moe.gate.weight": ("ffn_gate_inp.weight", "transpose_f32"),
}
# tensor_mapping.py :16 / :79 / :100; `tie_word_embeddings: false` measured, so lm_head is its own tensor.
MIXTRAL_GLOBAL_DENSE: dict[str, tuple[str, str]] = {
    "model.embed_tokens.weight": ("token_embd.weight", "transpose"),
    "lm_head.weight": ("output.weight", "transpose"),
    "model.norm.weight": ("output_norm.weight", "same"),
}

# tied embeddings (config `tie_word_embeddings: true` — AC2 fixture config measured):
# `lm_head` key ABSENT in the fixture headers, so it is not mapped (never invented).
GEMMA4_GLOBAL_DENSE: dict[str, tuple[str, str]] = {
    "model.language_model.embed_tokens.weight": ("token_embd.weight", "transpose"),
    "model.language_model.norm.weight": ("output_norm.weight", "same"),
}


# --- tokenizer metadata (p5-09 AC2) — measured, never invented ----------------
# Grounding (docs/work/p5-09.md 전제 실측, pin c566ff4 read-only):
# - Engine reader `tools/strata_tokenizer.py:175-190` `from_gguf`: needs
#   `tokenizer.ggml.tokens` + `tokenizer.ggml.merges` (raise "GGUF is missing …"
#   when absent), `tokenizer.ggml.model` ∈ MODELS {"gpt2": None, "gemma4": "gemma4"}
#   (line 106 — the `pre` name the model string implies; the reference GGUF ships
#   NO `tokenizer.ggml.pre` key, the model string carries the pre), collects every
#   `tokenizer.ggml.*_token_id` as special ids (line 186-187), and token types
#   3=CONTROL / 4=USER_DEFINED / 6=BYTE (`__init__` docstring, lines 138-146).
# - Engine C++ `src/poly/main.cpp:915-916` reads `tokenizer.ggml.eos/eot/eom_token_id`
#   as the default stop ids when no --eos-ids are given.
# - Reference GGUF header measured 2026-10-20 (HF `unsloth/gemma-4-26B-A4B-it-GGUF`,
#   Q8_0 file, first 40 MB Range, parsed by the engine's own `gguf_reader.py`):
#   model "gemma4", tokens 262,144 == HF vocab id-map, merges 514,906 == HF merges
#   ("left right" strings), scores all -1000.0, token_type {3:16, 1:261865, 4:7,
#   6:256}, bos 2 / eos 106 (`<turn|>`) / mask 4 / padding 0 / unknown 3,
#   add_bos_token True, add_space_prefix False. `<eos>`(1) is NORMAL(1) — the turn
#   end is the GGUF eos. HF `chat_template.jinja` (18,681 B) differs from the
#   reference template by 36 diff lines (converter-added) — we emit the HF ORIGINAL
#   (design rule: HF measured text, the reference conversion is not our source).
# token_type classes per special token: measured from the reference header id by
# id, transcribed below as the measured table (a shape rule is NOT invented).
SPECIAL_TYPE_MEASURED: dict[str, int] = {
    "<pad>": 3, "<bos>": 3, "<unk>": 3, "<mask>": 3,
    "<eos>": 1,  # reference header MEASURED: `<eos>` is NORMAL(1) — the GGUF eos
                 # id is `<turn|>`(106), the token text itself is ordinary text
    "<|tool>": 3, "<tool|>": 3, "<|think|>": 3,
    "<|turn>": 3, "<turn|>": 3,
    "<|image>": 3, "<|audio>": 3, "<|image|>": 3, "<|audio|>": 3,
    "<image|>": 3, "<audio|>": 3, "<|video|>": 3,
    "<|tool_call>": 4, "<tool_call|>": 4, "<|tool_response>": 4,
    "<tool_response|>": 4, "<|\"|>": 4, "<|channel>": 4, "<channel|>": 4,
}


def load_hf_tokenizer(hf_dir: str | pathlib.Path) -> dict:
    """HF tokenizer files MEASURED from a checkpoint directory (p5-09 AC1 files).

    Reads `tokenizer.json` (BPE vocab/merges/added_tokens), `tokenizer_config.json`
    (special token names), `chat_template.jinja` (verbatim text). Returns the raw
    measured pieces; the GGUF key builder below turns them into the engine key
    vocabulary. A missing `tokenizer.json` is a hard refusal — never invented.
    """
    d = pathlib.Path(hf_dir)
    tjp = d / "tokenizer.json"
    if not tjp.is_file():
        raise FileNotFoundError(f"{hf_dir}: tokenizer.json absent — tokenizer emit refused")
    tk = json.loads(tjp.read_text(encoding="utf-8"))
    if tk.get("model", {}).get("type") != "BPE":
        raise ValueError(f"tokenizer.json model.type must be BPE, got "
                         f"{tk.get('model', {}).get('type')!r}")
    vocab = tk["model"]["vocab"]
    added = tk["added_tokens"]
    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8")) \
        if (d / "tokenizer_config.json").is_file() else {}
    ct = (d / "chat_template.jinja").read_text(encoding="utf-8") \
        if (d / "chat_template.jinja").is_file() else None
    return {"vocab": vocab, "merges": tk["model"]["merges"], "added": added,
            "config": tc, "chat_template": ct}


def build_tokenizer_meta(hf_tok: dict,
                         vocab_guard: int | None = None) -> dict[str, tuple[str, object]]:
    """GGUF tokenizer metadata keys from MEASURED HF pieces (p5-09 AC2 emit rule).

    Returns {key: (gguf_type_name, value)} for the StreamingGGUFWriter, in the
    reference header's measured key set: model, tokens, merges, scores, token_type,
    add_bos_token, add_space_prefix, bos/eos/mask/padding/unknown_token_id,
    chat_template. The `pre` key is NOT emitted (the reference ships none; the
    engine derives the pre-tokenizer from `model` via MODELS, measured pin c566ff4).

    never-invented gates (raise, refuse the emit):
    - vocab ids must cover 0..n-1 exactly (measured HF fact 2026-10-20: coverage True);
    - vocab_size must EQUAL `vocab_guard` when the guard carries it (p5-08 rope gate
      shape: measure-equal or refuse);
    - every special id from `added_tokens` must exist in vocab (measured True);
    - every special text must be in the measured type table SPECIAL_TYPE_MEASURED
      (reference header, id by id) — an unmapped special is refused, not guessed;
    - every merge pair must name vocabulary tokens as measured (`a b` space-safe:
      measured HF fact: parts contain no space, 0 collisions).
    """
    vocab = hf_tok["vocab"]
    ids = sorted(vocab.values())
    if ids != list(range(len(vocab))):
        raise ValueError("tokenizer vocab does not cover ids 0..n-1 (never-invented)")
    n = len(vocab)
    if vocab_guard is not None and n != int(vocab_guard):
        raise ValueError(f"tokenizer vocab_size {n} != guard vocab_size {vocab_guard}")
    types = [1] * n
    for t in vocab:
        if t.startswith("<0x") and len(t) == 6:
            types[vocab[t]] = 6
    for a in hf_tok["added"]:
        text = a["content"]
        if text not in vocab:
            raise ValueError(f"special {text!r} is not in the vocabulary")
        ty = SPECIAL_TYPE_MEASURED.get(text)
        if ty is None:
            raise ValueError(f"special {text!r} has no measured token_type in the "
                             f"reference table — emit refused (never invented)")
        types[vocab[text]] = ty
    merges = [" ".join(m) for m in hf_tok["merges"]]
    for i, m in enumerate(merges):
        parts = m.split(" ")
        if len(parts) != 2:
            raise ValueError(f"merge {i} is not a pair: {m!r}")
        if parts[0] not in vocab or parts[1] not in vocab:
            raise ValueError(f"merge {i} names a token outside the vocabulary: {m!r}")
    cfg = hf_tok["config"]
    def sid(name: str) -> int | None:
        tok = cfg.get(name)
        return vocab.get(tok) if isinstance(tok, str) else None
    out: dict[str, tuple[str, object]] = {}
    out["tokenizer.ggml.model"] = ("string", "gemma4")
    out["tokenizer.ggml.tokens"] = ("array:string", list(vocab))
    out["tokenizer.ggml.merges"] = ("array:string", merges)
    out["tokenizer.ggml.scores"] = ("array:f32", [-1000.0] * n)
    out["tokenizer.ggml.token_type"] = ("array:i32", types)
    out["tokenizer.ggml.add_bos_token"] = ("bool", True)
    out["tokenizer.ggml.add_space_prefix"] = ("bool", False)
    for key, name in (("bos_token_id", "bos_token"), ("eos_token_id", "eos_token"),
                      ("mask_token_id", "mask_token"),
                      ("padding_token_id", "pad_token"),
                      ("unknown_token_id", "unk_token"),
                      ("eot_token_id", "eot_token")):
        i = sid(name)
        if i is not None:
            out[f"tokenizer.ggml.{key}"] = ("i32", i)
    if hf_tok["chat_template"] is not None:
        out["tokenizer.chat_template"] = ("string", hf_tok["chat_template"])
    return out


# arch prefix → (layer table, global table); absent arch → the p4-01 tables
ARCH_DENSE: dict[str, tuple[dict, dict]] = {
    "gemma4": (GEMMA4_LAYER_DENSE, GEMMA4_GLOBAL_DENSE),
    "mixtral": (MIXTRAL_LAYER_DENSE, MIXTRAL_GLOBAL_DENSE),
}


def hf_layer_id(name: str) -> int | None:
    """Layer id for an HF `model.layers.<L>.*` name, else None."""
    m = _LAYER_RE.match(name)
    return int(m.group(1)) if m else None


def map_dense(name: str, arch: str = "qwen4exp") -> tuple[int, str, str] | None:
    """(layer, engine_name, rule) for an HF dense tensor, engine prefix applied.

    rule: "transpose" (shape flipped, bytes identical), "same" (shape kept),
    "bf16_keep" (BF16 bytes verbatim — Bf16InF32 engine kind).
    Returns None for names not in the measured vocabulary of `arch`
    (p4-01 never-invented: unmeasured names are NOT mapped).

    arch "gemma4" uses the fixture-measured HF vocabulary + the engine Gemma4
    key table (GEMMA4_LAYER_DENSE / GEMMA4_GLOBAL_DENSE); arch "mixtral" the
    engine_ext Mixtral table (MIXTRAL_LAYER_DENSE / MIXTRAL_GLOBAL_DENSE, p6-01);
    any other arch keeps the p4-01 QSA table.
    """
    layer_table, global_table = ARCH_DENSE.get(arch, (LAYER_DENSE, GLOBAL_DENSE))
    m = _LAYER_RE.match(name)
    if m:
        layer, suffix = int(m.group(1)), m.group(2)
        hit = layer_table.get(suffix)
        if hit is None:
            return None
        return layer, f"blk.{layer}.{hit[0]}", hit[1]
    hit = global_table.get(name)
    if hit is None:
        return None
    return -1, hit[0], hit[1]


def fused_expert_role(name: str) -> tuple[int, str] | None:
    """(layer, gate_up|down) for a fused HF expert tensor, else None."""
    m = FUSED_EXPERT.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2).removesuffix("_proj")


def gdn_qkv_part(name: str) -> tuple[int, str] | None:
    """(layer, q|k|v) for a GDN projection feeding the fused `attn_qkv`, else None."""
    m = GDN_QKV.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2).removesuffix("_proj")


def is_hf_naming(meta: ModelMeta) -> bool:
    """HF naming auto-detection: any measured HF key hit (expert_map hits included)."""
    for t in meta.tensors:
        if fused_expert_role(t.name) or gdn_qkv_part(t.name) \
                or map_dense(t.name) is not None:
            return True
    return False


def derive_geometry(meta: ModelMeta, head_dim: int | None = None,
                    arch: str = "qwen4exp") -> dict:
    """Guard keys DERIVED from the HF header (never invented, p4-01 §Emit 규칙).

    embedding_length = embed_tokens shape[1]; block_count = layer count;
    expert_count = NE, expert_feed_forward_length = FF from the fused gate_up
    shape (NE, 2FF, H); head_dim from `self_attn.q_norm` shape[0] when present
    (else the caller-supplied head_dim); head_count = q_proj shape[0] // head_dim,
    head_count_kv = k_proj shape[0] // head_dim.

    arch "gemma4" (p5-06 AC3, header-measured): per-layer head_dim IS the q_norm
    shape (engine q_norm is f32(head_dim), gemma4.cpp:205-206 — every fixture layer
    has one, so head_dim needs no config), head_count_kv becomes the per-layer
    ARRAY the engine reads via of_layer (gemma4.cpp:64-68: sliding 8, full 2 —
    k_proj shape[0] // head_dim per layer), head_count is checked equal across
    layers (gemma4.cpp:67 `layers with different numbers of heads`).

    Returns engine-keyed ints (lists where the engine array key is measured),
    prefix stripped; the CLI cross-checks them against the preset guard keys and
    refuses a mismatch.
    """
    by_name = {t.name: t for t in meta.tensors}
    if arch == "gemma4":
        return _derive_gemma4(by_name)
    if arch == "mixtral":
        return _derive_mixtral(meta, by_name, head_dim)
    layers = sorted({l for l in (hf_layer_id(t.name) for t in meta.tensors) if l is not None})
    out: dict[str, int] = {}
    emb = by_name.get("model.embed_tokens.weight")
    if emb is not None:
        out["embedding_length"] = emb.shape[1]
    out["block_count"] = len(layers)
    gu = by_name.get(f"model.layers.{layers[0]}.mlp.experts.gate_up_proj") if layers else None
    if gu is not None:
        ne, two_ff, h = gu.shape
        out["expert_count"] = ne
        out["expert_feed_forward_length"] = two_ff // 2
        out["embedding_length"] = h  # fused gate_up row width IS n_embd
    if head_dim is None:
        qn = by_name.get(f"model.layers.{layers[0]}.self_attn.q_norm.weight") if layers else None
        if qn is not None:
            head_dim = qn.shape[0]
    if head_dim:
        q = by_name.get(f"model.layers.{layers[0]}.self_attn.q_proj.weight")
        k = by_name.get(f"model.layers.{layers[0]}.self_attn.k_proj.weight")
        if q is not None:
            out["attention.head_count"] = q.shape[0] // head_dim
        if k is not None:
            out["attention.head_count_kv"] = k.shape[0] // head_dim
    return out


def _derive_gemma4(by_name: dict) -> dict:
    """Gemma4 guard keys DERIVED from the measured fixture headers (p5-06 AC3)."""
    names = [n for n in by_name if n.startswith("model.language_model.layers.")]
    layers = sorted({int(n.split(".")[3]) for n in names if n.split(".")[3].isdigit()})
    out: dict = {}
    emb = by_name.get("model.language_model.embed_tokens.weight")
    if emb is not None:
        out["embedding_length"] = emb.shape[1]
    out["block_count"] = len(layers)
    p0 = f"model.language_model.layers.{layers[0]}."
    gu = by_name.get(p0 + "experts.gate_up_proj")
    if gu is not None:
        ne, two_ff, h = gu.shape
        out["expert_count"] = ne
        out["expert_feed_forward_length"] = two_ff // 2
        out["embedding_length"] = h  # fused gate_up row width IS n_embd
    # per-layer head_dim = q_norm length (engine f32(head_dim), gemma4.cpp:205-206)
    head_dims = {}
    for L in layers:
        qn = by_name.get(f"model.language_model.layers.{L}.self_attn.q_norm.weight")
        if qn is not None:
            head_dims[L] = qn.shape[0]
    counts, kvs = [], []
    for L in layers:
        hd = head_dims.get(L)
        q = by_name.get(f"model.language_model.layers.{L}.self_attn.q_proj.weight")
        k = by_name.get(f"model.language_model.layers.{L}.self_attn.k_proj.weight")
        if hd and q is not None:
            counts.append(q.shape[0] // hd)
        if hd and k is not None:
            kvs.append(k.shape[0] // hd)
    if counts:
        if len(set(counts)) > 1:
            raise ValueError(f"gemma4 head_count varies across layers {sorted(set(counts))} "
                             f"(engine gemma4.cpp:67 refuses 'layers with different numbers "
                             f"of heads')")
        out["attention.head_count"] = counts[0]
    if len(kvs) == len(layers):
        out["attention.head_count_kv"] = kvs
    return out


def _derive_mixtral(meta: ModelMeta, by_name: dict, head_dim: int | None) -> dict:
    """Mixtral guard keys DERIVED from the HF header (p6-01 AC2, fixture-measured names):
    embedding_length = embed_tokens shape[1]; block_count = layer count; expert_count =
    the experts of the first layer (= router `block_sparse_moe.gate` rows, checked equal);
    expert_feed_forward_length = w1 rows; with head_dim (config.json, read_hf_config):
    attention.key_length = head_dim, head_count = q rows // head_dim, head_count_kv =
    k rows // head_dim."""
    from stratashift.meta.expert_map import build_expert_map

    layers = sorted({l for l in (hf_layer_id(t.name) for t in meta.tensors) if l is not None})
    out: dict = {}
    emb = by_name.get("model.embed_tokens.weight")
    if emb is not None:
        out["embedding_length"] = emb.shape[1]
    out["block_count"] = len(layers)
    if not layers:
        return out
    p0 = f"model.layers.{layers[0]}."
    refs = [r for r in build_expert_map(meta) if r.layer_id == layers[0]]
    if refs:
        out["expert_count"] = len({r.expert_id for r in refs})
        gate0 = next((r for r in refs if r.role == "gate"), None)
        if gate0 is not None:
            out["expert_feed_forward_length"] = by_name[gate0.tensor_name].shape[0]
    router = by_name.get(p0 + "block_sparse_moe.gate.weight")
    if router is not None and "expert_count" in out and router.shape[0] != out["expert_count"]:
        raise ValueError(f"mixtral router {p0}block_sparse_moe.gate.weight has {router.shape[0]} rows "
                         f"but layer {layers[0]} has {out['expert_count']} experts")
    if head_dim:
        out["attention.key_length"] = head_dim
        q = by_name.get(p0 + "self_attn.q_proj.weight")
        k = by_name.get(p0 + "self_attn.k_proj.weight")
        if q is not None:
            out["attention.head_count"] = q.shape[0] // head_dim
        if k is not None:
            out["attention.head_count_kv"] = k.shape[0] // head_dim
    return out
