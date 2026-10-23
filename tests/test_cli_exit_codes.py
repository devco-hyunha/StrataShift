"""p5-02 CLI polish gate: auto-detected HF dispatch, CLI->verify final link, exit codes.

AC1: safetensors with fused HF naming routes to the HF emit path WITHOUT `--hf-emit`
(the p4-01 "auto-detected" claim, connected in dispatch — is_hf_naming existed in the
map module, the dispatch never used it).
AC2: `convert --out` (p2 pack manifest, `packed_rows` form) passes `verify <out>` —
the quickstart final link; the engine-form manifest verify path (p4 harness contract)
stays green through the GGUF->convert->verify hop.
AC3: missing model file exits 1 with `error:` on stderr and no traceback; argparse
usage keeps SystemExit(2); success exits 0 via cli_entry (p5-01 AC1 contract).
"""
from __future__ import annotations

import struct
from pathlib import Path

import pytest

import stratashift.cli.main as cli_mod
from stratashift.polystrata import engine_tools
from stratashift.ingest.gguf import GGUF_MAGIC, parse_gguf_header
from tests.test_header_parse import make_safetensors

GGUF_GATE = pytest.mark.skipif(
    not engine_tools.available(),
    reason="pinned engine tools checkout required (STRATASHIFT_ENGINE_TOOLS)",
)

# tiny fused HF fixture geometry: H=64, FF=64 (Q2_0 role rows are multiples of 64,
# engine 64-element Q2_0 block contract), NE=2, L=1, vocab=128, head_dim=32
H, FF, NE, V, HD, HEADS, KV = 64, 64, 2, 128, 32, 2, 2

FUSED_HF_TENSORS = [
    ("model.embed_tokens.weight", [V, H], "F32"),
    ("lm_head.weight", [V, H], "F32"),
    ("model.norm.weight", [H], "F32"),
    ("model.layers.0.input_layernorm.weight", [H], "F32"),
    ("model.layers.0.post_attention_layernorm.weight", [H], "F32"),
    ("model.layers.0.self_attn.q_proj.weight", [HEADS * HD, H], "F32"),
    ("model.layers.0.self_attn.k_proj.weight", [KV * HD, H], "F32"),
    ("model.layers.0.self_attn.v_proj.weight", [KV * HD, H], "F32"),
    ("model.layers.0.self_attn.o_proj.weight", [H, HEADS * HD], "F32"),
    ("model.layers.0.self_attn.q_norm.weight", [HD], "F32"),
    ("model.layers.0.self_attn.k_norm.weight", [HD], "F32"),
    ("model.layers.0.mlp.gate.weight", [NE, H], "F32"),
    ("model.layers.0.mlp.shared_expert.gate_proj.weight", [FF, H], "F32"),
    ("model.layers.0.mlp.shared_expert.up_proj.weight", [FF, H], "F32"),
    ("model.layers.0.mlp.shared_expert.down_proj.weight", [H, FF], "F32"),
    ("model.layers.0.mlp.experts.gate_up_proj", [NE, 2 * FF, H], "F32"),
    ("model.layers.0.mlp.experts.down_proj", [NE, H, FF], "F32"),
]


def make_hf_fused(path: Path) -> Path:
    """Fused HF naming fixture (Qwen1.5 style, p4-01 measured naming): every element
    is a safe F32 (~0.746, no inf/nan in the quantizer), blob bytes are deterministic.
    Optional tensors for the emit builder (shared_expert_gate BF16, GDN/QSA/indexer)
    are absent — the builder skips them (`entry is None: continue`).
    """
    header: dict[str, dict] = {}
    blob = bytearray()
    for name, shape, dtype in FUSED_HF_TENSORS:
        n = 1
        for d in shape:
            n *= d
        nb = n * 4  # F32
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(blob), len(blob) + nb]}
        blob += b"\x3f" * nb
    header["__metadata__"] = {"format": "pt"}
    make_safetensors(path, header, bytes(blob))
    return path


@GGUF_GATE
def test_cli_auto_detect_hf_emit(tmp_path: Path) -> None:
    """AC1: no `--hf-emit` flag — HF fused naming must route to the fused HF emit path."""
    model = make_hf_fused(tmp_path / "hf_fused.safetensors")
    out = tmp_path / "pack"
    gguf = tmp_path / "model.gguf"
    rc = cli_mod.main(["convert", str(model), "--out", str(out),
                       "--emit-gguf", str(gguf)])
    assert rc == 0
    with open(gguf, "rb") as f:
        assert struct.unpack("<I", f.read(4))[0] == GGUF_MAGIC
    meta = parse_gguf_header(str(gguf))
    names = {t.name for t in meta.tensors}
    # engine-named fused expert tensors prove the HF emit path ran (not the p2 stream)
    assert "blk.0.ffn_gate_exps.weight" in names
    assert "blk.0.ffn_up_exps.weight" in names
    assert "blk.0.ffn_down_exps.weight" in names


def test_cli_verify_p2_cli_pack(tmp_path: Path) -> None:
    """AC2a: `convert --out` pack passes `verify <out>` (p2 manifest contract)."""
    model = make_hf_fused(tmp_path / "hf_fused.safetensors")
    out = tmp_path / "pack"
    assert cli_mod.main(["convert", str(model), "--out", str(out)]) == 0
    assert cli_mod.main(["verify", str(out)]) == 0


@GGUF_GATE
def test_cli_engine_manifest_verify(tmp_path: Path) -> None:
    """AC2b: GGUF->convert writes the engine-form manifest; verify PASSes on it
    (p4 harness contract path stays green — the original KeyError 'tensors' scope).
    """
    model = make_hf_fused(tmp_path / "hf_fused.safetensors")
    out1 = tmp_path / "pack1"
    gguf = tmp_path / "model.gguf"
    assert cli_mod.main(["convert", str(model), "--out", str(out1),
                         "--emit-gguf", str(gguf)]) == 0
    out2 = tmp_path / "pack2"
    assert cli_mod.main(["convert", str(gguf), "--out", str(out2)]) == 0
    assert cli_mod.main(["verify", str(out2)]) == 0


def test_inspect_missing_exits_1_no_traceback(tmp_path: Path, capsys, monkeypatch) -> None:
    """AC3: missing model file -> exit 1, `error:` line on stderr, no traceback."""
    missing = tmp_path / "missing.safetensors"
    monkeypatch.setattr("sys.argv", ["stratashift", "inspect", str(missing)])
    assert cli_mod._run() == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err


def test_usage_error_keeps_exit_2(monkeypatch) -> None:
    """AC3: argparse usage error keeps SystemExit(2) — not swallowed into 1."""
    monkeypatch.setattr("sys.argv", ["stratashift", "convert"])  # missing required args
    with pytest.raises(SystemExit) as ei:
        cli_mod._run()
    assert ei.value.code == 2


def test_cli_entry_success_exit_0(monkeypatch, tmp_path: Path) -> None:
    """p5-01 AC1 contract stays green through the p5-02 wrapper: success exits 0."""
    model = make_hf_fused(tmp_path / "hf_fused.safetensors")
    out = tmp_path / "pack"
    monkeypatch.setattr("sys.argv", ["stratashift", "convert", str(model),
                                     "--out", str(out)])
    with pytest.raises(SystemExit) as ei:
        cli_mod.cli_entry()
    assert ei.value.code == 0


@GGUF_GATE
def test_engine_named_bf16_emit_user_message_exit_1(tmp_path: Path, capsys,
                                                     monkeypatch) -> None:
    """AC3: non-HF (engine-named) BF16 dense input to the F32-dense emit path must
    print a user message on stderr and exit 1 — no traceback (the p5-02 gap 2
    measurement: `ValueError: lm_head.weight: only F32 dense tensors supported`).

    AC4 regression (2026-10-18 root cause): the exception path MUST leave
    tracemalloc inactive — a leaked `tracemalloc.start()` (stop bypassed by the raise)
    made `test_no_full_load` measure the cumulative session peak and fail its
    header budget. `heap_gate()` pairs start/stop on every exit path.
    """
    import tracemalloc

    header = {
        "blk.0.token_embd.weight": {"dtype": "BF16", "shape": [128, 64],
                                    "data_offsets": [0, 16384]},
        "__metadata__": {"format": "pt"},
    }
    model = tmp_path / "engine_bf16.safetensors"
    make_safetensors(model, header, b"\x3f" * 16384)
    monkeypatch.setattr("sys.argv", ["stratashift", "convert", str(model), "--out",
                                     str(tmp_path / "pack"), "--emit-gguf",
                                     str(tmp_path / "m.gguf")])
    assert cli_mod._run() == 1
    assert not tracemalloc.is_tracing(), "tracemalloc leaked past the exception path"
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "Traceback" not in err
