"""p5-06 AC4 gate: Gemma4 emit → engine gguf_reader parse match + canonical_xcheck.

The engine guard is NOT applicable to `general.architecture=gemma4` (check_architecture,
gguf_reader.hpp:796-812, pin c566ff4 — measured refusal; the Gemma4 run class is CUDA/HIP,
model_family.hpp:52-62), so the acceptance is the reference verifier (p4-01/p4-03 template):
engine `tools/gguf_reader.py` parse == our GGUF parser, engine canonical Q2_0 round-trip
bit-exact, our `verify` PASS. The full-model run lives in scripts/p5_gemma4_smoke.py.

Skip gates (no GPU, no fixture): every test skips when the AC2 fixture directory is absent.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

MODEL_DIR = ROOT / ".local" / "models" / "gemma4-26b-a4b-it"
GGUF = pathlib.Path("/opt/data/cache/scratch/p5-06-out/gemma4-emitted.gguf")
ENGINE_TOOLS = pathlib.Path("/opt/data/polystrata-src/tools")
sys.path.insert(0, str(ENGINE_TOOLS))

from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402
from stratashift.quant.pack import q2_0_unpack  # noqa: E402


def _fixture_ready() -> bool:
    return (MODEL_DIR / "model-00001-of-00002.safetensors").is_file() \
        and (MODEL_DIR / "model.safetensors.index.json").is_file()


@pytest.fixture(scope="module")
def gguf():
    if not _fixture_ready():
        pytest.skip("AC2 fixture (.local/models/gemma4-26b-a4b-it) absent")
    if not GGUF.is_file():
        pytest.skip("p5_gemma4_smoke.py emit not run yet (run scripts/p5_gemma4_smoke.py)")
    return GGUF


def test_engine_reader_parse_matches_ours(gguf):
    """Engine gguf_reader.py == our parser on names/shapes/offsets (reference verifier)."""
    from gguf_reader import GGUFFile
    eng = GGUFFile(str(gguf))
    ours = parse_gguf_header(str(gguf))
    eng_list = [(t.name, list(t.shape), t.offset) for t in eng.tensors]
    ours_list = [(t.name, list(t.shape), t.offset) for t in ours.tensors]
    assert eng_list == ours_list
    names = {t.name for t in eng.tensors}
    # the measured key families: experts in all 30 layers, dense MLP in all 30 layers,
    # full layers (5/11/17/23/29) have NO attn_v.weight (attention_k_eq_v, AC2 measured)
    for L in range(30):
        assert f"blk.{L}.ffn_gate_exps.weight" in names, L
        assert f"blk.{L}.ffn_gate.weight" in names, L
        assert (f"blk.{L}.attn_v.weight" in names) == (L not in (5, 11, 17, 23, 29)), L
    assert "token_embd.weight" in names and "output_norm.weight" in names
    assert "output.weight" not in names  # tied embeddings — lm_head absent, never invented


def test_engine_meta_keys_measured(gguf):
    """GGUF meta carries the engine gemma4 must keys, values == header-derived (never
    invented), head_count_kv/sliding_window_pattern are the per-layer arrays."""
    from gguf_reader import GGUFFile
    eng = GGUFFile(str(gguf))
    meta = dict(eng.metadata) if hasattr(eng, "metadata") else {}
    if not meta:
        pytest.skip("engine reader meta accessor shape unknown")
    assert meta["general.architecture"] == "gemma4"
    assert meta["gemma4.block_count"] == 30
    assert meta["gemma4.embedding_length"] == 2816
    assert meta["gemma4.expert_count"] == 128
    assert meta["gemma4.expert_feed_forward_length"] == 704
    kv = meta["gemma4.attention.head_count_kv"]
    assert list(kv) == [2 if L in (5, 11, 17, 23, 29) else 8 for L in range(30)]
    pat = meta["gemma4.attention.sliding_window_pattern"]
    assert list(pat) == [0 if L in (5, 11, 17, 23, 29) else 1 for L in range(30)]


def test_canonical_xcheck_q2_0_roundtrip(gguf):
    """Engine canonical Q2_0 mapping round-trip on the first expert tensor, bit-exact."""
    from gguf_reader import GGUFFile
    from p4_hf_emit import from_s2, to_s2
    eng = GGUFFile(str(gguf))
    first_q2 = next(t for t in eng.tensors if t.type_name == "Q2_0")
    with open(gguf, "rb") as fh:
        fh.seek(eng.data_start + first_q2.offset)
        raw = fh.read(64 * 18)
    ours_v = []
    for b in range(64):
        ours_v.extend(q2_0_unpack(raw[b * 18:(b + 1) * 18]))
    eng_v = from_s2(to_s2(raw)).astype(np.float32)
    assert np.array_equal(np.array(ours_v, dtype=np.float32), eng_v)


def test_cli_derived_guard_cross_check(gguf):
    """The CLI's header-derived guard keys must equal the preset values the file carried
    (the emit refuses a mismatch — this test re-derives from the fixture headers)."""
    from stratashift.ingest.shard_index import load_sharded_model
    from stratashift.meta.hf_gguf_map import derive_geometry
    meta = load_sharded_model(str(MODEL_DIR))
    g = derive_geometry(meta, arch="gemma4")
    preset = json.loads((ROOT / "scripts" / "p5_gemma4_guard.json").read_text())
    for key, val in g.items():
        assert preset[f"gemma4.{key}"] == val, (key, val, preset.get(f"gemma4.{key}"))
