"""p5-09 AC2 gate: HF tokenizer files → GGUF `tokenizer.ggml.*` emit, engine round-trip.

Grounding (docs/work/p5-09.md, pin c566ff4 read-only, measured 2026-10-20):
- Engine `tools/strata_tokenizer.py:175-190` `from_gguf` needs
  `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`; `tokenizer.ggml.model` must be
  in MODELS {"gpt2": None, "gemma4": "gemma4"} (line 106); special ids are
  collected from every `tokenizer.ggml.*_token_id` key (line 186-187); token
  types 3=CONTROL / 4=USER_DEFINED / 6=BYTE drive the special matching.
- Reference GGUF header measured (unsloth gemma-4-26B-A4B-it-GGUF Q8_0 head,
  parsed by the engine's own gguf_reader.py): model "gemma4", 262,144 tokens ==
  HF vocab id-map, 514,906 merges == HF merges, scores all -1000.0, token_type
  {3:16, 1:261865, 4:7, 6:256}, bos 2 / eos 106 / mask 4 / padding 0 /
  unknown 3, add_bos_token True, add_space_prefix False, chat_template present.
  The builder (`meta/hf_gguf_map.py::build_tokenizer_meta`) emits exactly this
  key vocabulary from the MEASURED HF files.

Tests:
- unit: synthetic mini HF tokenizer files (measured file structure, mini-scaled
  V=2048 — the mini fixture guard vocab) → key/value/type contract, gates refuse
  (vocab coverage, guard mismatch, unmapped special, merge outside vocab);
- round-trip: mini emit with injected tokenizer files → our parser + engine
  `gguf_reader.py` parse match + engine `from_gguf` + `extract()` dir encode→
  decode identity corpus + expert/dense tensor bytes UNCHANGED vs the no-tokenizer
  emit (metadata-only growth, p5-09 AC2 byte-plan contract).

Skip gate: engine tools checkout (STRATASHIFT_ENGINE_TOOLS, p0-01 pin).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402
from stratashift.meta.hf_gguf_map import build_tokenizer_meta, load_hf_tokenizer  # noqa: E402
from stratashift.polystrata.engine_paths import (  # noqa: E402
    ENV_SRC, ENV_TOOLS, describe_missing, resolve_engine_paths)

ENGINE_TOOLS = resolve_engine_paths().tools
TOOLS_MISSING = ("pinned engine tools checkout required (p0-01): "
                 + describe_missing(ENGINE_TOOLS / "gguf_reader.py", ENV_SRC, ENV_TOOLS))

# synthetic mini tokenizer — V=2048 (the mini fixture guard vocab_size).
# File structure + key vocabulary are MEASURED HF/reference shapes; the pieces are
# mini-scaled exactly as the mini fixture weights are (docs/work/p5-09 rule 5).
V = 2048
SPECIALS = [("<pad>", 3), ("<eos>", 1), ("<bos>", 3), ("<unk>", 3),
            ("<mask>", 3), ("<|tool>", 3), ("<tool|>", 3), ("<|tool_call>", 4),
            ("<tool_call|>", 4), ("<|tool_response>", 4), ("<tool_response|>", 4),
            ("<|\"|>", 4), ("<|think|>", 3), ("<|channel>", 4), ("<channel|>", 4),
            ("<|turn>", 3), ("<turn|>", 3)]

_CHAT = ("{%- for m in messages -%}"
         "<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n"
         "{%- endfor -%}<|turn>assistant\n")


def mini_tokenizer_files() -> dict:
    """tokenizer.json / tokenizer_config.json / chat_template.jinja as measured HF
    structure: vocab dict covering ids 0..V-1, added_tokens specials, BPE model,
    merges as ["left","right"] pairs (HF measured form 2026-10-20).

    The vocab MUST cover every single character of the round-trip corpus, exactly
    as the measured HF gemma4 vocab does (single-char pieces 19,227 + 256 byte
    tokens `<0xNN>` + `▁`; measured 2026-10-20: ASCII 32..126 coverage 94/95, the
    missing one is the space — SentencePiece spells it `▁`). The engine raw-mode
    encoder (`_encode_raw`, strata_tokenizer.py:284-295) drops pieces the vocab
    lacks from the lookup and only byte-falls-back per PIECE — an unpaired char
    makes encode/decode lossy. The mini vocab mirrors that structure mini-scaled.
    """
    vocab: dict[str, int] = {}
    for text, _ty in SPECIALS:
        vocab[text] = len(vocab)
    vocab["▁"] = len(vocab)
    # single-char pieces: printable ASCII 33..126 (the HF measured structure —
    # coverage 94/95 over 32..126, the missing piece is the space, which
    # SentencePiece spells `▁`), plus the corpus's non-ASCII chars and the
    # newline/tab pieces the raw pre-tokenizer splits on
    for i in range(33, 127):
        c = chr(i)
        if c not in vocab:
            vocab[c] = len(vocab)
    for c in "éöü🎉\n\t":
        if c not in vocab:
            vocab[c] = len(vocab)
    for b in range(256):
        vocab[f"<0x{b:02x}>"] = len(vocab)
    while len(vocab) < V:
        vocab["▁tok" + str(len(vocab))] = len(vocab)
    names = sorted(vocab, key=lambda t: vocab[t])
    # merge pairs must name real vocabulary pieces (measured HF fact: every merge
    # pair is in-vocab, 514,906/514,906 — the gate refuses an outside piece)
    merges = [[names[274], names[275]], [names[276], names[277]],
              [names[275], names[278]]]
    added = [{"id": vocab[t], "content": t, "single_word": False, "lstrip": False,
              "rstrip": False, "normalized": False, "special": True}
             for t, _ty in SPECIALS]
    tok_json = {"version": "1.0",
                "truncation": None, "padding": None, "added_tokens": added,
                "model": {"type": "BPE", "vocab": vocab, "merges": merges,
                          "special_tokens": {},
                          "merges_cache": []}}
    tok_config = {"bos_token": "<bos>", "eos_token": "<eos>", "mask_token": "<mask>",
                  "pad_token": "<pad>", "unk_token": "<unk>", "eot_token": "<turn|>",
                  "tokenizer_class": "GemmaTokenizer"}
    return {"tokenizer.json": json.dumps(tok_json),
            "tokenizer_config.json": json.dumps(tok_config),
            "chat_template.jinja": _CHAT}


def write_tokenizer_dir(d: pathlib.Path) -> None:
    for name, text in mini_tokenizer_files().items():
        (d / name).write_text(text, encoding="utf-8")


@pytest.mark.skipif(not (ENGINE_TOOLS / "gguf_reader.py").is_file(),
                    reason=TOOLS_MISSING)
def test_build_tokenizer_meta_key_contract(tmp_path):
    """Measured HF files → reference header key vocabulary, types and values."""
    d = tmp_path / "hf"
    d.mkdir()
    write_tokenizer_dir(d)
    hf = load_hf_tokenizer(d)
    meta = build_tokenizer_meta(hf, vocab_guard=V)
    assert meta["tokenizer.ggml.model"] == ("string", "gemma4")
    toks = meta["tokenizer.ggml.tokens"]
    assert toks[0] == "array:string" and len(toks[1]) == V
    by_text = {t: i for i, t in enumerate(toks[1])}
    assert by_text["<pad>"] == 0
    merg = meta["tokenizer.ggml.merges"]
    assert merg[0] == "array:string" and all(" " in m for m in merg[1])
    assert meta["tokenizer.ggml.scores"][0] == "array:f32"
    assert meta["tokenizer.ggml.scores"][1] == [-1000.0] * V  # reference measured
    types = meta["tokenizer.ggml.token_type"][1]
    # measured reference classes (SPECIAL_TYPE_MEASURED): CONTROL 3, USER_DEFINED 4,
    # BYTE 6, NORMAL 1 — the type vector must equal the measured table.
    for text, ty in SPECIALS:
        assert types[by_text[text]] == ty, text
    byte_ids = [i for t, i in by_text.items() if t.startswith("<0x")]
    assert len(byte_ids) == 256 and all(types[i] == 6 for i in byte_ids)
    assert types.count(3) + types.count(4) + types.count(6) + types.count(1) == V
    # special ids (reference measured keys): bos 2, eos 1, mask 4, padding 0,
    # unknown 3, eot = <turn|> id
    assert meta["tokenizer.ggml.bos_token_id"] == ("i32", 2)
    assert meta["tokenizer.ggml.eos_token_id"] == ("i32", 1)
    assert meta["tokenizer.ggml.mask_token_id"] == ("i32", 4)
    assert meta["tokenizer.ggml.padding_token_id"] == ("i32", 0)
    assert meta["tokenizer.ggml.unknown_token_id"] == ("i32", 3)
    assert meta["tokenizer.ggml.eot_token_id"] == ("i32", by_text["<turn|>"])
    assert meta["tokenizer.ggml.add_bos_token"] == ("bool", True)
    assert meta["tokenizer.ggml.add_space_prefix"] == ("bool", False)
    assert meta["tokenizer.chat_template"] == ("string", _CHAT)


@pytest.mark.skipif(not (ENGINE_TOOLS / "gguf_reader.py").is_file(),
                    reason=TOOLS_MISSING)
def test_tokenizer_meta_gates_refuse(tmp_path):
    """never-invented gates (p5-08 rope gate shape): guard mismatch, unmeasured
    special, merge outside vocab, non-BPE model — all refuse, none guessed."""
    d = tmp_path / "hf"
    d.mkdir()
    write_tokenizer_dir(d)
    hf = load_hf_tokenizer(d)
    with pytest.raises(ValueError, match="vocab_size"):
        build_tokenizer_meta(hf, vocab_guard=V + 1)
    # a special text outside the measured reference table is refused (a byte token
    # promoted to special is the minimal in-vocab, unmeasured case)
    bad = dict(hf)
    tok = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    byte_id = tok["model"]["vocab"]["<0x00>"]
    bad["added"] = hf["added"] + [{"id": byte_id, "content": "<0x00>", "special": True}]
    with pytest.raises(ValueError, match="token_type"):
        build_tokenizer_meta(bad, vocab_guard=V)
    # a merge naming a piece outside the vocabulary is refused
    bad_m = dict(hf)
    bad_m["merges"] = hf["merges"] + [["zz", "outside"]]
    with pytest.raises(ValueError, match="outside the vocabulary"):
        build_tokenizer_meta(bad_m, vocab_guard=V)
    # non-BPE model type refuses (engine MODELS contract, measured)
    tjp = d / "tokenizer.json"
    tj = json.loads(tjp.read_text(encoding="utf-8"))
    tj["model"]["type"] = "Unigram"
    tjp.write_text(json.dumps(tj), encoding="utf-8")
    with pytest.raises(ValueError, match="BPE"):
        load_hf_tokenizer(d)


def _emit(tmp_path, with_tok: bool):
    """Run the HF emit path over the mini fixture; tokenizer files injected when
    with_tok. Returns the GGUF path."""
    from test_gemma4_mini_emit import build_mini, GUARD
    model = tmp_path / ("mini_tok" if with_tok else "mini_plain")
    build_mini(model)
    if with_tok:
        write_tokenizer_dir(model)
    guard = tmp_path / ("guard_tok.json" if with_tok else "guard_plain.json")
    guard.write_text(json.dumps(GUARD), encoding="utf-8")
    gguf = tmp_path / ("mini_tok.gguf" if with_tok else "mini_plain.gguf")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(model), "--out", str(tmp_path / ("pack" if with_tok else "packp")),
                        "--emit-gguf", str(gguf), "--hf-emit",
                        "--guard-keys", str(guard), "--dense-type", "Q4_K"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return gguf, json.loads(r.stdout)


@pytest.mark.skipif(not (ENGINE_TOOLS / "gguf_reader.py").is_file(),
                    reason=TOOLS_MISSING)
def test_emit_roundtrip_engine_parse_and_run(tmp_path):
    """AC2 round trip: emitted GGUF parses identically under the engine
    gguf_reader.py, engine `from_gguf` loads it, `extract()` writes the tokenizer
    dir, and encode→decode is identity over the boundary corpus
    (docs/strata_tokenizer corpus intent: multi-byte UTF-8, emoji, whitespace).
    Tensor bytes must be UNCHANGED vs the no-tokenizer emit (metadata-only growth)."""
    from gguf_reader import GGUFFile  # noqa: E402
    import strata_tokenizer as ST  # noqa: E402

    gguf, report = _emit(tmp_path, True)
    assert report["tokenizer_meta_keys"] > 0
    ours = parse_gguf_header(str(gguf))
    eng_md = GGUFFile(gguf).metadata
    # our parser == engine parser on every tokenizer key
    tk_keys = sorted(k for k in ours.metadata if "tokenizer" in k)
    assert tk_keys == sorted(k for k in eng_md if "tokenizer" in k)
    for k in tk_keys:
        assert ours.metadata[k] == eng_md[k], k
    tk = ST.Tokenizer.from_gguf(gguf)
    assert tk.pre == "gemma4" and tk.raw
    # specials: CONTROL + USER_DEFINED literals matched (engine measured classes)
    special = tk.special_ids
    assert special["tokenizer.ggml.bos_token_id"] == 2
    assert special["tokenizer.ggml.eos_token_id"] == 1
    assert special["tokenizer.ggml.eot_token_id"] >= 0
    corpus = ["Hello, world!",
              "héllo wörld ünicode",
              "emoji 🎉 test",
              "line\nbreak\ttab",
              "<|turn>user\nhi<turn|>"]
    for s in corpus:
        ids = tk.encode(s, parse_special=True)
        assert tk.decode(ids) == s, s
    # extract() → tokenizer dir → from_dir round trip (pack API path, server
    # load_tokenizer contract, serve/server.py:6853-6858 measured)
    out_dir = tmp_path / "packdir"
    cfg = ST.extract(gguf, out_dir)
    assert cfg["vocab_size"] == V and cfg["model"] == "gemma4"
    tok2 = ST.Tokenizer.from_dir(out_dir / "tokenizer")
    for s in corpus:
        ids = tok2.encode(s, parse_special=True)
        assert tok2.decode(ids) == s, s
    # tensor bytes unchanged: metadata-only growth (p5-09 AC2 byte-plan contract)
    gguf_plain, report0 = _emit(tmp_path, False)
    assert report0["tokenizer_meta_keys"] == 0
    ours0 = parse_gguf_header(str(gguf_plain))
    names0 = {t.name: t for t in ours0.tensors}
    names1 = {t.name: t for t in ours.tensors}
    assert set(names0) == set(names1)
    for name, t0 in names0.items():
        t1 = names1[name]
        assert t0.shape == t1.shape and t0.type_name == t1.type_name
        assert t0.offset == t1.offset, name  # data-relative offsets identical
    with open(gguf_plain, "rb") as fa, open(gguf, "rb") as fb:
        for t0 in names0.values():
            fa.seek(ours0.data_start + t0.offset)
            fb.seek(ours.data_start + t0.offset)
            b0 = fa.read(t0.expected_bytes())
            b1 = fb.read(t0.expected_bytes())
            assert b0 == b1, t0.name
    # the GGUF stays loadable by our own verify path (pack builder accepts it)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "convert", str(gguf), "--out", str(tmp_path / "ours2")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "src" / "stratashift" / "cli" / "main.py"),
                        "verify", str(tmp_path / "ours2")],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
