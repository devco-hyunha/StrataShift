"""p6-01 AC3: engine_ext overlay — pin file hashes, anchor edits, the copy, the diff, `build-engine --ext`.

The overlay tests use a small fake engine tree and a fake engine_ext folder; one test applies the real
engine_ext edits to the real pinned tree (in memory, skipped when the tree is absent). No compiler runs here.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib

import pytest

from stratashift.install import build
from stratashift.install import engine_ext as ext
from stratashift.install import ensure as ens
from stratashift.install import probe as probe_mod
from stratashift.polystrata.engine_paths import (DEFAULT_SRC, ENGINE_EXT_DIR, EXT_TARGET, EnginePaths,
                                                 ext_paths)

ENGINE_CPP = "#include \"a.hpp\"\r\n#include \"gemma4.hpp\"\r\nint f() {\r\n    return 1;\r\n}\r\n"


def make_tree(root: pathlib.Path) -> pathlib.Path:
    src = root / "engine"
    (src / "src" / "poly").mkdir(parents=True)
    (src / "src" / "poly" / "engine.cpp").write_bytes(ENGINE_CPP.encode())
    (src / "CMakeLists.txt").write_bytes(b"project(x)\n")
    return src


def make_ext(root: pathlib.Path, src: pathlib.Path) -> pathlib.Path:
    d = root / "ext"
    (d / "src" / "poly").mkdir(parents=True)
    (d / "src" / "poly" / "mixtral.cpp").write_text("// ours\n", encoding="utf-8")
    files = {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in ("src/poly/engine.cpp",
                                                                             "CMakeLists.txt")}
    (d / "pin_files.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    (d / "overlay.json").write_text(json.dumps({
        "copy": {"src/poly/mixtral.cpp": "src/poly/mixtral.cpp"},
        "edits": [{"file": "src/poly/engine.cpp", "anchor": "#include \"gemma4.hpp\"", "where": "before",
                   "lines": ["#include \"mixtral.hpp\""]},
                  {"file": "src/poly/engine.cpp", "anchor": "    return 1;", "where": "after",
                   "lines": ["    // after"]}]}), encoding="utf-8")
    return d


# ---------- edits ----------

def test_apply_edit_keeps_crlf_and_places_lines() -> None:
    out = ext.apply_edit(ENGINE_CPP, "#include \"gemma4.hpp\"", "before", ["#ifdef X", "#endif"])
    assert out == ENGINE_CPP.replace("#include \"gemma4.hpp\"", "#ifdef X\r\n#endif\r\n#include \"gemma4.hpp\"")
    out = ext.apply_edit("a\nb\n", "a", "after", ["x"])
    assert out == "a\nx\nb\n"


@pytest.mark.parametrize("text", ["a\nb\n", "anchor\nanchor\n", "  anchor\n"])
def test_apply_edit_needs_the_anchor_once(text: str) -> None:
    """A missing, repeated or differently indented anchor stops the overlay."""
    with pytest.raises(ext.OverlayError, match="anchor found"):
        ext.apply_edit(text, "anchor", "before", ["x"])


# ---------- stage_overlay ----------

def test_stage_overlay_copies_and_edits(tmp_path) -> None:
    src = make_tree(tmp_path)
    d = make_ext(tmp_path, src)
    dst = tmp_path / "ext-src"
    (dst / "stale").mkdir(parents=True)  # an old copy is replaced, not merged
    done = ext.stage_overlay(src, dst, d)
    assert not (dst / "stale").exists()
    assert (dst / "src" / "poly" / "mixtral.cpp").read_text(encoding="utf-8") == "// ours\n"
    got = (dst / "src" / "poly" / "engine.cpp").read_bytes().decode()
    assert got == ("#include \"a.hpp\"\r\n#include \"mixtral.hpp\"\r\n#include \"gemma4.hpp\"\r\nint f() {\r\n"
                   "    return 1;\r\n    // after\r\n}\r\n")
    assert (src / "src" / "poly" / "engine.cpp").read_bytes() == ENGINE_CPP.encode()  # pin untouched
    assert any("edited src/poly/engine.cpp" in line for line in done)


def test_stage_overlay_refuses_a_changed_pin(tmp_path) -> None:
    """A depended-on engine file that differs from pin_files.json stops before anything is copied."""
    src = make_tree(tmp_path)
    d = make_ext(tmp_path, src)
    (src / "CMakeLists.txt").write_bytes(b"project(y)\n")
    assert ext.changed_pin_files(src, d) == ["CMakeLists.txt"]
    with pytest.raises(ext.OverlayError, match="CMakeLists.txt"):
        ext.stage_overlay(src, tmp_path / "ext-src", d)
    assert not (tmp_path / "ext-src").exists()


def test_stage_overlay_refuses_a_lost_anchor_before_copying(tmp_path) -> None:
    src = make_tree(tmp_path)
    d = make_ext(tmp_path, src)
    spec = json.loads((d / "overlay.json").read_text(encoding="utf-8"))
    spec["edits"][0]["anchor"] = "#include \"gone.hpp\""
    (d / "overlay.json").write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ext.OverlayError):
        ext.stage_overlay(src, tmp_path / "ext-src", d)
    assert not (tmp_path / "ext-src").exists()


def test_diff_pin_files(tmp_path) -> None:
    old = make_tree(tmp_path / "old")
    new = make_tree(tmp_path / "new")
    d = make_ext(tmp_path, old)
    assert ext.diff_pin_files(old, new, d) == ""
    (new / "CMakeLists.txt").write_bytes(b"project(z)\n")
    diff = ext.diff_pin_files(old, new, d)
    assert "-project(x)" in diff and "+project(z)" in diff and "engine.cpp" not in diff


def test_real_overlay_fits_the_pinned_tree() -> None:
    """The committed engine_ext matches the pinned tree: every hash, and every anchor exactly once."""
    if not (DEFAULT_SRC / "src" / "poly" / "engine.cpp").is_file():
        pytest.skip(f"pinned engine tree absent: {DEFAULT_SRC}")
    assert ext.changed_pin_files(DEFAULT_SRC) == []
    spec = ext.load_spec("overlay.json")
    for e in spec["edits"]:
        text = (DEFAULT_SRC / e["file"]).read_bytes().decode("utf-8")
        assert ext.apply_edit(text, e["anchor"], e["where"], e["lines"]) != text
    for ours in spec["copy"]:
        assert (ENGINE_EXT_DIR / ours).is_file(), ours


# ---------- build-engine --ext ----------

def _paths(root: pathlib.Path) -> EnginePaths:
    src = root / "engine"
    return EnginePaths(src=src, tools=src / "tools", build=root / "polystrata" / "build",
                       gguf_py=root / "gguf-py", scratch=root / "scratch")


def _fake_env(monkeypatch) -> None:
    monkeypatch.setattr(ens, "ensure_need", lambda need, paths, **kw: None)
    monkeypatch.setattr(probe_mod, "find_vcvars", lambda: pathlib.Path("C:/vs/VC/Auxiliary/Build/vcvars64.bat"))
    monkeypatch.setattr(probe_mod, "find_nvcc", lambda: (pathlib.Path("C:/cuda/bin/nvcc.exe"), (13, 0)))
    monkeypatch.setattr(build, "find_ninja", lambda v: pathlib.Path("C:/vs/ninja.exe"))
    monkeypatch.setattr(build, "gpu_arch", lambda: "86")


def test_ext_needs_gpu(tmp_path) -> None:
    err = io.StringIO()
    assert build.build_engine(_paths(tmp_path), ext=True, err=err) == 2
    assert "--gpu" in err.getvalue()


def test_ext_paths_and_target(tmp_path) -> None:
    p = ext_paths(_paths(tmp_path))
    assert p.src == tmp_path / "polystrata" / "ext-src" and p.build == tmp_path / "polystrata" / "build-ext"
    a = build.build_args(p, gpu=True, ext=True)
    assert a[a.index("--target") + 1:a.index("-j")] == [EXT_TARGET]


def test_ext_build_stages_then_builds_the_copy(monkeypatch, tmp_path) -> None:
    _fake_env(monkeypatch)
    staged: list[tuple] = []
    monkeypatch.setattr(ext, "stage_overlay", lambda src, dst: staged.append((src, dst)) or ["staged"])
    calls: list[list[str]] = []
    err = io.StringIO()
    paths = _paths(tmp_path)
    rc = build.build_engine(paths, gpu=True, ext=True, err=err, run=lambda cmd: calls.append(cmd) or 0)
    assert rc == 0 and staged == [(paths.src, ext_paths(paths).src)] and len(calls) == 1
    bat = (ext_paths(paths).build / "stratashift-build.bat").read_text(encoding="utf-8")
    assert str(ext_paths(paths).src) in bat and "--target strata-shift-poly" in bat
    assert "strata-gguf" not in bat and EXT_TARGET in err.getvalue()


def test_ext_build_refused_overlay_builds_nothing(monkeypatch, tmp_path) -> None:
    _fake_env(monkeypatch)

    def refuse(src, dst):
        raise ext.OverlayError("the engine tree differs from the pin: CMakeLists.txt")

    monkeypatch.setattr(ext, "stage_overlay", refuse)
    calls: list = []
    err = io.StringIO()
    rc = build.build_engine(_paths(tmp_path), gpu=True, ext=True, err=err, run=lambda cmd: calls.append(cmd) or 0)
    assert rc == 2 and calls == [] and "CMakeLists.txt" in err.getvalue()


def test_ext_dry_run_names_the_copy(monkeypatch, tmp_path) -> None:
    _fake_env(monkeypatch)
    monkeypatch.setattr(ext, "changed_pin_files", lambda src: [])
    err = io.StringIO()
    rc = build.build_engine(_paths(tmp_path), gpu=True, ext=True, dry_run=True, err=err)
    out = err.getvalue()
    assert rc == 0 and "ext-src" in out and "strata-shift-poly" in out


def test_cli_passes_ext(monkeypatch) -> None:
    import importlib
    cli = importlib.import_module("stratashift.cli.main")
    seen: list[dict] = []
    monkeypatch.setattr(cli, "build_engine", lambda paths, **kw: seen.append(kw) or 0)
    assert cli.main(["build-engine", "--gpu", "--ext", "--dry-run"]) == 0
    assert seen[0]["ext"] is True and seen[0]["gpu"] is True
