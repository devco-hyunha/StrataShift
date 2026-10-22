"""p6-09 AC1: engine paths are decided in one place — option → environment → default."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

from stratashift.polystrata import engine_paths as ep

ENV_NAMES = (ep.ENV_SRC, ep.ENV_TOOLS, ep.ENV_BUILD, ep.ENV_GGUF_PY, ep.ENV_SCRATCH)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_live_under_repo_local(clean_env) -> None:
    p = ep.resolve_engine_paths()
    local = ep.REPO_ROOT / ".local" / "polystrata"
    assert p.src == local / "PolyStrata-main"
    assert p.tools == p.src / "tools"
    assert p.build == local / "build"
    assert p.gguf_py in (p.build / "_deps" / "strata_llamacpp-src" / "gguf-py",
                         ep.DEFAULT_GGUF_PY_UNPACKED)
    assert p.scratch.name == "stratashift"
    assert (ep.REPO_ROOT / "pyproject.toml").is_file()


def test_environment_beats_default(clean_env, tmp_path: pathlib.Path) -> None:
    clean_env.setenv(ep.ENV_SRC, str(tmp_path / "src"))
    clean_env.setenv(ep.ENV_BUILD, str(tmp_path / "build"))
    clean_env.setenv(ep.ENV_SCRATCH, str(tmp_path / "scratch"))
    p = ep.resolve_engine_paths()
    assert p.src == tmp_path / "src"
    assert p.tools == tmp_path / "src" / "tools"
    assert p.build == tmp_path / "build"
    assert p.gguf_py in (tmp_path / "build" / ep.GGUF_PY_IN_BUILD, ep.DEFAULT_GGUF_PY_UNPACKED)
    assert p.scratch == tmp_path / "scratch"


def test_tools_and_gguf_py_env_are_specific(clean_env, tmp_path: pathlib.Path) -> None:
    clean_env.setenv(ep.ENV_SRC, str(tmp_path / "src"))
    clean_env.setenv(ep.ENV_TOOLS, str(tmp_path / "tools"))
    clean_env.setenv(ep.ENV_GGUF_PY, str(tmp_path / "gguf-py"))
    p = ep.resolve_engine_paths()
    assert p.tools == tmp_path / "tools"
    assert p.gguf_py == tmp_path / "gguf-py"


def test_option_beats_environment(clean_env, tmp_path: pathlib.Path) -> None:
    clean_env.setenv(ep.ENV_SRC, str(tmp_path / "env-src"))
    clean_env.setenv(ep.ENV_TOOLS, str(tmp_path / "env-tools"))
    p = ep.resolve_engine_paths(str(tmp_path / "opt-src"))
    assert p.src == tmp_path / "opt-src"
    assert p.tools == tmp_path / "opt-src" / "tools"


@pytest.mark.parametrize("platform, expected", [("win32", "strata-gguf.exe"),
                                                ("linux", "strata-gguf")])
def test_exe_name(monkeypatch: pytest.MonkeyPatch, platform: str, expected: str) -> None:
    monkeypatch.setattr(ep.sys, "platform", platform)
    assert ep.exe_name("strata-gguf") == expected


def test_engine_binary_finds_release_subdir(clean_env, tmp_path: pathlib.Path) -> None:
    clean_env.setenv(ep.ENV_BUILD, str(tmp_path))
    p = ep.resolve_engine_paths()
    missing = ep.engine_binary(p, "strata-gguf")
    assert missing == tmp_path / ep.exe_name("strata-gguf")
    release = tmp_path / "Release" / ep.exe_name("strata-gguf")
    release.parent.mkdir()
    release.write_bytes(b"")
    assert ep.engine_binary(p, "strata-gguf") == release
    top = tmp_path / ep.exe_name("strata-gguf")
    top.write_bytes(b"")
    assert ep.engine_binary(p, "strata-gguf") == top


def test_scratch_dir_is_created(clean_env, tmp_path: pathlib.Path) -> None:
    clean_env.setenv(ep.ENV_SCRATCH, str(tmp_path / "a" / "b"))
    assert ep.scratch_dir() == tmp_path / "a" / "b"
    assert (tmp_path / "a" / "b").is_dir()


def test_describe_missing_names_path_and_env() -> None:
    msg = ep.describe_missing(pathlib.Path("x") / "y", ep.ENV_BUILD, ep.ENV_SRC)
    assert str(pathlib.Path("x") / "y") in msg
    assert ep.ENV_BUILD in msg and ep.ENV_SRC in msg


def test_cli_engine_src_option_reaches_refusal(tmp_path: pathlib.Path) -> None:
    """An --engine-src without tools/ refuses emit with rc 2 and names the path looked for.
    Runs in a subprocess: the engine import is once per process (first load wins)."""
    st = tmp_path / "m.safetensors"
    st.write_bytes(len(b"{}").to_bytes(8, "little") + b"{}")
    r = subprocess.run(
        [sys.executable, "-c", "import sys; from stratashift.cli.main import main; "
         "sys.exit(main(sys.argv[1:]))", "convert", str(st), "--out", str(tmp_path / "o"),
         "--emit-gguf", str(tmp_path / "o.gguf"), "--engine-src", str(tmp_path / "nowhere")],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ep.REPO_ROOT / "src")},
    )
    assert r.returncode == 2, r.stderr
    assert str(tmp_path / "nowhere" / "tools") in r.stderr
    assert "--engine-src" in r.stderr


def test_gguf_py_prefers_build_tree(clean_env, tmp_path: pathlib.Path) -> None:
    clean_env.setenv(ep.ENV_BUILD, str(tmp_path))
    in_build = tmp_path / ep.GGUF_PY_IN_BUILD
    in_build.mkdir(parents=True)
    assert ep.resolve_engine_paths().gguf_py == in_build
