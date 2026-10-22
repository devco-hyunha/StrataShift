"""p6-13 AC1: `build-engine` command assembly, prerequisites, dry run, retry (all fakes).

No network, winget, cmake or compiler runs here: subprocess, probe and installers are replaced.
"""
from __future__ import annotations

import importlib
import io
import os
import pathlib
import subprocess

from stratashift.install import build, ensure as ens
from stratashift.install import probe as probe_mod
from stratashift.polystrata import engine_paths
from stratashift.polystrata.engine_paths import EnginePaths


def make_paths(root: pathlib.Path) -> EnginePaths:
    src = root / "engine"
    return EnginePaths(src=src, tools=src / "tools", build=root / "build",
                       gguf_py=root / "gguf-py", scratch=root / "scratch")


NINJA = pathlib.Path("C:/vs/ninja.exe")
VCVARS = pathlib.Path("C:/vs/VC/Auxiliary/Build/vcvars64.bat")


# ---------- command assembly ----------

def test_configure_args_cpu(tmp_path) -> None:
    """p6-13 AC1: CPU configure turns CUDA off, tests off, points at the pinned ggml."""
    args = build.configure_args(make_paths(tmp_path), NINJA, gpu=False)
    assert "-DSTRATA_ENABLE_CUDA=OFF" in args and "-DSTRATA_BUILD_TESTS=OFF" in args
    assert any(a.startswith("-DSTRATA_GGML_DIR=") for a in args)
    assert not any("CUDA_ARCHITECTURES" in a or "CUDA_COMPILER" in a for a in args)
    assert "-DSTRATA_ENABLE_CUDA=ON" not in args


def test_configure_args_gpu(tmp_path) -> None:
    """p6-13 AC1: GPU configure turns CUDA on with arch and compiler."""
    nvcc = pathlib.Path("C:/cuda/bin/nvcc.exe")
    args = build.configure_args(make_paths(tmp_path), NINJA, gpu=True, arch="86", nvcc=nvcc)
    assert "-DSTRATA_ENABLE_CUDA=ON" in args and "-DCMAKE_CUDA_ARCHITECTURES=86" in args
    assert any(a.startswith("-DCMAKE_CUDA_COMPILER=") for a in args)
    assert "-DSTRATA_ENABLE_CUDA=OFF" not in args


def test_build_args_targets(tmp_path) -> None:
    """p6-13 AC1: CPU builds two targets, --gpu adds strata-poly."""
    paths = make_paths(tmp_path)

    def targets(gpu: bool) -> list[str]:
        a = build.build_args(paths, gpu)
        return a[a.index("--target") + 1:a.index("-j")]

    assert targets(False) == ["strata-gguf", "strata-dequant"]
    assert targets(True) == ["strata-gguf", "strata-dequant", "strata-poly"]


# ---------- gpu_arch ----------

def _fake_smi(monkeypatch, stdout: str | None) -> None:
    def run(cmd, **kw):
        if stdout is None:
            raise OSError("no nvidia-smi")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    monkeypatch.setattr(build.subprocess, "run", run)


def test_gpu_arch(monkeypatch) -> None:
    """p6-13 AC1: compute capability 8.6 -> 86; empty output or missing tool -> None."""
    _fake_smi(monkeypatch, "8.6\n")
    assert build.gpu_arch() == "86"
    _fake_smi(monkeypatch, "")
    assert build.gpu_arch() is None
    _fake_smi(monkeypatch, None)
    assert build.gpu_arch() is None


# ---------- find_ninja ----------

def test_find_ninja_bundled(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: the Ninja bundled with Visual Studio is preferred."""
    vcvars = tmp_path / "VS" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    ninja = (tmp_path / "VS" / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake"
             / "Ninja" / "ninja.exe")
    ninja.parent.mkdir(parents=True)
    ninja.write_bytes(b"")
    monkeypatch.setattr(build.shutil, "which", lambda name: "C:/path/ninja.exe")
    assert build.find_ninja(vcvars) == ninja


def test_find_ninja_none(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: no bundled Ninja and none on PATH -> None; PATH is used when present."""
    vcvars = tmp_path / "VS" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    assert build.find_ninja(vcvars) is None
    monkeypatch.setattr(build.shutil, "which", lambda name: "C:/path/ninja.exe")
    assert build.find_ninja(vcvars) == pathlib.Path("C:/path/ninja.exe")


# ---------- write_bat / run_build ----------

def test_write_bat(tmp_path) -> None:
    """p6-13 AC1: .bat starts quiet, calls vcvars, guards each step, uses CRLF."""
    bat = tmp_path / "b.bat"
    build.write_bat(bat, VCVARS, [["cmake", "-S", "a b"], ["cmake", "--build", "x"]])
    raw = bat.read_bytes().decode("utf-8")
    assert "\r\n" in raw and raw.count("\n") == raw.count("\r\n")
    lines = raw.splitlines()
    assert lines[0] == "@echo off"
    assert lines[1].startswith("call ") and "vcvars64.bat" in lines[1]
    assert len(lines) == 4
    assert all(line.endswith("|| exit /b 2") for line in lines[2:])


def _runner(codes: list[int]):
    calls: list[list[str]] = []

    def run(cmd: list[str]) -> int:
        calls.append(cmd)
        return codes[min(len(calls) - 1, len(codes) - 1)]

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_run_build_retries_once(tmp_path) -> None:
    """p6-13 AC1: first run fails, second passes -> 0 after two runs; bat is in the build dir."""
    paths = make_paths(tmp_path)
    run = _runner([1, 0])
    assert build.run_build(paths, VCVARS, [["cmake"]], run) == 0
    assert len(run.calls) == 2
    assert (paths.build / "stratashift-build.bat").is_file()


def test_run_build_gives_up_after_one_retry(tmp_path) -> None:
    """p6-13 AC1: failing twice -> 1 after exactly two runs."""
    run = _runner([1])
    assert build.run_build(make_paths(tmp_path), VCVARS, [["cmake"]], run) == 1
    assert len(run.calls) == 2


def test_run_build_success_runs_once(tmp_path) -> None:
    """p6-13 AC1: success on the first run does not repeat."""
    run = _runner([0])
    assert build.run_build(make_paths(tmp_path), VCVARS, [["cmake"]], run) == 0
    assert len(run.calls) == 1


# ---------- build_engine ----------

def _env(monkeypatch, *, need_rc=None, vcvars=VCVARS, ninja=NINJA, arch="86") -> list[str]:
    """Fake ensure/probe/ninja/arch. Returns the list of needs ensure_need was asked for."""
    needs: list[str] = []

    def fake_ensure(need, paths, **kw):
        needs.append(need)
        return need_rc

    monkeypatch.setattr(ens, "ensure_need", fake_ensure)
    monkeypatch.setattr(probe_mod, "find_vcvars", lambda: vcvars)
    monkeypatch.setattr(probe_mod, "find_nvcc",
                        lambda: (pathlib.Path("C:/cuda/bin/nvcc.exe"), (13, 0)))
    monkeypatch.setattr(build, "find_ninja", lambda v: ninja)
    monkeypatch.setattr(build, "gpu_arch", lambda: arch)
    return needs


def _call(tmp_path, run=None, **kw):
    err = io.StringIO()
    rc = build.build_engine(make_paths(tmp_path), err=err, run=run, **kw)
    return rc, err.getvalue()


def test_build_engine_returns_ensure_code(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: a numeric ensure_need result is returned as is, nothing is built."""
    needs = _env(monkeypatch, need_rc=2)
    run = _runner([0])
    assert _call(tmp_path, run)[0] == 2 and run.calls == []
    assert needs == ["engine-cpu"]


def test_build_engine_need_follows_gpu(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: --gpu asks for the engine-gpu bundle."""
    needs = _env(monkeypatch, need_rc=0)
    _call(tmp_path, _runner([0]), gpu=True)
    assert needs == ["engine-gpu"]


def test_build_engine_dry_run(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: dry run prints the cmake commands and runs nothing."""
    _env(monkeypatch)
    run = _runner([0])
    rc, out = _call(tmp_path, run, dry_run=True)
    assert rc == 0 and "cmake -G Ninja" in out and run.calls == []


def test_build_engine_gpu_without_arch(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: --gpu with unreadable compute capability -> 2 and a --cuda-arch hint."""
    _env(monkeypatch, arch=None)
    run = _runner([0])
    rc, out = _call(tmp_path, run, gpu=True)
    assert rc == 2 and "--cuda-arch" in out and run.calls == []


def test_build_engine_gpu_explicit_arch(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: --cuda-arch is used even when nvidia-smi cannot answer."""
    _env(monkeypatch, arch=None)
    rc, out = _call(tmp_path, _runner([0]), gpu=True, arch="75", dry_run=True)
    assert rc == 0 and "CMAKE_CUDA_ARCHITECTURES=75" in out and "strata-poly" in out


def test_build_engine_failure(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: build fails twice -> 1."""
    _env(monkeypatch)
    run = _runner([1, 1])
    rc, out = _call(tmp_path, run)
    assert rc == 1 and len(run.calls) == 2 and "failed" in out


def test_build_engine_success(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: success -> 0 and a 'built' line per target."""
    _env(monkeypatch)
    rc, out = _call(tmp_path, _runner([0]))
    assert rc == 0 and out.count("built ") == 2
    assert "strata-gguf" in out and "strata-dequant" in out


def test_build_engine_no_vcvars(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: no MSVC found after ensure -> 2, nothing run."""
    _env(monkeypatch, vcvars=None)
    run = _runner([0])
    assert _call(tmp_path, run)[0] == 2 and run.calls == []


# ---------- CLI / env ----------

def test_cli_build_engine_passes_options(monkeypatch) -> None:
    """p6-13 AC1: `build-engine --gpu --cuda-arch 86 --dry-run` reaches build_engine."""
    cli = importlib.import_module("stratashift.cli.main")
    seen: list[dict] = []

    def fake(paths, **kw):
        seen.append(kw)
        return 0

    monkeypatch.setattr(cli, "build_engine", fake)
    assert cli.main(["build-engine", "--gpu", "--cuda-arch", "86", "--dry-run"]) == 0
    assert len(seen) == 1
    assert seen[0]["gpu"] is True and seen[0]["arch"] == "86" and seen[0]["dry_run"] is True


def test_engine_tool_env(monkeypatch, tmp_path) -> None:
    """p6-13 AC1: gguf_py leads PYTHONPATH, is exported as STRATA_GGUF_PY, old value is kept."""
    paths = make_paths(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "keep_me")
    env = engine_paths.engine_tool_env(paths)
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(paths.gguf_py) and "keep_me" in parts[1:]
    assert env["STRATA_GGUF_PY"] == str(paths.gguf_py)
