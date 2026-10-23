"""Build the pinned engine executables with MSVC + CMake + Ninja (p6-13, `build-engine`).

Pattern: engine setup.py `cmake_build` :4096-4122 (pin c566ff4, read-only) — a .bat calls
vcvars64.bat, then CMake configure (Ninja) and build, retried once on failure. CPU-only
targets need no CUDA (engine CMakeLists.txt:245-250); strata-poly needs CUDA on (:1779).

`--ext` (p6-01): the same configure on a copy of the pinned tree with engine_ext/ put in
(install/engine_ext.py), building only strata-shift-poly in its own build folder.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
from typing import Callable

from stratashift.polystrata.engine_paths import EXT_TARGET, LLAMACPP_SRC, EnginePaths, ext_paths

CPU_TARGETS = ("strata-gguf", "strata-dequant")
GPU_TARGETS = ("strata-poly",)
EXT_TARGETS = (EXT_TARGET,)


def find_ninja(vcvars: pathlib.Path) -> pathlib.Path | None:
    """VS-bundled Ninja next to the vcvars install, else PATH."""
    vs_root = vcvars.parents[3]  # <vs>/VC/Auxiliary/Build/vcvars64.bat
    bundled = (vs_root / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake"
               / "Ninja" / "ninja.exe")
    if bundled.is_file():
        return bundled
    found = shutil.which("ninja")
    return pathlib.Path(found) if found else None


def gpu_arch() -> str | None:
    """`nvidia-smi` compute capability of the first GPU, e.g. "8.6" -> "86"."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.match(r"\s*(\d+)\.(\d+)", r.stdout)
    return f"{m.group(1)}{m.group(2)}" if m else None


def configure_args(paths: EnginePaths, ninja: pathlib.Path, gpu: bool,
                   arch: str | None = None, nvcc: pathlib.Path | None = None) -> list[str]:
    args = ["cmake", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
            "-S", str(paths.src), "-B", str(paths.build), "-DCMAKE_BUILD_TYPE=Release",
            "-DSTRATA_BUILD_TESTS=OFF",  # CMakeLists.txt:98-102
            f"-DSTRATA_GGML_DIR={LLAMACPP_SRC.as_posix()}"]  # CMakeLists.txt:1308-1309
    if gpu:
        # same switches as engine setup.py build_engine_poly (:5857-5918)
        args += ["-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                 f"-DCMAKE_CUDA_COMPILER={pathlib.Path(nvcc).as_posix()}"]
    else:
        args.append("-DSTRATA_ENABLE_CUDA=OFF")
    return args


def targets_of(gpu: bool, ext: bool = False) -> tuple[str, ...]:
    if ext:
        return EXT_TARGETS
    return CPU_TARGETS + (GPU_TARGETS if gpu else ())


def build_args(paths: EnginePaths, gpu: bool, ext: bool = False) -> list[str]:
    return ["cmake", "--build", str(paths.build), "--target", *targets_of(gpu, ext),
            "-j", str(max(2, (os.cpu_count() or 4) // 2))]  # setup.py:4113


def write_bat(bat: pathlib.Path, vcvars: pathlib.Path, steps: list[list[str]]) -> None:
    lines = ["@echo off", f'call "{vcvars}" >nul || exit /b 1']
    lines += [subprocess.list2cmdline(step) + " || exit /b 2" for step in steps]
    # bytes: text mode on Windows would turn each \r\n into \r\r\n (p6-13 review)
    bat.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))


def run_build(paths: EnginePaths, vcvars: pathlib.Path, steps: list[list[str]],
              run: Callable[[list[str]], int] | None = None) -> int:
    """Run configure+build in a vcvars shell; one retry on failure (setup.py:4114-4116)."""
    run = run or (lambda cmd: subprocess.run(cmd, check=False).returncode)
    paths.build.mkdir(parents=True, exist_ok=True)
    bat = paths.build / "stratashift-build.bat"
    write_bat(bat, vcvars, steps)
    rc = run(["cmd", "/c", str(bat)])
    return rc if rc == 0 else run(["cmd", "/c", str(bat)])


def build_engine(paths: EnginePaths, *, gpu: bool = False, arch: str | None = None,
                 yes: bool = False, dry_run: bool = False, ext: bool = False, err=None,
                 run: Callable[[list[str]], int] | None = None) -> int:
    """`stratashift build-engine`: ensure prerequisites (asks once), then configure + build.
    With `ext`, the pinned tree is copied with engine_ext/ put in and only strata-shift-poly
    is built there. 0 built (or dry run), 1 build failed, 2 refused / prerequisite missing."""
    import sys

    from stratashift.install.engine_ext import OverlayError, changed_pin_files, stage_overlay
    from stratashift.install.ensure import ensure_need
    from stratashift.install.probe import find_nvcc, find_vcvars
    from stratashift.polystrata.engine_paths import engine_binary

    err = err or sys.stderr
    if ext and not gpu:
        print("error: --ext builds strata-shift-poly, a CUDA executable: add --gpu", file=err)
        return 2
    need = "engine-gpu" if gpu else "engine-cpu"
    rc = ensure_need(need, paths, yes=yes, dry_run=dry_run, err=err)
    if rc is not None and not (dry_run and rc == 0):
        return rc
    vcvars = find_vcvars()
    ninja = find_ninja(vcvars) if vcvars else None
    if dry_run and (vcvars is None or ninja is None):
        print("dry run: the build commands are shown once MSVC and Ninja are installed", file=err)
        return 0
    if vcvars is None or ninja is None:
        print(f"error: MSVC (vcvars64.bat) or Ninja not found (vcvars={vcvars}, ninja={ninja})",
              file=err)
        return 2
    nvcc = None
    if gpu:
        arch = arch or gpu_arch()
        if not arch:
            print("error: cannot read the GPU compute capability (nvidia-smi); pass --cuda-arch "
                  "(e.g. 86 for RTX 30xx)", file=err)
            return 2
        found = find_nvcc()
        if found is None and not dry_run:
            print("error: nvcc (CUDA Toolkit) not found", file=err)
            return 2
        nvcc = found[0] if found else pathlib.Path("<nvcc after install>")
    target_paths = ext_paths(paths) if ext else paths
    steps = [configure_args(target_paths, ninja, gpu, arch, nvcc), build_args(target_paths, gpu, ext)]
    if dry_run:
        if ext:
            changed = changed_pin_files(paths.src)
            print(f"dry run: would copy {paths.src} to {target_paths.src} with engine_ext/ put in"
                  + (f" — refused now, the pin files differ: {', '.join(changed)}" if changed else ""),
                  file=err)
        print("dry run: would run (in a vcvars64 shell):", file=err)
        for step in steps:
            print("  " + subprocess.list2cmdline(step), file=err)
        return 0
    if ext:
        try:
            for line in stage_overlay(paths.src, target_paths.src):
                print(line, file=err)
        except OverlayError as e:
            print(f"error: {e}", file=err)
            return 2
    if run_build(target_paths, vcvars, steps, run) != 0:
        print(f"error: engine build failed (see the output above; build dir {target_paths.build})",
              file=err)
        return 1
    for name in targets_of(gpu, ext):
        print(f"built {engine_binary(target_paths, name)}", file=err)
    return 0
