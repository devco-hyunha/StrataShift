"""Is each install item already present? (p6-12)

Search order follows the pinned engine setup.py (pin c566ff4, read-only): `find_nvcc`
:1759-1793, `find_vcvars` :1796-1834. Known install folders are searched besides PATH
because PATH is not refreshed in the running shell right after a winget install.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

from stratashift.install.plan import CMAKE, CUDA, ENGINE_SRC, LLAMA_SRC, MSVC
from stratashift.polystrata.engine_paths import EnginePaths, LLAMACPP_SRC

MIN_CUDA = (12, 0)  # engine setup.py:4028 — 12.0 floor for cards other than sm_120


def _program_files(var: str, default: str) -> pathlib.Path:
    return pathlib.Path(os.environ.get(var, default))


def find_cmake() -> pathlib.Path | None:
    found = shutil.which("cmake")
    if found:
        return pathlib.Path(found)
    candidate = _program_files("ProgramFiles", r"C:\Program Files") / "CMake" / "bin" / "cmake.exe"
    return candidate if candidate.is_file() else None


def find_vcvars() -> pathlib.Path | None:
    """vcvars64.bat of the newest VS with the x64 C++ tools (setup.py:1812-1819)."""
    vswhere = (_program_files("ProgramFiles(x86)", r"C:\Program Files (x86)")
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.is_file():
        return None
    r = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property", "installationPath"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in r.stdout.splitlines():
        bat = pathlib.Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if line.strip() and bat.is_file():
            return bat
    return None


def nvcc_version(nvcc: pathlib.Path) -> tuple[int, int] | None:
    try:
        r = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = re.search(r"release (\d+)\.(\d+)", r.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_nvcc() -> tuple[pathlib.Path, tuple[int, int]] | None:
    """Newest nvcc among PATH, CUDA_PATH/CUDA_HOME, default toolkit folders (setup.py:1759-1793)."""
    candidates: list[pathlib.Path] = []
    if shutil.which("nvcc"):
        candidates.append(pathlib.Path(shutil.which("nvcc")))
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            candidates += [pathlib.Path(os.environ[var]) / "bin" / n for n in ("nvcc.exe", "nvcc")]
    root = (_program_files("ProgramFiles", r"C:\Program Files")
            / "NVIDIA GPU Computing Toolkit" / "CUDA")
    if root.is_dir():
        candidates += sorted(root.glob("*/bin/nvcc.exe"))
    found = [(c, v) for c in candidates if c.is_file() and (v := nvcc_version(c))]
    return max(found, key=lambda cv: cv[1]) if found else None


def engine_src_present(paths: EnginePaths) -> bool:
    return (paths.tools / "gguf_writer.py").is_file()


def probe(need_keys: list[str], paths: EnginePaths) -> dict[str, bool]:
    """Present/absent per item key — only the keys asked for are probed."""
    checks = {
        ENGINE_SRC.key: lambda: engine_src_present(paths),
        LLAMA_SRC.key: lambda: (LLAMACPP_SRC / "ggml" / "CMakeLists.txt").is_file(),
        CMAKE.key: lambda: find_cmake() is not None,
        MSVC.key: lambda: find_vcvars() is not None,
        CUDA.key: lambda: (found := find_nvcc()) is not None and found[1] >= MIN_CUDA,
    }
    return {key: checks[key]() for key in need_keys}
