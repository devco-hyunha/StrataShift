#!/usr/bin/env python3
"""p6-01 AC4: build llama_ref (scripts/p6_llama_parity/llama_ref.cpp) on the pinned llama.cpp, CPU only.

The full pinned source is unpacked once from `.local/llama.cpp/llama.cpp-3cf03257.zip` (the engine CMake pin,
already on disk; nothing is downloaded) into `.local/llama.cpp/full/`, then CMake (Ninja, in a vcvars64 shell)
builds `.local/llama.cpp/parity-build/llama_ref.exe`. Windows + MSVC, the same tools as `stratashift build-engine`.

Usage: python scripts/p6_llama_parity/build.py   -> prints the executable's path, exit 0 when built.
"""
from __future__ import annotations

import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stratashift.install.build import find_ninja, run_build  # noqa: E402
from stratashift.install.probe import find_vcvars  # noqa: E402
from stratashift.polystrata.engine_paths import LLAMACPP_PIN, EnginePaths  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ZIP = ROOT / ".local" / "llama.cpp" / f"llama.cpp-{LLAMACPP_PIN[:8]}.zip"
FULL_ROOT = ROOT / ".local" / "llama.cpp" / "full"
FULL = FULL_ROOT / f"llama.cpp-{LLAMACPP_PIN}"
BUILD = ROOT / ".local" / "llama.cpp" / "parity-build"
EXE = BUILD / "llama_ref.exe"


def unpack() -> None:
    if (FULL / "src" / "models" / "llama.cpp").is_file():
        return
    root = FULL_ROOT.resolve()
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():  # no member may land outside the target folder
            if not str((root / name).resolve()).startswith(str(root)):
                raise ValueError(f"unsafe zip member: {name}")
        z.extractall(root)


def main() -> int:
    if not ZIP.is_file():
        print(f"error: pinned llama.cpp zip not found: {ZIP}", file=sys.stderr)
        return 2
    vcvars = find_vcvars()
    ninja = find_ninja(vcvars) if vcvars else None
    if vcvars is None or ninja is None:
        print("error: MSVC (vcvars64.bat) or Ninja not found (stratashift build-engine installs them)",
              file=sys.stderr)
        return 2
    unpack()
    steps = [["cmake", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(HERE), "-B", str(BUILD),
              "-DCMAKE_BUILD_TYPE=Release", f"-DLLAMACPP_FULL={FULL.as_posix()}"],
             ["cmake", "--build", str(BUILD), "--target", "llama_ref"]]
    paths = EnginePaths(src=FULL, tools=FULL, build=BUILD, gguf_py=FULL / "gguf-py", scratch=BUILD)
    if run_build(paths, vcvars, steps) != 0:
        print(f"error: llama_ref build failed (build dir {BUILD})", file=sys.stderr)
        return 1
    print(EXE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
