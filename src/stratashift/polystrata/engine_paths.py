"""Engine checkout / build / gguf-py / scratch paths — the single place they are decided (p6-09).

Order per path: ① CLI option → ② environment variable → ③ in-repo default under `.local/`
(GITIGNORE_POLICY, never committed). Nothing here opens or writes files except
`scratch_dir()`, which creates the scratch directory it returns.

Boxes that used the old fixed Linux paths set the three environment variables instead
(values in docs/USAGE.md "엔진 경로").
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
from dataclasses import dataclass

# src/stratashift/polystrata/engine_paths.py -> repository root (editable install, p5-01).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

ENV_SRC = "STRATASHIFT_ENGINE_SRC"
ENV_TOOLS = "STRATASHIFT_ENGINE_TOOLS"
ENV_BUILD = "STRATASHIFT_ENGINE_BUILD"
ENV_GGUF_PY = "STRATASHIFT_GGUF_PY"
ENV_SCRATCH = "STRATASHIFT_SCRATCH"

# pinned engine checkout (p0-01, docs/rea/binary_pin.md c566ff4) — read-only.
DEFAULT_SRC = REPO_ROOT / ".local" / "polystrata" / "PolyStrata-main"
DEFAULT_BUILD = REPO_ROOT / ".local" / "polystrata" / "build"
# pinned llama.cpp checkout fetched by the engine CMake build (Q2_0=42); the pip `gguf`
# lacks Q2_0 and is not the contract (tests/test_q8_0.py).
GGUF_PY_IN_BUILD = pathlib.Path("_deps") / "strata_llamacpp-src" / "gguf-py"
# same llama.cpp commit unpacked without an engine build: FetchContent GIT_TAG,
# engine CMakeLists.txt:1314 (pin c566ff4).
LLAMACPP_PIN = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMACPP_SRC = REPO_ROOT / ".local" / "llama.cpp" / f"llama.cpp-{LLAMACPP_PIN}"
DEFAULT_GGUF_PY_UNPACKED = LLAMACPP_SRC / "gguf-py"


@dataclass(frozen=True)
class EnginePaths:
    src: pathlib.Path
    tools: pathlib.Path
    build: pathlib.Path
    gguf_py: pathlib.Path
    scratch: pathlib.Path


def _env_path(name: str) -> pathlib.Path | None:
    value = os.environ.get(name)
    return pathlib.Path(value) if value else None


def _first_existing(*candidates: pathlib.Path) -> pathlib.Path:
    """First candidate that exists, else the first one (so messages name it)."""
    return next((c for c in candidates if c.exists()), candidates[0])


def resolve_engine_paths(engine_src: str | None = None) -> EnginePaths:
    """Decide every engine-related path. `engine_src` (CLI option) beats the environment;
    with it the tools dir is `<engine_src>/tools` even if STRATASHIFT_ENGINE_TOOLS is set."""
    if engine_src:
        src = pathlib.Path(engine_src)
        tools = src / "tools"
    else:
        src = _env_path(ENV_SRC) or DEFAULT_SRC
        tools = _env_path(ENV_TOOLS) or src / "tools"
    build = _env_path(ENV_BUILD) or DEFAULT_BUILD
    gguf_py = _env_path(ENV_GGUF_PY) or _first_existing(
        build / GGUF_PY_IN_BUILD, DEFAULT_GGUF_PY_UNPACKED)
    scratch = _env_path(ENV_SCRATCH) or pathlib.Path(tempfile.gettempdir()) / "stratashift"
    return EnginePaths(src=src, tools=tools, build=build, gguf_py=gguf_py, scratch=scratch)


def exe_name(name: str) -> str:
    return name + ".exe" if sys.platform == "win32" else name


def engine_binary(paths: EnginePaths, name: str) -> pathlib.Path:
    """Built engine executable: `<build>/<name>` then `<build>/Release/<name>` (MSVC
    multi-config generators). Returns the first candidate when none exists, so skip
    and error messages can name the path that was looked for."""
    return _first_existing(paths.build / exe_name(name), paths.build / "Release" / exe_name(name))


def scratch_dir(paths: EnginePaths | None = None) -> pathlib.Path:
    scratch = (paths or resolve_engine_paths()).scratch
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


def engine_tool_env(paths: EnginePaths | None = None) -> dict[str, str]:
    """Environment for running the engine's Python tools as subprocesses: the pinned gguf-py
    on PYTHONPATH (tools/canonical_xcheck.py:590 imports `gguf` directly) and as
    STRATA_GGUF_PY (tools/_paths.py `gguf_py`), pin c566ff4. The pip `gguf` lacks Q2_0."""
    gguf_py = str((paths or resolve_engine_paths()).gguf_py)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (gguf_py, env.get("PYTHONPATH")) if p)
    env["STRATA_GGUF_PY"] = gguf_py
    return env


def describe_missing(path: pathlib.Path, *env_names: str) -> str:
    """One line for skip reasons / refusals: which path was looked for, how to change it."""
    return f"not found: {path} (set {' or '.join(env_names)})"
