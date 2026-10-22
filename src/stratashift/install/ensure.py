"""Probe → plan → ask once → install → probe again, for one need (p6-12).

Flow modelled on engine setup.py `install_build_tools` (:4007-4093, pin c566ff4): one
question for everything missing, `--yes` skips it, the exact winget command is printed
before it runs, and success is judged by probing again, not by winget's exit code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Callable, TextIO

from stratashift.install import fetch
from stratashift.install.plan import (
    ENGINE_SRC, LLAMA_SRC, NEEDS, Item, describe_plan, missing_items, winget_command)
from stratashift.install.probe import probe
from stratashift.polystrata.engine_paths import (
    DEFAULT_SRC, ENV_SRC, ENV_TOOLS, LLAMACPP_SRC, EnginePaths)

ENV_YES = "STRATASHIFT_YES"
MANUAL_DOC = "docs/USAGE.md, section '자동 설치'"
REFUSED = 2


def _say(err: TextIO, msg: str) -> None:
    print(msg, file=err, flush=True)


def _manual(items: list[Item]) -> str:
    return "\n".join(f"  - {item.label}: {item.manual}" for item in items)


def _ask(stdin: TextIO, err: TextIO) -> bool:
    if not stdin.isatty():
        _say(err, f"error: cannot ask (input is not a terminal); pass --yes or set {ENV_YES}=1")
        return False
    _say(err, "Install them now? [y/N] ")
    return stdin.readline().strip().lower() in ("y", "yes")


def _install(item: Item, paths: EnginePaths, err: TextIO,
             run: Callable[[list[str]], object]) -> None:
    if item is ENGINE_SRC:
        _say(err, f"> download {fetch.ENGINE_ZIP_URL} -> {paths.src}")
        fetch.install_engine_src(paths.src)
        return
    if item is LLAMA_SRC:
        _say(err, f"> download {fetch.LLAMACPP_ZIP_URL} -> {LLAMACPP_SRC}")
        fetch.install_llama_src(LLAMACPP_SRC)
        return
    cmd = winget_command(item)
    _say(err, "> " + subprocess.list2cmdline(cmd))
    run(cmd)


def ensure_need(need: str, paths: EnginePaths, *, yes: bool = False, dry_run: bool = False,
                stdin: TextIO | None = None, err: TextIO | None = None,
                platform: str | None = None,
                run: Callable[[list[str]], object] | None = None) -> int | None:
    """None when everything `need` requires is present (the command goes on);
    otherwise the exit code to stop with: 0 after a dry run, 2 when refused or failed."""
    stdin = stdin or sys.stdin
    err = err or sys.stderr
    platform = platform or sys.platform
    run = run or (lambda cmd: subprocess.run(cmd, check=False))
    yes = yes or os.environ.get(ENV_YES) == "1"

    missing = missing_items(need, probe([item.key for item in NEEDS[need]], paths))
    if not missing:
        return None
    _say(err, describe_plan(missing))
    if dry_run:
        _say(err, "dry run: nothing installed")
        return 0
    # a user-given path is reported on every OS (checked before the Windows-only gate:
    # p6-13 WSL run found the Linux message dropped the looked-for path)
    if ENGINE_SRC in missing and (paths.src, paths.tools) != (DEFAULT_SRC, DEFAULT_SRC / "tools"):
        _say(err, f"error: engine tools not found at {paths.tools / 'gguf_writer.py'}; the path "
                  f"was given (--engine-src / {ENV_SRC} / {ENV_TOOLS}), so nothing is installed "
                  f"into it. Fix the path, or unset it to use the default {DEFAULT_SRC}")
        return REFUSED
    if platform != "win32":
        _say(err, f"error: automatic install is Windows-only; install by hand:\n{_manual(missing)}")
        return REFUSED
    if any(item.winget for item in missing) and shutil.which("winget") is None:
        _say(err, "error: winget (Windows package manager) is not available; install "
                  "'App Installer' from the Microsoft Store, or install by hand:\n"
                  + _manual(missing))
        return REFUSED
    if not yes and not _ask(stdin, err):
        _say(err, f"not installed; the command needs them. Manual steps: {MANUAL_DOC}")
        return REFUSED
    for item in missing:
        try:
            _install(item, paths, err, run)
        except (OSError, fetch.FetchError) as e:
            _say(err, f"error: {item.label} did not install: {e}\n{_manual([item])}")
            return REFUSED
    # re-probe only what was missing; items present before stay present (p6-12 review)
    after = probe([item.key for item in missing], paths)
    still = [item for item in missing if not after[item.key]]
    if still:
        _say(err, "error: still missing after install:\n" + _manual(still)
             + f"\nIf it was just installed, open a new terminal (PATH) and run again. "
               f"Manual steps: {MANUAL_DOC}")
        return REFUSED
    _say(err, "prerequisites installed")
    return None
