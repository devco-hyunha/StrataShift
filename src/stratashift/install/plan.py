"""Install items, the needs that group them, and the plan text (pure, no I/O, p6-12)."""
from __future__ import annotations

from dataclasses import dataclass

# winget arguments, engine setup.py:4048 (pin c566ff4)
WINGET_BASE = ("install", "-e", "--source", "winget", "--accept-package-agreements",
               "--accept-source-agreements", "--disable-interactivity")


@dataclass(frozen=True)
class Item:
    key: str
    label: str
    size: str
    duration: str
    admin: bool
    winget: tuple[str, ...] = ()  # args after WINGET_BASE; empty = not a winget item
    manual: str = ""  # how to install it by hand


ENGINE_SRC = Item(
    key="engine-src", label="PolyStrata engine source (pin c566ff4)", size="about 25 MB",
    duration="about 1 minute", admin=False,
    manual="download https://github.com/VecSzn/PolyStrata/archive/"
           "c566ff4787c8ab188ecf33c6f1907a9f90d9f152.zip and unpack it as "
           ".local/polystrata/PolyStrata-main (backup: https://huggingface.co/spaces/VecSzn/PolyStrata, "
           "usable only if its content digest matches the pin)")
CMAKE = Item(
    key="cmake", label="CMake", size="about 40 MB", duration="a few minutes", admin=True,
    winget=("--id", "Kitware.CMake"), manual="winget install -e --id Kitware.CMake")
# MSVC + CUDA together: "about 8-10 GB, 15-40 minutes, Windows will ask for permission"
# (engine setup.py:4041-4043)
MSVC = Item(
    key="msvc", label="Visual Studio 2022 Build Tools (C++)", size="about 8-10 GB with CUDA",
    duration="15-40 minutes with CUDA", admin=True,
    winget=("--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
            "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools "
            "--includeRecommended"),  # engine setup.py:4049-4052
    manual="Visual Studio 2022 Build Tools with the 'Desktop development with C++' workload")
CUDA = Item(
    key="cuda", label="NVIDIA CUDA Toolkit 13.0", size="about 8-10 GB with MSVC",
    duration="15-40 minutes with MSVC", admin=True,
    winget=("--id", "Nvidia.CUDA", "--version", "13.0"),  # engine setup.py:4054
    manual="https://developer.nvidia.com/cuda-downloads (13.0)")

NEEDS: dict[str, tuple[Item, ...]] = {
    "emit": (ENGINE_SRC,),  # convert --emit-gguf: engine tools/gguf_writer.py
    "engine-build": (CMAKE, MSVC, CUDA),  # engine executables (wired to a command in 3e)
}


def missing_items(need: str, present: dict[str, bool]) -> list[Item]:
    """Items of `need` that are not present; present ones are never reinstalled."""
    return [item for item in NEEDS[need] if not present.get(item.key, False)]


def describe_plan(items: list[Item]) -> str:
    lines = ["missing prerequisites:"]
    for item in items:
        admin = "Windows will ask for permission" if item.admin else "no admin rights"
        lines.append(f"  - {item.label}: {item.size}, {item.duration}, {admin}")
    return "\n".join(lines)


def winget_command(item: Item) -> list[str]:
    return ["winget", *WINGET_BASE, *item.winget]
