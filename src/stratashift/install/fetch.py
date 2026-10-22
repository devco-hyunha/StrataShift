"""Download the pinned engine source and check it by content digest (p6-12).

The recorded pin zip hash (docs/rea/binary_pin.md, 3ecc0e4c…) cannot be reused: that zip
was GitHub's `main` archive, and a commit archive names its top folder
`PolyStrata-<sha>`, so the file bytes always differ. The content digest ignores the top
folder: sha256 over "<path>\\0<sha256 of file>\\n" for every file, sorted by path.
Measured 2026-10-22 on the recorded pin zip AND the unpacked .local tree: equal.
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import tempfile
import urllib.request
import zipfile

ENGINE_PIN_COMMIT = "c566ff4787c8ab188ecf33c6f1907a9f90d9f152"  # docs/rea/binary_pin.md
ENGINE_TREE_SHA256 = "905d413e21aecce5f643eac307a818adea143cfa58a626788982087e35c605ae"
ENGINE_TREE_FILES = 3261
ENGINE_ZIP_URL = f"https://github.com/VecSzn/PolyStrata/archive/{ENGINE_PIN_COMMIT}.zip"
# pinned llama.cpp (engine CMakeLists.txt:1314, setup.py:181-182). Digest measured 2026-10-22 on
# .local/llama.cpp/llama.cpp-3cf03257.zip (a GitHub commit zip: its zip comment is the commit).
LLAMACPP_PIN_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
LLAMACPP_TREE_SHA256 = "87b1443f593e9b9159a61fa933e0fd4a2bc29e95f5ad6da422789d1c7cc515ae"
LLAMACPP_TREE_FILES = 3608
LLAMACPP_ZIP_URL = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMACPP_PIN_COMMIT}.zip"
CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


def download(url: str, dest: pathlib.Path, timeout: float = 60.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, CHUNK)


def _members(z: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    """(entry, path below the top folder) for every file; refuses paths that leave it."""
    out = []
    for info in z.infolist():
        if info.is_dir():
            continue
        parts = info.filename.split("/", 1)
        if len(parts) < 2 or not parts[1]:
            continue
        rel = pathlib.PurePosixPath(parts[1])
        if rel.is_absolute() or ".." in rel.parts or ":" in parts[1]:
            raise FetchError(f"zip entry leaves the target folder: {info.filename}")
        out.append((info, rel.as_posix()))
    return out


def zip_tree_digest(zip_path: pathlib.Path) -> tuple[str, int]:
    with zipfile.ZipFile(zip_path) as z:
        rows = sorted((rel, hashlib.sha256(z.read(info)).hexdigest()) for info, rel in _members(z))
    h = hashlib.sha256()
    for rel, digest in rows:
        h.update(f"{rel}\0{digest}\n".encode("utf-8"))
    return h.hexdigest(), len(rows)


def unpack(zip_path: pathlib.Path, dest: pathlib.Path) -> None:
    """Unpack below the top folder into `dest` via a sibling temp dir, then rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=dest.name + ".", dir=dest.parent))
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info, rel in _members(z):
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def install_engine_src(dest: pathlib.Path, url: str = ENGINE_ZIP_URL,
                       expected: str = ENGINE_TREE_SHA256) -> None:
    install_pinned_zip(dest, url, expected)


def install_llama_src(dest: pathlib.Path, url: str = LLAMACPP_ZIP_URL,
                      expected: str = LLAMACPP_TREE_SHA256) -> None:
    install_pinned_zip(dest, url, expected)


def install_pinned_zip(dest: pathlib.Path, url: str, expected: str) -> None:
    """Download, verify the content digest, unpack to `dest`. A mismatch deletes the download."""
    if dest.exists():
        raise FetchError(f"{dest} already exists; not overwriting")
    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / f"{dest.name}.download.zip"
    try:
        download(url, zip_path)
        digest, files = zip_tree_digest(zip_path)
        if digest != expected:
            raise FetchError(f"content digest mismatch for {url}: got {digest} ({files} files), "
                             f"pin expects {expected}; download deleted")
        unpack(zip_path, dest)
    finally:
        zip_path.unlink(missing_ok=True)
