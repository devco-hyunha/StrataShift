"""p6-12: automatic install of missing prerequisites (plan, ask, fetch, probe, CLI).

Nothing here touches the network or runs winget: probe, winget, download and the
installer are replaced by fakes.
"""
from __future__ import annotations

import hashlib
import importlib
import io
import pathlib
import shutil
import subprocess
import zipfile

import pytest

from stratashift.install import ensure as ens
from stratashift.install import fetch, plan
from stratashift.install import probe as probe_mod
from stratashift.polystrata.engine_paths import EnginePaths

REPO = pathlib.Path(__file__).resolve().parent.parent
PIN_ZIP = REPO / ".local" / "polystrata" / "polystrata-main-c566ff4.zip"


# ---------- helpers ----------

class FakeStdin:
    def __init__(self, tty: bool, answer: str = "") -> None:
        self.tty = tty
        self.answer = answer
        self.reads = 0

    def isatty(self) -> bool:
        return self.tty

    def readline(self) -> str:
        self.reads += 1
        return self.answer + "\n"


class Runner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> int:
        self.calls.append(cmd)
        return 0


def make_paths(root: pathlib.Path) -> EnginePaths:
    src = root / "engine"
    return EnginePaths(src=src, tools=src / "tools", build=root / "build",
                       gguf_py=root / "gguf-py", scratch=root / "scratch")


@pytest.fixture(autouse=True)
def clean_yes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ens.ENV_YES, raising=False)


def set_probe(monkeypatch: pytest.MonkeyPatch,
              present: dict[str, bool] | list[dict[str, bool]]) -> None:
    """probe() answers `present`; given a list, one entry per call (the last one repeats)."""
    seq = present if isinstance(present, list) else [present]
    state = {"i": 0}

    def fake(keys: list[str], paths: EnginePaths) -> dict[str, bool]:
        cur = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return {k: cur.get(k, False) for k in keys}

    monkeypatch.setattr(ens, "probe", fake)


def set_winget(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(ens.shutil, "which", lambda name: "C:/winget.exe" if available else None)


def run_ensure(paths: EnginePaths, need: str = "engine-build", *, tty: bool = False,
               answer: str = "", platform: str = "win32", **kw):
    err, stdin, runner = io.StringIO(), FakeStdin(tty, answer), Runner()
    rc = ens.ensure_need(need, paths, stdin=stdin, err=err, platform=platform, run=runner, **kw)
    return rc, err.getvalue(), stdin, runner


ALL_MISSING: dict[str, bool] = {}
ALL_PRESENT = {"cmake": True, "msvc": True, "cuda": True, "engine-src": True}
ONLY_CUDA_MISSING = {"cmake": True, "msvc": True}


# ---------- AC1: plan ----------

def test_missing_items_skips_present() -> None:
    """p6-12 AC1: items already present are not in the plan."""
    items = plan.missing_items("engine-build", {"cmake": True, "msvc": False})
    assert [i.key for i in items] == ["msvc", "cuda"]
    assert plan.missing_items("emit", {"engine-src": True}) == []


def test_describe_plan_has_name_size_time_permission() -> None:
    """p6-12 AC1: the plan text names each item with size, time and permission."""
    text = plan.describe_plan([plan.CMAKE, plan.ENGINE_SRC])
    assert plan.CMAKE.label in text and plan.CMAKE.size in text and plan.CMAKE.duration in text
    assert "Windows will ask for permission" in text
    assert plan.ENGINE_SRC.label in text and "no admin rights" in text


def test_winget_command_order() -> None:
    """p6-12 AC1: winget command is base args, then the item's id/version."""
    cmd = plan.winget_command(plan.CUDA)
    assert cmd[:6] == ["winget", "install", "-e", "--source", "winget", "--accept-package-agreements"]
    assert cmd[1:1 + len(plan.WINGET_BASE)] == list(plan.WINGET_BASE)
    assert cmd[-4:] == ["--id", "Nvidia.CUDA", "--version", "13.0"]


# ---------- AC2 / AC4: ensure_need ----------

def test_all_present_is_silent(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: everything present -> None, no output."""
    set_probe(monkeypatch, ALL_PRESENT)
    rc, out, _, runner = run_ensure(make_paths(tmp_path))
    assert rc is None and out == "" and runner.calls == []


def test_dry_run_prints_plan_only(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: --dry-run shows the plan, installs nothing, rc 0."""
    set_probe(monkeypatch, ALL_MISSING)
    set_winget(monkeypatch, True)
    rc, out, stdin, runner = run_ensure(make_paths(tmp_path), dry_run=True)
    assert rc == 0 and runner.calls == [] and stdin.reads == 0
    assert "dry run" in out and plan.MSVC.label in out


def test_not_a_terminal_without_yes_refuses(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: no terminal and no --yes -> rc 2 with a --yes hint."""
    set_probe(monkeypatch, ALL_MISSING)
    set_winget(monkeypatch, True)
    rc, out, _, runner = run_ensure(make_paths(tmp_path), tty=False)
    assert rc == 2 and "--yes" in out and runner.calls == []


def test_answer_n_refuses(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: answering n -> rc 2, nothing installed."""
    set_probe(monkeypatch, ALL_MISSING)
    set_winget(monkeypatch, True)
    rc, _, stdin, runner = run_ensure(make_paths(tmp_path), tty=True, answer="n")
    assert rc == 2 and stdin.reads == 1 and runner.calls == []


def test_answer_y_installs_then_reprobes(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: answering y runs winget once per missing item; present after -> None."""
    set_probe(monkeypatch, [ONLY_CUDA_MISSING, {"cuda": True}])
    set_winget(monkeypatch, True)
    rc, out, stdin, runner = run_ensure(make_paths(tmp_path), tty=True, answer="y")
    assert rc is None and stdin.reads == 1
    assert runner.calls == [plan.winget_command(plan.CUDA)]
    assert "Nvidia.CUDA" in out  # the command is shown before it runs


def test_yes_flag_does_not_ask(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: yes=True never reads stdin."""
    set_probe(monkeypatch, [ONLY_CUDA_MISSING, {"cuda": True}])
    set_winget(monkeypatch, True)
    rc, _, stdin, runner = run_ensure(make_paths(tmp_path), tty=True, yes=True)
    assert rc is None and stdin.reads == 0 and len(runner.calls) == 1


def test_env_yes_does_not_ask(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: STRATASHIFT_YES=1 behaves like --yes."""
    monkeypatch.setenv(ens.ENV_YES, "1")
    set_probe(monkeypatch, [ONLY_CUDA_MISSING, {"cuda": True}])
    set_winget(monkeypatch, True)
    rc, _, stdin, runner = run_ensure(make_paths(tmp_path), tty=True)
    assert rc is None and stdin.reads == 0 and len(runner.calls) == 1


def test_no_winget_refuses(monkeypatch, tmp_path) -> None:
    """p6-12 AC4: winget missing -> rc 2, App Installer hint, nothing run."""
    set_probe(monkeypatch, ALL_MISSING)
    set_winget(monkeypatch, False)
    rc, out, _, runner = run_ensure(make_paths(tmp_path), yes=True)
    assert rc == 2 and "App Installer" in out and runner.calls == []


def test_still_missing_after_install(monkeypatch, tmp_path) -> None:
    """p6-12 AC4: winget ran but the item is still absent -> rc 2, new-terminal hint."""
    set_probe(monkeypatch, ALL_MISSING)
    set_winget(monkeypatch, True)
    rc, out, _, runner = run_ensure(make_paths(tmp_path), yes=True)
    assert rc == 2 and len(runner.calls) == 3
    assert "new terminal" in out


def test_non_windows_refuses(monkeypatch, tmp_path) -> None:
    """p6-12 AC4: Linux gets manual steps only, rc 2, nothing run."""
    set_probe(monkeypatch, ALL_MISSING)
    set_winget(monkeypatch, True)
    rc, out, _, runner = run_ensure(make_paths(tmp_path), platform="linux", yes=True)
    assert rc == 2 and "Windows-only" in out and runner.calls == []


def test_user_chosen_engine_path_not_installed(monkeypatch, tmp_path) -> None:
    """p6-12 AC4: a missing engine source at a user-given path is never auto-installed."""
    set_probe(monkeypatch, ALL_MISSING)
    called: list[pathlib.Path] = []
    monkeypatch.setattr(ens.fetch, "install_engine_src", lambda dest: called.append(dest))
    paths = make_paths(tmp_path)
    rc, out, _, runner = run_ensure(paths, "emit", yes=True)
    assert rc == 2 and called == [] and runner.calls == []
    assert str(paths.src) in out


def _default_paths(monkeypatch, tmp_path) -> EnginePaths:
    paths = make_paths(tmp_path)
    monkeypatch.setattr(ens, "DEFAULT_SRC", paths.src)
    return paths


def test_engine_src_installs_to_default_path(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: emit with the default path downloads the engine source, then passes."""
    paths = _default_paths(monkeypatch, tmp_path)
    got: list[pathlib.Path] = []

    def fake_install(dest: pathlib.Path) -> None:
        got.append(dest)
        (dest / "tools").mkdir(parents=True)
        (dest / "tools" / "gguf_writer.py").write_text("# fake\n", encoding="utf-8")

    monkeypatch.setattr(ens.fetch, "install_engine_src", fake_install)
    rc, _, _, runner = run_ensure(paths, "emit", yes=True)
    assert rc is None and got == [paths.src] and runner.calls == []


def test_engine_src_fetch_error(monkeypatch, tmp_path) -> None:
    """p6-12 AC3: a FetchError during install -> rc 2 with 'did not install'."""
    paths = _default_paths(monkeypatch, tmp_path)

    def boom(dest: pathlib.Path) -> None:
        raise fetch.FetchError("digest mismatch")

    monkeypatch.setattr(ens.fetch, "install_engine_src", boom)
    rc, out, _, _ = run_ensure(paths, "emit", yes=True)
    assert rc == 2 and "did not install" in out and "digest mismatch" in out


# ---------- AC3: fetch ----------

def make_zip(path: pathlib.Path, top: str, files: dict[str, bytes]) -> pathlib.Path:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(f"{top}/{name}", data)
    return path


FILES = {"tools/gguf_writer.py": b"print(1)\n", "README.md": b"hello\n", "src/a.c": b"int a;\n"}


def test_digest_ignores_top_folder_name(tmp_path) -> None:
    """p6-12 AC3: the content digest does not depend on the top folder name."""
    a = make_zip(tmp_path / "a.zip", "A", FILES)
    b = make_zip(tmp_path / "b.zip", "B", FILES)
    assert fetch.zip_tree_digest(a) == fetch.zip_tree_digest(b)
    assert fetch.zip_tree_digest(a)[1] == len(FILES)


def test_digest_changes_with_content(tmp_path) -> None:
    """p6-12 AC3: one changed byte changes the digest."""
    a = make_zip(tmp_path / "a.zip", "A", FILES)
    b = make_zip(tmp_path / "b.zip", "A", {**FILES, "README.md": b"hellp\n"})
    assert fetch.zip_tree_digest(a)[0] != fetch.zip_tree_digest(b)[0]


def test_zip_with_parent_path_is_rejected(tmp_path) -> None:
    """p6-12 AC3: an entry that climbs out of the folder is refused."""
    bad = make_zip(tmp_path / "bad.zip", "Top", {**FILES, "../evil": b"x"})
    with pytest.raises(fetch.FetchError):
        fetch.zip_tree_digest(bad)
    with pytest.raises(fetch.FetchError):
        fetch.unpack(bad, tmp_path / "dest")
    assert not (tmp_path / "dest").exists()


def _fake_download(monkeypatch, source: pathlib.Path) -> None:
    monkeypatch.setattr(fetch, "download",
                        lambda url, dest, timeout=60.0: shutil.copyfile(source, dest))


def test_install_engine_src_unpacks_without_top_folder(monkeypatch, tmp_path) -> None:
    """p6-12 AC3: matching digest -> files below dest without the top folder, zip removed."""
    src_zip = make_zip(tmp_path / "src.zip", "PolyStrata-abc", FILES)
    _fake_download(monkeypatch, src_zip)
    dest = tmp_path / "out" / "PolyStrata-main"
    fetch.install_engine_src(dest, url="http://fake", expected=fetch.zip_tree_digest(src_zip)[0])
    for name, data in FILES.items():
        assert (dest / name).read_bytes() == data
    assert not (dest / "PolyStrata-abc").exists()
    assert list(dest.parent.iterdir()) == [dest]  # no download zip, no temp dir


def test_install_engine_src_wrong_digest(monkeypatch, tmp_path) -> None:
    """p6-12 AC3: wrong digest -> FetchError, neither dest nor the download remain."""
    src_zip = make_zip(tmp_path / "src.zip", "Top", FILES)
    _fake_download(monkeypatch, src_zip)
    dest = tmp_path / "out" / "PolyStrata-main"
    with pytest.raises(fetch.FetchError, match="digest mismatch"):
        fetch.install_engine_src(dest, url="http://fake",
                                 expected=hashlib.sha256(b"x").hexdigest())
    assert not dest.exists()
    assert list(dest.parent.iterdir()) == []


def test_install_engine_src_refuses_existing_dest(monkeypatch, tmp_path) -> None:
    """p6-12 AC3: an existing destination is never overwritten."""
    src_zip = make_zip(tmp_path / "src.zip", "Top", FILES)
    _fake_download(monkeypatch, src_zip)
    dest = tmp_path / "PolyStrata-main"
    dest.mkdir()
    (dest / "keep.txt").write_bytes(b"mine")
    with pytest.raises(fetch.FetchError, match="already exists"):
        fetch.install_engine_src(dest, url="http://fake",
                                 expected=fetch.zip_tree_digest(src_zip)[0])
    assert (dest / "keep.txt").read_bytes() == b"mine"


@pytest.mark.skipif(not PIN_ZIP.is_file(), reason=f"pinned zip not found: {PIN_ZIP}")
def test_pinned_zip_matches_constants() -> None:
    """p6-12 AC3: the recorded pin zip gives the digest and file count baked into fetch.py."""
    assert fetch.zip_tree_digest(PIN_ZIP) == (fetch.ENGINE_TREE_SHA256, fetch.ENGINE_TREE_FILES)


# ---------- AC1: probe ----------

def _fake_nvcc_run(stdout: str):
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    return run


def test_nvcc_version_parsed(monkeypatch) -> None:
    """p6-12 AC1: nvcc --version output gives (major, minor)."""
    out = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.0, V13.0.48\n"
    monkeypatch.setattr(probe_mod.subprocess, "run", _fake_nvcc_run(out))
    assert probe_mod.nvcc_version(pathlib.Path("nvcc")) == (13, 0)
    monkeypatch.setattr(probe_mod.subprocess, "run", _fake_nvcc_run("garbage"))
    assert probe_mod.nvcc_version(pathlib.Path("nvcc")) is None


@pytest.mark.parametrize("version,expected", [((11, 8), False), ((12, 0), True), ((13, 0), True)])
def test_probe_cuda_needs_12_or_newer(monkeypatch, tmp_path, version, expected) -> None:
    """p6-12 AC1: CUDA below 12.0 counts as missing."""
    monkeypatch.setattr(probe_mod, "find_nvcc", lambda: (pathlib.Path("nvcc"), version))
    assert probe_mod.probe(["cuda"], make_paths(tmp_path)) == {"cuda": expected}


def test_probe_cuda_absent(monkeypatch, tmp_path) -> None:
    """p6-12 AC1: no nvcc at all -> missing."""
    monkeypatch.setattr(probe_mod, "find_nvcc", lambda: None)
    assert probe_mod.probe(["cuda"], make_paths(tmp_path)) == {"cuda": False}


def test_probe_engine_src_checks_gguf_writer(tmp_path) -> None:
    """p6-12 AC1: the engine source counts as present only with tools/gguf_writer.py."""
    paths = make_paths(tmp_path)
    assert probe_mod.probe(["engine-src"], paths) == {"engine-src": False}
    paths.tools.mkdir(parents=True)
    (paths.tools / "gguf_writer.py").write_text("", encoding="utf-8")
    assert probe_mod.probe(["engine-src"], paths) == {"engine-src": True}


# ---------- AC2: CLI wiring ----------

def test_cli_emit_dry_run_calls_ensure(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: convert --emit-gguf --dry-run asks ensure_need for 'emit' with dry_run, rc 0."""
    cli = importlib.import_module("stratashift.cli.main")
    seen: list[dict] = []

    def fake(need, paths, *, yes=False, dry_run=False, **kw):
        seen.append({"need": need, "yes": yes, "dry_run": dry_run})
        return 0

    monkeypatch.setattr(cli, "ensure_need", fake)
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"x")
    rc = cli.main(["convert", str(model), "--out", str(tmp_path / "out"),
                   "--emit-gguf", str(tmp_path / "o.gguf"), "--dry-run"])
    assert rc == 0
    assert seen == [{"need": "emit", "yes": False, "dry_run": True}]


def test_cli_dry_run_without_emit_skips_ensure(monkeypatch, tmp_path) -> None:
    """p6-12 AC2: --dry-run alone checks nothing and converts nothing, rc 0."""
    cli = importlib.import_module("stratashift.cli.main")
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "ensure_need", lambda *a, **k: calls.append((a, k)) or 0)
    model = tmp_path / "m.safetensors"
    model.write_bytes(b"x")
    rc = cli.main(["convert", str(model), "--out", str(tmp_path / "out"), "--dry-run"])
    assert rc == 0 and calls == []
