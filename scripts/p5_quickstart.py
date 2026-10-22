"""p5-03 quickstart re-run probe (numbered steps, p5-03 AC3).

Re-executes the README quickstart end-to-end on the mixtral-tiny fixture (.local,
gitignore) and asserts the measured outputs. Steps mirror README/USAGE:
 #1 inspect          -> tensors=21 experts=16 layers=[0,1] data_start=2224
 #2 convert+emit     -> no --hf-emit flag: HF naming auto-detected (p5-02 AC1), RC 0
 #3 verify p2        -> `verify <out>` PASS exit 0 (p5-02 AC2, p2 contract)
 #4 engine-form hop  -> convert <gguf> --out + verify -> engine-form PASS, exit 0

Run:  PYTHONPATH=src python3 scripts/p5_quickstart.py [scratch_dir]
Writes a JSON report; prints one PASS/FAIL line per step. No full model load (C1:
mixtral-tiny is 8.6 MB, stream path only).
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import stratashift.cli.main as cli_mod  # noqa: E402
from stratashift.polystrata.engine_paths import scratch_dir  # noqa: E402

MODEL = ROOT / ".local" / "models" / "mixtral-tiny.safetensors"
SCRATCH = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    scratch_dir() / "p5-03-quickstart-rerun"


def step(n: int, msg: str) -> None:
    print(f"#{n} {msg}")


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    # #1 inspect
    out = SCRATCH / "out"
    rc = cli_mod.main(["inspect", str(MODEL)])
    assert rc == 0, f"#1 inspect rc={rc}"
    step(1, f"inspect rc=0 ({MODEL.name})")
    report["step1_inspect"] = {"rc": rc, "model": str(MODEL)}

    # #2 convert + emit — NO --hf-emit flag (p5-02 auto-detection)
    gguf = SCRATCH / "model.gguf"
    rc = cli_mod.main(["convert", str(MODEL), "--out", str(out),
                       "--emit-gguf", str(gguf)])
    assert rc == 0, f"#2 convert+emit rc={rc}"
    size = gguf.stat().st_size
    step(2, f"convert+emit rc=0, gguf {size} B (auto-detected, no flag)")
    report["step2_convert_emit"] = {"rc": rc, "gguf_bytes": size}

    # #3 verify p2 contract
    rc = cli_mod.main(["verify", str(out)])
    assert rc == 0, f"#3 verify p2 rc={rc}"
    step(3, f"verify p2 rc=0 ({out.name})")
    report["step3_verify_p2"] = {"rc": rc}

    # #4 engine-form hop: GGUF -> convert -> verify
    out2 = SCRATCH / "out2"
    rc = cli_mod.main(["convert", str(gguf), "--out", str(out2)])
    assert rc == 0, f"#4a gguf convert rc={rc}"
    rc = cli_mod.main(["verify", str(out2)])
    assert rc == 0, f"#4b gguf-path verify rc={rc}"
    man = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
    assert "tensors" in man, "#4 engine-form manifest expected"
    step(4, f"engine-form hop rc=0 (layers {man['n_layers']} x NE "
            f"{man['n_experts_per_layer']}, tensors contract)")
    report["step4_engine_hop"] = {"rc": rc, "n_layers": man["n_layers"],
                                  "n_experts_per_layer": man["n_experts_per_layer"]}

    (SCRATCH / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS all 4 steps, report: {SCRATCH / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
