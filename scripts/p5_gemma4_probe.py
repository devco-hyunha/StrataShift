"""p5-06 AC2 header probe (numbered steps, header-only — C1: never full model).

 #1 config.json fetch (HF CDN)
 #2 shard list + HEAD Content-Length verification
 #3 shard1 safetensors header via HTTP Range (bytes 0..65535 — header JSON only)
 #4 HF naming key inventory + geometry derivation (hf_gguf_map reuse, header only)
 #5 local fixture header probe (.local/models/gemma4-26b-a4b-it — parse_header,
    headers only, C1 gate: no payload read; AC3 measured inventory basis)
 #6 index.json HEAD presence check (p4-02 multi-shard input gate)

REPO is the MoE variant `google/gemma-4-26B-A4B-it` (2026-10-18): the first probe
run (2026-10-18 morning) measured `google/gemma-4-31B-it`, which is DENSE
(config hidden_size 5376, 60 layers, no experts keys in its header — the dense
report stays in report.json); AC3 mapping targets the MoE fixture.

Run: python3 scripts/p5_gemma4_probe.py  (network + HF CDN, writes scratch report)
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from stratashift.polystrata.engine_paths import scratch_dir  # noqa: E402

REPO = "google/gemma-4-26B-A4B-it"
SCRATCH = scratch_dir() / "p5-06-gemma4"
FIXTURE_DIR = pathlib.Path("/workspace/.local/models/gemma4-26b-a4b-it")


def fetch(url: str, rng: str | None = None) -> bytes:
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # #1 config.json — composite: text geometry lives in text_config
    cfg = json.loads(fetch(f"https://huggingface.co/{REPO}/resolve/main/config.json"))
    tc = cfg.get("text_config", cfg)
    step1 = {k: tc.get(k) for k in
            ("hidden_size", "num_hidden_layers", "num_attention_heads",
             "num_key_value_heads", "head_dim", "intermediate_size",
             "hidden_activation", "vocab_size", "sliding_window",
             "sliding_window_pattern", "final_logit_softcapping")}
    step1["model_type"] = cfg.get("model_type")
    print(f"#1 config.json rc=0 {step1}")

    # #2 shard list + HEAD Content-Length
    files = json.loads(fetch(f"https://huggingface.co/api/models/{REPO}"))
    names = sorted({f["rfilename"] for f in files["siblings"]
                    if f["rfilename"].endswith(".safetensors")})
    sizes = []
    for n in names:
        req = urllib.request.Request(
            f"https://huggingface.co/{REPO}/resolve/main/{n}", method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            cl = r.headers.get("Content-Length")
        sizes.append((n, int(cl) if cl else -1))
    print(f"#2 shards rc=0 n={len(sizes)} total={sum(s for _, s in sizes)}")

    # #3 shard1 header via Range — header JSON only, payload NOT downloaded
    raw = fetch(f"https://huggingface.co/{REPO}/resolve/main/{names[0]}",
                rng="0-1048575")
    hlen = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + hlen])
    print(f"#3 shard1 header rc=0 hlen={hlen} tensors={len(header) - 1}")

    # #4 naming inventory: HF keys, shapes, fused/per-expert patterns
    tkeys = {k: v for k, v in header.items() if k != "__metadata__"}
    pats: dict[str, int] = {}
    for k in tkeys:
        base = k.split(".")
        pats[".".join(base[:3]) if len(base) > 3 else k] = pats.get(
            ".".join(base[:3]) if len(base) > 3 else k, 0) + 1
    report = {"config": step1, "shards": sizes,
             "shard1_tensor_count": len(tkeys),
             "shard1_header": tkeys}
    (SCRATCH / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"#4 inventory rc=0 keys={len(tkeys)} written {SCRATCH / 'report.json'}")
    for pat, cnt in sorted(pats.items()):
        print(f"   {pat}: {cnt}")

    # #5 local fixture header probe — headers only (8B length + header JSON), C1 gate.
    inv: dict[str, dict] = {}
    for p in sorted(FIXTURE_DIR.glob("*.safetensors")):
        with open(p, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
        t = {k: v for k, v in header.items() if k != "__metadata__"}
        for k, v in t.items():
            inv[k] = {"dtype": v["dtype"], "shape": v["shape"],
                      "offsets": v["data_offsets"], "shard": p.name}
        print(f"#5 {p.name} rc=0 hlen={hlen} tensors={len(t)}")
    pats5: dict[str, int] = {}
    for k in inv:
        base = k.split(".")
        p5 = ".".join(base[:3]) if len(base) > 3 else k
        pats5[p5] = pats5.get(p5, 0) + 1
    (SCRATCH / "report_moe.json").write_text(json.dumps(
        {"fixture": str(FIXTURE_DIR), "config": json.loads(
            (FIXTURE_DIR / "config.json").read_text())["text_config"],
         "inventory": inv}, indent=1) + "\n")
    print(f"#5 local inventory rc=0 keys={len(inv)} written {SCRATCH / 'report_moe.json'}")
    for pat, cnt in sorted(pats5.items()):
        print(f"   {pat}: {cnt}")

    # #6 index.json HEAD presence (p4-02 multi-shard input needs weight_map)
    idx = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
    req = urllib.request.Request(idx, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"#6 index.json rc=0 status={r.status} CL={r.headers.get('Content-Length')}")
    except urllib.error.HTTPError as e:
        print(f"#6 index.json rc=1 status={e.code} (p4-02 input gate: weight_map needed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
