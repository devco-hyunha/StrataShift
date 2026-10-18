"""p5-06 AC2 header probe (numbered steps, header-only — C1: never full model).

 #1 config.json fetch (HF CDN)
 #2 shard list + HEAD Content-Length verification
 #3 shard1 safetensors header via HTTP Range (bytes 0..65535 — header JSON only)
 #4 HF naming key inventory + geometry derivation (hf_gguf_map reuse, header only)

Run: python3 scripts/p5_gemma4_probe.py  (network + HF CDN, writes scratch report)
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import urllib.request

REPO = "google/gemma-4-31B-it"
SCRATCH = pathlib.Path("/opt/data/cache/scratch/p5-06-gemma4")


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
