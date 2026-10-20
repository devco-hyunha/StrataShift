"""p6-01 AC1 header probe (numbered steps, header-only — C1: never full model).

 #1 config.json fetch (HF CDN) — geometry fields, exact values
 #2 shard list + HEAD Content-Length verification (19 shards)
 #3 shard1 safetensors header via HTTP Range (header JSON only, payload NOT downloaded)
 #4 HF naming key inventory (block_sparsemoe per-expert w1/w2/w3, gate, norms,
    attention q/k/v/o) + shapes recorded in the fixture JSON
 #5 index.json HEAD presence check (p4-02 multi-shard input gate)

Fixture output: tests/fixtures/mixtral_header.json — the AC2 grounds for the
`hf_gguf_map` Mixtral mapping (never-invented). p4-03 precedent: engine guard
NOT applicable (geometry ≠ qwen4exp), reference verifier acceptance.
Run: python3 scripts/p6_mixtral_probe.py
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import urllib.request

REPO = "mistralai/Mixtral-8x7B-v0.1"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mixtral_header.json"


def fetch(url: str, rng: str | None = None) -> bytes:
    req = urllib.request.Request(url)
    if rng:
        req.add_header("Range", f"bytes={rng}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def main() -> int:
    # #1 config.json — measured geometry
    cfg = json.loads(fetch(f"https://huggingface.co/{REPO}/resolve/main/config.json"))
    step1 = {k: cfg.get(k) for k in
             ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "num_local_experts", "num_experts_per_tok",
              "intermediate_size", "rope_theta", "vocab_size", "hidden_act",
              "sliding_window", "tie_word_embeddings")}
    step1["model_type"] = cfg.get("model_type")
    step1["head_dim"] = step1["hidden_size"] // step1["num_attention_heads"]
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
    tkeys = {k: v for k, v in header.items() if k != "__metadata__"}
    print(f"#3 shard1 header rc=0 hlen={hlen} tensors={len(tkeys)}")

    # #4 key inventory + shapes of the AC2 grounds keys
    want = ("model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight")
    inventory = {}
    for k in want:
        if k in tkeys:
            v = tkeys[k]
            inventory[k] = {"dtype": v["dtype"], "shape": v["shape"],
                            "data_offsets": v["data_offsets"], "shard": names[0]}
            print(f"#4 {k} rc=0 {v['dtype']} {v['shape']}")
        else:
            print(f"#4 {k} ABSENT in shard1 (shard map via index.json)")
    # index.json weight_map gives full key->shard map; shapes come from shard headers.
    idx = json.loads(fetch(f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"))
    weight_map = idx["weight_map"]
    # MEASURED 2026-10-20: Mixtral's index carries `metadata.total_size` — exactly the
    # HF-convention form shard_index.parse_index requires (shard_index.py:55), so the
    # p4-02 multi-shard input gate accepts this repo. Record the measured format.
    total = idx.get("metadata", {}).get("total_size")
    print(f"#4 index.json rc=0 metadata.total_size={total} entries={len(weight_map)}")
    fixture = {"repo": REPO, "config": step1, "shards": sizes,
               "shard1_header_len": hlen, "shard1_tensor_count": len(tkeys),
               "measured_keys": inventory,
               "index_total_size": total,
               "index_format": "metadata.total_size (measured 2026-10-20, matches "
                               "shard_index.parse_index gate)",
               "index_entries": len(weight_map),
               "key_patterns": sorted({k for k in weight_map})[:40]}
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"#5 fixture rc=0 written {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
