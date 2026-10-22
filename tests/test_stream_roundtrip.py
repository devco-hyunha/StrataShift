"""Stream roundtrip: synthetic Q2_0 safetensors → convert → experts.bin/index.txt
re-parsed by polystrata.index; per-row blob rejoin must unpack bit-exact to the
source slice values. AC2/AC4."""
from __future__ import annotations

import json
import struct

from stratashift.ingest.safetensors_header import parse_header
from stratashift.meta.expert_map import build_expert_map
from stratashift.polystrata.headers import Q2_0_BLOCK_BYTES
from stratashift.polystrata.index import parse_index_header, parse_index_row
from stratashift.quant.pack import q2_0_unpack

BLOCKS = 32  # 576 bytes per expert slice


def make_synthetic_model(path) -> None:
    """Fused layout: gate_up_proj (2 layers, 4 experts) + down_proj, U8 Q2_0 blocks."""
    n_experts, n_layers = 4, 2
    slice_bytes = BLOCKS * Q2_0_BLOCK_BYTES
    tensors = {}
    data = bytearray()
    for name, rows in (("model.layers.0.mlp.experts.gate_up_proj", n_experts),
                       ("model.layers.1.mlp.experts.gate_up_proj", n_experts),
                       ("model.layers.0.mlp.experts.down_proj", n_experts),
                       ("model.layers.1.mlp.experts.down_proj", n_experts)):
        blob = bytearray()
        for e in range(n_experts):
            d = float(1 << e)  # per-expert scale
            for b in range(BLOCKS):
                qs = bytes((b + e) % 256 for _ in range(16))
                blob += struct.pack("<e", d) + qs
        begin = len(data)
        data += blob
        tensors[name] = {"dtype": "U8", "shape": [n_experts, BLOCKS * Q2_0_BLOCK_BYTES],
                         "data_offsets": [begin, begin + len(blob)]}
    header = json.dumps({**tensors, "__metadata__": {"format": "PyTorch"}}).encode("utf-8")
    pad = (-len(header)) % 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header) + pad))
        f.write(header + b" " * pad)
        f.write(bytes(data))


def test_convert_roundtrip(tmp_path) -> None:
    model = tmp_path / "synthetic.safetensors"
    make_synthetic_model(model)
    out_dir = tmp_path / "out"

    from stratashift.cli.main import cmd_convert
    assert cmd_convert(str(model), str(out_dir), None) == 0

    experts = (out_dir / "experts.bin").read_bytes()
    index_lines = (out_dir / "index.txt").read_text(encoding="utf-8").splitlines()
    hdr = parse_index_header(index_lines[0])
    rows = [parse_index_row(line) for line in index_lines[1:]]
    # rows are per tensor (engine index is tensor-keyed); expert map drives slicing
    assert hdr.tensors == len(rows) == 4
    assert hdr.pool == len(experts)

    # per-row: codes+scales rejoin == source slice, unpack matches source unpack
    meta = parse_header(str(model))
    refs = build_expert_map(meta)
    src = {r.tensor_name: r for r in refs}
    with open(model, "rb") as f:
        f.seek(meta.data_start)
        for row in rows:
            codes_bytes = row.values["codes_bytes"]
            scales_bytes = row.values["scales_bytes"]
            dst = row.values["dst_off"]
            blob = experts[dst:dst + codes_bytes + scales_bytes]
            src_slice = None
            for r in refs:
                if r.tensor_name == row.values["name"]:
                    begin, end = r.data_offsets
                    f.seek(meta.data_start + begin)
                    src_slice = f.read(end - begin)
                    break
            assert src_slice is not None
            # rejoin planes → blocks == source blocks bit-exact
            rebuilt = b"".join(
                blob[codes_bytes + 2 * i:codes_bytes + 2 * i + 2]
                + blob[16 * i:16 * i + 16]
                for i in range(BLOCKS)
            )
            assert rebuilt == src_slice
            for i in range(BLOCKS):
                assert q2_0_unpack(rebuilt[18 * i:18 * i + 18]) == \
                    q2_0_unpack(src_slice[18 * i:18 * i + 18])
