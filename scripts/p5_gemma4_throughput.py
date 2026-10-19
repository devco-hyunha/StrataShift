"""p5-06 AC4 time gate: measured Q2_0 quantization throughput on the Gemma4 fixture.

Reads TEN expert slices (header + seek+read, C1 gate: one slice alive at a time),
runs quantize_q2_0_stream exactly as the emit builder's gate/up/down chunks do, and
reports bytes/element and MB/s. The AC4 full-emit ETA is the measured total input
bytes divided by this measured rate — never an invented number.

Run: python3 scripts/p5_gemma4_throughput.py
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from stratashift.ingest.shard_index import load_sharded_model  # noqa: E402
from stratashift.quant.pack import quantize_q2_0_stream  # noqa: E402

MODEL_DIR = REPO / ".local" / "models" / "gemma4-26b-a4b-it"


def main() -> int:
    meta = load_sharded_model(str(MODEL_DIR))
    by_name = {t.name: t for t in meta.tensors}
    gu = by_name["model.language_model.layers.0.experts.gate_up_proj"]
    ne, two_ff, h = gu.shape
    row_elems = (two_ff // 2) * h
    esz = 2  # BF16 measured
    shard = str(MODEL_DIR / gu.shard)
    base = meta.shard_data_start[gu.shard] + gu.data_offsets[0]
    t0 = time.perf_counter()
    out_bytes = 0
    n = 10
    with open(shard, "rb") as fh:
        for e in range(n):
            lo = base + e * two_ff * h * esz
            fh.seek(lo)
            raw = fh.read(row_elems * esz)
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            f32 = u.view("<f4").tobytes()
            out_bytes += len(quantize_q2_0_stream(f32))
            del raw, u, f32
    dt = time.perf_counter() - t0
    in_bytes = n * row_elems * esz
    rate = in_bytes / dt
    total_in = sum(t.nbytes() for t in meta.tensors)
    print(f"throughput rc=0 experts={n} in={in_bytes} B out={out_bytes} B "
          f"dt={dt:.2f}s rate={rate / 1e6:.1f} MB/s")
    print(f"total fixture tensors = {total_in} B -> full-emit ETA "
          f"{total_in / rate / 3600:.1f} h at the measured rate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
