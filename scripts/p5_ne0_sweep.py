#!/usr/bin/env python3
"""p5-08 AC3 gate: engine LOAD gate sweep over an emitted GGUF.

Engine `tensor_payload_bytes` (`include/strata/artifact/gguf_reader.hpp:505`, pin
c566ff4): the payload is computed ONLY when the ENGINE shape's first dim is a
multiple of the type's block elems (`be`); a failing tensor gets payload 0 and the
loader throws "does not fit its file" at load (measured rc=1, p5-07 AC5 — 30
`ffn_down [2112,2816]` Q4_K, 2112 % 256 = 64). `be` is the measured
`block_geometry` switch — `polystrata/headers.py::ENGINE_LOAD_BE` (Q3_K/Q4_K/Q5_K/
Q6_K be=256, Q4_0/Q8_0 be=32, F32 be=1). Types outside the switch are counted
UNKNOWN and held at be=1 (the engine carries them verbatim; measured F16/BF16
pass-through, p4-01 §dtype).

Rope gate (p5-08 rope fix): when the architecture's sliding pattern carries a FULL
layer (pattern value 0), the engine loads `rope_freqs.weight` F32 [n_rot/2, 1] from
the FILE (`gemma4.cpp:207`, pin c566ff4 — measured TK1 RUN FAIL rc=1 "the GGUF has
no tensor rope_freqs.weight"). Presence + F32 + shape + VALUES (the engine float64
formula `theta ** (-2i/n_rot)` cast to float32, `src/kernels/rope_parity.cpp`
header) are all checked.

Usage: python3 scripts/p5_ne0_sweep.py <gguf>   → exit 0 when FAIL=0.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stratashift.ingest.gguf import parse_gguf_header  # noqa: E402
from stratashift.polystrata.headers import ENGINE_LOAD_BE  # noqa: E402


def main() -> int:
    gguf = pathlib.Path(sys.argv[1])
    meta = parse_gguf_header(str(gguf))
    arch = meta.metadata.get("general.architecture")
    prefix = arch if arch else ""

    fails = 0
    unknown = []
    for t in meta.tensors:
        be = ENGINE_LOAD_BE.get(t.type_name)
        if be is None:
            unknown.append(t.type_name)
            be = 1  # engine carries non-switch types verbatim
        ne0 = t.shape[0]
        if ne0 % be != 0:
            fails += 1
            print(f"FAIL: {t.name} {list(t.shape)} {t.type_name} — ne0={ne0} % be={be} = "
                  f"{ne0 % be}")
    counts: dict[str, int] = {}
    for t in meta.tensors:
        counts[t.type_name] = counts.get(t.type_name, 0) + 1

    # rope gate: a FULL layer in the pattern requires the file tensor
    pattern = meta.metadata.get(f"{prefix}.attention.sliding_window_pattern")
    full_layers = [i for i, v in enumerate(pattern) if int(v) == 0] if pattern else []
    rope_ok = True
    if full_layers:
        rot = meta.metadata.get(f"{prefix}.rope.dimension_count")
        key_len = meta.metadata.get(f"{prefix}.attention.key_length")
        base = meta.metadata.get(f"{prefix}.rope.freq_base")
        if rot is None:
            rot = key_len  # engine default rot_full = dim_full (gemma4.cpp:53)
        if rot is None or base is None:
            rope_ok = False
            print(f"FAIL: full layers {full_layers} — GGUF meta lacks "
                  f"{prefix}.rope.dimension_count/key_length or {prefix}.rope.freq_base")
        else:
            rot = int(rot)
            want_shape = [rot // 2, 1]
            t = next((t for t in meta.tensors if t.name == "rope_freqs.weight"), None)
            if t is None:
                rope_ok = False
                print(f"FAIL: full layers {full_layers} require tensor rope_freqs.weight "
                      f"F32 {want_shape} (engine gemma4.cpp:207) — absent")
            elif t.type_name != "F32" or list(t.shape) != want_shape:
                rope_ok = False
                print(f"FAIL: rope_freqs.weight is {t.type_name} {list(t.shape)}, "
                      f"expected F32 {want_shape}")
            else:
                with open(gguf, "rb") as fh:
                    fh.seek(meta.data_start + t.offset)
                    raw = fh.read(t.expected_bytes())
                got = np.frombuffer(raw, dtype="<f4")
                base_f = float(base)
                want = np.power(np.float64(base_f),
                                -2.0 * np.arange(rot // 2) / np.float64(rot)) \
                    .astype("<f4")
                if not np.array_equal(got, want):
                    rope_ok = False
                    d = np.flatnonzero(got != want)
                    print(f"FAIL: rope_freqs values diverge from float64 "
                          f"theta**(-2i/n_rot) at {d[:4]} ({len(d)} of {got.size})")
                else:
                    print(f"rope gate PASS: rope_freqs.weight F32 {want_shape}, "
                          f"{got.size} values bit-exact (base={base_f}, rot={rot}), "
                          f"full layers {full_layers}")

    print(f"sweep: {len(meta.tensors)} tensors, types {json_types(counts)}, "
          f"FAIL={fails}, rope {'PASS' if rope_ok else 'FAIL'}")
    if unknown:
        print(f"unknown types (be=1 verbatim): {sorted(set(unknown))}")
    return 0 if (fails == 0 and rope_ok) else 1


def json_types(counts: dict) -> str:
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) + "}"


if __name__ == "__main__":
    sys.exit(main())
