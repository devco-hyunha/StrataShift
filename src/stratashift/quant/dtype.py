"""Payload dtype normalization for the quantizers (moved out of cli/main.py, p6-08)."""
from __future__ import annotations

import numpy as np


def to_f32(raw: bytes, dtype: str) -> bytes:
    """Normalize a tensor payload to F32 bytes (BF16 = top 16 bits of F32, F16 → F32).

    Vectorized numpy (peak = tensor slice, C1/C2); bit-identical to the scalar
    engine_manifest.bf16_to_f32_bytes loop (tests/test_hf_gguf_map.py AC2).
    """
    if dtype == "F32":
        return raw
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view("<f4").tobytes()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype("<f4").tobytes()
    raise ValueError(f"unsupported HF dtype {dtype} for emit")
