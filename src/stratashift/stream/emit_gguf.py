"""safetensors float experts → engine GGUF v3 with Q2_0 tensors (p3-02), moved out of
cli/main.py and split into steps (p6-08).

Expert tensors are named `blk.{L}.ffn_{gate,up,down}_exps.weight` with safetensors
shape (out, NE, in) row-major; the GGUF fused shape is (in, out, NE) with blocks
running along dim 0 (engine gguf_writer add_q2_0 contract). One tensor is alive at
a time (CONSTRAINTS C1/C2). Guard keys come from the preset input — never invented
(docs/rea/struct_layouts.md §1 architecture guard, VERIFIED p3-01).

`dense_type` (p3-03): "F32" = pass-through (p3-02 behaviour); "Q4_0" quantizes each
dense F32 tensor to Q4_0 blocks (18 B / 32 elements) with the SAME shape recorded —
engine gguf_reader expected_bytes = elements × 18 / 32, so the element count must be
a multiple of 32 (refused with an explicit error otherwise).
"""
from __future__ import annotations

import json
import sys

import numpy as np

from stratashift.ingest.safetensors_header import parse_header, numel
from stratashift.meta.imatrix import load_imatrix_stats
from stratashift.monitor.heap import heap_gate, heap_peak_bytes
from stratashift.monitor.sampler import Sampler
from stratashift.polystrata import engine_tools
from stratashift.polystrata.headers import PLE_BYTES, PLE_HEAD_DIM, PLE_NAME, PLE_ROWS
from stratashift.quant.pack import quantize_q2_0_stream
from stratashift.quant.pack_imatrix import broadcast_imatrix, quantize_q4_k_imatrix_stream
from stratashift.quant.select import PLAIN_QUANTIZERS, select_pack_quant
from stratashift.stream.emit_common import finalize_gguf, require_engine_tools
from stratashift.stream.reader import SliceReader


def emit_gguf(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
              dense_type: str = "F32", imatrix_file: str | None = None) -> int:
    with heap_gate():
        return _emit(model, out_path, guard_keys, max_rss_gb, dense_type, imatrix_file)


def _emit_expert(w, reader: SliceReader, model: str, meta, entry, sampler: Sampler) -> bool:
    """One `*_exps.weight` tensor, one expert run at a time. True = RSS ceiling hit."""
    if entry.dtype != "F32":
        raise ValueError(f"{entry.name}: expert tensor must be F32 float, got {entry.dtype}")
    out_dim, ne, in_dim = entry.shape
    # AC6 no-full-load gate: slice ONE expert run (out_dim × in_dim floats),
    # never the whole tensor (3.35 GB) — peak = header + single expert slice.
    run_bytes = out_dim * in_dim * 4
    q = bytearray()
    for e in range(ne):
        raw = reader.read(model, meta.data_start + entry.data_offsets[0] + e * run_bytes,
                          run_bytes)
        q += quantize_q2_0_stream(raw)
        del raw
        if sampler.sample():
            return True
    w.tensors.append(engine_tools._Tensor(entry.name, [in_dim, out_dim, ne], "Q2_0", bytes(q)))
    return False


def _emit_ple(w, reader: SliceReader, model: str, meta, entry) -> None:
    """PLE table: raw IQ4_NL block bytes pass through verbatim. Engine ple_formats() has
    no F32 (ngram.hpp PleFormat enum); strata_pack build:301 requires
    per_layer_token_embd.weight in shard2, and ngram.hpp:32 requires GGUF shape[0] ==
    PLE_HEAD_DIM (160). safetensors stores the 720 raw bytes as U8[720] (IQ4_NL: 32
    elements per 18 B block → GGUF shape [160, 8], p3-01 VERIFIED)."""
    if list(entry.shape) != [PLE_BYTES]:
        raise ValueError(f"{entry.name}: PLE payload must be U8[{PLE_BYTES}], "
                         f"got U8{entry.shape}")
    begin, end = entry.data_offsets
    raw = reader.read(model, meta.data_start + begin, end - begin)
    w.tensors.append(engine_tools._Tensor(entry.name, [PLE_HEAD_DIM, PLE_ROWS], "IQ4_NL", raw))


def _emit_dense(w, reader: SliceReader, model: str, meta, entry, dense_type: str,
                imatrix_stats: dict | None) -> int | None:
    """One dense tensor. Returns 3 on an imatrix refusal, else None.

    Type selection, including the Q4_0 whole-block refusal and the engine pack gates
    (p3-03~p3-09: codes→scales planes CONTIGUOUS at strata_pack.py:43 ALIGN=64; 1-D
    tensors stay F32 for Q8_0, pack_index.py:174-186), lives in
    `quant.select.select_pack_quant`; tensors below a type's gate stay F32 pass-through."""
    if entry.dtype != "F32":
        raise ValueError(f"{entry.name}: only F32 dense tensors supported "
                         f"for emit, got {entry.dtype}")
    begin, end = entry.data_offsets
    raw = reader.read(model, meta.data_start + begin, end - begin)
    try:
        qtype = select_pack_quant(entry.shape, dense_type)
    except ValueError as e:
        raise ValueError(f"{entry.name}: {e}") from None
    if qtype == "F32":
        w.add_f32(entry.name, np.frombuffer(raw, dtype="<f4").reshape(entry.shape))
        return None
    # p6-03 AC3: imatrix opt-in (Q4_K only) consumes the normalized vector over the
    # input dimension; row-major tensor [out, in] tiles the vector per row.
    # Plain RTN path is byte-unchanged when imatrix_stats is None (AC gate).
    # A dense stats entry has nmat=1 (imatrix.cpp:723 single count), so the
    # vector is the flat column plane [ne0].
    vec = imatrix_stats.get(entry.name) if imatrix_stats and qtype == "Q4_K" else None
    if vec is not None:
        if vec.shape[0] != 1:
            print(f"{entry.name}: imatrix dense path expects nmat=1, "
                  f"got {vec.shape[0]} — refusing", file=sys.stderr)
            return 3
        q = quantize_q4_k_imatrix_stream(
            raw, broadcast_imatrix(vec.reshape(-1), numel(entry.shape), entry.shape[-1]))
    else:
        q = PLAIN_QUANTIZERS[qtype](raw)
    w.tensors.append(engine_tools._Tensor(entry.name, list(entry.shape), qtype, q))
    return None


def _emit(model: str, out_path: str, guard_keys: dict, max_rss_gb: float | None,
          dense_type: str, imatrix_file: str | None) -> int:
    if not require_engine_tools():
        return 2
    # p6-03 AC3 opt-in: imatrix stats file (GGUF format, loader contract transcribed
    # from llama.cpp common/imatrix-loader.cpp + quantize.cpp:204-215). Loaded ONCE as
    # normalized per-matrix vectors; the stats file is small (calibration product), not
    # a model body — C1-safe. None (default) → plain RTN path is byte-unchanged.
    imatrix_stats = load_imatrix_stats(imatrix_file) if imatrix_file else None
    meta = parse_header(model)
    # heap peak, not VmRSS, gates the streaming slice (p3-03 AC5). The engine GGUFWriter
    # accumulates all tensor bytes in w.tensors and write() copies the full body
    # (tools/gguf_writer.py:151-180, pin c566ff4, read-only), so the total heap peak
    # includes that writer accumulation — it is the engine's write design, not a full
    # model load.
    sampler = Sampler(max_rss_gb)
    w = engine_tools.GGUFWriter()
    w.add("general.architecture", guard_keys.get("architecture", "qwen4exp"))
    w.add("general.name", guard_keys.get("name", "stratashift emit"))
    for key, value in guard_keys.items():
        if key in ("architecture", "name"):
            continue
        w.add(key, value)
    with SliceReader() as reader:
        for entry in meta.tensors:
            if entry.name.startswith("blk.") and entry.name.endswith("_exps.weight"):
                if _emit_expert(w, reader, model, meta, entry, sampler):
                    return 3
            elif entry.dtype == "U8" and entry.name == PLE_NAME:
                _emit_ple(w, reader, model, meta, entry)
                sampler.record(heap=True)
            else:
                if _emit_dense(w, reader, model, meta, entry, dense_type, imatrix_stats) == 3:
                    return 3
                sampler.record(heap=True)
            if sampler.over_limit():
                return 3
    out = finalize_gguf(w, out_path)
    print(json.dumps({
        "model": model,
        "gguf": str(out),
        "tensors": len(w.tensors),
        "guard_keys": len(guard_keys),
        "rss_samples_bytes": sampler.rss,
        "rss_peak_bytes": sampler.rss_peak(),
        "heap_peak_bytes": heap_peak_bytes(),
        "slice_heap_peak_bytes": sampler.slice_heap_peak(),
    }, indent=2))
    return 0
