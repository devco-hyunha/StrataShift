#!/usr/bin/env python3
"""p6-01 AC4: a copy of a mixtral GGUF that the pinned llama.cpp loads as its `llama` architecture.

Tensor bytes are copied as they are (the same file for both engines); only metadata changes:
general.architecture mixtral -> llama, `mixtral.*` -> `llama.*`, i64/f64 values -> u32/f32 (our writer stores
i64/f64; strata-poly reads either, llama.cpp's hparams want u32/f32), plus the keys llama.cpp's loader needs and
the engine does not: `llama.context_length`, `llama.feed_forward_length` (= expert_feed_forward_length, which
llama.cpp sizes the experts by) and `tokenizer.ggml.model = none`. Copy method: gguf-py's gguf_new_metadata.py.

Usage: python scripts/p6_llama_parity/to_llama.py IN.gguf OUT.gguf   (the pinned gguf-py on PYTHONPATH;
compare.py sets it)
"""
from __future__ import annotations

import sys

from gguf import GGUFReader, GGUFValueType, GGUFWriter


def to_llama(src: str, dst: str, context_length: int = 1024) -> None:
    r = GGUFReader(src)
    w = GGUFWriter(dst, "llama")
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture":
            continue
        name = "llama." + f.name[len("mixtral."):] if f.name.startswith("mixtral.") else f.name
        t = f.types[0]
        if name.startswith("llama.") and t == GGUFValueType.INT64:
            t = GGUFValueType.UINT32
        if name.startswith("llama.") and t == GGUFValueType.FLOAT64:
            t = GGUFValueType.FLOAT32
        if t == GGUFValueType.ARRAY:
            w.add_key_value(name, f.contents(), t, sub_type=f.types[-1])
        else:
            w.add_key_value(name, f.contents(), t)
    w.add_context_length(context_length)
    w.add_feed_forward_length(int(r.fields["mixtral.expert_feed_forward_length"].contents()))
    w.add_string("tokenizer.ggml.model", "none")
    for t in r.tensors:
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    to_llama(sys.argv[1], sys.argv[2])
