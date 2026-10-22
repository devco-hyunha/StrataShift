"""Gemma4 derived global tensor `rope_freqs.weight` (moved out of cli/main.py, p6-08)."""
from __future__ import annotations

import numpy as np


def gemma4_rope_freqs(guard_keys: dict, cfg_json: dict | None) -> np.ndarray:
    """Derived global tensor `rope_freqs.weight` (p5-08 rope fix).

    Engine gemma4.cpp:207 (pin c566ff4) loads the FULL layers' rotary frequency table
    from the FILE: `upload(gg_->find("rope_freqs.weight"), "rope_freqs.weight",
    B.n_rot / 2, 1, kF32)` — measured TK1 gemma4 RUN FAIL rc=1 "the GGUF has no tensor
    rope_freqs.weight" (docs/work/p5-08 §AC4, 2026-10-19). Engine `upload`
    (engine.cpp:884-890, pin c566ff4) gate: shape[0] == n_rot/2, rest == 1 → the
    tensor is F32 [n_rot/2, 1].

    VALUES are the engine's own float64 reference formula, `theta ** (-2i / n_rot)`
    (`src/kernels/rope_parity.cpp` header, pin c566ff4 — "exactly as ref/qsa.py::
    rope_freqs"; consistent with gemma4.cpp:194 `pow(base, -2/n_rot)`), computed in
    float64 and cast to float32. base = guard `gemma4.rope.freq_base`, n_rot = guard
    `gemma4.rope.dimension_count` (engine default rot_full = dim_full, gemma4.cpp:53).

    NEVER-INVENTED gate (p4-01 rule): when the checkpoint's config.json carries
    rope_parameters, they must MEASURE-equal the preset — full_attention.rope_theta
    == freq_base, partial_rotary_factor · global_head_dim == dimension_count,
    sliding_attention.rope_theta == freq_base_swa. Engine gate gemma4.cpp:77:
    rot even, rot ≤ key_length.
    """
    base = guard_keys.get("gemma4.rope.freq_base")
    if base is None:
        raise ValueError("gemma4 emit requires guard key gemma4.rope.freq_base "
                         "(engine gemma4.cpp:207 loads rope_freqs.weight from the file)")
    base = float(base)
    dim_full = guard_keys.get("gemma4.attention.key_length")
    rot = guard_keys.get("gemma4.rope.dimension_count")
    if rot is None:
        rot = dim_full  # engine default: rot_full = dim_full (gemma4.cpp:53, pin c566ff4)
    if rot is None:
        raise ValueError("gemma4 emit needs rope.dimension_count or attention.key_length")
    rot = int(rot)
    if rot % 2 == 1 or (dim_full is not None and rot > int(dim_full)):
        raise ValueError(f"engine gate gemma4.cpp:77 refuses rope.dimension_count={rot} "
                         f"(even, <= key_length={dim_full})")
    if cfg_json is not None:
        tc = cfg_json.get("text_config")
        tc = tc if isinstance(tc, dict) else cfg_json
        rp = tc.get("rope_parameters")
        if isinstance(rp, dict):
            full = rp.get("full_attention")
            if isinstance(full, dict):
                theta = float(full.get("rope_theta", base))
                if theta != base:
                    raise ValueError(f"config full rope_theta {theta} != guard freq_base {base} "
                                     f"(never invented)")
                prf = full.get("partial_rotary_factor")
                if prf is not None:
                    ghd = tc.get("global_head_dim")
                    if ghd is None:
                        ghd = cfg_json.get("head_dim")
                    if ghd is None:
                        ghd = dim_full
                    if ghd is not None and int(round(float(prf) * int(ghd))) != rot:
                        raise ValueError(f"config partial_rotary_factor {prf} · global_head_dim "
                                         f"{ghd} != guard dimension_count {rot}")
            swa = rp.get("sliding_attention")
            bswa = guard_keys.get("gemma4.rope.freq_base_swa")
            if isinstance(swa, dict) and bswa is not None:
                if float(swa.get("rope_theta", bswa)) != float(bswa):
                    raise ValueError(f"config sliding rope_theta {swa.get('rope_theta')} != "
                                     f"guard freq_base_swa {bswa}")
    i = np.arange(rot // 2, dtype=np.float64)
    freqs = np.power(np.float64(base), -2.0 * i / np.float64(rot))
    return freqs.astype("<f4").reshape(rot // 2, 1)
