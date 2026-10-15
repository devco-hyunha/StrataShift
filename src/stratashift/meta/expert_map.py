"""Expert map — MoE FFN expert tensor candidates from header metadata.

Name rules (draft, pattern-based — "후보" per ACCEPTANCE Phase 1; engine-load
confirmation happens in Phase 3, CONSTRAINTS C4):

- Mixtral HF:   model.layers.<L>.experts.<E>.(gate|up|down)_proj.weight
- Qwen2/3 MoE:  model.layers.<L>.mlp.experts.<E>.(gate|up|down)_proj.weight
- GPT-OMoe:     model.layers.<L>.moe.layers.<E>.mlp.(gate|up|down)_proj.weight
- legacy:       model.layers.<L>.experts.<E>.w1|w2|w3
- fused (HF transformers parallelized, e.g. tiny-random-Mixtral fixture):
  model.layers.<L>.mlp.experts.(gate_up_proj|down_proj) with shape[0] = num experts;
  expert e is the contiguous row block [e·prod(shape[1:]), (e+1)·prod(shape[1:]))
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from stratashift.ingest.safetensors_header import DTYPE_SIZES, ModelMeta

# role normalization: canonical FFN roles are gate / up / down (gate_up = fused pair)
_ROLE_CANON = {
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
    "gate": "gate", "up": "up", "down": "down",
    "w1": "gate", "w3": "up", "w2": "down",
    "gate_up_proj": "gate_up",
}

_EXPERT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)(?:\.weight)?$"),
    re.compile(r"(?:^|.)layers\.(\d+)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|gate|up|down)(?:\.weight)?$"),
    re.compile(r"(?:^|.)layers\.(\d+)\.moe\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)(?:\.weight)?$"),
    re.compile(r"(?:^|.)layers\.(\d+)\.experts\.(\d+)\.(w1|w2|w3)$"),
]

_FUSED_PATTERN = re.compile(r"(?:^|.)layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


@dataclass(frozen=True)
class ExpertRef:
    layer_id: int
    expert_id: int
    role: str            # gate | up | down
    tensor_name: str
    data_offsets: tuple[int, int]  # relative to data_start


def classify(name: str) -> tuple[int, int, str] | None:
    """Return (layer, expert, role) for an expert tensor name, else None."""
    for pattern in _EXPERT_PATTERNS:
        m = pattern.search(name)
        if m:
            layer, expert, raw = int(m.group(1)), int(m.group(2)), m.group(3)
            return layer, expert, _ROLE_CANON[raw]
    return None


def build_expert_map(meta: ModelMeta) -> list[ExpertRef]:
    """List FFN expert candidate refs (layer, expert, role) from header metadata."""
    refs: list[ExpertRef] = []
    for t in meta.tensors:
        hit = classify(t.name)
        if hit is not None:
            layer, expert, role = hit
            refs.append(ExpertRef(layer_id=layer, expert_id=expert, role=role,
                                  tensor_name=t.name, data_offsets=t.data_offsets))
            continue
        fused = _FUSED_PATTERN.search(t.name)
        if fused and len(t.shape) >= 2:
            layer = int(fused.group(1))
            role = _ROLE_CANON[fused.group(2)]
            row_elems = 1
            for dim in t.shape[1:]:
                row_elems *= dim
            row_bytes = row_elems * DTYPE_SIZES[t.dtype]
            for expert in range(t.shape[0]):
                refs.append(ExpertRef(layer_id=layer, expert_id=expert, role=role,
                                      tensor_name=t.name,
                                      data_offsets=(expert * row_bytes,
                                                   (expert + 1) * row_bytes)))
    refs.sort(key=lambda r: (r.layer_id, r.expert_id, r.role))
    return refs


def group_by_expert(refs: list[ExpertRef]) -> dict[tuple[int, int], list[ExpertRef]]:
    """Group refs by (layer, expert) — one expert = its gate/up/down rows."""
    grouped: dict[tuple[int, int], list[ExpertRef]] = {}
    for ref in refs:
        grouped.setdefault((ref.layer_id, ref.expert_id), []).append(ref)
    return grouped
