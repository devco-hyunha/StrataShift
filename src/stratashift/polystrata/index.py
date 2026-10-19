"""`index.txt` schema — dense tensor index written by tools/pack_index.py and
read by WeightTable::load (docs/rea/struct_layouts.md §4; DECOMPILED F9).

Header line:  `# align <align> pool <bytes> tensors <count>`
Rows: 19 space-separated fields, no quoting. A row that does not yield exactly
19 fields is REFUSED (weights.cpp:204-207: a wrong byte offset decodes to a
plausible weight — refuse loudly, never partial-parse).
"""
from __future__ import annotations

from dataclasses import dataclass

from stratashift.polystrata.headers import INDEX_FIELDS

# field order per pack_index.py:44-60 / weights.cpp:190-205 (DECOMPILED sscanf)
ROW_FIELDS: tuple[str, ...] = (
    "name", "file", "kind", "src_off", "src_bytes", "dst_off", "dst_bytes",
    "ne0", "ne1", "code_bits", "code_bias", "group_elems", "codebook",
    "has_offset", "codes_bytes", "scales_bytes", "offset_bytes",
    "scales_fp16", "act_kind",
)

# kind codes (pack_index.py:81, weights.cpp:211-214)
KIND_VERBATIM = 0
KIND_BF16_IN_F32 = 1
KIND_F32_COPY = 2
KIND_F16_IN_F32 = 3
KIND_BF16_RAW16 = 4
KIND_F16_RAW16 = 5

_INT_FIELDS = {  # field → signed? (sscanf %d/%lld are signed, %llu unsigned)
    "file": False, "kind": True, "src_off": False, "src_bytes": False,
    "dst_off": False, "dst_bytes": False, "ne0": True, "ne1": True,
    "code_bits": True, "code_bias": True, "group_elems": True, "codebook": True,
    "has_offset": True, "codes_bytes": False, "scales_bytes": False,
    "offset_bytes": False, "scales_fp16": True, "act_kind": True,
}


@dataclass(frozen=True)
class IndexHeader:
    align: int
    pool: int
    tensors: int


@dataclass(frozen=True)
class IndexRow:
    values: dict  # field → int/str, 19 keys


def parse_index_header(line: str) -> IndexHeader:
    parts = line.split()
    if len(parts) != 7 or parts[0] != "#" or parts[1] != "align" \
            or parts[3] != "pool" or parts[5] != "tensors":
        raise ValueError(f"index.txt: bad header line {line!r}")
    try:
        align = int(parts[2])
        pool = int(parts[4])
        tensors = int(parts[6])
    except ValueError as exc:
        raise ValueError(f"index.txt: header not numeric: {line!r}") from exc
    return IndexHeader(align=align, pool=pool, tensors=tensors)


def parse_index_row(line: str) -> IndexRow:
    tokens = line.split()
    if len(tokens) != INDEX_FIELDS:
        raise ValueError(
            f"index.txt: row has {len(tokens)} fields, need exactly {INDEX_FIELDS} "
            f"(refused — partial parse means a wrong byte offset)"
        )
    values: dict = {"name": tokens[0]}
    for field, token in zip(ROW_FIELDS[1:], tokens[1:]):
        try:
            values[field] = int(token)
        except ValueError as exc:
            raise ValueError(f"index.txt: field {field!r} not numeric: {token!r}") from exc
    return IndexRow(values=values)


# --- engine-format index writer (p3-01) ---------------------------------------
# Byte-compatible with tools/pack_index.py output (pin c566ff4), which is the
# contract WeightTable::load sscanf-parses (struct_layouts §4, DECOMPILED F9).

# weights.cpp:66-72 file_name(): id → pack file
FILE_ID: dict[str, int] = {"dense.bin": 0, "embd.bin": 1, "experts.bin": 2, "extra.bin": 3}

# pack_index.py:74-79 — activation kind is a property of source_type; the S-form
# cannot recover it (Q5_0 vs Q5_K, IQ4_NL vs IQ4_XS are indistinguishable there).
ACT_Q8_0, ACT_Q8_K = 0, 1
VDT: dict[str, int] = {
    "Q2_0": ACT_Q8_0, "Q4_0": ACT_Q8_0, "Q5_0": ACT_Q8_0, "Q8_0": ACT_Q8_0, "IQ4_NL": ACT_Q8_0,
    "Q3_K": ACT_Q8_K, "Q4_K": ACT_Q8_K, "Q5_K": ACT_Q8_K, "Q6_K": ACT_Q8_K, "IQ4_XS": ACT_Q8_K,
}

_PLANE_KEYS = ("codes", "scales", "offsets")


def engine_rows(man: dict) -> list[tuple]:
    """19-field rows for man["tensors"] in insertion order, as pack_index.py emits them.

    Engine-form plane sizes are emitted, not re-derived (the width bug lived in
    re-deriving — pack_index.py docstring), and the loader checks what it was told:
    scales_bytes/2 when scales_fp16, three planes summing to src_bytes and dst_bytes.
    """
    align = int(man.get("align", 64))
    rows: list[tuple] = []
    dst = 0
    for name, v in man["tensors"].items():
        f = v.get("file")
        if f not in FILE_ID:
            raise ValueError(f"index.txt: unknown file {f!r} for {name}")
        shape = v.get("shape") or [0, 0]
        ne0 = int(shape[0])
        ne1 = int(shape[1]) if len(shape) > 1 else 0
        elems = int(v.get("elements", 0))
        st = v.get("source_type") or "?"

        codes_bytes = scales_bytes = offset_bytes = 0
        scales_fp16 = 0
        widen_here = 0
        act_kind = VDT.get(st, ACT_Q8_0)
        if "values" in v:
            src_off = int(v["values"]["offset"])
            src_bytes = int(v["values"]["bytes"])
        else:
            planes = [k for k in _PLANE_KEYS if isinstance(v.get(k), dict)]
            lo = min(int(v[k]["offset"]) for k in planes)
            hi = max(int(v[k]["offset"]) + int(v[k].get("bytes", 0)) for k in planes)
            src_off, src_bytes = lo, hi - lo
            # planes must be CONTIGUOUS in order — the loader reads them as one span
            expect = src_off
            for k in _PLANE_KEYS:
                if k not in planes:
                    continue
                if int(v[k]["offset"]) != expect:
                    raise ValueError(f"index.txt: {name} planes not contiguous")
                expect += int(v[k].get("bytes", 0))
            if expect != src_off + src_bytes:
                raise ValueError(f"index.txt: {name} plane span does not fill the span")
            if v.get("has_offset") and "offsets" not in planes:
                raise ValueError(f"index.txt: {name} has_offset without offsets plane")
            code_bits = int(v.get("code_bits") or 0)
            group_elems = int(v.get("group_elems") or 0)
            if code_bits == 0 or group_elems == 0:
                raise ValueError(f"index.txt: {name} quantized with code_bits={code_bits}")
            if ne0 % group_elems:
                raise ValueError(f"index.txt: {name} ne0 {ne0} not a multiple of {group_elems}")
            n_groups = ne0 // group_elems
            # engine pack_codes (strata_pack.py:60, pin c566ff4): per = 8//bits codes per
            # byte, LSB-first. bits=5 → per=1 (one byte per code, 3 bits wasted), so the
            # codes plane is ceil(elems/per), NOT elems*bits//8.
            per = 8 // code_bits
            codes_bytes = (elems + per - 1) // per
            scales_fp16 = 1 if v.get("scales_fp16") else 0
            offs_fp16 = 1 if v.get("offsets_fp16") else 0
            # engine form: scales are ALWAYS 4 B per scale (fp16 in the pack is widened, exact)
            scales_bytes = ne1 * n_groups * 4
            offset_bytes = ne1 * n_groups * 4 if v.get("has_offset") else 0
            src_planes = (codes_bytes
                          + ne1 * n_groups * (2 if scales_fp16 else 4)
                          + (ne1 * n_groups * (2 if offs_fp16 else 4) if v.get("has_offset") else 0))
            plane_sum = (int(v["codes"]["bytes"]) + int(v["scales"]["bytes"])
                         + int(v.get("offsets", {}).get("bytes", 0)))
            if src_planes != plane_sum:
                raise ValueError(f"index.txt: {name} derived planes {src_planes} != manifest {plane_sum}")
            if src_planes != src_bytes:
                raise ValueError(f"index.txt: {name} derived planes {src_planes} != span {src_bytes}")
            if scales_fp16:
                widen_here = scales_bytes - ne1 * n_groups * 2
            if offs_fp16 and v.get("has_offset"):
                # the offsets plane is widened too: engine form has 4 B per group
                # (weights.cpp §3 S4+offset f32), the pack stores fp16 — both planes widen
                widen_here = widen_here + (offset_bytes - ne1 * n_groups * 2)

        vf = bool(v.get("values_fp16"))
        if st == "BF16":
            kind, dst_bytes = 1, elems * 2
        elif st == "F16" and not vf:
            kind, dst_bytes = 3, elems * 2
        elif st == "F16":
            kind, dst_bytes = 0, src_bytes
        elif st == "F32":
            kind, dst_bytes = 2, elems * 4
        else:
            kind, dst_bytes = 0, codes_bytes + scales_bytes + offset_bytes
            if dst_bytes != src_bytes + widen_here:
                raise ValueError(f"index.txt: {name} engine form {dst_bytes} != pack {src_bytes} + widen {widen_here}")

        cb = v.get("codebook")
        rows.append((name, FILE_ID[f], kind, src_off, src_bytes, dst, dst_bytes, ne0, ne1,
                     int(v.get("code_bits") or 0), int(v.get("code_bias") or 0),
                     int(v.get("group_elems") or 0), 1 if cb == "IQ4NL" else 0,
                     1 if v.get("has_offset") else 0,
                     codes_bytes, scales_bytes, offset_bytes, scales_fp16, act_kind))
        dst = (dst + dst_bytes + align - 1) // align * align
    return rows


def write_engine_index(path, man: dict) -> int:
    """Write index.txt byte-compatible with tools/pack_index.py (3 header lines + rows).

    Returns the pool bytes (the engine checks arena >= pool, weights.cpp:235).
    """
    align = int(man.get("align", 64))
    rows = engine_rows(man)
    pool = 0
    lines = ["# strata pack index v3  --  generated by tools/pack_index.py; do not hand-edit",
            "# align %d pool %d tensors %d" % (align, pool, len(rows)),
            "# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias "
            "group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind"]
    dst = 0
    for r in rows:
        lines.append("%s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d" % r)
        dst = (dst + r[6] + align - 1) // align * align
    lines[1] = "# align %d pool %d tensors %d" % (align, dst, len(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return dst
