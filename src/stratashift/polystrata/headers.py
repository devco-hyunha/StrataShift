"""PolyStrata format constants — DRAFT.

Every constant cites `docs/rea/` (SOURCE/VERIFIED/DECOMPILED evidence) and the
pinned PolyStrata commit. Status: DRAFT — promotion to final requires Phase 3
engine-load verification (CONSTRAINTS C4). Do not "finalize" from REA alone.

Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` (docs/REFERENCES.md).
"""

# --- GGUF header (docs/rea/struct_layouts.md §1; loader_entrypoints.md F1-F4) ---
GGUF_MAGIC = 0x46554747        # "GGUF" little-endian u32; SOURCE gguf_reader.hpp:443, DECOMPILED F2
GGUF_VERSION = 3               # v3 only; SOURCE gguf_reader.hpp:446, DECOMPILED F3
GGUF_ALIGN_DEFAULT = 32        # general.alignment default; SOURCE gguf_reader.hpp:472-475, DECOMPILED 0x20 (F4)

# --- safetensors data section alignment (HF spec; p1-01 ingest) ---
SAFETENSORS_ALIGN = 8

# --- STRP expert profile (docs/rea/struct_layouts.md §5; loader_entrypoints.md F6-F7) ---
STRP_MAGIC = b"STRP"           # u32 0x50525453 LE; VERIFIED data bytes + DECOMPILED F6
STRP_VERSION = 1               # version 1 only; SOURCE expert_cache.cpp:97, DECOMPILED F6
STRP_HEADER_U32_COUNT = 5      # version, n_layers, n_expert, slots, n_ranked

# --- expert blob paging (docs/rea/paging_model.md §2; loader_entrypoints.md F11, F13-F14) ---
BLOB_BYTES = 1_382_400         # 0x151800; SOURCE load_main.cpp:19, DECOMPILED F11
BLOBS_PER_LAYER = 512          # 0x200; SOURCE load_main.cpp:21, DECOMPILED F11
LOAD_CHUNK_DEFAULT = 16 << 20  # 0x1000000; SOURCE load_main.cpp, DECOMPILED F11

# --- Q2_0 bitpack contract (docs/rea/bitpack_rules.md §1; loader_entrypoints.md F5) ---
Q2_0_BLOCK_BYTES = 18          # fp16 scale (2 B) + codes (16 B); SOURCE dequant.hpp, DECOMPILED F5
QK2_0 = 64                     # elements per Q2_0 block (PROVEN by artifact brackets)
Q2_0_CODES_PER_BYTE = 4        # LSB-first: byte j//4, bits (j%4)*2, mask 3, symbol code-1

# --- Q4_0 bitpack contract (docs/rea/struct_layouts.md §2; p3-03) ---
Q4_0_BLOCK_BYTES = 18          # fp16 scale (2 B) + qs (16 B); SOURCE dequant.hpp:88-95,
                               # canonical_xcheck.py MAPPINGS:565 (block_bytes 18)
QK4_0 = 32                     # elements per Q4_0 block; SOURCE dequant.hpp:91 (j<16, j+16),
                               # canonical_xcheck.py MAPPINGS:565 (block_elems 32)
Q4_0_CODES_PER_BYTE = 2        # pack_codes LSB-first 2 codes/byte; SOURCE strata_pack.py:60-67
Q4_0_CODE_BIAS = -8            # codebook -8..7, bias inside the integer; SOURCE
                               # canonical_xcheck.py:107-110 from_s4_q4_0, dequant.hpp:92-93
Q4_0_MIN_PACK_ELEMENTS = 128   # engine pack gate: pack_index.py:144-152 requires codes→scales
                               # planes CONTIGUOUS and strata_pack.py:43 ALIGN=64 pads every plane
                               # start to 64 B; codes plane = n_el/2 B → contiguity iff
                               # n_el % 128 == 0 (p3-03: engine refuses attn_norm 64 elems)

# --- Q4_K bitpack contract (canonical_xcheck.py MAPPINGS:567; p3-04) ---
Q4_K_BLOCK_BYTES = 144         # d fp16(2) + dmin fp16(2) + 12 packed scale bytes + qs(128);
                               # SOURCE canonical_xcheck.py:567 MAPPINGS (block_bytes 144)
QK_K = 256                     # elements per K-quant block; SOURCE canonical_xcheck.py:567
                               # (block_elems 256), dequantize_row_q4_K (L1529)
Q4_K_GROUPS = 8                # 8 groups of 32 per block (group_elems 32); SOURCE
                               # canonical_xcheck.py:567 + to_s4_q4_k:128-160
Q4_K_SCALE_BYTES = 12          # packed 6-bit scales+mins; SOURCE get_scale_min_k4 (ggml-quants.c
                               # L880, canonical_xcheck.py:111)
Q4_K_MIN_PACK_ELEMENTS = 512   # engine pack gate (p3-04): codes plane 128 B/block (64-mult),
                               # scales 32 B/block, offsets 32 B/block; offsets start = 160n,
                               # contiguity needs 160n % 64 == 0 → n even → elements % 512 == 0

# --- Q5_K bitpack contract (canonical_xcheck.py MAPPINGS:568; p3-05) ---
Q5_K_BLOCK_BYTES = 176         # d fp16(2) + dmin fp16(2) + 12 packed scale bytes + qh(32) + qs(128);
                               # SOURCE canonical_xcheck.py:568 MAPPINGS (block_bytes 176)
Q5_K_QH_BYTES = 32             # qh[QK_K/8] — the 5th bit plane; SOURCE to_s8_q5_k:370 (qh=a[:,16:48])
Q5_K_MIN_PACK_ELEMENTS = 512   # engine pack gate (p3-05): codes plane 256 B/block (S8, 64-mult),
                               # scales 32 B/block, offsets 32 B/block; offsets start = 288n,
                               # contiguity needs 288n % 64 == 0 → n even → elements % 512 == 0

# --- index.txt schema (docs/rea/struct_layouts.md §4; loader_entrypoints.md F9) ---
INDEX_FIELDS = 19              # sscanf refuses n != 19; SOURCE weights.cpp:190-205, DECOMPILED F9
