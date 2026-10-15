# polystrata_spec_draft — REA ↔ 핀 소스 ↔ Python 상수 매핑

Phase 0~1 산출물 (MODULE_DESIGN 예고 파일). 근거 접두어 규칙은 `docs/rea/README.md`.
Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` · Ghidra 12.1.4.

**상태: DRAFT.** 상수 초안은 `src/stratashift/polystrata/headers.py`에 고정되어 있고,
확정 승격은 Phase 3 엔진 로드 성공 후에만 (CONSTRAINTS C4).

## 1. 매핑 표

| 규격 항목 | REA 근거 | 핀 소스 | Python 상수 (`polystrata/`) |
|-----------|----------|---------|------------------------------|
| GGUF magic `0x46554747` | `struct_layouts.md §1` + `loader_entrypoints F2` (DECOMPILED) | `gguf_reader.hpp:443` | `headers.GGUF_MAGIC` |
| GGUF version 3 전용 | `struct_layouts.md §1` + F3 | `gguf_reader.hpp:446` | `headers.GGUF_VERSION` |
| GGUF alignment 기본 32 | `struct_layouts.md §1` + F4 (`0x20` init) | `gguf_reader.hpp:472-476` | `headers.GGUF_ALIGN_DEFAULT`, `alignment.gguf_data_start` |
| safetensors data 정렬 8 | p1-01 (HF spec) | — | `headers.SAFETENSORS_ALIGN`, `alignment.safetensors_data_start` |
| STRP magic `0x50525453` | `struct_layouts.md §5` (VERIFIED 바이트) + F6 | `expert_cache.cpp:85-90` | `headers.STRP_MAGIC` |
| STRP version 1 | `struct_layouts.md §5` + F6 | `expert_cache.cpp:97` | `headers.STRP_VERSION` |
| STRP 헤더 5×u32 (version, nl, ne, slots, n_ranked) | `struct_layouts.md §5` VERIFIED | `expert_cache.cpp:84` | `headers.STRP_HEADER_U32_COUNT` |
| STRP ranked pair u16·(layer,expert) | `struct_layouts.md §5` VERIFIED | `expert_cache.cpp:120-137` | (p0-04 builder) |
| STRP rank int32[nl×ne] (−1 non-resident) | `struct_layouts.md §5` VERIFIED | `expert_cache.hpp:68` | (p0-04 builder) |
| expert blob `1,382,400` (`0x151800`) | `paging_model.md §2` + F11 | `load_main.cpp:19` | `headers.BLOB_BYTES` |
| blobs/layer 512 (`0x200`) | `paging_model.md §2` + F11 | `load_main.cpp:21` | `headers.BLOBS_PER_LAYER` |
| chunk 기본 16 MiB (`0x1000000`) | `paging_model.md §2` + F11 | `load_main.cpp:20` | `headers.LOAD_CHUNK_DEFAULT` |
| Q2_0 block 18 B / QK 64 | `bitpack_rules.md §1` + F5 (DECOMPILED 루프) | `dequant.hpp:69-78` | `headers.Q2_0_BLOCK_BYTES`, `headers.QK2_0` |
| Q2_0 unpack `(qs[j/4] >> (j%4)*2) & 3`, `v=(code−1)·d` | `bitpack_rules.md §1` + F5 | `dequant.hpp:72-77` | `headers.Q2_0_CODES_PER_BYTE` (Phase 2 pack.py가 구현) |
| scale fp16(pack) ↔ f32(engine), 필드 `scales_fp16` | `bitpack_rules.md §4` | `pack_index.py:31-40` | `index.ROW_FIELDS` |
| `index.txt` 헤더 `# align %d pool %llu tensors %d` | `struct_layouts.md §4` + F9 | `weights.cpp:190` | `index.parse_index_header` |
| `index.txt` 19-field 행, n≠19 거부 | `struct_layouts.md §4` + F9 | `weights.cpp:196-207` | `headers.INDEX_FIELDS`, `index.parse_index_row` |
| kind 0..5 (VERBATIM..F16 raw16) | `struct_layouts.md §4` | `pack_index.py:81`, `weights.cpp:211-214` | `index.KIND_*` |

## 2. HYPOTHESIS (상수화하지 않음)

| 항목 | 상태 |
|------|------|
| VMM `gran` 실측값 | GPU 런타임 의존 — `paging_model.md §5` HYPOTHESIS 유지 |
| AVX512 SIMD bitplane 매핑 (`vpmultishiftqb` 순서) | `.cu` 커널 미빌드 — `bitpack_rules.md §5` HYPOTHESIS |
| struct 메모리 오프셋(vtable/패딩) | stripped 실행 파일에서 미복원 — 백로그 |
| StrataShift 출력 == engine `experts.bin`+`index.txt` 동등성 | Phase 3 로드 테스트로만 확정 |

## 3. 검증 게이트

`tests/test_polystrata_constants.py` — 상수 ↔ docs/rea 기록값 대조, 정렬 ceil,
index 헤더/행 파싱(19-field 거부 포함). pytest 전수 PASS (2026-10-15).
