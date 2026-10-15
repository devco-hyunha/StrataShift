# loader_entrypoints — 로더 진입점 표

Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` (GitHub `main`).
모든 항목은 **핀된 소스 인용**이다. 디컴파일로 얻은 주소·심볼은 0개 (p0-02에서 채움).

## A. GGUF 헤더·메타데이터 (입력 레이어)

| # | file:line | function | role |
|---|-----------|----------|------|
| A1 | `include/strata/artifact/gguf_reader.hpp:442` | `GgufFile::open` 마직 검사 | `magic != 0x46554747u` → "not a GGUF file" 거부. v3만 허용 (`:446`) |
| A2 | `gguf_reader.hpp:256,262` | `Cursor` / `Cursor::read<T>` | 헤더 스트리밍 리터 (파일 전체 로드 없음) |
| A3 | `gguf_reader.hpp:284,298` | `MetaType` / `read_value` | KV 메타데이터 전 타입 (U64/I64/F64 포함, 배열 depth≤2) |
| A4 | `gguf_reader.hpp:472-476` | `alignment_`, `data_start_` | `general.alignment` 기본 **32**, `data_start = ceil(pos, align)` |
| A5 | `gguf_reader.hpp:456-482` | tensor 디렉토리 파싱 | `nd` 1..4, 중복 tensor명 거부, shard-aware (`split.*` keys) |
| A6 | `gguf_reader.hpp:503` | `tensor_payload_bytes` | type id → tensor 바이트 수 |
| A7 | `gguf_reader.hpp:639,752,796` | `read_geometry` / `geometry_unsupported` / `check_architecture` | 메타데이터 → `ModelGeometry`, 불일치는 **거부** (mis-run 안 함) |

## B. dense 가중치 로드 (pack → arena)

| # | file:line | function | role |
|---|-----------|----------|------|
| B1 | `src/core/weights.cpp:93,125,174` | `WeightTable::load` | `<pack>/index.txt`를 읽고 tensor를 arena에 복사 |
| B2 | `weights.cpp:190-205` | row parse | `sscanf` **19 fields** 고정 — 부족하면 거부 (오프셋 오판 방지) |
| B3 | `weights.cpp:196` | 헤더 라인 | `# align %d pool %d tensors %d` |
| B4 | `weights.cpp:222-236` | skip set / arena compact | 건너뛴 행은 index 순서로 재배치, `arena_bytes < pool`이면 거부 |
| B5 | `weights.cpp:240-263` | pinned staging 2버퍼 | 5 GB pageable upload 방지 (pinned reuse) |

## C. Expert 아레나 로드 (스트리밍 루프의 원형)

| # | file:line | function | role |
|---|-----------|----------|------|
| C1 | `src/core/load_main.cpp:19-21` | `strata-load` main | `blob_bytes=1382400`, `blobs_per_layer=512`, `chunk=16 MiB`, `threads=8` |
| C2 | `load_main.cpp:47-60` | `PinnedArena` | `layers × 512 × 1,382,400` 바이트 pinned 할당 |
| C3 | `src/core/concurrent_main.cpp:88`, `overlap_main.cpp:77` | `f.seekg(i * 1382400)` | expert `i`는 **정확히 blob stride**로 접근 |

## D. Expert 소스·캐시 (페이징 레이어)

| # | file:line | function | role |
|---|-----------|----------|------|
| D1 | `include/strata/core/expert_source.hpp:1-24` | contract | layer별 `x_f`(pinned, 주소 불변), routed 10 expert id + router weight, `k×n_embd` 반환 |
| D2 | `expert_source.hpp:515` | arena 산수 | `experts.bin` = 33,973,862,400 B = `48 × 512` blobs |
| D3 | `src/core/expert_cache.cpp:72-137` | `read_expert_profile` | `STRP` 마직 + version 1 + (layer,expert) ranked |
| D4 | `expert_cache.cpp:170-206` | `write_expert_profile` | little-endian (`make_profile.py`의 `<`) |
| D5 | `expert_cache.cpp:374-378` | `slots_within(bytes)` | 오프셋 배열 `off_`로 slot 수 계산 |
| D6 | `src/core/vmm.cpp:14,62-72` | `vmm_granularity` | `cuMemGetAllocationGranularity(MINIMUM)` — VMM chunk 단위 |

## E. Python 측 참조 구현 (StrataShift가 따라야 할 정본)

| # | file | role |
|---|------|------|
| E1 | `tools/gguf_reader.py` | `GGUFFile` — C++ 리더의 참조 (P0.S6부터) |
| E2 | `tools/pack_layer.py` | blob 레이아웃 산수·assert (struct_layouts의 근거) |
| E3 | `tools/pack_index.py` | `index.txt` 19열 스키마 생성 |
| E4 | `tools/make_profile.py` | `STRP` 프로필 생성 |

**p0-02 게이트**: 이 표의 file:line은 소스 인용이다. Ghidra로 얻는 함수 주소·심볼은
`DECOMPILED:` 접두어로 구분해 추가하고, 소스 인용과 충돌하면 소스를 정본으로 삼지
**말 것** — 런타임 검증 전 확정 금지 (CONSTRAINTS C4).

## F. DECOMPILED 진입점 (p0-02, Ghidra 12.1.4)

분석: Ghidra 프로젝트 `StrataShift-PolyStrata-c566ff4` (`/opt/data/ghidra-projects/`),
Auto-analyze + Decompiler, postScript `DecompileDump.java`. 덤프: `/opt/data/ghidra-decomp/*.decomp.txt`.
`.o`는 ELF relocatable — **주소는 재배치 전(base 0x100000 계열)**, 심볼은 demangled 그대로.
`strata-gguf`/`strata-dequant`는 static·stripped — 파싱 코드가 `main`에 inline.

| # | 산출물 | address | symbol (demangled) | role | 소스 대응 |
|---|--------|---------|--------------------|------|-----------|
| F1 | strata-gguf | `001025c0` | `main` | GGUF open·헤더 파싱 (inline) | A1~A7 |
| F2 | strata-gguf | inline @ main | magic check `!= 0x46554747` → "not a GGUF file (bad magic)" | GGUF magic | A1 (`:443`) |
| F3 | strata-gguf | inline @ main | `version != 3` → "GGUF v…, this reader handles v3" | v3 gate | A1 (`:446`) |
| F4 | strata-gguf | inline @ main | `local_120 = 0x20` (open 전 초기화) | alignment 기본 32 | A4 (`:472-475`) |
| F5 | strata-dequant | main (inline) | Q2_0 scalar unpack 루프 (64 elem, `>> ((j&3)*2) & 3`, `-1`, `*d`) | 2-bit LSB unpack | E2, bitpack §1 |
| F6 | expert_cache.o | `001048f0` | `strata::core::read_expert_profile` | STRP 프로필 파싱 (`magic == 0x50525453`, version==1) | D3 (`expert_cache.cpp:72`) |
| F7 | expert_cache.o | `00103b40` | `strata::core::write_expert_profile` | STRP 생성 (LE) | D4 |
| F8 | expert_cache.o | `00103200` | `ExpertCache::slots_within(bytes)` | offset 배열로 mapped slot 수 | D5 (`:374-378`) |
| F9 | weights.o | `001020e0` | `WeightTable::load` | `index.txt` 19-field sscanf 파싱·arena 복사 | B1~B4 |
| F10 | weights.o | `00100530` | `WeightTable::find` | tensor명 lookup | B1 |
| F11 | load_main.o | `00100690` | `main` | blob stride `0x151800`(=1,382,400)·`0x200`(=512)·chunk 기본 `0x1000000`(16 MiB) | C1·C2 |
| F12 | vmm.o | `00100510` | `strata::core::vmm_granularity` | `cuMemGetAllocationGranularity(MINIMUM)` (driver entry point v12000) | D6 (`vmm.cpp:62-72`) |
| F13 | concurrent_main.o | thread lambda | `lVar += 0x151800` 루프 | expert i stride | C3 (`:88`) |
| F14 | overlap_main.o | `001005c0` | `main` | `+ 0x151800` stride 2곳 | C3 (`:77`) |

**충돌 없음**: F2~F5·F6·F9·F11의 상수·문자열·형식은 핀 소스 A1·A4·B2·C1·E2와 **일치**.
F12의 `gran` 실측값은 런타임(GPU) 의존 → `HYPOTHESIS:` 유지 (paging_model §5).
