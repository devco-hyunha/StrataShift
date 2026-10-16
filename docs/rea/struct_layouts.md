# struct_layouts — 구조체·레이아웃 오프셋

Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152`.
`SOURCE:` = 핀 소스 인용. `VERIFIED:` = 실데이터 바이트 대조. `HYPOTHESIS:` = 미검증.

## 1. GGUF 파일 헤더

`SOURCE: include/strata/artifact/gguf_reader.hpp:442-482`

| 오프셋 | 필드 | 근거 |
|--------|------|------|
| 0 | `magic: u32 = 0x46554747` ("GGUF", LE) | `:443` — 다르면 거부 |
| 4 | `version: u32` — v3만 허용 | `:446` |
| 8 | `n_tensors: u64` | tensor 디렉토리 루프 |
| … | KV 메타데이터 (`MetaType` 0..11: U8/I8/U16/I16/U32/I32/F32/BOOL/STRING/ARRAY/U64/I64/F64) | `:284` |
| 헤더 끝 | `data_start = ceil(pos, alignment)` | `:476` |
| `general.alignment` | 기본 **32** | `:472-475` |

tensor 엔트리: `name(string)`, `n_dims(u32, 1..4)`, `dtype(u32)`, `offset(u64, data_start 기준)`, shape.
중복 tensor명 → 거부 (`:463`).

`VERIFIED (p3-01, 2026-10-16):` 레코드 순서는 `name → n_dims(u32) → shape(u64×nd) → dtype(u32) →
offset(u64)` (`gguf_reader.hpp:463-469`). 우리 파서(`ingest/gguf.py`)가 `dtype, offset, shape`
순으로 읽은 fixture가 `strata-gguf --check`의 tensor 계수/오프셋 불일치를 만들었고, 순서 수정 후
7 tensors/`data_start 1568`가 엔진과 정확히 일치. 헤더는 magic+version+n_tensors+n_kv = 24 B.

`VERIFIED (p3-01):` 아키텍처 가드 — `read_geometry`+`geometry_unsupported`는 6 필수 키와 모든
pinned 키가 compiled geometry와 같음을 요구. `embedding_length`는 compiled 값 **2560**
(`geometry.hpp`), `block_count` pinned **48**. 그 외 ssm/attention/indexer/hc 키는
`docs/work/p3-01.md` Result의 fixture 값과 동일.

`VERIFIED (p3-02, 2026-10-16):` **data section EOF flush** — 엔진 verify는
`data_section_offset = filelen − max(offset + expected_bytes)`
(`tools/canonical_xcheck.py:619-625`)로 data_start를 tensor 테이블만으로 역산한다.
엔진 writer는 모든 tensor 뒤를 alignment로 pad(`gguf_writer.py:177-179`)하므로 마지막
tensor가 alignment 배수가 아니면 trailing padding이 남아 verify가 전 tensor를 그만큼
어긋나게 읽는다 (p3-02 첫 FAIL: trailing 16 B → expert 48 checks 전부 MISMATCH).
p3-01 fixture GGUF는 마지막 tensor(262,144 B)가 64 배수라 flush. 우리 emit는 write 후
`data_start + max(offset+expected_bytes)`로 truncate하여 flush로 고정 — truncate 후
strata_pack verify 0 bad, dense/PLE bit-exact.

`VERIFIED (p3-02):` fused expert tensor layout — shape `(in, out, NE)`, GGML dim0 최속이라
expert `e`는 role tensor 안 **연속 run** (`pack_layer.py:19-21` "expert `e` is one
CONTIGUOUS 460,800-byte run", `strata_pack.py:353-355` `t.offset + e·nbytes`).
safetensors expert tensor `(out, NE, in)`의 expert `e` slice도 연속
(`[e·out·in, (e+1)·out·in)`) → Q2_0 stream을 expert 순서로 append하면 엔진이 읽는
expert-major와 정확히 일치 (verify 0 bad로 확인).

## 2. ggml type id → 블록 기하 (양자화 tensor 폭)

`SOURCE: gguf_reader.hpp:125-175` (`block_geometry`)

| type id | 이름 | elems/block | bytes/block |
|---------|------|-------------|-------------|
| 0 | F32 | 1 | 4 |
| 1 / 30 | F16 / BF16 | 1 | 2 |
| 2 | Q2_0 | **64** | **18** |
| 3 | Q3_0 | 32 | 20 |
| 6 | Q4_0 | 32 | 22 |
| 7 | Q5_0 | 32 | 24 |
| 8 | Q8_0 | 32 | 34 |
| 9 | Q1_0 | 32 | 36 |
| 10 | Q4_1 | 256 | 84 |
| 11 | Q4_2 | 256 | 110 |
| 12 | Q4_3 | 256 | 144 |
| 13 | Q5_1 | 256 | 176 |

`QK2_0 = 64`는 **assumption이 아니라 아티팩트 오프셋 bracket으로 증명**된 값
(`dequant.hpp:12-14`; llama.cpp PR은 128을 썼고 이 아티팩트는 64).

## 3. Expert blob (1,382,400 B) 내부 레이아웃

`SOURCE: include/strata/kernels/cpu/expert.hpp:44-58` + `tools/pack_layer.py:44-64`
`VERIFIED:` 산수 재계산 → `end == 1,382,400` (본 세션에서 독립 재계산)

| 상수 | 값 | 의미 |
|------|-----|------|
| `H` | 2560 | n_embd |
| `FF` | 640 | expert intermediate width |
| `NE` | 512 | routed experts per layer |
| `QK` | 64 | weights per fp16 scale (Q2_0) |
| `QKA` | 32 | activation elements per scale (Q8_0/Q8_1) |
| `BB` | 18 | bytes per 64-weight Q2_0 block |
| `ROW_GU` | 640 | codes bytes per gate/up row |
| `ROW_D` | 160 | codes bytes per down row |
| `SC_GU` | 40 | weight blocks per gate/up row |
| `SC_D` | 10 | weight blocks per down row |
| `BLOB` | 1,382,400 | `3 × (H·FF·BB / QK)` |

**오프셋 (blob 기준, byte):**

| 필드 | 오프셋 | 크기 |
|------|--------|------|
| `gate_up.codes` | 0 | 819,200 (rows interleaved g0,u0,g1,u1,…) |
| `down.codes` | 819,200 | 409,600 (2,560 rows × 160 B) |
| `gate_up.scales` (fp16) | 1,228,800 | 102,400 (2 × 640 × 40 × 2 B) |
| `down.scales` (fp16) | 1,331,200 | 51,200 (2,560 × 10 × 2 B) |
| end | 1,382,400 | — |

`SOURCE: pack_layer.py:88-93` — 변환은 **pure relayout**: Q2_0 codes와 fp16 scale은
그대로 복사되므로 `dequant(pack) == dequant(ggml)` bit-exact.

GGUF 원본 블록은 `[fp16 scale (2 B)][codes (16 B)]` 순서
(`pack_layer.py:88-91`: `gu[:,:,:,2:]` = codes, `gu[:,:,:2]` = scales).

## 4. `index.txt` (dense tensor 인덱스)

`SOURCE: tools/pack_index.py:44-60` + `src/core/weights.cpp:190-205`

- 헤더 라인: `# align <align> pool <bytes> tensors <count>`
- 행: **19 fields**, space-separated, quoting 없음 (name에 공백 있으면 거부)
- 열 순서: `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias
  group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind`
- `kind`: 0 VERBATIM, 1 BF16_IN_F32, 2 F32_COPY, 3 F16_IN_F32, 4 BF16(raw16), 5 F16(raw16)
  (`pack_index.py:81`, `weights.cpp:211-214`)
- loader는 plane 크기를 **재계산하지 않고 지시대로 검증**: `scales_bytes/2` when `scales_fp16`,
  3 plane 합 = `src_bytes` = `dst_bytes` (`pack_index.py:62-70`, `weights.cpp:204`)

## 5. `STRP` expert 프로필

`SOURCE: include/strata/core/expert_cache.hpp:42-46,68` + `src/core/expert_cache.cpp:170-206`
`VERIFIED: data/expert-profile.bin` 실바이트 대조 (본 세션, 실행 없음)

| 오프셋 | 필드 | 실측 |
|--------|------|------|
| 0 | `magic "STRP"` (4 B) | `b'STRP'` ✓ |
| 4 | `version, n_layers, n_expert, slots, n_ranked` (5 × u32, LE) | `(1, 48, 512, 24576, 24576)` ✓ |
| 24 | `n_ranked × (u16 layer, u16 expert)` | 첫 pair `(28,288),(46,185),(47,242)` ✓ |
| 24+4·n_ranked | `int32[n_layers × n_expert]` = slot rank (−1 = non-resident) | 98,304 B, 파일 끝과 정확히 일치 ✓ |

파일 크기 196,632 B = 24 + 4·24,576 + 48·512·4. **`HYPOTHESIS:` 없음 — 전항 소스+바이트 이중 확인.**

## 6. 미수집 (p0-02)

- Ghidra 디컴파일로 복원하는 C++ struct 레이아웃 (vtable, 패딩, 필드 순서)
- `PinnedArena`, `ExpertCache`, `ActQ`의 실제 메모리 오프셋
- CUDA VMM granularity 실측값 (`cuMemGetAllocationGranularity`는 런타임 값)

## 7. DECOMPILED 교차 검증 (p0-02, Ghidra 12.1.4)

분석: 프로젝트 `StrataShift-PolyStrata-c566ff4`, Auto-analyze + Decompiler.
덤프 `/opt/data/ghidra-decomp/`. 진입점 표는 `loader_entrypoints.md` §F.

- `DECOMPILED: strata-gguf main @ 001025c0` — `*(int*)hdr != 0x46554747` → "not a GGUF file
  (bad magic)"; `version != 3` → "GGUF v…, this reader handles v3". §1의 magic·v3 게이트 **일치**.
- `DECOMPILED: strata-gguf main` — open 경로에서 `local_120 = 0x20` 초기화 → alignment 기본 **32**
  (§1 `general.alignment` 기본값 **일치**). `.rodata`에 `general.alignment` 문자열 존재 확인.
- `DECOMPILED: expert_cache.o read_expert_profile @ 001048f0` — `magic == 0x50525453` ("STRP" LE),
  `version != 1` 거부, `nl/ne` 모델 대조, `n_ranked > want` 거부. §5 레이아웃 **일치**
  (바이트 VERIFIED + 디컴파일 이중 확인).
- `DECOMPILED: weights.o WeightTable::load @ 001020e0` — `sscanf` 서식 문자열
  `"# align %d pool %llu tensors %d"`와 19-field `"%255s %d %d %llu %llu %llu %llu %lld %lld %d …"`
  실재. §4 스키마 **일치**.
- 충돌 없음. §6의 struct 메모리 오프셋(vtable/패딩)은 stripped 실행 파일에서 복원 불가 →
  백로그 (p0-03에서 `.o` 디컴파일 심볼 타입 복원 시도).
