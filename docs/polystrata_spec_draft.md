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
| STRP ranked pair u16·(layer,expert) | `struct_layouts.md §5` VERIFIED | `expert_cache.cpp:120-137` | `polystrata.strp.build_strp` (p0-04, 엔진 실물 byte-identical) |
| STRP rank int32[nl×ne] (−1 non-resident) | `struct_layouts.md §5` VERIFIED | `expert_cache.hpp:68` | `polystrata.strp` (p0-04; 불변량 `rank[ranked[i]]==i` 테스트) |
| expert blob `1,382,400` (`0x151800`) | `paging_model.md §2` + F11 | `load_main.cpp:19` | `headers.BLOB_BYTES` |
| blobs/layer 512 (`0x200`) | `paging_model.md §2` + F11 | `load_main.cpp:21` | `headers.BLOBS_PER_LAYER` |
| chunk 기본 16 MiB (`0x1000000`) | `paging_model.md §2` + F11 | `load_main.cpp:20` | `headers.LOAD_CHUNK_DEFAULT` |
| Q2_0 block 18 B / QK 64 | `bitpack_rules.md §1` + F5 (DECOMPILED 루프) | `dequant.hpp:69-78` | `headers.Q2_0_BLOCK_BYTES`, `headers.QK2_0` |
| Q2_0 unpack `(qs[j/4] >> (j%4)*2) & 3`, `v=(code−1)·d` | `bitpack_rules.md §1` + F5 | `dequant.hpp:72-77` | `headers.Q2_0_CODES_PER_BYTE` (Phase 2 pack.py가 구현) |
| scale fp16(pack) ↔ f32(engine), 필드 `scales_fp16` | `bitpack_rules.md §4` | `pack_index.py:31-40` | `index.ROW_FIELDS` |
| `index.txt` 헤더 `# align %d pool %llu tensors %d` | `struct_layouts.md §4` + F9 | `weights.cpp:190` | `index.parse_index_header` |
| `index.txt` 19-field 행, n≠19 거부 | `struct_layouts.md §4` + F9 | `weights.cpp:196-207` | `headers.INDEX_FIELDS`, `index.parse_index_row` |
| kind 0..5 (VERBATIM..F16 raw16) | `struct_layouts.md §4` | `pack_index.py:81`, `weights.cpp:211-214` | `index.KIND_*` |
| GGUF tensor 레코드 순서 `name→n_dims→shape(u64×nd)→dtype→offset(u64)` | `struct_layouts.md §1` VERIFIED p3-01 | `gguf_reader.hpp:463-469` | `ingest/gguf.py` |
| 아키텍처 가드 compiled `embedding_length=2560`, `block_count=48` (+ssm/att/indexer/hc 키) | `struct_layouts.md §1` VERIFIED p3-01 | `geometry.hpp`, `gguf_reader.hpp` read_geometry | fixture guard 키 |
| PLE `per_layer_token_embd.weight`은 pack에 없음; shard2에 `shape[0]=160`(PLE_HEAD_DIM), 타입 IQ4_NL/Q5_0/I8 | `layer.hpp:552`, `ngram.cpp:242-267` | `strata_pack.py:301` | fixture PLE tensor |
| IQ4_NL → S4 + IQ4NL codebook 16값(−127..113), code_bias 0, group 32, block 18 B | `canonical_xcheck.py:571` MAPPINGS | pin c566ff4 | `engine_manifest.FORM_INFO`, `iq4_nl_planes` |
| manifest 삽입 순서 = dense group 먼저, embd 나중; index.txt는 삽입 순 행 | `strata_pack.py:268-270` | `pack_index.py` | `cli.main` groups |
| fused expert GGUF layout: shape `(in, out, NE)`, expert `e`는 role tensor 안 **연속 460,800 B run** (GGML dim0 최속) | `struct_layouts.md §1` VERIFIED p3-02 | `pack_layer.py:19-21`, `strata_pack.py:353-355` | `cli.main.cmd_emit_gguf` expert-major 스트림 |
| safetensors expert tensor `(out, NE, in)` → expert `e` slice = `[e·out·in, (e+1)·out·in)` 연속, Q2_0 stream은 expert-major append | p3-02 AC1 (엔진 quantize와 byte-identical) | `tools/gguf_writer.py quantize_q2_0` | `quant/pack.py quantize_q2_0_stream` |
| GGUF data section **EOF flush**: 마지막 tensor 뒤 trailing padding 금지 — verify는 `data_section_offset = filelen − max(offset+expected_bytes)`로 data_start를 역산하므로 trailing 16 B가 전 tensor를 +16 어긋나게 읽는다 (p3-02 FAIL 원인, truncate로 닫음) | `struct_layouts.md §1` VERIFIED p3-02 | `canonical_xcheck.py:619-625`, `gguf_writer.py:177-179` | `cli.main.cmd_emit_gguf` write 후 truncate |
| Q4_0 → S4: block 18 B / 32 elems, 저니블=앞반·고니블=뒤반, codebook −8..7(bias integer 안), scale fp16 | `struct_layouts.md §2` + `bitpack_rules.md` | `canonical_xcheck.py:86-110,566` MAPPINGS, `dequant.hpp:88-95` | `headers.Q4_0_*`, `engine_manifest.FORM_INFO["Q4_0"]`, `q4_0_planes`, `quant/pack.py q4_0_unpack` |
| Q4_0 엔진 pack gate: codes plane(n_el/2 B)가 64 배수여야 plane 연속 — **elements % 128 == 0** 필요; 미만은 엔진 self-build도 pack_index 거부 → emit은 F32 pass-through 유지 | `struct_layouts.md §2` CONFLICT/pack gate (p3-03 실측 FAIL) | `pack_index.py:144-152`, `strata_pack.py:43` ALIGN=64 | `headers.Q4_0_MIN_PACK_ELEMENTS`, `cli.main.cmd_emit_gguf` dense 분기 |
| PLE safetensors `U8[720]` (IQ4_NL raw block bytes) → GGUF `IQ4_NL` shape `[160, 8]` verbatim pass-through | `struct_layouts.md §1` VERIFIED p3-02 | `ngram.hpp:32`, `strata_pack.py:301` | `cli.main.cmd_emit_gguf` PLE 분기 |

## 2. HYPOTHESIS (상수화하지 않음)

| 항목 | 상태 |
|------|------|
| VMM `gran` 실측값 | GPU 런타임 의존 — `paging_model.md §5` HYPOTHESIS 유지 |
| AVX512 SIMD bitplane 매핑 (`vpmultishiftqb` 순서) | `.cu` 커널 미빌드 — `bitpack_rules.md §5` HYPOTHESIS |
| struct 메모리 오프셋(vtable/패딩) | stripped 실행 파일에서 미복원 — 백로그 |
| StrataShift 출력 == engine `experts.bin`+`index.txt` 동등성 | **확정 (p3-01)** — fixture 1레이어에서 experts.bin/dense.bin/embd.bin/index.txt byte-identical, 엔진 verify PASS |
| Q4_0 양자화 규칙 divergence: 엔진에 quantizer 없음(`gguf_writer.py` Q2_0 전용, `dequant.hpp` dequant만). 우리 producer 규칙 = llama.cpp ref(`d=amax/7`, `q=clip(trunc(w/d+8.5),0,15)`); gguf-py `quantize(Q4_0)`는 signed-max(`d=max/−8`, `quants.py:220-237`)로 **바이트 동일하지 않음** — 두 인코딩 모두 엔진 dequant에서 d/2 바운드 내. acceptance 게이트는 strata_pack verify(gguf-py dequantize reference) | HYPOTHESIS (p3-03) — pack format 상수(18 B/32 elems/nibble/bias −8)는 pin-verified, quantizer 규칙은 pin-verified 아님 |

## 3. 검증 게이트

`tests/test_polystrata_constants.py` — 상수 ↔ docs/rea 기록값 대조, 정렬 ceil,
index 헤더/행 파싱(19-field 거부 포함). pytest 전수 PASS (2026-10-15).
