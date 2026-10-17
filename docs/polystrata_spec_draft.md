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
| Q4_K → S4: block 144 B / 256 elems, 8 groups of 32, scale=d*sc, offset=−(dmin*m), codebook 0..15(bias 0), 3-plane(codes+scales+offsets); 저니블=group is+0·고니블=group is+1 | `struct_layouts.md §2` + `bitpack_rules.md` | `canonical_xcheck.py:128-160,567` MAPPINGS, `get_scale_min_k4`(:111, ggml-quants.c L880) | `headers.Q4_K_*`, `engine_manifest.FORM_INFO["Q4_K"]`, `q4_k_planes`, `quant/pack.py q4_k_unpack/get_scale_min_k4` |
| Q4_K 엔진 pack gate: codes 128 B/block(64배수) + scales 32 B + offsets 32 B; offsets 시작 160n 연속 → **elements % 512 == 0** 필요; 미만은 F32 pass-through | `struct_layouts.md §2` pack gate (p3-04 실측) | `pack_index.py:144-152`, `strata_pack.py:43` ALIGN=64 | `headers.Q4_K_MIN_PACK_ELEMENTS`, `cli.main.cmd_emit_gguf` dense 분기 |
| Q4_K 양자화 규칙 divergence: 엔진에 quantizer 없음. 우리 producer 규칙 = d=fp16(amax/63), group sc=clip(round((gmax−gmin)/d),1,63), m=clip(round(−gmin/d),0,63) (groups 4..7는 scb[8..11] high 4 bits → m<16), code=clip(round((w+d*m)/(d*sc)),0,15). ggml `make_qkx2` joint search는 미재현 → groups 4..7의 d/2 bound는 큰 range group에서 미보장. acceptance 게이트는 strata_pack verify + engine dequant 왕복 | HYPOTHESIS (p3-04) — pack format 상수(144 B/256 elems/nibble/bias 0/get_scale_min_k4 packing)는 pin-verified, quantizer 규칙은 pin-verified 아님 |
| Q5_K → S8: block 176 B / 256 elems, 8 groups of 32, scale=d*sc, offset=−(dmin*m), codebook 0..31(bias 0), 3-plane(codes 1 B/elem + scales + offsets); 저니블=group is+0·고니블=group is+1, qh 5th bit=+16 (u1=1<<(2j), u2=2<<(2j) — 같은 32 B를 4 steps가 재읽고 bit walk) | `struct_layouts.md §2` + `bitpack_rules.md` | `canonical_xcheck.py:370-416,569` MAPPINGS, `get_scale_min_k4`(:111) | `headers.Q5_K_*`, `engine_manifest.FORM_INFO["Q5_K"]`(code_bits=8 form S8, code_bias 0), `q5_k_planes`, `quant/pack.py q5_k_unpack` |
| Q5_K 엔진 pack gate: codes 256 B/block(64배수) + scales 32 B + offsets 32 B; offsets 시작 288n 연속 → **elements % 512 == 0** 필요; 미만은 F32 pass-through | `struct_layouts.md §2` pack gate (p3-05 실측) | `pack_index.py:144-152`, `strata_pack.py:43` ALIGN=64 | `headers.Q5_K_MIN_PACK_ELEMENTS`, `cli.main.cmd_emit_gguf` dense 분기 |
| Q5_K codes plane 산수: 엔진 `pack_codes`는 per=8//bits codes/byte — bits=5가 아니라 **form S8의 bits=8** → per=1, codes plane = elems (1 B/code). engine_rows는 ceil(elems/per) 사용 (elems*bits//8 아님) | VERIFIED (p3-05 golden byte-identical) | `strata_pack.py:60` pack_codes, `:147` bits=code_bits(form) | `polystrata/index.py engine_rows codes_bytes`, `engine_manifest.q5_k_planes` |
| Q6_K → S8: block 210 B / 256 elems, 16 groups of 16, scale=d*sc (signed int8 sc), codebook 0..63 bias −32, 2-plane(codes 1 B/elem + scales, offsets 없음); ql[l]=(q1 low|q3 high), ql[l+32]=(q2 low|q4 high), qh[l] bit pair (2qidx, 2qidx+1), quads land at STRIDES of 32 → group = element/16 | `struct_layouts.md §2` + `bitpack_rules.md` | `canonical_xcheck.py:422-468,570` MAPPINGS | `headers.Q6_K_*`, `engine_manifest.FORM_INFO["Q6_K"]`(code_bits=8, code_bias −32, has_offset False), `q6_k_planes`, `quant/pack.py q6_k_unpack` |
| Q6_K 엔진 pack gate: codes 256 B/block(S8 per=1) + scales 16×4 = 64 B/block, offsets 없음; scales 시작 256n % 64 == 0 항상 → **elements % 256 == 0** | `struct_layouts.md §2` pack gate (p3-06 실측) | `pack_index.py:144-152`, `strata_pack.py:43` ALIGN=64 | `headers.Q6_K_MIN_PACK_ELEMENTS`, `cli.main.cmd_emit_gguf` dense 분기 |
| Q6_K 양자화 규칙 divergence: 엔진에 quantizer 없음. 우리 producer 규칙 = d=fp16(amax/63), sc_g=clip(rint(gmax_g/d),−63,63) (signed), code=clip(rint(w/(d·sc_g))+32,0,63), sc_g==0 → code 32. bound |w−v| ≤ d·|sc_g|/2는 codebook이 group range를 완전히 덮는 group(0/63 클립 없음)에서만 보장 — ggml joint search 미재현. acceptance 게이트 = strata_pack verify + engine dequant 왕복 | HYPOTHESIS (p3-06) — pack format 상수(210 B/256 elems/signed sc/bias −32/stride-32 quad 배치/pack_codes per=1)는 pin-verified, quantizer 규칙은 pin-verified 아님 |
| Q5_K 양자화 규칙 divergence: 엔진에 quantizer 없음. 우리 producer 규칙 = d=fp16(amax/63), dmin=fp16(max(d,−min/63)) (양수 유지 → m=round(−gmin/dmin) ≤ 63), sc_g=clip(ceil((gmax−gmin)/(31d)),1,63) (ceil → 31·d·sc ≥ rng → code 클립 없음), m_g=clip(round(−gmin/dmin),0,63) (groups 4..7 m<16), code=clip(round((w+dmin*m)/(d*sc)),0,31). ggml `make_qxq5_k` joint search 미재현 → bound는 groups 0..3만 assert. acceptance 게이트 = strata_pack verify + engine dequant 왕복 | HYPOTHESIS (p3-05) — pack format 상수(176 B/256 elems/nibble+bit/bias 0/get_scale_min_k4/pack_codes per=1)는 pin-verified, quantizer 규칙은 pin-verified 아님 |
| Q3_K → S4: block 110 B / 256 elems, 16 groups of 16, scale=d*(sc6−32) (6-bit sc, bias −32), codebook 0..7 bias −4, 2-plane(codes 4-bit per=2 + scales, offsets 없음); group g=8·half+2·j+sub (elements 16g..16g+15), qs[half·32+sub·16+l] bit 2j = c2, hmask[sub·16+l] bit (4·half+j) = m, code = c2+4·m — hmask는 두 half가 같은 32 B 재읽기, bit만 walk (m 리셋 안 함) | `struct_layouts.md §2` + `bitpack_rules.md` | `canonical_xcheck.py:209-247,565` MAPPINGS, `unpack_q3_k_scales`(:170-189) | `headers.Q3_K_*`, `engine_manifest.FORM_INFO["Q3_K"]`(code_bits=4, code_bias −4, has_offset False), `q3_k_planes`, `quant/pack.py q3_k_unpack` |
| Q3_K 엔진 pack gate: codes 128 B/block(S4 per=2, 64배수) + scales 16×4 = 64 B/block, offsets 없음; 연속성 128n % 64 == 0 항상 → **elements % 256 == 0** | `struct_layouts.md §2` pack gate (p3-07 실측) | `pack_index.py:144-152`, `strata_pack.py:43` ALIGN=64 | `headers.Q3_K_MIN_PACK_ELEMENTS`, `cli.main.cmd_emit_gguf` dense 분기 |
| Q3_K 양자화 규칙 divergence: 엔진에 quantizer 없음. 우리 producer 규칙 = d=fp16(amax/63), sc6_g=clip(rint(gmax_g/d)+32,0,63) (signed −32..31), code=clip(rint(w/(d·(sc6−32)))+4,0,7), dl==0 → code 4. bound |w−v| ≤ |dl_g|/2는 codebook(−4..3)이 group range를 완전히 덮는 group(0/7 클립 없음)에서만 보장 — ggml joint search 미재현. acceptance 게이트 = strata_pack verify + engine dequant 왕복 | HYPOTHESIS (p3-07) — pack format 상수(110 B/256 elems/6-bit scales/bias −4/hmask bit-walk/pack_codes per=2)는 pin-verified, quantizer 규칙은 pin-verified 아님 |
| Q8_0 → S8: block 34 B / 32 elems, d fp16 + qs[32] int8, codebook −128..127 (bias −128 inside the integer), 2-plane(codes plane = int8(qs)+128 1 B/elem + fp16 scales 2 B/block, offsets 없음) — canonical code plane는 UNSIGNED, plane bytes는 raw qs bytes와 음수 code에서 128 차이 | `struct_layouts.md §2` + `bitpack_rules.md` | `canonical_xcheck.py:58-84,571` MAPPINGS (code_bias −128, has_offset False) | `headers.Q8_0_*`, `engine_manifest.FORM_INFO["Q8_0"]`(code_bits=8, code_bias −128, has_offset False), `q8_0_planes`, `quant/pack.py q8_0_unpack` |
| Q8_0 엔진 pack gate: codes 32 B/block(S8 per=1) + scales 2 B/block fp16, offsets 없음; scales 시작 align64(32n) 연속 → 32n % 64 == 0 → n even → **elements % 64 == 0**; 그리고 ndim ≥ 2 — pack_index scales 산수 = ne1·n_groups (ne1 = GGUF shape[1]) → 1-D 텐서(ne1=0)는 derived scales 0 → pack_index 거부; 미만/1-D는 F32 pass-through | `struct_layouts.md §2` pack gate (p3-09 실측) | `pack_index.py:144-152`, `strata_pack.py:43` ALIGN=64, `pack_index.py:174-186` (scales = ne1·n_groups) | `headers.Q8_0_MIN_PACK_ELEMENTS`, `cli.main.cmd_emit_gguf` dense 분기 |
| Q8_0 양자화 규칙: 엔진에 quantizer 존재 — `mtp_pack.py:100 q8_0` (d = amax/127 f32, inv = 1/d (d==0 → 0), q = np.rint(x·inv).astype(int8), 저장 d = fp16(d)). 우리 `quantize_q8_0_stream`는 이 quantizer와 **바이트 동일** (random + edges: all-zero, tiny amax, large amax, fp16-boundary amax = 65504·127 — 경계 밖은 d가 fp16 overflow, contract domain 아님). `mtp_rt.py:34`는 sign·floor(|v|+0.5) (round-half-away) — tie에서 mtp_pack과 divergence, 계약은 mtp_pack | VERIFIED (p3-09, test_q8_0 바이트 동일 게이트) — pack format 상수(34 B/32 elems/bias −128/plane +128)와 quantizer 규칙 모두 pin-verified | `mtp_pack.py:100-108`, `canonical_xcheck.py:58-84` | `quant/pack.py quantize_q8_0_stream`, `tests/test_q8_0.py` |
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

## 4. HF naming → engine GGUF 키 매핑 (p4-01, 2026-10-17)

엔진 키 어휘 정본: `src/core/layout.cpp:60-110` want2/want1 suffix, prefix `blk.<layer>.`
(`LayerView::name`, layout.cpp:137-139), pin c566ff4. HF naming 실측: mixtral-tiny 헤더
(21 tensors, BF16) + qwen4exp HF fixture. shape 규약: HF `[out,in]` row-major = engine GGUF
`(ne0=in, ne1=out)` row-major — 바이트 동일, shape metadata 전치 (struct_layouts §1 VERIFIED,
p3-02 fused expert 계약).

| HF key | engine GGUF key | shape 변환 | dtype |
|--------|-----------------|-----------|-------|
| `model.embed_tokens.weight` | `token_embd.weight` | verbatim | pass-through |
| `lm_head.weight` | `output.weight` | verbatim | pass-through |
| `model.layers.L.input_layernorm.weight` | `blk.L.attn_norm.weight` | 1-D | F32 |
| `model.layers.L.post_attention_layernorm.weight` | `blk.L.ffn_norm.weight` | 1-D | F32 |
| `model.layers.L.mlp.gate.weight` | `blk.L.ffn_gate_inp.weight` | verbatim | dense preset |
| `model.layers.L.mlp.experts.gate_up_proj` (NE,2FF,H) | `blk.L.ffn_gate_exps.weight` (H,FF,NE) | expert e gate rows `[0,FF)` | Q2_0 |
| 〃 | `blk.L.ffn_up_exps.weight` (H,FF,NE) | expert e up rows `[FF,2FF)` | Q2_0 |
| `model.layers.L.mlp.experts.down_proj` (NE,H,FF) | `blk.L.ffn_down_exps.weight` (FF,H,NE) | expert e 연속 run | Q2_0 |
| `model.layers.L.mlp.shared_expert.{gate,up,down}_proj.weight` | `blk.L.ffn_{gate,up,down}_shexp.weight` | 전치 | dense preset |
| `model.layers.L.mlp.shared_expert_gate.weight` | `blk.L.ffn_gate_inp_shexp.weight` | verbatim | Bf16InF32 (BF16 바이트 verbatim) |
| `model.layers.L.self_attn.{q,k,v,o}_proj.weight` | `blk.L.{attn_q,attn_k,attn_v,attn_output}.weight` | 전치 | dense preset |
| `model.layers.L.self_attn.q_gate_proj.weight` | `blk.L.attn_q_gate.weight` (part; fused attn_q = q rows + q_gate rows, `q_full`=n_head·2·head_dim, layer.hpp:363) | 전치 | dense preset |
| `model.layers.L.linear_attn.{q,k,v}_proj.weight` | `blk.L.attn_qkv.weight` (concat q,k,v rows, bytes verbatim) | fused | dense preset |
| `model.layers.L.linear_attn.{g_proj,o_proj,conv1d,A_log,dt_bias,norm}` | `blk.L.{attn_gate,ssm_out,ssm_conv1d,ssm_a,ssm_dt.bias,ssm_norm}` | 1-D→2-D broadcast rows (fixture 실측) | F32 |
| `model.layers.L.indexer.{q,k}_proj.weight` | `blk.L.indexer.{q,k}_proj.weight` | 전치 | dense preset |

가드 keys HF 헤더 산출(never invented): `embedding_length`=embed_tokens shape[1],
`block_count`=layer 수(≤ preset — partial fixture), `expert_count`=NE,
`expert_feed_forward_length`=FF, `attention.head_count`=q_proj shape[0]/head_dim,
`attention.head_count_kv`=k_proj shape[0]/head_dim.

미확인 HF naming은 매핑하지 않는다 (p4-01 리스크: HYPOTHESIS 확장 금지). 전 engine
키 집합 커버리지·multi-shard는 p4-02 백로그.

## 5. HF multi-shard 입력 — 엔진 split 검증 미러 (p4-02, 2026-10-17)

HF transformers checkpoint: `model-00001-of-NNNNN.safetensors` … + `model.safetensors.index.json`
(`{"metadata": {"total_size": N}, "weight_map": {tensor: shard file}}`). tensor 이름 → shard
파일; 오프셋은 shard별 `data_offsets`(shard data_start 기준, p1-01 헤더 규격).

엔진 split 모델 규칙 (pin c566ff4, read-only) — `ingest/shard_index.py` 게이트 미러:

| 엔진 근거 (실측 2026-10-17) | 규칙 | 우리 게이트 |
|------------------------------|------|-------------|
| `gguf_split.hpp:18-50` | shard 번호 `-NNNNN-of-MMMMM` (5자리), 누락 shard는 이름을 지목하는 오류 | weight_map이 지목한 shard 파일 전수 존재 확인, 누락 시 이름 지목 오류 |
| `gguf_reader.hpp:565-590` (validate_split) | shard tensor 디렉토리 합계 ≠ 선언 총계 → 거부; metadata shard(00001) 우선 | Σ shard 헤더 tensor 수 == weight_map 수 (count gate, 구축 순서 미러) |
| `gguf_reader.hpp:536-540` (GgufModel ctor) | 한 tensor가 두 shard에 있으면 거부 (GGUF에 index 없음) | tensor 이름 → shard 전사, 중복 시 거부 |
| `gguf_reader.hpp:557-562` (in_bounds) | payload가 자기 shard 파일 안에 들어야 | shard 헤더 data_start + max_end ≤ file_size (parse_header 게이트) |
| HF 컨벤션 (엔진 근거 없음 — 명시) | `total_size` = Σ tensor bytes | total_size == Σ tensor payload bytes |

HF는 index.json이 authoritative (shard 파일명 임의 허용) — 엔진의 5자리 family 패턴은
HF 강제 대상 아님. shard 헤더가 dtype/shape/오프셋의 정본, weight_map은 name → shard
전사만 담는다. emit 스트리밍: shard 파일 handle 캐시 + seek+read (mmap 금지, p4-01
streaming hygiene VERIFIED).
