# RELEASE_NOTES

## v0.4.0-moe (후보, Phase 4 임의 MoE 입력 일반화) — 2026-10-17

- HF MoE 입력 일반화 (`p4-01`): CLI `convert --emit-gguf --hf-emit` — HF naming
  (Mixtral/Qwen2MoE/fused/Qwen3-Next) → engine GGUF 키 매핑(`meta/hf_gguf_map.py`,
  layout.cpp:60-110 want2/want1 suffix + `blk.<L>.` prefix), fused gate_up_proj
  `(NE,2FF,H)` expert별 gate rows `[0,FF)` / up rows `[FF,2FF)` split, down_proj
  `(NE,H,FF)` expert 연속 run → Q2_0 stream, BF16/F16 → F32 정규화 후 quantizer 계약 유지,
  `ffn_gate_inp_shexp.weight` Bf16InF32(BF16 바이트 verbatim).
- StreamingGGUFWriter + `add_tensor_stream(name, shape, type, chunks)`: tensor payload를
  temp file에 chunk 스트리밍, heap peak = 단일 chunk. `slice_read` + `posix_fadvise(DONTNEED)`
  로 mmap slice 읽기 후 page cache drop. no-full-load gate: emit slice heap peak 126 MB,
  convert heap peak 377 MB (< 2 GiB).
- 가드 keys HF 헤더 산출(never invented): `embedding_length`/`block_count`(≤ preset, partial
  fixture)/`expert_count`/`expert_feed_forward_length`/`attention.head_count(_kv)`.
- 버그 수정: fused tensor 존재 layer에서 per-expert 경로 재emit 차단 — `build_expert_map`의
  `_FUSED_PATTERN`이 fused `down_proj`를 512 `down` refs로 만들어 `ffn_down_exps` 중복 tensor
  name + full-tensor payload(gguf tmp 79 GB 폭증, OOM RC=137) 발생 → fused 경로가 gate/up/down
  전 role 커버하므로 per-expert refs 비우기 가드.
- acceptance: qwen4exp geometry HF fixture 엔진 체인 SMOKE PASS — strata-gguf guard PASS,
  strata-dequant PASS + CHECKSUM 교차, strata_pack build+verify 0 bad, ours experts.bin
  (707,788,800 B)/dense.bin(525,869,648 B)/embd.bin(10,485,760 B)/index.txt golden
  byte-identical, 엔진 verify OUR pack PASS, 2회 convert sha256 동일. Mixtral geometry
  (엔진 가드 applicable 아님): mixtral-tiny BF16 fused → gguf_reader.py parse tensor
  수/shape/오프셋 일치 + canonical_xcheck dequant 왕복 bit-exact + our verify PASS.
- pytest: `tests/test_hf_gguf_map.py` 9 GREEN (AC1 naming/shape, AC2 BF16→F32 ==
  numpy reference + quantize_q2_0_stream byte-identical), `tests/test_hf_emit_chain.py`
  GREEN (325.56s, 엔진 binaries skipif).
- HF multi-shard 입력 (`p4-02`): `ingest/shard_index.py` — `model-00001-of-NNNNN.safetensors`
  + `model.safetensors.index.json`(weight_map) 입력. 엔진 split 검증 미러 게이트: 누락 shard
  이름 지목(gguf_split.hpp:44-46), shard 헤더 전사 count/양방향 일치(validate_split 미러
  gguf_reader.hpp:565-590), 한 tensor 두 shard 거부(GgufModel ctor gguf_reader.hpp:536-540),
  payload in_bounds(gguf_reader.hpp:557-562), total_size == Σ tensor bytes(HF 컨벤션).
  `TensorMeta.shard`/`ModelMeta.shard_data_start` 레지스트리, emit 오프셋 shard 경유
  (`abs_offset`), shard handle 캐시 + seek+read(mmap 금지). CLI `convert <dir> --emit-gguf`
  (multi-shard convert pack path는 단일 파일 한정 — 게이트 명시). acceptance: qwen4exp HF
  fixture 2-shard 분할 → GGUF sha256 단일 파일 입력과 동일 + 2회 emit 동일 + 엔진 체인
  SMOKE PASS(chain_a 재사용) + slice heap peak < 2 GiB. pytest: `tests/test_shard_index.py`
  8 GREEN(게이트 FAIL 케이스 포함), `tests/test_shard_chain.py` GREEN.
  회귀 교정: `cmd_convert`가 entry 메타데이터(shape×dtype)로 Q2_0-shaped tensor만
  seek+read로 읽음 — 전체 payload yield OOM(RC=137) 제거, `stream/mmap_reader.py` 삭제.
- 실모델 스모크 (`p4-03`): HF 실체크포인트 `Qwen/Qwen1.5-MoE-A2.7B`(8 shards, 27 GB,
  BF16, per-expert Qwen2MoE naming, H=2048, moe_FF=1408, NE=60, layers=24,
  head_dim=128 config.json 실측 산출) → CLI `convert <dir> --emit-gguf --hf-emit` →
  엔진 가드 applicable 아님(geometry≠qwen4exp) + reference verifier acceptance:
  엔진 `gguf_reader.py` parse match PASS(267 tensors), canonical_xcheck 4096 values
  bit-exact, our verify PASS(pool 7,031,070,720 B), slice heap peak 116,821,170 B
  (< 2 GiB), 2회 emit sha256 동일(7fca86bc2234, GGUF 10,534,643,712 B).
  다운로드 게이트: Content-Length HEAD 검증(shard truncation 재다운로드).
  Goal 내 교정 3건: F32 passthrough 스트리밍 BF16 chunk `_to_f32` widening(GGUF 크기
  delta 검출), `add_tensor`/`add_tensor_stream` 256 MB 주기 fsync(dirty page ENOMEM),
  sha256 1 MB chunk 스트리밍(read_bytes full-load OOM).
  pytest: `tests/test_real_model_smoke.py` gate GREEN (1 passed/1982.82s,
  skipif 모델 미다운로드·엔진 tools 부재; 전수 suite와 병렬 실행 금지 convention).

## v0.3.0-runtime (후보, Phase 3 최소 연동) — 2026-10-16

- GGUF v3 파서: 헤더 24 B(magic/version/n_tensors/n_kv), tensor 레코드 순서
  `name→n_dims→shape(u64×nd)→dtype→offset` (gguf_reader.hpp:463-469 VERIFIED).
- 아키텍처 가드: compiled geometry 키 전체 수용 (`embedding_length=2560`, `block_count=48` 등).
- 엔진 로드 스모크 (`scripts/p3_engine_smoke.py`, tests/test_engine_smoke.py):
  strata-gguf --check PASS, strata-dequant PASS(Q2_0 symbolic invariant 0 violations),
  CHECKSUM 엔진↔Python 교차 일치, 엔진 golden build+verify PASS,
  우리 convert 산출물 experts.bin/dense.bin/embd.bin/index.txt golden과 byte-identical,
  엔진 verify가 OUR pack을 acceptance, 2회 convert sha256 재현.
- IQ4_NL dense 경로: S4 + IQ4NL codebook(16값 −127..113, canonical_xcheck.py:571),
  `iq4_nl_planes` 평면 빌더, PLE `per_layer_token_embd.weight`(shape[0]=160) fixture 수용.
- `stratashift verify` 로컬 self-check PASS.
- pytest 21/21 GREEN (GGUF no-full-load 게이트 2건 추가: 헤더 파싱 peak < 2 MB,
  단일 expert-role slice peak = slice + margin).
- safetensors float expert → GGUF Q2_0 emit (`p3-02`): CLI `convert --emit-gguf OUT --guard-keys JSON`
  — expert F32 `(out,NE,in)`를 expert run 단위 mmap slice로 Q2_0 양자화해 GGUF `(in,out,NE)`
  expert-major 스트림 기록, PLE `U8[720]` → IQ4_NL `[160,8]` verbatim, dense F32 pass-through.
  `quantize_q2_0_stream` 엔진 `quantize_q2_0`와 바이트 동일. data section EOF flush convention
  VERIFIED (trailing padding → verify +16 어긋남 → truncate로 닫음).
- 엔진 체인 acceptance on emitted GGUF (`scripts/p3_safetensors_gguf.py` SMOKE PASS):
  strata-gguf guard PASS, strata-dequant PASS + CHECKSUM 교차 일치, strata_pack build+verify
  **0 bad**, 우리 convert 산출 experts/dense/embd/index golden과 byte-identical.
- pytest 31/31 GREEN.
- dense tensor Q4_0 preset (`p3-03`): CLI `convert --emit-gguf --dense-type Q4_0` — dense F32를
  Q4_0 18 B/32 elems 블록으로 양자화해 같은 shape로 기록. `quantize_q4_0_stream` + `q4_0_planes`
  (S4 codes 16 B + fp16 scales 2 B, code_bias −8). 엔진 pack gate VERIFIED: `pack_index.py:144-152`
  plane 연속 + `strata_pack.py:43` ALIGN=64 → **elements % 128 == 0** 필요, 미만은 엔진 self-build도
  pack_index 거부 → emit은 F32 pass-through 유지 (`headers.Q4_0_MIN_PACK_ELEMENTS`).
  엔진 체인 acceptance (`scripts/p3_q4_dense.py` SMOKE PASS): strata-gguf guard PASS, strata_pack
  build+verify **0 bad** (Q4_0 dense bit-exact: output.weight 65536 elements S4), 우리 convert 산출
  experts/dense/embd/index golden과 byte-identical. quantizer 규칙은 HYPOTHESIS — 엔진에 quantizer
  없음(gguf_writer Q2_0 전용), gguf-py signed-max divergence. no-full-load gate: 스트리밍 slice
  heap peak(tracemalloc) — VmRSS는 mmap page cache 포함으로 부풀려짐(WSL 실측).
- dense tensor Q4_K preset (`p3-04`): CLI `convert --emit-gguf --dense-type Q4_K` — dense F32를
  Q4_K 144 B/256 elems 블록(8 groups of 32)으로 양자화해 같은 shape로 기록. `quantize_q4_k_stream`
  + `q4_k_planes` 3-plane(S4 codes 128 B + f32 scales 32 B + f32 offsets 32 B, code_bias 0,
  has_offset True). `get_scale_min_k4` 전사 + group별 min m 시프트(groups 4..7는 scb[8..11] high
  4 bits → m<16). 엔진 pack gate VERIFIED: codes 128 B/block + scales 32 B + offsets 32 B, offsets
  시작 160n 연속 → **elements % 512 == 0** 필요, 미만은 F32 pass-through (`Q4_K_MIN_PACK_ELEMENTS`).
  엔진 체인 acceptance (`scripts/p3_q4_k.py` SMOKE PASS): strata-gguf guard PASS, strata-dequant
  PASS+CHECKSUM, strata_pack build+verify **0 bad** (Q4_K dense bit-exact), 우리 convert 산출
  golden byte-identical, round-trip vs engine to_s4_q4_k+from_s4_q4_k bit-exact. quantizer 규칙은
  HYPOTHESIS — ggml make_qkx2 joint search 미재현(groups 4..7 bound 미보장).
- dense tensor Q5_K preset (`p3-05`): CLI `convert --emit-gguf --dense-type Q5_K` — dense F32를
  Q5_K 176 B/256 elems 블록(8 groups of 32)으로 양자화해 같은 shape로 기록. `quantize_q5_k_stream`
  + `q5_k_planes` 3-plane(S8 codes 1 B/elem + f32 scales 32 B + f32 offsets 32 B, code_bias 0,
  has_offset True). 5-bit codebook: nibble=code&0xF, qh 5th bit=+16 (u1/u2 walk 전사). producer
  규칙 HYPOTHESIS: d=fp16(amax/63), dmin=fp16(max(d,−min/63)), sc=clip(ceil((gmax−gmin)/(31d)),1,63),
  m=clip(round(−gmin/dmin),0,63) (groups 4..7 m<16). 엔진 pack gate VERIFIED: codes 256 B/block +
  scales 32 B + offsets 32 B, offsets 시작 288n 연속 → **elements % 512 == 0** 필요, 미만은 F32
  pass-through (`Q5_K_MIN_PACK_ELEMENTS`). 엔진 index 정합: FORM_INFO code_bits=8(form S8,
  pack_codes per=1), code_bias=0 (설치 핀 MAPPINGS:569). 엔진 체인 acceptance (`scripts/p3_q5_k.py`
  SMOKE PASS): strata-gguf guard PASS, strata-dequant PASS+CHECKSUM, strata_pack build+verify
  **0 bad** (Q5_K dense bit-exact), 우리 convert 산출 golden byte-identical, round-trip vs engine
  to_s8_q5_k+from_s8_q5_k bit-exact.
- dense tensor Q6_K preset (`p3-06`): CLI `convert --emit-gguf --dense-type Q6_K` — dense F32를
  Q6_K 210 B/256 elems 블록(16 groups of 16, signed int8 sc, codebook 0..63 bias −32)으로 양자화.
  `quantize_q6_k_stream` + `q6_k_planes` 2-plane(codes 1 B/elem + signed scales, offsets 없음).
  packing nibble 짝: ql[l]=(q1 low|q3 high), ql[l+32]=(q2 low|q4 high), qh[l] bit pair (2qidx) —
  quads가 stride 32로 landing → group = element/16. producer 규칙 HYPOTHESIS: d=fp16(amax/63),
  sc=clip(rint(gmax/d),−63,63), code=clip(rint(w/(d·sc))+32,0,63) (sc==0 → code 32). pack gate:
  **elements % 256 == 0** (`Q6_K_MIN_PACK_ELEMENTS`, offsets 없음 → scales 시작 256n % 64 == 0 항상).
  FORM_INFO Q6_K code_bits=8, code_bias=−32, has_offset False. 엔진 체인 acceptance
  (`scripts/p3_q6_k.py` SMOKE PASS): guard PASS, dequant+CHECKSUM, strata_pack verify **0 bad**,
  golden 4개 산출물 byte-identical, 왕복 vs engine to_s8_q6_K+from_s8_q6_K bit-exact.
  전수 pytest 56/56 GREEN.
- dense tensor Q3_K preset (`p3-07`): CLI `convert --emit-gguf --dense-type Q3_K` — dense F32를
  Q3_K 110 B/256 elems 블록(16 groups of 16, 6-bit sc bias −32, codebook 0..7 bias −4)으로 양자화.
  `quantize_q3_k_stream` + `q3_k_planes` 2-plane(codes 4-bit S4 per=2 + signed scales, offsets 없음).
  packing: group g=8·half+2·j+sub (elements 16g..16g+15), qs[half·32+sub·16+l] bit 2j = c2,
  hmask[sub·16+l] bit (4·half+j) = m, code = c2+4·m — hmask는 두 half가 같은 32 B를 재읽고 bit만
  walk. producer 규칙 HYPOTHESIS: d=fp16(amax/63), sc6=clip(rint(gmax/d)+32,0,63),
  code=clip(rint(w/(d·(sc6−32)))+4,0,7) (dl==0 → code 4). pack gate: **elements % 256 == 0**
  (`Q3_K_MIN_PACK_ELEMENTS`). FORM_INFO Q3_K code_bits=4, code_bias=−4, has_offset False.
  엔진 체인 acceptance (`scripts/p3_q7_k.py` SMOKE PASS): guard PASS, dequant+CHECKSUM,
  strata_pack verify **0 bad**, golden 4개 산출물 byte-identical, 왕복 vs engine
  to_s4_q3_k+from_s4_q3_k bit-exact. test_q3_k 6/6 GREEN.
- dense tensor Q8_0 preset (`p3-09`): CLI `convert --emit-gguf --dense-type Q8_0` — dense F32를
  Q8_0 34 B/32 elems 블록(codebook −128..127)으로 양자화, GGUF type id 8로 기록.
  `quantize_q8_0_stream` + `q8_0_planes` 2-plane(codes plane = int8(qs)+128 1 B/elem +
  fp16 scales 2 B/block, offsets 없음 — MAPPINGS:571 code_bias −128, has_offset False).
  엔진에 Q8_0 quantizer 존재(`mtp_pack.py:100 q8_0`, pin c566ff4): producer 규칙
  d = amax/127(f32), q = rint(w·(1/d)) (ties-to-even)는 **VERIFIED — 우리 스트림이 엔진
  quantizer와 바이트 동일**(random + edges: all-zero, tiny amax, large amax, fp16-boundary
  amax = 65504·127; `mtp_rt.py:34`의 round-half-away 변형은 tie divergence, 계약은 mtp_pack).
  pack gate VERIFIED: codes 32 B/block(S8 per=1), scales 시작 align64(32n) 연속
  (`pack_index.py:144-152`, `strata_pack.py:43`) → **elements % 64 == 0**, 그리고
  ndim ≥ 2(`pack_index.py:174-186` scales = ne1·n_groups → 1-D는 pack_index 거부);
  미만/1-D는 F32 pass-through (`Q8_0_MIN_PACK_ELEMENTS`). 엔진 체인 acceptance (`scripts/p3_q8_0.py`
  SMOKE PASS): guard PASS, dequant+CHECKSUM, strata_pack build+verify **0 bad** (Q8_0 dense
  bit-exact), 우리 convert 산출 golden byte-identical + manifest planes agree + 엔진 verify
  --out ours PASS + 2회 convert sha256 동일, 왕복 vs engine to_s8_q8_0+from_s8_q8_0 bit-exact.
  test_q8_0 7/7 GREEN.
- pytest 69/69 GREEN.
- REA 인터페이스 전환 (`p0-05`): REA(morluto/rea) 6.5.0 전역 설치·핀(package.json SHA256),
  provider=ghidra 게이트 healthy(`GHIDRA_INSTALL_DIR`+`JAVA_HOME`), CLI 스모크 PASS
  (`rea inspect strata-dequant` evidence procedures 144/strings 233, `rea search dequant` 9 hits,
  `rea function dequantize_q3_k` pseudocode confidence `observed`), Hermes MCP 등록 실측
  (serverInfo `rea 6.5.0`, tools/list 207건), 근거 접두어 `REA:` 규칙 신설
  (REA_GHIDRA §6, docs/rea/README). Ghidra는 provider 엔진 유지 — 엔진 교체 없음.
- REA 백로그 해소 (`p0-06`, 2026-10-17): `.o` 6개 SHA256 binary_pin §4 전수 일치 게이트.
  `.o` DWARF 부재(Release) → 필드 접근 오프셋으로 struct 레이아웃 복원: `WeightTable`
  (load @ 0x1020e0) map 0x00-0x27 / LoadReport 0x28-0x5F / pack_dir_ 0x60-0x77, 총 120B,
  vptr 없음; `ExpertCache` (admit @ 0x1036b0) base_ 0x00 … admitted_ 0xe8, 총 240B,
  vptr 없음 — 오프셋 전수 소스 정의와 일치 (`struct_layouts` §8, `REA:` evidence 병기).
  `ActQ`·`PinnedArena`는 `SOURCE:` derived 오프셋 표. bitpack_rules §5-5 HYPOTHESIS 해소:
  `qp/psum/pscale` = ActQ 활성화 양자화 plane (`expert.hpp:75-77`) — pack 바이트 아님,
  AVX-512 kernel은 image를 읽지 않음, bitplane 매핑 정본 = `q2_bitplane_parity.cpp`
  (scalar, engine-verified) — pack byte contract 스칼라 유지 확정, 코드 변경 없음.
  잔여 백로그: VMM granularity(런타임 값, CUDA/GPU 부재).

## v0.2.0-stream (Phase 2 완료) — 2026-10-15

- safetensors 헤더 파싱, Expert 단위 스트리밍 루프, Q2_0 blob relayout(pack_layer 산수),
  메모리 안정 모니터(`--max-rss-gb`), CLI `inspect`/`convert`.
