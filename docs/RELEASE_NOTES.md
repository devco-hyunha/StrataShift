# RELEASE_NOTES

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
  test_q6_k 6/6 GREEN (전수 pytest는 잔여 — 커밋 후 전수 실행이 DoD 마감 조건).
- pytest 50/50 GREEN.

## v0.2.0-stream (Phase 2 완료) — 2026-10-15

- safetensors 헤더 파싱, Expert 단위 스트리밍 루프, Q2_0 blob relayout(pack_layer 산수),
  메모리 안정 모니터(`--max-rss-gb`), CLI `inspect`/`convert`.
