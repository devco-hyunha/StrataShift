# StrataShift 개발 로드맵

## 개요

| Phase | 이름 | 목표 | 산출물 |
|-------|------|------|--------|
| Docs | 문서 베이스라인 | 기획·제약·에이전트 지침 고정 | docs / AGENTS.md |
| **0** | **엔진 바이너리 역설계 (REA)** | 가중치 로드 저수준 규칙 파악 | `docs/rea/*`, 규격 상수 초안 |
| 1 | PoC·메타데이터 | 전체 적재 없이 구조 파악 증명 | 헤더 파서, Expert 맵, REA 규격 매핑 |
| 2 | 스트리밍·양자화 | 청크 트랜스파일러 | 변환 CLI, 모니터, 샘플 출력 |
| 3 | 엔진 연동·피드백 | PolyStrata에서 로드·구동 검증 | 주입 테스트, AI 디버그 루프 |
| **4** | **임의 MoE 모델 일반화** | 엔진특화 분석을 타 MoE 가정에 적용 | HF naming→GGUF 매핑, fused/BF16/shard 입력, reference verifier acceptance |
| **5** | **v1.0.0 공개화** | 공개 가능한 CLI·문서·안정성 (마일스톤 정의대로) | packaging·CLI polish·사용 문서·CI·릴리즈 태깅 |

---

## Docs — 문서 베이스라인 (완료)

- [x] 제품 명세서 / 아키텍처 / 제약 / 참조 / REA 절차
- [x] AGENTS.md 프롬프트 (Ghidra 포함)
- [x] AI 자동화 워크플로 (`AI_DEV_WORKFLOW` / `DOC_SYNC` / `docs/work`)
- [x] PolyStrata·Ghidra 핀 버전 기록 (p0-01: PolyStrata `c566ff4` · Ghidra 12.1.4 · JDK 21)

---

## Phase 0 — 엔진 바이너리 역설계 (REA 분석 레이어)

### 목표

PolyStrata 엔진이 가중치를 읽어들이는 **저수준 바이트·패킹·오프셋 규칙**을 파악한다.

### 작업

0. **Ghidra 설치** (OS/도구 경로, 저장소 벤더 금지) — [REA_GHIDRA.md](REA_GHIDRA.md) §2-1 · Goal `p0-01` — **완료** (`/opt/data/tools/ghidra_12.1.4_PUBLIC`, JDK 21, headless 확인)
1. PolyStrata 확보·핀 + (가능하면) 공개 소스 정찰 → `docs/rea/` 골격 — `p0-01` — **완료** (zip SHA256 + 골격 5파일, 소스 정찰)
2. PolyStrata 실행 파일/로더를 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)에 임포트 — `p0-02` — **완료** (9개 빌드 산출물 임포트·Auto-analyze·디컴파일, `DECOMPILED:` 14건)
3. 가중치 파일 파싱 함수·메모리 블록 할당 루틴 디컴파일 — **완료** (GGUF magic/정렬, STRP, index.txt 19열, blob stride)
4. 바이트 패킹, 정렬, Struct 오프셋, 페이징 규칙 역설계 — **완료 (p0-06, 2026-10-17)**
   (Q2_0 스칼라 contract 이중 근거; struct 오프셋 `.o` 실측 복원 — `struct_layouts` §8;
   AVX512·bitplane HYPOTHESIS 해소 — `bitpack_rules` §5-5. 잔여: VMM granularity는
   런타임 값 — CUDA/GPU 부재 백로그)
5. (병행) 공개 소스가 있으면 소스↔바이너리 교차 검증
6. LLM 에이전트에 디컴파일/Struct 표를 넘겨 Python 상수·패커 초안 유도
7. 결과를 `docs/rea/` 및 `polystrata` 상수 초안으로 고정 — **완료** (`p0-03` 상수 + `p0-04` STRP writer: 엔진 실물 `expert-profile.bin` byte-identical 재생산, `polystrata/strp.py`)
8. **REA 인터페이스 전환** — `p0-05` — **완료 (2026-10-17)** (REA 6.5.0 핀·package.json SHA 기록,
   provider=ghidra 게이트 healthy, CLI 스모크 inspect/search/function PASS, Hermes MCP 등록
   207 tools 실측, `REA:` 접두어 규칙, REA_GHIDRA §6 — Ghidra는 provider 엔진 유지)

상세 절차: [REA_GHIDRA.md](REA_GHIDRA.md)

### 완료 기준

→ [ACCEPTANCE.md — Phase 0](ACCEPTANCE.md#phase-0)

### 의존성

- **Ghidra 설치** (1번) — **완료** (p0-01)
- PolyStrata 바이너리 또는 분석 가능한 로더 (GitHub / HF Space) — 소스 확보 완료, **빌드 산출물 필요** (p0-02)

---

## Phase 1 — PoC 및 메타데이터 파싱 검증 (입력 레이어) — **완료 (2026-10-15, ACCEPTANCE Phase 1 게이트 PASS)**

### 목표

모델 전체를 VRAM/RAM에 올리지 않고 구조를 파악할 수 있음을 증명한다.

### 작업

1. **safetensors 헤더 파싱** — `p1-01` — **완료** (`src/stratashift/ingest/safetensors_header.py`, 헤더+8B 정렬만 읽음)
2. **레이어·Expert 오프셋 추출** — `p1-01` — **완료** (`src/stratashift/meta/expert_map.py`, per-tensor + fused 규칙)
3. **Phase 0 REA 규격 매핑** — `p0-03` — **완료** (`docs/polystrata_spec_draft.md` + `src/stratashift/polystrata/` 상수 초안)
4. **PoC 스크립트** — `p1-01` — **완료** (`scripts/poc_parse_header.py`, 단일 expert 슬라이스 mmap 데모)

### 완료 기준

→ [ACCEPTANCE.md — Phase 1](ACCEPTANCE.md#phase-1)

### 의존성

- Phase 0 규격 초안 (최소: 헤더 정렬·매직/버전 가설)
- 소형/중형 테스트용 MoE safetensors (또는 샤드 일부)

---

## Phase 2 — 디스크 스트리밍 및 양자화 파이프라인 (처리 레이어) — **완료 (2026-10-15, ACCEPTANCE Phase 2 게이트 PASS)**

### 목표

순차적 청크 기반 트랜스파일러를 구현한다.

### 작업

1. **제너레이터 루프** — **완료** (`p2-01`, `stream/mmap_reader.py::iter_tensor_payloads`)
   ```
   [디스크에서 Expert/레이어 1개 로드]
     → [CPU에서 비트 패킹·양자화]
     → [타깃 포맷 스트리밍 쓰기]
     → [메모리 해제]
   ```
2. **양자화·패킹 모듈** — **완료** (`p2-01`, `quant/pack.py` Q2_0 contract; 프리셋 1 = Q2_0. Q4_0 dense는 `p3-03`에서 해소 — Q4/K-quant는 Backlog)
3. **인덱스·매니페스트 writer** — **완료** (`p2-01`, `meta/manifest.py` 19열 index.txt + manifest.json)
4. **메모리 안정성 모니터** — **완료** (`p2-01`, `monitor/rss.py` VmRSS 샘플 + `--max-rss-gb` 안전 중단; VRAM은 GPU 의존 — Phase 3)
5. **CLI** — **완료** (`p2-01`, `stratashift inspect`/`convert`; `verify`는 Phase 3)

### 완료 기준

→ [ACCEPTANCE.md — Phase 2](ACCEPTANCE.md#phase-2)

---

## Phase 3 — PolyStrata 연동 및 피드백 루프 (실행 레이어)

### 목표

변환 결과물이 PolyStrata 런타임에서 정상 로드·구동되는지 확인한다.

### 작업

1. 변환 테스트 가중치를 PolyStrata 로더에 주입 — **완료** (`p3-01`: fixture GGUF + 엔진 strata-gguf/strata-dequant/strata_pack acceptance)
2. 로드/추론 스모크 테스트 — **완료** (`p3-01`: 엔진 golden build+verify PASS, 우리 산출물 golden과 byte-identical, 엔진 verify가 OUR pack PASS; `scripts/p3_engine_smoke.py` + `tests/test_engine_smoke.py`)
3. AI 디버깅 루프 — **완료** (`p3-01`: 엔진 FAIL 로그 4건 → REA 재대조 → 패치 → 재검증 PASS; `docs/work/p3-01.md` Result)
4. 회귀 테스트: 동일 입력 → 동일 출력 해시 — **완료** (`p3-01`: 2회 convert sha256 identical)
5. **safetensors float expert → GGUF Q2_0 emit** — `p3-02` — **완료** (CLI `--emit-gguf`/`--guard-keys`,
   expert-major Q2_0 스트림, PLE U8[720]→IQ4_NL[160,8] verbatim, data section EOF flush VERIFIED;
   엔진 체인 acceptance: strata-gguf guard PASS, strata-dequant PASS+CHECKSUM 교차, strata_pack
   verify 0 bad, 우리 convert 산출 golden byte-identical, quantize 바이트 동일; pytest 31/31)
6. **dense tensor Q4_0 preset** — `p3-03` — **완료** (CLI `--dense-type Q4_0`, `quantize_q4_0_stream`
   + `q4_0_planes`; 엔진 pack gate VERIFIED: plane 연속 + ALIGN=64 → elements % 128 == 0 필요
   (`pack_index.py:144-152`, `strata_pack.py:43`), 미만은 F32 pass-through; 엔진 체인 acceptance
   PASS — strata-gguf guard PASS, strata_pack build+verify PASS, 우리 convert 산출 golden
   byte-identical; quantizer 규칙은 HYPOTHESIS (엔진 quantizer 없음, gguf-py signed-max divergence);
   pytest 38/38)
7. **dense tensor Q4_K preset** — `p3-04` — **완료** (CLI `--dense-type Q4_K`, `quantize_q4_k_stream`
   + `q4_k_planes` 3-plane(codes+scales+offsets); `get_scale_min_k4` 전사 + group별 min m 시프트
   (groups 4..7는 scb[8..11] high 4 bits → m<16 제약); 엔진 pack gate VERIFIED: codes 128 B/block
   + scales 32 B + offsets 32 B, offsets 시작 160n 연속 → elements % 512 == 0 필요
   (`pack_index.py:144-152`, `strata_pack.py:43`), 미만은 F32 pass-through; 엔진 체인 acceptance
   PASS — strata-gguf guard PASS, strata-dequant PASS+CHECKSUM, strata_pack build+verify 0 bad,
   우리 convert 산출 golden byte-identical, round-trip vs engine to_s4_q4_k+from_s4_q4_k bit-exact;
   quantizer 규칙은 HYPOTHESIS (엔진 quantizer 없음, ggml make_qkx2 joint search 미재현 —
   groups 4..7 bound는 m<16으로 미보장); pytest 44/44)
8. **dense tensor Q5_K preset** — `p3-05` — **완료** (CLI `--dense-type Q5_K`, `quantize_q5_k_stream`
   + `q5_k_planes` 3-plane(codes 1 B/elem + f32 scales + f32 offsets); `to_s8_q5_k` qh u1/u2 walk 전사,
   5-bit codebook: nibble=code&0xF, qh bit=code≥16(+16); producer 규칙 HYPOTHESIS: d=fp16(amax/63),
   dmin=fp16(max(d,−min/63)), sc=clip(ceil((gmax−gmin)/(31d)),1,63) (ceil → 31·d·sc ≥ rng, code 클립 없음),
   m=clip(round(−gmin/dmin),0,63) (groups 4..7 m<16); 엔진 pack gate VERIFIED: codes 256 B/block + scales
   32 B + offsets 32 B, offsets 시작 288n 연속 → elements % 512 == 0 필요, 미만은 F32 pass-through
   (`Q5_K_MIN_PACK_ELEMENTS`); 엔진 index 정합: FORM_INFO code_bits=8(form S8, pack_codes per=1),
   code_bias=0 (설치 핀 c566ff4 MAPPINGS:569); 엔진 체인 acceptance PASS — strata-gguf guard PASS,
   strata-dequant PASS+CHECKSUM, strata_pack build+verify 0 bad (Q5_K dense bit-exact), 우리 convert
   산출 golden byte-identical, round-trip vs engine to_s8_q5_k+from_s8_q5_k bit-exact; pytest 50/50)
9. **dense tensor Q6_K preset** — `p3-06` — **완료 (2026-10-16)** (CLI `--dense-type Q6_K`,
   `quantize_q6_k_stream` + `q6_k_planes` 2-plane(codes 1 B/elem + signed scales, offsets 없음);
   6-bit codebook code 0..63 bias −32, signed int8 sc; producer 규칙 HYPOTHESIS: d=fp16(amax/63),
   sc=clip(rint(gmax/d),−63,63), code=clip(rint(w/(d·sc))+32,0,63) (sc==0 → code 32); packing
   nibble 짝 ql[l]=(q1|q3<<4), ql[l+32]=(q2|q4<<4), qh[l] bit pair (2qidx); pack gate
   elements % 256 == 0 (`Q6_K_MIN_PACK_ELEMENTS`); FORM_INFO Q6_K code_bits=8, code_bias=−32,
   has_offset False; 엔진 체인 SMOKE PASS — guard PASS, dequant+CHECKSUM, strata_pack verify 0 bad,
   golden 4개 산출물 byte-identical, 왕복 bit-exact; 전수 pytest 56/56 GREEN)
10. **dense tensor Q3_K preset** — `p3-07` — **완료 (2026-10-16)** (CLI `--dense-type Q3_K`,
    `quantize_q3_k_stream` + `q3_k_planes` 2-plane(codes 4-bit S4 per=2 + signed scales, offsets 없음);
    5-bit codebook code 0..7 bias −4, 6-bit sc (bias −32); producer 규칙 HYPOTHESIS: d=fp16(amax/63),
    sc6=clip(rint(gmax/d)+32,0,63), code=clip(rint(w/(d·(sc6−32)))+4,0,7) (dl==0 → code 4); packing
    group g=8·half+2·j+sub (elements 16g..16g+15), qs[half·32+sub·16+l] bit 2j = c2,
    hmask[sub·16+l] bit (4·half+j) = m (hmask 두 half 재읽기, bit walk); pack gate
    elements % 256 == 0 (`Q3_K_MIN_PACK_ELEMENTS`); FORM_INFO Q3_K code_bits=4, code_bias=−4,
    has_offset False; 엔진 체인 SMOKE PASS — guard PASS, dequant+CHECKSUM, strata_pack verify 0 bad,
    golden 4개 산출물 byte-identical, 왕복 bit-exact; 전수 pytest 62/62 GREEN)
11. **dense tensor Q2_K preset** — `p3-08` — **blocked (2026-10-16)** — Goal 기록:
    [docs/work/p3-08.md](work/p3-08.md) (AC1 전제 게이트 FAIL, 재개 조건 명문화). 설치 핀 c566ff4의
    `canonical_xcheck.py` MAPPINGS에 Q2_K canonical mapping이 없음 (gguf_reader.py:22 type id 10과
    block (256, 84)는 존재하나, strata_pack build/verify는 MAPPINGS[tname]로 canonical화 → Q2_K
    acceptance 게이트 실행 불가). 엔진이 mapping을 추가하면 재개. producer만 선구현하지 말 것
    (AGENTS: PolyStrata 없이 임의 포맷 확정 금지).
12. **dense tensor Q8_0 preset** — `p3-09` — **완료 (2026-10-17)** (CLI `--dense-type Q8_0`,
    `quantize_q8_0_stream` + `q8_0_planes` 2-plane(codes plane = int8(qs)+128, 1 B/elem +
    fp16 scales 2 B/block, offsets 없음); codebook −128..127, code_bias −128 (MAPPINGS:571),
    GGUF type id 8. 엔진에 Q8_0 quantizer 존재(`mtp_pack.py:100 q8_0`) — producer 규칙
    d = amax/127(f32), q = rint(w·(1/d)) (ties-to-even)는 **VERIFIED (바이트 동일 게이트)**,
    K-quant의 HYPOTHESIS와 구별; `mtp_rt.py:34` round-half-away 변형은 tie에서 divergence —
    계약은 mtp_pack. pack gate VERIFIED: codes 32 B/block(S8 per=1), scales 시작 align64(32n)
    연속 → **elements % 64 == 0** 필요(`pack_index.py:144-152`, `strata_pack.py:43`;
    `Q8_0_MIN_PACK_ELEMENTS`), 그리고 ndim ≥ 2 필요(`pack_index.py:174-186` scales =
    ne1·n_groups → 1-D는 derived scales 0, pack_index 거부); 미만/1-D는 F32 pass-through;
    FORM_INFO Q8_0 code_bits=8, code_bias=−128, has_offset False; 엔진 체인 acceptance
    PASS — strata-gguf guard PASS, strata-dequant PASS+CHECKSUM, strata_pack build+verify
    0 bad (Q8_0 dense bit-exact), 우리 convert 산출 golden byte-identical + manifest planes
    agree + 엔진 verify --out ours PASS + 2회 convert sha256 동일, 왕복 vs engine
    to_s8_q8_0+from_s8_q8_0 bit-exact; Goal 기록 [docs/work/p3-09.md](work/p3-09.md);
    전수 pytest 69/69 GREEN)

### 완료 기준

→ [ACCEPTANCE.md — Phase 3](ACCEPTANCE.md#phase-3)

---

## Phase 4 — 임의 MoE 모델 일반화 (입력 레이어 확장)

### 목표

Phase 0~3은 엔진과 엔진 특화 모델(qwen4exp geometry)의 분석·수락이었다. AGENTS.md 목표
("임의의 MoE .safetensors 모델")의 "임의" 칸을 개척한다: HF naming·fused gate_up·BF16·
multi-shard 입력을 스트리밍으로 엔진 GGUF 포맷에 적용한다.

### 작업

1. **HF naming → engine GGUF 키 매핑 + fused split + BF16** — `p4-01` — **완료 (2026-10-17)**
   (`docs/work/p4-01.md`; 근거: `src/core/layout.cpp:60-110` 키 어휘, mixtral-tiny 실측
   헤더, struct_layouts §1 가드). acceptance: qwen4exp geometry HF fixture = 엔진 체인
   full, Mixtral geometry = reference verifier (엔진 가드 applicable 아님 — compiled
   geometry 근거 기록). CLI `convert --emit-gguf --hf-emit`; StreamingGGUFWriter +
   add_tensor_stream(chunks) + slice_read+fadvise로 heap peak emit 126 MB / convert 377 MB
   (< 2 GiB gate). 엔진 체인 SMOKE PASS: strata-gguf guard PASS, dequant+CHECKSUM 교차,
   strata_pack build+verify 0 bad, ours experts/dense/embd/index golden byte-identical,
   엔진 verify OUR pack PASS, 2회 convert sha256 동일; mixtral-tiny gguf_reader parse
   match + canonical_xcheck 왕복 + our verify PASS. 전수 pytest GREEN.
2. **multi-shard index.json 입력** — `p4-02` — **완료 (2026-10-17)**
   (`docs/work/p4-02.md`; 엔진 split 검증 미러 게이트 — gguf_split.hpp:44-46 누락
   shard 이름 지목, validate_split 미러 gguf_reader.hpp:565-590 전사 count/양방향,
   GgufModel ctor gguf_reader.hpp:536-540 두 shard 중복 거부, in_bounds gguf_reader.hpp:557-562,
   total_size HF 컨벤션). `ingest/shard_index.py` + `TensorMeta.shard`/`shard_data_start`
   레지스트리 + CLI `convert <dir> --emit-gguf --hf-emit`. acceptance 실측: qwen4exp fixture
   2-shard 분할 → GGUF sha256 단일 파일과 동일 (2a69f2ca1056…) + 2회 emit 동일 + 엔진 체인
   SMOKE PASS(guard PASS, CHECKSUM 교차, strata_pack verify 0 bad, ours golden
   byte-identical, 엔진 verify OUR PASS) + heap gate (emit 126 MB / convert 377 MB < 2 GiB).
   전수 pytest GREEN (89 passed/2195.94s). 원인은 같은 Goal에서 닫음: `iter_tensor_payloads`
   전체 payload yield + `continue` blob 잔존 → transient 6.7 GB OOM (RC=137) —
   entry 메타데이터 게이트 + seek+read로 교체, `stream/mmap_reader.py` 삭제(MODULE_DESIGN 동기).
3. **실모델 스모크** — `p4-03` — **완료 (2026-10-17)**: HF 실체크포인트
   `Qwen/Qwen1.5-MoE-A2.7B`(8 shards, 27 GB, BF16, per-expert Qwen2MoE naming,
   H=2048, moe_FF=1408, NE=60, layers=24, head_dim=128 config.json 실측) 다운로드
   (Content-Length HEAD 검증) → CLI `convert <dir> --emit-gguf --hf-emit` →
   엔진 가드 applicable 아님(geometry≠qwen4exp) 기록 + reference verifier acceptance:
   엔진 `gguf_reader.py` parse match PASS(267 tensors names/shapes/offsets),
   canonical_xcheck 4096 values bit-exact, our verify PASS(pool 7,031,070,720 B),
   slice heap peak 116,821,170 B (< 2 GiB), 2회 emit sha256 동일(7fca86bc2234,
   GGUF 10,534,643,712 B). 전수 pytest GREEN (90 passed/3982.87s). 원인은 같은 Goal에서
   닫음: F32 passthrough BF16 미확대(GGUF 크기 delta로 검출), dirty page ENOMEM(256 MB
   주기 fsync), sha256 full-load OOM(chunk 스트리밍). 산출물 `.local/` 전용(GITIGNORE).

### 완료 기준

- AC는 각 Goal 파일의 체크 항목; 엔진 가드 밖의 모델은 reference verifier acceptance로 고정.
- 엔진이 수락하지 않는 임의 커스텀 포맷은 최종 타깃으로 확정하지 않는다 (AGENTS).

→ [ACCEPTANCE.md — Phase 4](ACCEPTANCE.md#phase-4) (Goal AC 요약 + 인간 서명 게이트)

---

## Phase 5 — v1.0.0 공개화 (프로젝트 완성 종점)

### 목표

마일스톤 정의 `v1.0.0 = 공개 가능한 CLI·문서·안정성`을 만족한다. **이 Phase의 AC 전수 PASS가 프로젝트 완성의 종점이다** (인간 서명 게이트는 별도 — ACCEPTANCE).

### 완료 정의 (v1.0.0 AC)

- [ ] `pip install .` (또는 `pip install -e .`) 후 `stratashift` CLI가 설치·실행된다 (console entry point).
- [ ] CLI 전 명령(`inspect`/`convert`/`verify`)에 `--help`·종료 코드·에러 메시지가 사용자 수준으로 정돈된다.
- [ ] README에 quickstart(설치→inspect→convert→verify)가 실측 예시로 존재한다.
- [ ] CI(pytest 전수 + no-full-load 가드)가 PR/푸시 게이트로 동작한다.
- [ ] RELEASE_NOTES에 `v1.0.0` 릴리즈 노트가 확정(후보 아님)되고 git 태그 `v1.0.0`이 찍힌다.
- [ ] 전수 pytest GREEN + CONSTRAINTS PR 체크리스트 통과.
- [ ] **엔진 구동 + API 종점 (사용자 정의 정본, 2026-10-20)**: 우리 변환 산출물이 PolyStrata
  `serve/server.py`의 OpenAI `/v1/chat/completions`로 **텍스트 왕복 응답**한다 (tokenizer
  메타 emit + server 스모크 실측 — Goal `p5-09`, docs/work/p5-09.md).
  변환만으로는 v1.0.0 완료로 보지 않는다.
- [ ] **목적 정본 (정박 2026-10-20)**: 이 프로젝트의 목적은 "임의 MoE 모델을 PolyStrata에서
  **동작**하게 하는 변환"이다. 완성 판정 게이트 5종 — (1) byte 계약 통과 ✅(p5-08)
  (2) 엔진 load ✅ (3) RUN token 생성 ✅(gemma4 실model rc=0) (4) API 텍스트 왕복 ❌
  (p5-09 AC3·AC4) (5) 임의 아키텍처 RUN ❌ (p6-01 첫 실증 = Mixtral 8x7B). 게이트 4·5
  미실측 상태에서 `v1.0.0` 태그 금지 (변환 CLI 완성 ≠ 목적 완성).

### 작업 (Goal 세분 — planned scope)

1. **packaging·entry point** — `p5-01`: `pyproject.toml` 버전·console script(`stratashift = stratashift.cli.main:main`)·설치 게이트 테스트. 런타임 의존 추가 금지(C1: safetensors/torch 금지 유지). — **완료 (2026-10-18)**: version 0.4.0, entry `cli_entry`(exit code 전파), `dependencies = ["numpy"]`(src 전수 실측 — 유일한 3rd-party import; safetensors/torch 금지 유지), 실 설치 게이트 `pip install . --target` + `stratashift --help` exit 0 + import graph C1 GREEN (`tests/test_packaging.py` 3 GREEN).
2. **CLI polish** — `p5-02`: exit code 규격(0/1/2), `--help` 전 계층, 에러 메시지, `verify` CLI 최종 연결. CLI 테스트 추가. — **완료 (2026-10-18)**: dispatch auto-detection 연결(HF naming → fused HF emit, 플래그 없이 mixtral-tiny RC 0 실측), verify 두 계약 dispatch(engine-form/p2-form — `_verify_p2`), exit code 규격 `_run()` wrapper(stderr 한 줄 + 1, usage 2, max-rss 3 유지), `tests/test_cli_exit_codes.py` 8 GREEN. 전수 suite 1st 1 failed/99 passed → root cause(tracemalloc start/stop 예외 경로 leak) `heap_gate()` fix → deterministic 재현 25 GREEN → **전수 재실행 100 passed / 0 failed GREEN**.
3. **사용자 문서** — `p5-03`: README quickstart 실측 실행 기록, `docs/USAGE.md`(inspect/convert/verify 플래그 표), DOC_SYNC. — **완료 (2026-10-18)**: README quickstart 실측 4단계 고정(inspect 실출력, convert+emit 플래그 없이 auto-detection rc 0 GGUF 16,508,352 B, verify p2 exit 0, engine-form hop exit 0), USAGE.md 전 플래그 표 --help 실측 + verify 두 계약 섹션, `scripts/p5_quickstart.py` 재실행 GREEN(4 steps PASS).
4. **CI 게이트** — `p5-04`: GitHub Actions `pytest` + 금지 API grep 게이트(전체 load grep 0). PR 템플릿(AC·테스트·DOC_SYNC). — **done (2026-10-23, PR #5 머지 `84970e3`)**: AC1~AC5 PASS — `workflow` scope 부여 후 ci.yml 복원(원본 `58b02bc`), Actions run `39896690157` gate success. 수집·픽스처 실패는 엔진·`.local` 의존 테스트 skip 가드로 해결(`863954a`, `2c7771b`).
5. **릴리즈** — `p5-05`: v1.0.0 노트 확정 → 태그 → RELEASE_NOTES 후보 표기 제거. 인간 서명(ACCEPTANCE)는 병행 게이트. Nuitka single-file packaging 백로그(2026-10-20 논의: 실측 병목은 I/O이며 numpy 연산은 Nuitka로 가속 불가 — 배포 편의 용도).
6. **Gemma4 매핑 확장** — `p5-06`: Gemma4 arch 매핑 + 실모델 chain acceptance — **done (2026-10-19, 커밋 979eaf7)**.
7. **CUDA enablement + 엔진 RUN 게이트** — `p5-07`: host WSL CUDA 빌드 + mini RUN rc=0 — **RUN closure (2026-10-20)**, AC5 인간 서명 잔여.
8. **dense gate 엔진 정합 + rope fix** — `p5-08`: `_dense_select`(ne0 gate) + `_gemma4_rope_freqs` 발행, 실model RUN rc=0 — **done (AC1~AC4, 2026-10-20)**, AC5 인간 서명·speed gate TK1 백로그.
9. **엔진 구동 + API 종점** — `p5-09`: GGUF tokenizer 메타 emit + `serve/server.py` OpenAI
   `/v1/chat/completions` 텍스트 왕복 스모크 — **AC1~AC4 closure (2026-10-20)**, AC5 인간
   서명 잔여. 실측: tokenizer.json 32,169,626 B download; emit 12+ keys (mini tokenizer_meta_keys
   14, 실model tokenizer keys 14); mini API 스모크 R1~R7 rc=0 (HTTP 200 + 비어 있지 않은 텍스트
   왕복); 실model probe.gguf 7,877,904,096 B 688 tensors API 왕복 HTTP 200 (usage total 51),
   engine 실측 VmRSS load 2,467,764 kB → gen 3,789,720 kB, VmSwap 0 kB, GPU 3,746→3,787 MiB;
   root cause 같은 Goal에서 닫음: `gemma4-emitted.gguf` 32 B tail truncation (rope_freqs
   does not fit its file) → 재발현 probe delta 0 (writer 장부·tmp 자기 일치), 재-emit
   probe.gguf complete (short 0) — 원인은 그날 emit의 partial 완주. docs/work/p5-09.md.
10. **문서 베이스라인 정리** — `p5-10`: 명명 통일·ROADMAP 슬림화·goal 등록 일관화·ACCEPTANCE 정합·언어 순도 sweep — **in_progress (2026-10-20)**.
11. **K-quant producer 전사 (후보)** — `p5-11`: `k_quants.c` `make_qkx2_quants` joint search 전사로 p3-04~p3-07 HYPOTHESIS를 VERIFIED 승격 (근거: REFERENCES §4.1 계열 1 — pinned URL). 엔진 quantizer 없으므로 producer측만; 왕복 bit-exact 게이트는 유지. AC는 코드+테스트 전수 GREEN + 엔진 체인 acceptance 재실행. — **planned 후보 (2026-10-20)**

### 진행 규칙

- Goal은 ROADMAP planned scope 안에서만 생성·실행한다. scope 밖 발명 금지(AGENTS).
- upstream 차단(Q2_K·GPU)·인간 서명은 게이트 대기 — Phase 5 Goal의 전제 조건이 아닌 항목으로 분리.

### 완료 기준

→ [ACCEPTANCE.md — Phase 5](ACCEPTANCE.md#phase-5)

---

## Phase 6 — 엔진 실행 확장 (architecture 클래스·AMD/HIP)

### 목표

Phase 4·5는 변환 산출물 일반화까지였다. 엔진 RUN 계층의 두 외부를 뚫는다:
(1) Mixtral 실행 클래스 (p6-01, 로컬 엔진 빌드 — `.local` 전용, 저장소 커밋 금지 유지),
(2) AMD ROCm/HIP enablement·Strix Halo 실측 (p6-02, pin 상향 + gfx1151 게이트 + 인간 서명).

### 작업 (Goal 세분 — planned scope 등록 2026-10-20)

1. **Mixtral 실행 클래스 (로컬 엔진 빌드)** — `p6-01` (`docs/work/p6-01.md`,
   design 2026-10-20): pin `c566ff4` 트리에 `Mixtral final : public Engine` 작성 —
   engine.hpp hook 15종 집합의 최소 부분집합(전 layer full attention rope + routed SiLU
   experts top-2, shared·dense·swa·recurrent·drafter·vision 없음 — 근거 `engine.cpp:808-818`
   dispatch·`engine.hpp` base 기계 실측 2026-10-20, `gemma4.hpp` hook 실례).
   AC1 hook 전사·AC2 `hf_gguf_map` Mixtral 매핑(실측 fixture grounds)·AC3 hipcc 전이 아닌
   nvcc 컴파일 전용 게이트(sandbox GPU 부재)—AC4 host WSL CUDA 빌드 + mini RUN rc=0
   (`/opt/p5-07` 전례)—AC5 실모델 8x7B TK1 RUN(speed)—AC6 지원 매트릭스 `experimental`
   열 신설(UPSTREAM_VERSIONING §2 한 줄 개정, `supported` 아님) + 인간 서명.
   엔진 코드는 `.local` 전용(gitignored) — StrataShift 저장소 엔진 커밋 금지 유지(AGENTS).
2. **HIP enablement·Strix Halo 실측** — `p6-02` (`docs/work/p6-02.md`,
   design 2026-10-20): pin `c566ff4`는 CUDA 전용 실측(CMakeLists grep HIP 0건,
   2026-10-20). upstream PR #7 `feat(hip): AMD ROCm/HIP port`(RX 7900 XT gfx1100 실증,
   maintainer merge-as-beta 코멘트, `CMAKE_HIP_ARCHITECTURES` 기본 gfx906) 병합 커밋으로
   pin 상향(UPSTREAM_VERSIONING §3) → REA 재검(VMM 가드 `!defined(STRATA_USE_HIP)`
   dmalloc fallback, `mind_vram` HIP guard) → hipcc 컴파일 전용 게이트(gfx1151, PC) →
   Strix Halo ROCm 실측 + mini/실모델 HIP RUN rc(인간 병행) → ACCEPTANCE AMD beta 서명.
   Strix Halo 반복 오류(2026-10-19, p5-08 실행 장소 TK1 이동)의 root cause 해소 경로.
   AMD RUN은 Strix Halo 전용(지금 PC/TK1는 NVIDIA — HIP 물리 불가).
3. **imatrix streaming calibration producer** — `p6-03` (`docs/work/p6-03.md`,
   **AC1~AC3 complete 2026-10-22**): `in_sum2`/`counts` additive plane accumulator
   (imatrix.cpp 전사) + Q4_K imatrix producer + 단일 파일 emit opt-in `--imatrix`.
   plain RTN golden byte gate 불변. **AC4 스트리밍 축적 백로그** (p6-04 `llama-imatrix`
   산출물로 대체, 2026-10-22 사용자 결정; 재개 조건: 외부 PC 없이 다른 모델 imatrix 필요 시.
   참고: `llama-imatrix`는 `-ngl 99`로 모델 전체 GPU/RAM 적재 — 저장소 밖 산출물).
   AC5~AC7 → `p6-07` 이관.
4. **imatrix full-sweep calibration product** — `p6-04` (`docs/work/p6-04.md`,
   **done 2026-10-21**): RTX 4070 외부 발화 bundle(`run-all.sh`/`run-all.bat`) 완주 —
   product `p6-04-imatrix-20261021/imatrix.gguf` 5,409 chunks / 2,769,408 tokens /
   plane 295 engine-key base / unvisited unit 24 (chunks=8 대비 486 → 24). engine-key
   plane loader consume 왕복 실측. 코퍼스: `Salesforce/wikitext` wikitext-103-raw-v1
   (CC BY-SA).
5. **imatrix-consuming HF fused emit (Q2_0 폐기)** — `p6-05` (`docs/work/p6-05.md`,
   **AC1~AC5 complete 2026-10-22**): 사용자 ruling "Q2_0는 하지마" — Q4_0/Q4_K imatrix
   producer + expert ladder + dense engine-key lookup + CLI 전달. fixture gate
   (fused·dense) PASS. AC6~AC7·full product emit → `p6-07` 이관.
6. **IQ codebook producer** — `p6-06` 백로그 (codebook 원문 REA 전사 필요).
7. **imatrix quality · engine acceptance · 인간 서명** — `p6-07` (`docs/work/p6-07.md`,
   design 2026-10-22): p6-03 AC5~AC7 + p6-05 AC6~AC7·full product emit 이관.
   PPL/KLD·serve·엔진 왕복·ACCEPTANCE 서명.
8. **Windows 지원 3a — 엔진 경로 한 곳 설정** — `p6-09` (`docs/work/p6-09.md`):
   `polystrata/engine_paths.py`(옵션→환경 변수→`.local/` 기본값), `convert --engine-src`,
   `src`·`scripts`·`tests`의 `/opt/data` 고정 경로 0건. 출력 바이트 불변. 이어서 3b(RSS)·
   3c(인코딩·`tomllib`·추가 패키지)·3d(자동 설치)·3e(엔진 빌드·Windows 문서).
9. **Windows 지원 3b — 메모리 감시(RSS)** — `p6-10` (`docs/work/p6-10.md`): Windows Working Set 측정,
   측정 불가 OS는 `None`·`--max-rss-gb` 거절, `test_rss_flat` 0 비교 통과 방지.
10. **Windows 지원 3c — 호환성** — `p6-11` (`docs/work/p6-11.md`): Python 3.11+(`tomllib`, 사용자 결정),
   텍스트 I/O `encoding="utf-8"` 전수 + 가드 테스트, 선택 의존성 `test`, `test_q4_0` 핀 gguf-py.
11. **Windows 지원 3d — 실행할 때 자동 설치** — `p6-12` (`docs/work/p6-12.md`): `install/` 영역, 묶음 `emit`(핀 엔진 소스, 내용 해시)·
   `engine-build`(CMake·MSVC·CUDA, 3e에서 명령 연결), `convert --yes`/`--dry-run`.
12. **Windows 지원 3e — 엔진 빌드·실행·문서** — `p6-13` (`docs/work/p6-13.md`): `stratashift build-engine [--gpu]`,
   Windows 네이티브 엔진 빌드(MSVC·CMake·CUDA 13.0), `strata-poly.exe` mini fixture RUN, Windows 설치·실행 문서.

### 진행 규칙

- Phase 6는 Phase 5 잔여(인간 서명 게이트)와 병행 가능. p6-01 AC2 매핑은 실측 fixture
  전제(never-invented) — p4-03 acceptance 유지.
- 실험 엔진 빌드는 `experimental` 표기만 — 지원 매트릭스 `supported`는 단일 pin 계약 유지.
- 변환기 산출물 sha256 regression: pin 상향 전후 our convert 산출 동일해야 함(변환기는
  엔진 pin 무관 — 상수 패치 시 REA 근거, p6-02 테스트 계획).

### 완료 기준

→ 각 Goal 파일 AC. 엔진 RUN 실측 로그·인간 서명 게이트 포함 (ACCEPTANCE.md AMD·실험 빌드 항목 신설).

---

## 우선순위·의존 그래프

```
Docs baseline
    │
    ▼
Phase 0  Ghidra REA → bitpack / struct / paging rules
    │
    ▼
Phase 1  safetensors header + expert map + REA 규격 매핑
    │
    ▼
Phase 2  stream loop + quant(pack) + monitor
    │
    ▼
Phase 3  runtime inject + AI feedback (+ REA 재분석)
    │
    ▼
Phase 4  HF naming 매핑 + fused/BF16/shard 입력 + 실모델 스모크
    │
    ▼
Phase 5  v1.0.0 공개화 (packaging·CLI·CI·릴리즈)
    │
    ▼
Phase 6  엔진 실행 확장 — p6-01 Mixtral 클래스(로컬 빌드) · p6-02 HIP enablement(Strix)
```

## 마일스톤 태그 (제안)

| 태그 | 의미 |
|------|------|
| `v0.0.1-rea` | Phase 0 규격 초안 고정 |
| `v0.1.0-poc` | Phase 1 완료 |
| `v0.2.0-stream` | Phase 2 완료 |
| `v0.3.0-runtime` | Phase 3 최소 연동 |
| `v0.4.0-moe` | Phase 4 임의 MoE 입력 일반화 |
| `v1.0.0` | 공개 가능한 CLI·문서·안정성 |

## 업스트림 추적 (상시)

PolyStrata는 제품 수명과 함께 **따라가야** 한다.  
일상은 핀 유지, 엔진 로더/패킹 변경 시에만 [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) 업그레이드 절차를 수행한다.

- 지원 매트릭스·매니페스트 핀 필드 유지
- Ghidra는 도구 — 필수 추적 대상 아님

## 현재 상태 한 줄 표 (정본은 각 Goal 파일 — 진행 기록은 Goal 파일에만)

| Goal | 상태 | 잔여 |
|------|------|------|
| p3-08 Q2_K | blocked | upstream MAPPINGS에 Q2_K 없음 → 핀 상향 시 개시 (upstream 감시) |
| p5-04 CI 게이트 | done | PR #5 머지 `84970e3`, Actions gate success |
| p5-05 릴리즈 | planned (human 종속) | v1.0.0 태그는 Phase 5 전수 PASS 후; Nuitka single-file packaging 백로그 |
| p5-06 Gemma4 매핑 | **done** (2026-10-19, 커밋 979eaf7) | — |
| p5-07 CUDA enablement + RUN | **RUN closure** (2026-10-20) | AC5 ACCEPTANCE 인간 서명 |
| p5-08 dense gate + rope fix | **done** (AC1~AC4 closure 2026-10-20, sha 29fa0890…) | AC5 인간 서명 · speed gate (12 GB TK1) 백로그 |
| p5-09 엔진 구동 + API 종점 | **AC1~AC4 closure** (2026-10-20) | AC5 ACCEPTANCE 인간 서명 · 전수 suite 재검(106/107, 재실행 중) |
| p5-10 문서 베이스라인 | in_progress (2026-10-20) | 명명 통일·ROADMAP 슬림화·goal 등록 일관화·ACCEPTANCE 정합·언어 sweep |
| p6-01 Mixtral 클래스 | design (2026-10-20) | AC1 fixture 실측 확정 후 코딩 (`.local` 전용 빌드) |
| p6-02 HIP enablement | design → **AC1 전제 열림** (2026-10-20 실사) | PR #11 병합 `ba1cf5f` (`0.1.42-poly.4`) + PR #53 Windows AMD 병합 `a418c11` 실측 기록 → pin 상향(헤르메스 AC2·AC3) → Strix Halo 실측 (인간 병행 AC4~AC7) |
| p6-03 imatrix producer | **AC1~AC3 complete** (2026-10-22) | AC4 스트리밍 축적 백로그 · AC5~AC7 → p6-07 |
| p6-04 imatrix full-sweep | **done** (2026-10-21) | — |
| p6-05 HF fused imatrix emit | **AC1~AC5 complete** (2026-10-22) | AC6~AC7·full product emit → p6-07 |
| p6-06 IQ codebooks | backlog | codebook 원문 REA 전사 |
| p6-07 imatrix quality gate | design (2026-10-22) | PPL/KLD·serve·엔진 왕복·ACCEPTANCE 서명 |
| p6-08 cli/main.py 구조 정리 | **done** (2026-10-22) | 동작 불변 리팩터링 (`docs/work/p6-08.md`), `cli/main.py` 1,703→189줄, AC1 31/31, `main` fast-forward 머지 |
| p6-13 Windows 3e 엔진 빌드 | **done** (2026-10-23) | `build-engine`·`strata-poly` mini RUN·Windows 문서, junction 없이 226/0/5 (`docs/work/p6-13.md`) — Windows 지원 단계 3 완료 |
| p6-12 Windows 3d 자동 설치 | **done** (2026-10-22) | 실행 시 확인→한 번 묻기→설치, 핀 소스 실제 다운로드 확인 (`docs/work/p6-12.md`) → 다음 3e |
| p6-11 Windows 3c 호환성 | **done** (2026-10-22) | Python 3.11+·utf-8·`test` extra, 3.11 실패 0 (`docs/work/p6-11.md`) → 다음 3d 자동 설치 |
| p6-10 Windows 3b RSS | **done** (2026-10-22) | Windows RSS 측정·측정 불가 거절, 기준 해시 31/31 (`docs/work/p6-10.md`) → 다음 3c |
| p6-09 Windows 3a 엔진 경로 | **done** (2026-10-22) | 경로 단일 지점·`/opt/data` 0건, 기준 해시 31/31 (`docs/work/p6-09.md`) → 다음 3b RSS |

## 다음 즉시 액션 (순서)

1. **p5-10 정리 마무리**: 명명 통일(rename)·Phase 5 goal 항목 등록·`p5-05.md` 생성·ACCEPTANCE Phase 0~3 정합 → docs 커밋.
2. **p5-09 AC1**: HF hub에서 gemma4 tokenizer 파일(`tokenizer.json`·`chat_template.jinja`·`special_tokens_map.json`) Content-Length HEAD 실측 다운로드 → `.local/models/` 저장.
3. **p5-09 AC2** (헤르메스 몫): `cli/main.py` tokenizer 메타 emit + round-trip tests GREEN → main 머지.
4. **p5-09 AC3·AC4** (host 세션): mini API 스모크 → host RTX 3050 Ti 실model OpenAI `/v1/chat/completions` 텍스트 왕복 실측.
5. **인간 게이트 일괄**: ACCEPTANCE 서명 (Phase 0~3 게이트 + RUN closure + API 종점), p5-05 릴리즈 태그.
6. 병행 가능: p6-01 AC1 fixture 실측, p6-02 upstream 감시 (PR #7 병합 확인), speed gate (TK1 수집 패키지 재가동 시).

스코프 밖 발명 금지 (AGENTS): Nuitka·Rust·tokenizer 외 커스텀 포맷 등 — 실측 게이트가 ROADMAP에 있는 것만 Goal로 만든다.
