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

## 다음 즉시 액션

1. **Ghidra 설치** (OS 경로) + 프로젝트 루트 `/opt/data/ghidra-projects/` ([REA_GHIDRA.md](REA_GHIDRA.md) §2-1) — **완료** (p0-01)
2. HF Space / GitHub에서 PolyStrata 바이너리·소스 핀 + 지원 표 `planned` · `docs/rea/` 골격 (`p0-01`) — **완료**
3. Ghidra에 로더 임포트·디컴파일 (`p0-02`) → Struct·bitpack 정본 — **완료** (빌드 9개, DECOMPILED 14건)
4. Phase 1: `ingest/safetensors_header.py` PoC (`p1-01`) — **완료** (Phase 1 전체 완료, ACCEPTANCE Phase 1 게이트 PASS)
5. Phase 2·3·4 — **완료** (p2-01, p3-01~p3-07·p3-09, p4-01~p4-03). Phase 0 백로그 해소 — **완료** (p0-06)
6. **잔여는 외부 차단뿐**: (a) p3-08 Q2_K — upstream main MAPPINGS에 Q2_K 없음(2026-10-18 재확인)
   → 엔진 추가 시 UPSTREAM_VERSIONING 절차로 핀 상향 후 개시; (b) GPU 의존 백로그(VMM granularity
   실측, WeightTable::load GPU 경로, VRAM 모니터) — CUDA/GPU 부재; (c) ACCEPTANCE.md — 인간 서명 전용.
   (a)는 upstream 감시, (b)는 GPU 확보, (c)는 인간 서명 대기. 코드·설계로 진행 가능한 Goal은 전수 소진.
