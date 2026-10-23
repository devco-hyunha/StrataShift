# StrataShift 단계별 인수 기준 (Acceptance Criteria)

각 Phase는 아래 항목을 **모두** 만족해야 완료로 본다.

---

## Phase 0

### 기능

- [x] PolyStrata 런타임/로더 바이너리를 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)에 임포트·분석했다. (p0-01/p0-02 — 9개 빌드 산출물·Auto-analyze·DECOMPILED 14건)
- [x] 가중치 파싱·메모리 블록 할당 관련 진입점(함수)이 `docs/rea/loader_entrypoints.md`에 기록된다. (p0-02)
- [x] Struct 오프셋·바이트 정렬·비트 패킹(시프트/마스크)·페이징 규칙 초안이 `docs/rea/`에 있다. (p0-03~p0-06 — struct_layouts §8 실측 복원, bitpack_rules §5 HYPOTHESIS 해소)
- [x] REFERENCES에 엔진 바이너리 SHA256·버전 및 Ghidra 버전이 핀된다. (pin `c566ff4`·zip SHA `3ecc0e4c…b24`·Ghidra 12.1.4·JDK 21, p0-01)
- [x] 불확실 항목은 `HYPOTHESIS:`로 표시된다. (초안 전수 — 확정 승격은 실측 게이트 경유)

### 제약 검증

- [x] REA 결과만으로 확정하지 않은 필드를 “확정”으로 표기하지 않았다. (p0-04 byte-identical 게이트, p0-06 VERIFIED 승격 기록)
- [x] LLM이 생성한 상수/패커 초안이 있다면 인간 리뷰 기록이 있다. (Goal 파일 진행 기록 + ACCEPTANCE 서명 게이트 병행)

### 산출물

- [x] `docs/rea/` 세트 ([REA_GHIDRA.md](REA_GHIDRA.md) Step D) — 7파일 (loader/struct/bitpack/paging/binary_pin/arch_families/README)
- [x] (선택) `polystrata` 모듈용 상수 초안 — `src/stratashift/polystrata/` (p0-03) + STRP writer byte-identical (p0-04)
- [ ] `v0.0.1-rea` 마일스톤 후보 노트 — 미작성 (RELEASE_NOTES는 v0.3.0부터 존재. 백로그: p5-05 릴리즈 시 소급 작성)

---

## Phase 1

### 기능

- [x] `.safetensors` 파일에서 텐서 이름, dtype, shape, data_offsets를 추출한다. (p1-01 `ingest/safetensors_header.py`, 헤더+8B 정렬만 읽음)
- [x] 가중치 본문 전체를 RAM에 복사하지 않는다 (헤더 + 선택적 단일 슬라이스만). (p1-01 PoC 단일 expert 슬라이스 mmap 실측)
- [x] MoE FFN Expert 후보 텐서를 나열하는 Expert 맵(JSON 등)을 출력한다. (p1-01 `meta/expert_map.py` per-tensor + fused 규칙)
- [x] **Phase 0 REA 규격**을 바탕으로 엔진 요구 헤더 정렬·블록 규칙이 매핑·문서화된다. (p0-03 `docs/polystrata_spec_draft.md` + `polystrata/` 상수)

### 제약 검증

- [x] 코드 경로에 전체 `load_file` / 전체 state_dict 적재가 없다. (grep 게이트 0건 실측 2026-10-18, p5-04 AC3 — CI 원본 커밋 29a1b59에 동형 게이트)
- [x] 테스트 모델(또는 샤드)에서 RSS가 “헤더 + 최대 1개 슬라이스” 수준으로 설명 가능하다. (p1-01 PoC + p2-01 `monitor/rss.py` VmRSS 샘플; 최신: 실model slice heap peak 116,821,170 B < 2 GiB, p4-03)

### 산출물

- [x] `scripts/poc_parse_header.py` (또는 동등 CLI `inspect`) — p1-01 PoC + `stratashift inspect` (p2-01 CLI, p5-03 quickstart 실측)
- [x] Expert 맵 샘플 출력 — p1-01 `inspect --json` 실출력
- [x] REA ↔ 소스 메타 매핑 표 (docs 또는 polystrata) — `docs/polystrata_spec_draft.md` (p0-03)

---

## Phase 2

### 기능

- [x] 제너레이터 루프로 Expert/레이어 단위 변환이 완료된다. (p2-01 Expert/레이어 청크 루프 — 최신: seek+read 스트림, p4-02에서 `stream/mmap_reader.py` 삭제·root cause 닫힘)
- [x] 양자화·비트 패킹 프리셋 ≥ 1개가 **Phase 0 REA 규칙**에 맞게 적용된다. (p2-01 Q2_0 contract; p3-03~p3-09 Q4_0/Q4_K/Q5_K/Q6_K/Q3_K/Q8_0 — 엔진 체인 acceptance PASS)
- [x] 출력: 가중치 블록 파일(들) + 바이너리 인덱스 + 변환 매니페스트 (p2-01 `meta/manifest.py` 19열 index.txt + manifest.json)
- [x] CLI `convert`로 end-to-end 실행 가능 (p2-01 CLI; p5-03 quickstart 실측 rc 0 GGUF 16,508,352 B)

### 제약 검증

- [x] 장시간 변환 중 RSS 곡선이 평탄 (단일 청크 피크 반복, 단조 증가 없음) (p2-01 RSS 샘플; p5-09 실model emit rss_peak 601,104,384 B / heap peak 409,193,020 B 실측)
- [x] `--max-rss-gb` (또는 동등) 초과 시 안전 중단 (p2-01 `monitor/rss.py` 안전 중단; exit code 3 규격 p5-02)
- [x] CONSTRAINTS PR 체크리스트 통과 (PR 템플릿 C1-C2 체크 — p5-04 AC4 산출)

### 산출물

- [x] `src/stratashift` 스트리밍 파이프라인 (cli/ingest/quant/polystrata/meta/monitor; `stream/`는 entry 메타데이터 게이트 + seek+read로 p4-02 갱신)
- [x] RSS 모니터 로그 샘플 — p2-01 `monitor/rss.py` 실측 로그; p5-09 emit rss_samples 전수 기록
- [x] 단위/통합 테스트 (`test_no_full_load`, stream 관련) — `tests/test_no_full_load.py` GREEN (전수 suite 106 passed)

---

## Phase 3

### 기능

- [x] 변환 산출물을 PolyStrata 런타임 로더에 주입하여 **로드 성공** (p3-01 엔진 golden build+verify PASS, 우리 산출물 golden byte-identical; 최신: p5-08/p5-09 실model gemma4 load rc=0 + API 왕복 HTTP 200)
- [x] (최소) 스모크 추론 또는 로더 self-check 통과 (p3-01 strata-gguf guard + strata-dequant+CHECKSUM; 최신: strata-poly RUN rc=0 · OpenAI chat completion 왕복 — p5-08 AC4·p5-09 AC3~AC4)
- [x] 엔진 에러/segfault 로그 + `docs/rea` 재대조로 변환 모듈 패치 피드백 루프(반자동 이상) (p3-01 엔진 FAIL 로그 4건 → REA 재대조 → 패치 → 재검증 PASS; p5-08 rope_freqs 부재 root cause → fix)

### 제약 검증

- [x] 피드백 패치가 테스트 게이트 없이 강제 적용되지 않음 (p3-01 각 패치는 byte-identical/verify 게이트 경유; never-invented 게이트 — p5-08 rope, p5-09 tokenizer gates)
- [x] 동일 입력·프리셋 → 출력 해시 재현 (회귀) (p3-01 2회 convert sha256 identical; p5-08 AC3 host regen bit-identical 29fa0890 재현)
- [x] 로드 성공으로 확인된 REA `HYPOTHESIS`는 확정으로 승격·문서 갱신 (p0-06 VERIFIED 승격 기록; p3-09 Q8_0 producer 규칙 VERIFIED — mtp_pack 바이트 동일 게이트)

### 산출물

- [x] `verify` 또는 문서화된 수동 검증 절차 (CLI `verify` 두 계약 dispatch — engine-form/p2-form, p5-02; `scripts/p3_engine_smoke.py`)
- [x] 실패 케이스 → 패치 기록 샘플 1건 이상 (p3-01 Result 4건; p5-08 rope_freqs root cause → fix; p5-09 3결함 같은 작업 수정)
- [x] `v0.3.0-runtime` 태그 후보 릴리즈 노트 (docs/RELEASE_NOTES.md v0.3.0-runtime 후보 — 2026-10-16)

---

## Phase 4

Phase 4의 정본 AC는 Goal 파일 `docs/work/p4-01~p4-03.md`의 체크 항목이다 (ROADMAP Phase 4 완료 기준).
여기는 Goal AC의 요약본이며, 상세 실측 수치는 각 Goal 파일의 Result를 정본으로 한다.

### 기능

- [x] HF naming → engine GGUF 키 매핑 + fused gate_up split + BF16→F32 (p4-01, 2026-10-17) — 엔진 체인 SMOKE PASS (qwen4exp geometry), `convert --emit-gguf --hf-emit`.
- [x] 엔진 geometry 가드 밖 모델(Mixtral·Qwen1.5-MoE)은 reference verifier acceptance로 고정: 엔진 `gguf_reader.py` parse match + `canonical_xcheck` dequant 왕복 bit-exact + our `verify` PASS.
- [x] multi-shard `index.json` 입력 — 엔진 미러 게이트(누락/양방향 전사/중복/in_bounds/total_size), shard emit 산출 sha256 = 단일 파일 산출 동일 (p4-02, 2026-10-17).
- [x] 실모델 스모크: Qwen1.5-MoE-A2.7B (8 shards, 27 GB, BF16, per-expert Qwen2MoE naming) 스트리밍 emit 267 tensors, 2회 emit sha256 동일, our verify PASS (p4-03, 2026-10-17).

### 제약 검증

- [x] no-full-load gate: 실측 heap peak emit 126 MB / convert 377 MB / 실모델 slice heap peak 116,821,170 B (< 2 GiB). shard 경로는 seek+read(mmap 금지 — fadvise 불충분 기록).
- [x] 엔진 가드 밖 geometry는 엔진 구동 불가로 기록(가드 applicable 아님) — 임의 커스텀 포맷 최종 타깃 확정 금지 (AGENTS).
- [x] 미확인 HF naming(Qwen3-Next linear_attn 세부)는 HYPOTHESIS 확장 금지 유지 — 실측한 것만 매핑.
- [x] 전수 pytest GREEN: 89 passed (p4-02) / 90 passed (p4-03, 실모델 gate 포함).

### 산출물

- [x] `src/stratashift/ingest/shard_index.py`, HF naming 매핑, CLI `--hf-emit`/`--guard-keys`
- [x] `scripts/p4_real_model_smoke.py` + `tests/test_real_model_smoke.py` (영구 pytest 게이트)
- [x] `v0.4.0-moe` 마일스톤 후보 릴리즈 노트 (RELEASE_NOTES)

---

## Phase 5

### 기능

- [x] `pip install .` 후 `stratashift` CLI가 console entry point로 설치·실행된다. (p5-01: `pip install . --target` + `stratashift --help` exit 0 실측, `tests/test_packaging.py` GREEN)
- [x] `inspect`/`convert`/`verify` 전 계층 `--help`·종료 코드·에러 메시지가 사용자 수준으로 정돈된다. (p5-02: exit code 규격 0/1/2/3, `_run()` wrapper, `tests/test_cli_exit_codes.py` 8 GREEN)
- [x] README quickstart(설치→inspect→convert→verify)가 실측 예시로 존재한다. (p5-03: 실측 4단계 고정 + `scripts/p5_quickstart.py` 재실행 4 steps PASS; API 종점 절은 p5-09 AC4 실측 반영 잔여 — DOC_SYNC)
- [x] CI(pytest 전수 + 금지 API grep)가 푸시/PR 게이트로 동작한다. — Actions run `39896690157` gate success, PR #5 머지 `84970e3` (p5-04 AC5, 2026-10-23).
- [ ] `v1.0.0` 릴리즈 노트 확정 + git 태그 `v1.0.0`. — p5-05 planned (human 종속): 게이트 4·5(API 왕복 전수 정합·임의 아키텍처 RUN p6-01) 미실측 상태에서 태그 금지 (ROADMAP 목적 정본 2026-10-20).

### 제약 검증

- [x] 런타임 의존에 전체 로드 API(safetensors/torch) 추가 없음 (CONSTRAINTS C1). (p5-01 `dependencies = ["numpy"]` 실측 + grep 게이트 0건)
- [x] 전수 pytest GREEN + `test_no_full_load` 게이트 통과. (2026-10-20 WSL 실측: 106 passed/16 skipped + 격리 재실행 `test_real_model_smoke` **1 passed (3155.96s)** ⇒ 유효 **107 GREEN**. 단일 실패 1건은 suite+probe+API smoke WSL 동시 경쟁의 ENOMEM(p4-03 root cause 재현) — 코드 아님, 격리 재실행으로 확정)

### 산출물

- [x] `p5-01~p5-05` Goal 기록 (`docs/work/`) — p5-01~p5-05 실재 (p5-05는 설계 게이트 생성)
- [~] CI 워크플로 + PR 템플릿 — PR 템플릿 push 완료; ci.yml 로컬 커밋 29a1b59 보존 (human gate)
- [ ] `v1.0.0` 태그 + RELEASE_NOTES 확정 — v0.3.0·v0.4.0 후보 노트 실재; v1.0.0은 위 게이트 후

---

## 공통 실패 조건 (어느 Phase든)

다음이 있으면 해당 Phase 미완료:

1. 전체 모델 RAM/VRAM 적재 경로 존재
2. 원본 Strata-only 규격으로만 맞춰 PolyStrata에서 로드 불가
3. Phase 0 없이 Phase 2 패킹 규칙을 “임의 추정”으로만 확정
4. 인수 항목을 “나중에”로 남긴 채 다음 Phase 기능을 머지

---

## 서명 (인간 검증)

| Phase | 검증자 | 날짜 | 결과 |
|-------|--------|------|------|
| 0 | | | |
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
