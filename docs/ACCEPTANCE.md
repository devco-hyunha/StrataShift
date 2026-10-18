# StrataShift 단계별 인수 기준 (Acceptance Criteria)

각 Phase는 아래 항목을 **모두** 만족해야 완료로 본다.

---

## Phase 0

### 기능

- [ ] PolyStrata 런타임/로더 바이너리를 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)에 임포트·분석했다.
- [ ] 가중치 파싱·메모리 블록 할당 관련 진입점(함수)이 `docs/rea/loader_entrypoints.md`에 기록된다.
- [ ] Struct 오프셋·바이트 정렬·비트 패킹(시프트/마스크)·페이징 규칙 초안이 `docs/rea/`에 있다.
- [ ] REFERENCES에 엔진 바이너리 SHA256·버전 및 Ghidra 버전이 핀된다.
- [ ] 불확실 항목은 `HYPOTHESIS:`로 표시된다.

### 제약 검증

- [ ] REA 결과만으로 확정하지 않은 필드를 “확정”으로 표기하지 않았다.
- [ ] LLM이 생성한 상수/패커 초안이 있다면 인간 리뷰 기록이 있다.

### 산출물

- [ ] `docs/rea/` 세트 ([REA_GHIDRA.md](REA_GHIDRA.md) Step D)
- [ ] (선택) `polystrata` 모듈용 상수 초안
- [ ] `v0.0.1-rea` 마일스톤 후보 노트

---

## Phase 1

### 기능

- [ ] `.safetensors` 파일에서 텐서 이름, dtype, shape, data_offsets를 추출한다.
- [ ] 가중치 본문 전체를 RAM에 복사하지 않는다 (헤더 + 선택적 단일 슬라이스만).
- [ ] MoE FFN Expert 후보 텐서를 나열하는 Expert 맵(JSON 등)을 출력한다.
- [ ] **Phase 0 REA 규격**을 바탕으로 엔진 요구 헤더 정렬·블록 규칙이 매핑·문서화된다.

### 제약 검증

- [ ] 코드 경로에 전체 `load_file` / 전체 state_dict 적재가 없다.
- [ ] 테스트 모델(또는 샤드)에서 RSS가 “헤더 + 최대 1개 슬라이스” 수준으로 설명 가능하다.

### 산출물

- [ ] `scripts/poc_parse_header.py` (또는 동등 CLI `inspect`)
- [ ] Expert 맵 샘플 출력
- [ ] REA ↔ 소스 메타 매핑 표 (docs 또는 polystrata)

---

## Phase 2

### 기능

- [ ] 제너레이터 루프로 Expert/레이어 단위 변환이 완료된다.
- [ ] 양자화·비트 패킹 프리셋 ≥ 1개가 **Phase 0 REA 규칙**에 맞게 적용된다.
- [ ] 출력: 가중치 블록 파일(들) + 바이너리 인덱스 + 변환 매니페스트
- [ ] CLI `convert`로 end-to-end 실행 가능

### 제약 검증

- [ ] 장시간 변환 중 RSS 곡선이 평탄 (단일 청크 피크 반복, 단조 증가 없음)
- [ ] `--max-rss-gb` (또는 동등) 초과 시 안전 중단
- [ ] CONSTRAINTS PR 체크리스트 통과

### 산출물

- [ ] `src/stratashift` 스트리밍 파이프라인
- [ ] RSS 모니터 로그 샘플
- [ ] 단위/통합 테스트 (`test_no_full_load`, stream 관련)

---

## Phase 3

### 기능

- [ ] 변환 산출물을 PolyStrata 런타임 로더에 주입하여 **로드 성공**
- [ ] (최소) 스모크 추론 또는 로더 self-check 통과
- [ ] 엔진 에러/segfault 로그 + `docs/rea` 재대조로 변환 모듈 패치 피드백 루프(반자동 이상)

### 제약 검증

- [ ] 피드백 패치가 테스트 게이트 없이 강제 적용되지 않음
- [ ] 동일 입력·프리셋 → 출력 해시 재현 (회귀)
- [ ] 로드 성공으로 확인된 REA `HYPOTHESIS`는 확정으로 승격·문서 갱신

### 산출물

- [ ] `verify` 또는 문서화된 수동 검증 절차
- [ ] 실패 케이스 → 패치 기록 샘플 1건 이상
- [ ] `v0.3.0-runtime` 태그 후보 릴리즈 노트

---

## Phase 5

### 기능

- [ ] `pip install .` 후 `stratashift` CLI가 console entry point로 설치·실행된다.
- [ ] `inspect`/`convert`/`verify` 전 계층 `--help`·종료 코드·에러 메시지가 사용자 수준으로 정돈된다.
- [ ] README quickstart(설치→inspect→convert→verify)가 실측 예시로 존재한다.
- [ ] CI(pytest 전수 + 금지 API grep)가 푸시/PR 게이트로 동작한다.
- [ ] `v1.0.0` 릴리즈 노트 확정 + git 태그 `v1.0.0`.

### 제약 검증

- [ ] 런타임 의존에 전체 로드 API(safetensors/torch) 추가 없음 (CONSTRAINTS C1).
- [ ] 전수 pytest GREEN + `test_no_full_load` 게이트 통과.

### 산출물

- [ ] `p5-01~p5-05` Goal 기록 (`docs/work/`)
- [ ] CI 워크플로 + PR 템플릿
- [ ] `v1.0.0` 태그 + RELEASE_NOTES 확정

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
