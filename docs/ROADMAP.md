# StrataShift 개발 로드맵

## 개요

| Phase | 이름 | 목표 | 산출물 |
|-------|------|------|--------|
| Docs | 문서 베이스라인 | 기획·제약·에이전트 지침 고정 | docs / AGENTS.md |
| **0** | **엔진 바이너리 역설계 (REA)** | 가중치 로드 저수준 규칙 파악 | `docs/rea/*`, 규격 상수 초안 |
| 1 | PoC·메타데이터 | 전체 적재 없이 구조 파악 증명 | 헤더 파서, Expert 맵, REA 규격 매핑 |
| 2 | 스트리밍·양자화 | 청크 트랜스파일러 | 변환 CLI, 모니터, 샘플 출력 |
| 3 | 엔진 연동·피드백 | PolyStrata에서 로드·구동 검증 | 주입 테스트, AI 디버그 루프 |

---

## Docs — 문서 베이스라인 (완료)

- [x] 제품 명세서 / 아키텍처 / 제약 / 참조 / REA 절차
- [x] AGENTS.md 프롬프트 (Ghidra 포함)
- [x] AI 자동화 워크플로 (`AI_DEV_WORKFLOW` / `DOC_SYNC` / `docs/work`)
- [ ] PolyStrata·Ghidra 핀 버전 기록 (Phase 0에서 채움)

---

## Phase 0 — 엔진 바이너리 역설계 (REA 분석 레이어)

### 목표

PolyStrata 엔진이 가중치를 읽어들이는 **저수준 바이트·패킹·오프셋 규칙**을 파악한다.

### 작업

1. PolyStrata 실행 파일/로더를 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)에 임포트
2. 가중치 파일 파싱 함수·메모리 블록 할당 루틴 디컴파일
3. 바이트 패킹, 정렬, Struct 오프셋, 페이징 규칙 역설계
4. (병행) 공개 소스가 있으면 소스↔바이너리 교차 검증
5. LLM 에이전트에 디컴파일/Struct 표를 넘겨 Python 상수·패커 초안 유도
6. 결과를 `docs/rea/` 및 `polystrata` 상수 초안으로 고정

상세 절차: [REA_GHIDRA.md](REA_GHIDRA.md)

### 완료 기준

→ [ACCEPTANCE.md — Phase 0](ACCEPTANCE.md#phase-0)

### 의존성

- PolyStrata 바이너리 또는 분석 가능한 로더 (GitHub / HF Space)
- Ghidra 설치

---

## Phase 1 — PoC 및 메타데이터 파싱 검증 (입력 레이어)

### 목표

모델 전체를 VRAM/RAM에 올리지 않고 구조를 파악할 수 있음을 증명한다.

### 작업

1. **safetensors 헤더 파싱**
   - Python으로 헤더 JSON + tensor metadata 추출
   - mmap 사용, data blob 전체 로드 금지
2. **레이어·Expert 오프셋 추출**
   - 이름, shape, dtype, data_offsets
   - FFN Expert 텐서 식별 규칙 초안
3. **Phase 0 REA 규격 매핑**
   - Ghidra로 얻은 헤더 정렬·블록 규칙과 소스 메타데이터를 연결
   - `docs/polystrata_spec_draft.md` (또는 `docs/rea` + 코드 상수) 갱신
4. **PoC 스크립트**
   - `scripts/poc_parse_header.py` (가칭)
   - 단일 레이어 슬라이스 읽기 데모 (변환 전)

### 완료 기준

→ [ACCEPTANCE.md — Phase 1](ACCEPTANCE.md#phase-1)

### 의존성

- Phase 0 규격 초안 (최소: 헤더 정렬·매직/버전 가설)
- 소형/중형 테스트용 MoE safetensors (또는 샤드 일부)

---

## Phase 2 — 디스크 스트리밍 및 양자화 파이프라인 (처리 레이어)

### 목표

순차적 청크 기반 트랜스파일러를 구현한다.

### 작업

1. **제너레이터 루프**
   ```
   [디스크에서 Expert/레이어 1개 로드]
     → [CPU에서 비트 패킹·양자화]
     → [타깃 포맷 스트리밍 쓰기]
     → [메모리 해제]
   ```
2. **양자화·패킹 모듈** — Phase 0 REA 비트 패킹 규칙 반영 프리셋 ≥ 1
3. **인덱스·매니페스트 writer**
4. **메모리 안정성 모니터** — RSS 샘플링, 평탄 곡선 검증
5. **CLI** — `stratashift convert ...`

### 완료 기준

→ [ACCEPTANCE.md — Phase 2](ACCEPTANCE.md#phase-2)

---

## Phase 3 — PolyStrata 연동 및 피드백 루프 (실행 레이어)

### 목표

변환 결과물이 PolyStrata 런타임에서 정상 로드·구동되는지 확인한다.

### 작업

1. 변환 테스트 가중치를 PolyStrata 로더에 주입
2. 로드/추론 스모크 테스트
3. AI 디버깅 루프
   - 엔진 에러 코드·segfault 로그 수집
   - LLM 에이전트 피드백 → 변환 스크립트 패치 제안
   - Ghidra REA 메모(`docs/rea/`)와 교차 대조 후 오프셋/패킹 수정
4. 회귀 테스트: 동일 입력 → 동일 출력 해시

### 완료 기준

→ [ACCEPTANCE.md — Phase 3](ACCEPTANCE.md#phase-3)

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
```

## 마일스톤 태그 (제안)

| 태그 | 의미 |
|------|------|
| `v0.0.1-rea` | Phase 0 규격 초안 고정 |
| `v0.1.0-poc` | Phase 1 완료 |
| `v0.2.0-stream` | Phase 2 완료 |
| `v0.3.0-runtime` | Phase 3 최소 연동 |
| `v1.0.0` | 공개 가능한 CLI·문서·안정성 |

## 업스트림 추적 (상시)

PolyStrata는 제품 수명과 함께 **따라가야** 한다.  
일상은 핀 유지, 엔진 로더/패킹 변경 시에만 [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) 업그레이드 절차를 수행한다.

- 지원 매트릭스·매니페스트 핀 필드 유지
- Ghidra는 도구 — 필수 추적 대상 아님

## 다음 즉시 액션

1. HF Space / GitHub에서 PolyStrata 바이너리·소스 핀 + 지원 표 `planned`
2. Ghidra 설치 후 로더 임포트 ([REA_GHIDRA.md](REA_GHIDRA.md))
3. `docs/rea/`에 Struct·bitpack 초안 기록
4. Phase 1: `ingest/safetensors_header.py` PoC
