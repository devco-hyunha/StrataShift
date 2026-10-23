# 업스트림 버전 정책 (PolyStrata 추적)

StrataShift는 **PolyStrata 호환 브릿지**다.  
장기적으로 엔진 버전을 따라가야 하며, 운영은 **핀(고정) → 지원 표 → 계획적 업그레이드**로 한다.

관련 핀 표: [REFERENCES.md](REFERENCES.md)  
로컬 엔진 배치: [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md)

---

## 1. 원칙

| 원칙 | 내용 |
|------|------|
| 종속 | 타깃 런타임은 항상 **PolyStrata** (원본 Strata-only 아님) |
| 핀 우선 | 개발·릴리스는 **특정 PolyStrata 버전/커밋/바이너리 SHA**에 맞춤 |
| 따라가기 | upstream 변경을 **즉시 자동 추적하지 않음**. 지원 버전을 올릴 때 의식적으로 마이그레이션 |
| 산출물 표기 | 변환 매니페스트에 `polystrata_version` / `binary_sha256` / `spec_version` 기록 |
| Ghidra | 도구일 뿐. 엔진과 달리 **매 릴리스 추적이 필수는 아님** (분석 막힐 때만 업그레이드) |

---

## 2. 지원 매트릭스 (작성·갱신)

StrataShift 릴리스마다 아래 표를 채운다. (초기값은 Phase 0 이후)

### 현재 지원 (Current)

| 항목 | 값 |
|------|-----|
| StrataShift 버전 | _TBD_ |
| PolyStrata 핀 (태그/커밋) | `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` (GitHub `main`, 2026-10-15) — `planned` |
| 바이너리 SHA256 | `TBD(p0-02)` — zip에 사전 빌드 바이너리 없음, 소스 빌드 예정 |
| 규격 문서 | `docs/rea/` (p0-01 골격 5파일) + spec draft |
| 상태 | `planned` |
| 비고 | zip SHA256 `3ecc0e4c922cbef0a86b7dc8317c67a5545ddd06c430994ba56afde8ffa86b24` · Ghidra 12.1.4 · JDK 21 |

### 지원 이력

| StrataShift | PolyStrata 핀 | 상태 | 종료일 | 비고 |
|-------------|---------------|------|--------|------|
| _TBD_ | _TBD_ | | | |

### 상태 정의

| 상태 | 의미 |
|------|------|
| `planned` | 핀 예정, 아직 로드 검증 전 |
| `supported` | 변환 + 런타임 로드 검증됨. 버그픽스 대상 |
| `deprecated` | 새 이슈는 받지 않음. 문서에 후속 핀 안내 |
| `unsupported` | 제거. 과거 핀만 히스토리에 남김 |

**규칙:** 동시에 `supported`인 PolyStrata 메이저/로더 ABI는 **가능하면 1개** (유지비).  
이전 핀은 `deprecated`로 남기고, 필요 시 브랜치·태그로만 보존한다.

---

## 3. 일상 개발 vs 업그레이드

### 일상 (핀 유지)

1. `.local/polystrata`는 **핀된 커밋/zip**만 사용
2. `REFERENCES` 핀과 불일치하면 작업을 멈춘다 (silent drift 금지)
3. 변환기 변경이 있어도 **매니페스트의 엔진 핀 필드는 유지**

### 업그레이드 (PolyStrata 따라가기)

upstream에 로더·패킹·페이징 등 **호환에 영향 있는 변경**이 있으면:

```text
1. 변경 노트 / diff / 바이너리 확보
2. .local/polystrata 를 새 핀으로 교체 + REFERENCES 갱신
3. Phase 0 재검: docs/rea/ (HYPOTHESIS 재평가)
4. polystrata/quant 상수·패커 패치
4-1. engine_ext (p6-01): python scripts/engine_ext_diff.py <옛 트리> <새 트리> 로 우리가 기대는 엔진 파일만
     비교 → engine_ext 클래스·overlay.json anchor 수정 → --hashes 출력으로 pin_files.json 갱신
     → build-engine --gpu --ext 로 strata-shift-poly 다시 빌드 (해시가 다르면 빌드가 멈춘다)
5. 회귀: convert → PolyStrata 로드 스모크 (+ strata-shift-poly mini Mixtral RUN)
6. 지원 매트릭스: 새 핀 = supported, 구 핀 = deprecated
7. StrataShift 버전/태그 및 CHANGELOG(추후)에 “requires PolyStrata X” 명시
```

호환이 **깨지지 않은** 패치 릴리스면 REA 전체 재실행 없이 로드 스모크만으로 핀을 올릴 수 있다.  
판단이 애매하면 **깨진 것으로 가정**하고 REA 재검한다.

---

## 4. 무엇을 “따라가야 하는” 변경인가

| 변경 유형 | 따라가기 | 작업량 |
|-----------|----------|--------|
| 로더 매직/헤더/Struct 오프셋 | 필수 | REA + 코드 |
| 비트 패킹·양자화 dtype | 필수 | REA + quant |
| Expert 페이징·인덱스 | 필수 | REA + index writer |
| API 서버·UI만 변경 | 보통 불필요 | 핀만 확인 |
| 문서·설치 스크립트만 | 불필요 | — |
| Ghidra 메이저 업 | 선택 | 도구 핀만 |

---

## 5. 매니페스트·CLI 계약 (구현 시)

변환 출력 매니페스트에 최소 포함:

```text
stratashift_version: ...
polystrata_pin: <tag-or-commit>
polystrata_binary_sha256: ...
spec_version: ...          # docs/rea 와 동기
conversion_preset: ...
```

CLI는 가능하면:

- `stratashift convert --require-polystrata-pin <sha>`
- 또는 매니페스트와 엔진 불일치 시 `verify`에서 경고/실패

---

## 6. 공개(public) 전환 시

- README에 **지원 PolyStrata 버전**을 한눈에 표시
- “임의 최신 engine 미보장 — 핀된 버전만 보장”을 명시
- deprecated 핀은 일정 후 문서에서 unsupported로 정리

---

## 7. 에이전트용 한 줄

> PolyStrata는 결국 따라간다. 다만 **핀으로 개발하고, 업그레이드는 REA→코드→로드 검증→지원 표 갱신** 순으로만 한다.  
> 핀 없이 “최신 엔진”에 맞춘다고 가정하지 말 것.

관련: [REFERENCES.md](REFERENCES.md), [REA_GHIDRA.md](REA_GHIDRA.md), [ROADMAP.md](ROADMAP.md), [ACCEPTANCE.md](ACCEPTANCE.md)
