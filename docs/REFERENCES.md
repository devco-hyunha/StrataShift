# 참조 레코드 및 툴킷 경로 (References & Tooling)

엔진 바이트 규격 역설계(REA)와 런타임 분석을 위한 공식 소스·툴킷.

---

## 1. PolyStrata (타깃 엔진)

| 구분 | URL | 용도 |
|------|-----|------|
| GitHub 공식 | https://github.com/VecSzn/PolyStrata | 소스, 이슈, 로더 로직 |
| Hugging Face Space | https://huggingface.co/spaces/VecSzn/PolyStrata | 최신 소스 `.zip`, 사전 빌드 바이너리 백업 |

**비고**

- GitHub 계정 검토 등으로 접근이 제한될 수 있음 → **HF Space를 공식 백업/배포 허브로 사용**
- 지원 모델 예시(엔진 README 기준): Qwen3.8-Flash-Next, GLM 5.3 Flash, Qwen3.6, Ornith 1.5, Gemma 4 등
- Built on Strata; OpenAI/Anthropic API localhost, Windows/Linux one-click install

### 엔진 핀 기록 (Phase 0에서 채움)

| 필드 | 값 |
|------|-----|
| 확보 경로 | GitHub / HF Space zip (택일) |
| 커밋 SHA / zip 날짜 | _TBD_ |
| zip SHA256 | _TBD_ |
| 바이너리 버전 | _TBD_ |
| 분석 대상 실행 파일 | _TBD_ |
| 바이너리 SHA256 | _TBD_ |
| 로더 관련 심볼/파일 | _TBD_ |
| 지원 상태 | `planned` → 이후 [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) 매트릭스 |

**버전 추적:** PolyStrata는 장기적으로 따라가되, **핀 → 계획적 업그레이드**로 운영한다.  
지원 표·마이그레이션 절차: **[UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md)**

## 2. 역설계 및 바이너리 분석 툴킷 (REA) — 필수

| 구분 | URL | 용도 |
|------|-----|------|
| Ghidra (NSA) | https://github.com/NationalSecurityAgency/ghidra | PolyStrata 런타임/로더 바이너리 분석, 메모리 페이징·비트 패킹·Struct 오프셋 추출 |

**역할**

- 가중치 파일 파싱 함수·메모리 블록 할당 루틴 디컴파일
- 바이트 정렬, 오프셋 구조, 비트 시프트/마스크 규칙 역설계
- LLM 에이전트와 연동해 Python 변환 스크립트에 규격 반영 ([REA_GHIDRA.md](REA_GHIDRA.md))

**로컬 배치:** PolyStrata·Ghidra 프로젝트는 `.local/`에만 두고 **git에 올리지 않는다.**  
커밋은 `docs/rea/` 메모와 핀 표만. 상세: [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md)

### Ghidra 핀 기록

| 필드 | 값 |
|------|-----|
| Ghidra 릴리즈/커밋 | _TBD_ |
| 프로젝트명 | _TBD_ (예: `StrataShift-PolyStrata-<ver>`) |
| 분석일 | _TBD_ |

## 3. 기반 원본 엔진

| 항목 | 내용 |
|------|------|
| 이름 | Strata |
| 저작 | Niko1221 및 기여자들 |
| 관계 | PolyStrata의 기반. StrataShift는 **PolyStrata 확장 규격**을 타깃으로 함 |

원본 Strata만 가정한 변환은 금지. 확장 로더·페이징·패킹 차이는 **Phase 0 REA + 소스 대조**로 문서화한다.

## 4. 입력 포맷 참조

| 포맷 | 참조 | StrataShift 사용 |
|------|------|------------------|
| safetensors | https://huggingface.co/docs/safetensors | Phase 1 우선 |
| GGUF | ggml-org / llama.cpp GGUF 스펙 | Phase 2+ |
| Hugging Face MoE 체크포인트 | 모델별 `config.json`, `model*.safetensors` | Expert 이름 규칙 추출 |

## 5. 보조 분석 도구

| 도구 | 용도 | Phase |
|------|------|-------|
| 디버거 / 런타임 로그 | segfault·에러 코드 수집 | 3 |
| 소스 텍스트 검색 | 공개 로더와 REA 결과 교차 검증 | 0~1 |

REA 산출물 경로: `docs/rea/` ([REA_GHIDRA.md](REA_GHIDRA.md) §Step D)

## 6. 본 프로젝트 문서 맵

| 문서 | 역할 |
|------|------|
| [PRODUCT_SPEC.md](PRODUCT_SPEC.md) | 무엇을 만드는가 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 어떻게 흐르는가 |
| [REA_GHIDRA.md](REA_GHIDRA.md) | Ghidra REA 절차 |
| [MODULE_DESIGN.md](MODULE_DESIGN.md) | 모듈·API |
| [ROADMAP.md](ROADMAP.md) | 언제·어떤 순서로 |
| [CONSTRAINTS.md](CONSTRAINTS.md) | 무엇을 하면 안 되는가 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 완료 판정 |
| [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) | PolyStrata 핀·지원 표·업그레이드 |
| [DOC_SYNC.md](DOC_SYNC.md) | 코드↔문서 동기화 |
| [AI_DEV_WORKFLOW.md](AI_DEV_WORKFLOW.md) | AI 자동화 개발 루프 |
| [../AGENTS.md](../AGENTS.md) | AI 에이전트 지침 |

## 7. 인용 시 원칙

1. 엔진·바이너리 규격 인용 시 **핀된 커밋/zip/SHA256**을 명시한다.
2. GitHub와 HF Space 내용이 다르면 **핀한 쪽을 정본**으로 한다.
3. REA 추측은 `HYPOTHESIS:`로 표시하고, 런타임 검증 전 확정하지 않는다.
4. 라이선스·크레딧은 README 및 배포물에 PolyStrata/Strata/Ghidra 고지를 반영한다.
5. PolyStrata 핀을 올릴 때는 [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) 업그레이드 절차를 따른다.
