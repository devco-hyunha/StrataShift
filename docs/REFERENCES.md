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

### 엔진 핀 기록 (p0-01 채움)

| 필드 | 값 |
|------|-----|
| 확보 경로 | GitHub `main` zip (GitHub API `commits/main`로 커밋 확인) |
| 커밋 SHA | `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` (2026-10-15T03:48:30Z) |
| zip SHA256 | `3ecc0e4c922cbef0a86b7dc8317c67a5545ddd06c430994ba56afde8ffa86b24` |
| zip 로컬 경로 | `.local/polystrata/polystrata-main-c566ff4.zip` (git 미추적) |
| 바이너리 버전 | zip에 사전 빌드 실행 파일 없음 (`*.exe/.so/.dll` 0개) — p0-02에서 소스 빌드 **완료** |
| 분석 대상 실행 파일 | p0-02 빌드: `strata-gguf` `514ba645…7344e` · `strata-dequant` `e1d39994…aed37` · `strata-plan` `f20f8e0a…b1162` + 컴파일 전용 `.o` 6개 — 전 표: [docs/rea/binary_pin.md](rea/binary_pin.md) §4 |
| 바이너리 SHA256 | 위 표 (9개 산출물 전수) |
| 로더 관련 파일 | `src/artifact/gguf_reader.cpp`, `src/core/weights.cpp`, `src/core/load_main.cpp`, `src/core/expert_cache.cpp`, `src/core/vmm.cpp`, `tools/pack_layer.py`, `tools/pack_index.py` — 표: [docs/rea/loader_entrypoints.md](rea/loader_entrypoints.md) |
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

**로컬 배치:** PolyStrata zip·압해제는 `.local/`에만 두고 **git에 올리지 않는다.**
Ghidra **프로젝트 파일은 저장소 밖에 둔다** — Ghidra 12.1은 `.`로 시작하는 경로 요소를
거부하므로 `.local/ghidra-projects/`는 **사용 불가** (실증: `NamingUtilities.checkName`).
프로젝트 루트: `/opt/data/ghidra-projects/`. 상세: [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md)

### Ghidra 핀 기록 (p0-01 채움)

| 필드 | 값 |
|------|-----|
| Ghidra 릴리즈 | **12.1.4** (`Ghidra_12.1.4_build`, 2026-09-21) |
| zip SHA256 | `ddac49f903da9d5bac833e5cc79395098b9c33cfd3279be5f31bd00387d2d4db` (GitHub 공개값과 일치 확인) |
| 설치 경로 | `/opt/data/tools/ghidra_12.1.4_PUBLIC/` (저장소 밖, git 미추적) |
| JDK | Temurin **21.0.12.1+1** `/opt/data/tools/jdk-21/` (Ghidra 12.1 요구 = Java 21) |
| headless 검증 | `analyzeHeadless /opt/data/ghidra-projects StrataShift-Smoke -import /bin/ls -NoAnalysis` → Import succeeded |
| 프로젝트명 | `StrataShift-PolyStrata-c566ff4` (p0-02 생성 — 9개 산출물 임포트·Auto-analyze) |
| 분석일 | **2026-10-15** — 디컴파일 덤프 `/opt/data/ghidra-decomp/*.decomp.txt` |

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
