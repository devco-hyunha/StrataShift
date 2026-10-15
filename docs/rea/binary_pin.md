# binary_pin — 분석 대상 핀

Goal: `p0-01` · Pin: PolyStrata `c566ff4` · Ghidra `12.1.4`

## 1. PolyStrata (분석 대상 소스)

| 필드 | 값 |
|------|-----|
| 확보 경로 | GitHub `main` zip (GitHub API `commits/main`로 커밋 확인) |
| 저장소 | https://github.com/VecSzn/PolyStrata |
| 커밋 SHA | `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` |
| 커밋 일시 | 2026-10-15T03:48:30Z (`readme: Tiel Coder's workstation numbers …`) |
| zip 파일 | `.local/polystrata/polystrata-main-c566ff4.zip` (git 미추적) |
| zip SHA256 | `3ecc0e4c922cbef0a86b7dc8317c67a5545ddd06c430994ba56afde8ffa86b24` |
| zip 크기 | 25,263,408 B · 3,728 entries |
| 압해제 위치 | `.local/polystrata/PolyStrata-main/` |
| 바이너리 버전 | 미확정 — zip에 사전 빌드 실행 파일 없음 (`*.exe/*.so/*.dll` 0개) |

**zip에 들어 있는 `.bin`은 가중치/프로필 데이터지 실행 파일이 아니다**
(`data/expert-profile.bin`, `data/draft_vocab*.bin` 등). 따라서 REA의 정본 대상은
**핀된 소스에서 빌드한 로더 ELF**다.

- `HYPOTHESIS:` p0-02의 분석 대상은 `CMakeLists.txt`로 빌드한 로더 계열
  (`strata-load`, `src/core/weights.cpp`, `src/artifact/gguf_reader.cpp`)이며,
  빌드 산출물의 SHA256을 이 파일에 추가 기록한다.
- CONSTRAINTS C4: zip 안의 바이너리·모델 가중치 **실행 금지** (해시·hexdump만).

## 2. Ghidra (도구)

| 필드 | 값 |
|------|-----|
| 릴리즈 | Ghidra **12.1.4** (`Ghidra_12.1.4_build`, 2026-09-21) |
| zip | `/opt/data/tools/ghidra_12.1.4_PUBLIC_20260921.zip` |
| zip SHA256 | `ddac49f903da9d5bac833e5cc79395098b9c33cfd3279be5f31bd00387d2d4db` — GitHub 릴리즈에 공개된 값과 **일치 확인** |
| 설치 경로 | `/opt/data/tools/ghidra_12.1.4_PUBLIC/` (저장소 밖, git 미추적) |
| headless | `/opt/data/tools/ghidra_12.1.4_PUBLIC/support/analyzeHeadless` (기동 확인) |
| JDK | Temurin **21.0.12.1+1** `/opt/data/tools/jdk-21/` (Ghidra 12.1 요구는 Java 21 — GettingStarted.md 명문화) |
| JDK tar.gz SHA256 | `ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94` |

## 3. 프로젝트 루트 (환경 제약)

Ghidra 12.1은 **`.`로 시작하는 경로 요소를 거부한다**
(`NamingUtilities.checkName` → "Path element starting with '.' is not permitted").
`.local/ghidra-projects/`는 **사용 불가**가 실증으로 확인되었다 (REA_GHIDRA §2-1 갱신).

- 프로젝트 루트: `/opt/data/ghidra-projects/` (저장소 밖 → git 미추적)
- 스모크 테스트: `analyzeHeadless /opt/data/ghidra-projects StrataShift-Smoke -import /bin/ls -NoAnalysis`
  → `REPORT: Import succeeded` (`.gpr`/`.rep` 생성)

## 4. p0-02에 넘기는 것

- 빌드 대상: 핀된 소스의 로더 계열 (CMake)
- Ghidra 프로젝트명: `StrataShift-PolyStrata-c566ff4`
- 데이터 파일의 정적 분석 대상: `data/expert-profile.bin` (STRP, §3 of `paging_model.md`에서 바이트 검증됨)
