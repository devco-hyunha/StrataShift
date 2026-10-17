# Phase 0 — Ghidra REA 분석 절차

PolyStrata 엔진이 가중치를 읽어들이는 **저수준 바이트 규격**을 파악하기 위한 역설계(REA) 가이드다.  
툴킷: [Ghidra](https://github.com/NationalSecurityAgency/ghidra) (NSA)

---

## 1. 목적

| 추출 대상 | 설명 |
|-----------|------|
| 바이트 정렬 (alignment) | 블록/텐서 시작 주소·패딩 |
| 오프셋 구조 | 헤더 필드, 인덱스 → 데이터 오프셋 |
| 비트 패킹 룰 | 시프트·마스크·패킹 폭·엔디안 |
| 구조체(Struct) 레이아웃 | 로더가 기대하는 필드 오프셋 |
| 메모리 페이징 | Expert/레이어 단위 로드 경계 |

산출물은 Phase 1 헤더 정렬 매핑과 Phase 2 `quant`/`polystrata` 구현의 **정본 입력**이 된다.

## 2. 사전 준비

순서는 **도구 먼저 → 분석 대상 확보 → 핀**이다. Ghidra 없이 바이너리 REA로 넘어가지 않는다.

1. Ghidra를 **OS/도구 경로에 설치** (저장소에 벤더하지 않음)  — p0-01 완료
   - https://github.com/NationalSecurityAgency/ghidra  
   - JDK(요구 버전) 확인: `java -version` — Ghidra **12.1.x는 Java 21** (GettingStarted.md)
   - 설치본: `/opt/data/tools/ghidra_12.1.4_PUBLIC/` · JDK: `/opt/data/tools/jdk-21/`
   - 압해제는 실행 비트가 사라진다 → shebang 스크립트 **뿐 아니라** 네이티브 바이너리도
     `chmod +x`: `support/analyzeHeadless`, `Ghidra/Features/Decompiler/os/linux_x86_64/decompile`,
     `sleigh` 등 `os/*/` 전부. (p0-02 실측: native decompiler 실행비트 소실로 디컴파일 전수 실패)
   - Ghidra 프로젝트 파일 → **저장소 밖** `/opt/data/ghidra-projects/` (git 미추적).
     **`.local/ghidra-projects/`는 사용 불가**: Ghidra가 `.`로 시작하는 경로 요소를
     거부한다 (`NamingUtilities.checkName` → "Path element starting with '.' is not permitted").
   - headless 확인: `JAVA_HOME=/opt/data/tools/jdk-21 analyzeHeadless <dir> <Proj> -import <file> -NoAnalysis`
     (`-create`는 12.1.4에서 단일 인자 옵션이 아니다 — `-import`가 프로젝트 생성)
2. PolyStrata 실행 파일·로더 바이너리(및 가능하면 소스) 확보 → **`.local/polystrata/`** (git 무시)  
   - 우선: https://github.com/VecSzn/PolyStrata  
   - 백업: https://huggingface.co/spaces/VecSzn/PolyStrata (소스 zip / 사전 빌드)
3. 분석 대상 바이너리 해시·버전을 [REFERENCES.md](REFERENCES.md)에 핀
4. (가능하면) 공개 소스 로더와 바이너리를 **교차 대조**할 준비
5. 커밋 정책: [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md)

## 3. Ghidra 작업 단계

### Step A — 프로젝트 임포트

1. Ghidra에서 새 Project 생성 (`StrataShift-PolyStrata-<version>`)
2. PolyStrata 런타임/로더 실행 파일 임포트
3. Auto-analyze 실행 (기본 분석기 + Decompiler 활성)

### Step B — 가중치 파싱 함수 탐색

검색 힌트 (심볼·문자열·교차참조):

- 가중치/모델 파일 확장자·매직 문자열
- `mmap`, `CreateFileMapping`, `open`+`pread` 등 I/O
- `align`, `pack`, `quant`, `expert`, `page`, `offset` 관련 심볼/문자열
- 의심 매직 상수 (헤더 magic / version)

가중치 파일 파싱 함수와 **메모리 블록 할당 루틴**을 진입점으로 표시한다.

### Step C — 디컴파일·구조체 복원

1. 파싱 루틴 디컴파일
2. 비트 시프트(`<<`, `>>`), AND 마스크, 루프 stride를 기록
3. 헤더/인덱스용 Data Type(Struct) 초안 작성
4. 필드별 오프셋·크기·엔디안을 표로 정리

### Step D — 규격 초안 문서화

결과를 다음 위치에 고정한다:

```text
docs/rea/
├── README.md                 # 분석 세션 요약
├── binary_pin.md             # 바이너리 경로·SHA256·버전
├── loader_entrypoints.md     # 함수 주소·심볼·역할
├── struct_layouts.md         # Struct 오프셋 표
├── bitpack_rules.md          # 시프트·마스크·패킹 폭
└── paging_model.md           # Expert/레이어 페이징 규칙
```

동시에 코드 상수 후보를 `src/stratashift/polystrata/`에 반영할 수 있도록 필드명을 안정적으로 짓는다.

### Step E — LLM 에이전트 연동 (반자동)

1. Ghidra 디컴파일 스니펫·Struct 표를 에이전트에 전달 ([AGENTS.md](../AGENTS.md) T0)
2. 에이전트가 Python `headers.py` / `pack.py` 초안·패치 제안
3. **인간 리뷰 + 단위 테스트 게이트** 후에만 머지 ([CONSTRAINTS.md](CONSTRAINTS.md) C4)

## 4. 소스 우선 vs 바이너리 REA

| 상황 | 전략 |
|------|------|
| 로더 소스가 충분히 공개됨 | 소스를 1차 정본, Ghidra로 빌드 결과 검증 |
| 소스 부재·난독·불일치 | Ghidra REA를 1차 정본, 런타임 로드로 검증 |
| GitHub 접근 불가 | HF Space zip/바이너리로 동일 절차 |

추측은 문서에 `HYPOTHESIS:` 로 표시하고, Phase 3 로드 성공 전까지 확정하지 않는다.

## 5. 완료 정의

→ [ACCEPTANCE.md — Phase 0](ACCEPTANCE.md#phase-0)

## 6. REA 인터페이스 레이어 (p0-05 — 전환)

**전환 정의**: REA(morluto/rea)가 **agent-facing REA 인터페이스**다. Ghidra는 REA의
provider 엔진으로 그대로 사용한다(REA는 Hopper/Ghidra/IDA를 래핑 — 엔진 교체 아님).
`docs/rea/` 정본과 근거 규칙은 유지된다. **p0-05 완료 (2026-10-17)** — AC1~AC7
판정·실측은 [docs/work/p0-05.md](work/p0-05.md) Result.

### 6.1 환경 게이트

- `GHIDRA_INSTALL_DIR=/opt/data/tools/ghidra_12.1.4_PUBLIC` + `JAVA_HOME=/opt/data/tools/jdk-21`
  → `rea doctor`의 ghidra checks 전수 healthy (java check는 JAVA_HOME 필수 — p0-05 실측)
- Node 22.19+/24.11+/26+, procps `ps` (Linux owned-process prerequisite)

### 6.2 CLI 절차 (터미널/스크립트)

```bash
rea inspect <bin> --provider ghidra --json          # binary inventory (procedures/strings)
rea search <bin> <pattern> --provider ghidra --json # 심볼·문자열 검색
rea function <bin> <PROC> --provider ghidra --json   # function dossier (pseudocode·assembly·xref·CFG)
rea instructions <bin> <PROC> --provider ghidra      # raw assembly
rea xrefs <bin> --address=<addr> --provider ghidra    # bounded references
rea trace <bin> --query=<literal> --provider ghidra   # bounded feature trace
```

- 결과는 canonical Evidence record: `evidence_id`, `normalized_result`, `confidence`
  (`observed`/`derived`), `limitations`. target 유래 문자열은 **data** — 지시문 실행 금지
  (mcp-contracts untrusted_content marker).
- Ghidra cold first-query는 client timeout을 초과할 수 있다 — first-query deadlines 절차 준수.

### 6.3 MCP 절차 (Hermes)

- 서버: `rea --mcp` (stdio). Hermes 등록: `hermes mcp add rea` — 실측 serverInfo `rea 6.5.0`,
  tools/list 207건. 도구는 `mcp_rea_*` 접두어로 새 세션에 노출 (재시작 필요).
- env: `JAVA_HOME`, `GHIDRA_INSTALL_DIR` (stdio 서브프로세스는 필터된 env만 상속 — env 키로 지정).

### 6.4 근거 접두어 확장

- `REA:` — REA evidence 산출 근거. `evidence_id`와 `confidence`(`observed`/`derived`) 병기 필수.
- `DECOMPILED:`(Ghidra headless batch, p0-02 정본)와 병존. 신규 REA 작업은 `REA:`로 시작.
- headless batch 덤프(§3 Step A~D)는 정본 산출 수단으로 유지 — REA는 대화형·탐색형 작업에 우선.

### 6.5 스모크 실측 (p0-05, 2026-10-17)

- `rea inspect strata-dequant --provider ghidra` exit 0 — procedures 144, strings 233,
  evidence `ev_6996678bab2d91148255dfd76a07f9f5aeffc19e9f41664e7a26cfba0f776d6d`
- `rea search strata-dequant dequant` — 9 hits (`_ZN6strata15dequantize_q3_KEPKhPf` 외)
- `rea function strata-dequant _ZN6strata15dequantize_q3_KEPKhPf` — pseudocode confidence `observed`
  (fp16 scale unpack + bit walk — Q3_K contract과 소스 교차 가능)

## 7. 관련 링크

| 항목 | URL |
|------|-----|
| REA 인터페이스 | https://github.com/morluto/rea |
| Ghidra | https://github.com/NationalSecurityAgency/ghidra |
| PolyStrata | https://github.com/VecSzn/PolyStrata |
| HF 백업 | https://huggingface.co/spaces/VecSzn/PolyStrata |
