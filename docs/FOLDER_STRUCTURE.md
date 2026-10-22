# StrataShift 폴더 구조 규칙

저장소 **디렉터리 레이아웃은 고정 계약**이다.  
새 폴더·패키지를 만들기 전에 이 문서를 따른다. 상세 모듈 API는 [MODULE_DESIGN.md](MODULE_DESIGN.md).

Cursor: `.cursor/rules/folder-structure.mdc` (alwaysApply)

---

## 1. 고정 루트 트리 (허용 골조)

```text
StrataShift/
├── .cursor/rules/          # Cursor 룰만
├── .gitignore
├── AGENTS.md
├── README.md
├── pyproject.toml          # Phase 1+
├── docs/                   # 기획·규격·REA 메모 (코드 금지)
│   ├── rea/                # Ghidra 분석 결과 텍스트만 (커밋)
│   └── work/               # Goal 단위 설계·AC·결과 (AI 워크플로)
├── src/stratashift/        # 유일한 라이브러리 패키지 루트
│   ├── cli/
│   ├── ingest/
│   ├── meta/
│   ├── stream/
│   ├── quant/
│   ├── polystrata/
│   ├── monitor/
│   ├── install/            # p6-12: 없는 프로그램 확인·자동 설치 (I/O 전용)
│   └── feedback/           # Phase 3
├── scripts/                # PoC·일회성 (재사용 시 src로 승격)
├── tests/                  # 테스트만
└── .local/                 # ❌ gitignore — 엔진·Ghidra프로젝트·모델 (커밋 금지)
    ├── polystrata/         # 핀 엔진 소스(PolyStrata-main) + 빌드(build/) — p6-09 기본 경로
    ├── llama.cpp/          # 엔진 CMake 핀 llama.cpp(gguf-py) 압축 해제본 — p6-09 기본 경로
    ├── ghidra-projects/
    ├── models/
    └── out/
```

**Git에 올리는 최상위**는 위 추적 대상만.  
`.local/` 은 분석용 로컬 드롭이며 [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md)를 따른다.  
그 밖의 최상위 폴더를 만들지 않는다. 필요하면 이 문서 + `MODULE_DESIGN.md`를 먼저 갱신한다.

---

## 2. 영역별 책임 (어디에 무엇을 두나)

| 경로 | 넣는다 | 넣지 않는다 |
|------|--------|-------------|
| `docs/` | 명세서, 로드맵, 인수 기준 | `.py`, 가중치, 바이너리 |
| `docs/rea/` | Ghidra Struct/bitpack/paging 메모 | 실행 코드, 모델 파일 |
| `docs/work/` | Goal 설계·AC·리뷰 결과 (`AI_DEV_WORKFLOW`) | 프로덕션 코드 |
| `src/stratashift/cli/` | CLI 엔트리·인자 파싱 | 양자화·mmap 본체 |
| `src/stratashift/ingest/` | safetensors/GGUF 헤더·오프셋 뷰 | 전체 load, 패킹 |
| `src/stratashift/meta/` | Expert 맵, 매니페스트 | 가중치 본문 I/O |
| `src/stratashift/stream/` | 슬라이스 read(seek+read)·emit/convert 단계 루프 (p6-08) | 비트 패킹 수식 |
| `src/stratashift/quant/` | 양자화·비트 패킹 (순수) | 파일 open/경로 I/O |
| `src/stratashift/polystrata/` | 헤더·정렬·인덱스 상수/빌더 | HF 포맷 파서 |
| `src/stratashift/monitor/` | RSS/VRAM 샘플·게이트 | 변환 로직 |
| `src/stratashift/install/` | 필요한 프로그램 확인·내려받기·winget 설치 (p6-12) | 변환·양자화 로직 |
| `src/stratashift/feedback/` | 엔진 로그 수집·패치 힌트 (P3) | 임의 바이너리 실행 |
| `scripts/` | PoC, 수동 실험 스크립트 | 장기 유지 라이브러리 |
| `tests/` | `test_*.py`, 픽스처 | 프로덕션 코드 |
| `.cursor/rules/` | `*.mdc` 룰 | 앱 코드 |

---

## 3. 새 폴더·패키지 추가 규칙

### MUST

1. **라이브러리 코드**는 오직 `src/stratashift/` 아래
2. 하위 패키지는 **허용 영역 목록** 중 하나이거나, 문서에 영역을 추가한 뒤 생성
3. 새 패키지마다 `__init__.py` 포함 (명시적 패키지)
4. 디렉터리명 = `snake_case` 소문자, 단수 개념 (`quants/` ❌ → `quant/` ✅)
5. 깊이는 `src/stratashift/<area>/<file.py>` 를 기본으로 한다. **3단 이상** 하위 폴더는 예외 시에만 (문서에 사유)

### MUST NOT

| 금지 | 이유 |
|------|------|
| 루트에 `src/` 밖 패키지 (`lib/`, `core/`, `app/`) | 패키지 루트 단일화 |
| 루트/`src/`에 `utils/`, `helpers/`, `common/` | 책임 없는 쓰레기통 |
| `docs/` 아래 `src` 미러 코드 | 문서와 코드 혼선 |
| `tests/` 안에 프로덕션 패키지 복사 | drift |
| `data/`, `models/`, `weights/` 를 추적 트리에 둠 | → `.local/models/` + gitignore |
| PolyStrata/Ghidra 본체를 `src/`나 루트에 벤더 | → `.local/` 또는 OS 설치 경로 |
| `tmp/`, `old/`, `backup/` | 로컬 전용, git 제외 |
| 영역 간 동일 이름 파일 난립 (`pack.py`를 ingest에도) | import·책임 혼란 |

---

## 4. 파일 → 폴더 매핑 빠른표

| 만들고 싶은 것 | 넣을 곳 |
|----------------|---------|
| safetensors 헤더 파서 | `src/stratashift/ingest/` |
| Expert 이름 규칙 | `src/stratashift/meta/` |
| 슬라이스 리더 (seek+read, mmap 아님) | `src/stratashift/stream/` |
| Q4 비트 패킹 | `src/stratashift/quant/` |
| 블록 매직/Struct | `src/stratashift/polystrata/` + `docs/rea/` |
| `stratashift convert` | `src/stratashift/cli/` |
| 헤더 파싱 실험 | `scripts/poc_*.py` → 안정화 후 ingest로 승격 |
| Ghidra Struct 표 | `docs/rea/struct_layouts.md` |
| 단위 테스트 | `tests/test_<feature>.py` |

매핑이 안 되면 **새 최상위 영역을 만들지 말고** 기존 영역에 맞추거나 `MODULE_DESIGN`/`FOLDER_STRUCTURE`를 먼저 수정한다.

---

## 5. 승격·이동 규칙

```text
scripts/poc_foo.py
    │ 재사용·테스트 필요
    ▼
src/stratashift/<area>/foo.py  +  tests/test_foo.py
    │ 규격 상수 확정
    ▼
docs/rea/... 와 polystrata 상수 동기화
```

- PoC를 `scripts/`에 남긴 채 다른 모듈에서 import 하지 말 것
- 이동 시 기존 import 경로를 한 번에 정리 (호환 셰임 레이어 남기지 않음)

---

## 6. 변경 절차 (폴더 구조 자체)

폴더 골조를 바꿀 때 순서:

1. `docs/FOLDER_STRUCTURE.md` 수정
2. `docs/MODULE_DESIGN.md` 트리 동기화
3. `.cursor/rules/folder-structure.mdc` 필요 시 갱신
4. 실제 디렉터리 생성/이동
5. `docs/INDEX.md` / README 링크 확인

코드만 먼저 새 경로에 두지 않는다.

---

## 7. 에이전트용 한 줄

> 코드는 `src/stratashift/<area>/`, 문서는 `docs/`, REA 메모는 `docs/rea/`, 엔진·모델은 `.local/`(무시).  
> `utils/`·루트 py·엔진 벤더 커밋 = 거부. 구조 변경은 문서 먼저.

관련: [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md), [CODING_STANDARDS.md](CODING_STANDARDS.md), [MODULE_DESIGN.md](MODULE_DESIGN.md)
