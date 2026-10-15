# Git에 올리지 말 것 · 로컬 분석 배치

`.gitignore`와 함께 읽는 **커밋 금지 / 로컬 전용** 정책이다.

---

## 1. 결론 먼저 — PolyStrata·Ghidra를 이 저장소에?

| 대상 | 로컬에 받아서 분석 | Git에 커밋 |
|------|-------------------|------------|
| **PolyStrata** 소스·바이너리 | **예** — `.local/polystrata/` 등 | **아니오** |
| **Ghidra** 툴 본체 | **예** — OS에 설치 또는 도구 전용 경로 | **아니오** |
| **Ghidra 프로젝트** (`.gpr` 등) | **예** — `.local/ghidra-projects/` | **아니오** |
| **REA 분석 메모** (Struct, bitpack) | — | **예** — `docs/rea/`만 |
| **테스트용 MoE 가중치** | `.local/models/` | **아니오** |

**권장:** 엔진·도구는 **받아서 분석하되, 저장소 밖(또는 `.local/` gitignore)에 둔다.**  
StrataShift git에는 **규격 메모·변환 코드만** 올린다.

이유:

- 라이선스·용량·바이너리 배포 이슈
- PolyStrata/Ghidra는 upstream이 따로 있음 (서브모듈로 고정해도 “분석용 스냅샷”이 자주 바뀜)
- private여도 실수로 가중치·실행 파일을 올리면 복구가 어렵다

---

## 2. 권장 로컬 레이아웃

저장소 루트에 **gitignore된** `.local/`만 사용한다. (커밋 대상 골조가 아님)

```text
StrataShift/                    # git 추적
├── docs/rea/                   # ✅ 커밋 — 분석 결과 메모
├── src/stratashift/            # ✅ 커밋 — 변환기 코드
├── .gitignore
└── .local/                     # ❌ 커밋 금지
    ├── polystrata/             # clone 또는 HF zip 풀기
    │   ├── src/                # 있으면 소스 열람
    │   └── bin/                # 런타임/로더 실행 파일
    ├── ghidra-projects/        # Ghidra 프로젝트 저장
    ├── models/                 # 테스트 MoE (.safetensors 등)
    └── out/                    # 변환 출력물
```

Ghidra 설치본은 예:

- Windows: `C:\Tools\ghidra_*` 또는 공식 설치 경로  
- StrataShift 안에 Ghidra 전체를 넣지 말 것

핀(버전·SHA)은 항상 [REFERENCES.md](REFERENCES.md)에 기록한다.

---

## 3. 커밋해도 되는 것 / 안 되는 것

### ✅ 올려도 됨

- `docs/`, `AGENTS.md`, `.cursor/rules/`
- `src/stratashift/`, `scripts/`(PoC 코드), `tests/`
- `docs/rea/*.md` (텍스트 규격·오프셋 표)
- `.gitignore`, `pyproject.toml`, `.env.example`

### ❌ 올리면 안 됨

| 종류 | 예시 |
|------|------|
| 엔진 트리 | PolyStrata clone, 사전 빌드 exe/dll, 배포 zip |
| REA 도구·프로젝트 | Ghidra 설치본, `*.gpr`, `*.rep/` |
| 모델·가중치 | `*.safetensors`, `*.gguf`, `*.bin`, `models/` |
| 변환 산출물 | `outputs/`, `.local/out/` |
| 비밀 | `.env`, 키, 토큰 |
| 가상환경·캐시 | `.venv/`, `__pycache__/`, `.pytest_cache/` |
| OS/IDE 잡파일 | `.DS_Store`, `.idea/` (팀 공유 설정만 예외적으로 허용 시 명시) |

---

## 4. 분석 워크플로 (요약)

1. Ghidra를 **시스템/도구 경로**에 설치  
2. PolyStrata를 `.local/polystrata/`에 받아 **핀 기록** (`REFERENCES`)  
3. Ghidra에서 로더 임포트 → 프로젝트는 `.local/ghidra-projects/`  
4. 발견한 Struct/bitpack만 `docs/rea/`에 문서화 → **이것만 commit**  
5. 변환기 코드는 `src/stratashift/`에 구현  

상세: [REA_GHIDRA.md](REA_GHIDRA.md)

---

## 5. 실수 방지

- `git add .` 전에 `git status`로 `.local/`, `*.safetensors`, zip이 안 잡히는지 확인  
- 이미 추적 중이면: `git rm -r --cached <path>` 후 커밋  
- private → public 전환 전에 히스토리에 바이너리/가중치가 없는지 검사

관련: [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md), [REFERENCES.md](REFERENCES.md)
