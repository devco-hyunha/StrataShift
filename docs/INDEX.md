# StrataShift 문서 인덱스

기획·개발에 필요한 문서의 읽는 순서와 역할.  
**문서 버전:** 0.3.0 (AI 개발 워크플로 반영)

## 권장 읽기 순서

| 순서 | 문서 | 한 줄 설명 |
|------|------|------------|
| 1 | [PRODUCT_SPEC.md](PRODUCT_SPEC.md) | 문제·목표·입출력·기능/비기능 요구 |
| 2 | [CONSTRAINTS.md](CONSTRAINTS.md) | 전체 적재 금지·REA 안전 규칙 |
| 3 | [AI_DEV_WORKFLOW.md](AI_DEV_WORKFLOW.md) | 목표→브랜치→테스트→리뷰→머지 |
| 4 | [DOC_SYNC.md](DOC_SYNC.md) | 코드 변경 시 문서 동시 갱신 |
| 5 | [REA_GHIDRA.md](REA_GHIDRA.md) | Phase 0 Ghidra 임포트·디컴파일·산출물 |
| 6 | [ARCHITECTURE.md](ARCHITECTURE.md) | Source → REA → Stream → Target |
| 7 | [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) | 고정 폴더 골조·배치·금지 경로 |
| 8 | [MODULE_DESIGN.md](MODULE_DESIGN.md) | 디렉터리·API·테스트 경계 |
| 9 | [CODING_STANDARDS.md](CODING_STANDARDS.md) | 파일·함수 생성·네이밍·스타일 |
| 10 | [ROADMAP.md](ROADMAP.md) | Phase 0~3 작업과 마일스톤 |
| 11 | [ACCEPTANCE.md](ACCEPTANCE.md) | Phase 완료 판정 체크리스트 |
| 12 | [REFERENCES.md](REFERENCES.md) | PolyStrata / Ghidra / HF 핀 |
| 13 | [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) | PolyStrata 따라가기·지원 매트릭스 |
| 14 | [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md) | 커밋 금지·`.local/` 분석 배치 |

## 저장소 루트

| 문서 | 역할 |
|------|------|
| [../README.md](../README.md) | 프로젝트 진입점 |
| [../AGENTS.md](../AGENTS.md) | AI 코딩 에이전트 지침·프롬프트 |
| [../.cursor/rules/](../.cursor/rules/) | Cursor 룰 (workflow / folder / core / doc-sync / …) |

## Phase·작업별 주 문서

- **모든 AI 작업:** [AI_DEV_WORKFLOW.md](AI_DEV_WORKFLOW.md) + [DOC_SYNC.md](DOC_SYNC.md) + `docs/work/<goal-id>.md`
- **Phase 0:** [REA_GHIDRA.md](REA_GHIDRA.md), `docs/rea/`, REFERENCES 핀
- **Phase 1~3:** ROADMAP / ACCEPTANCE
- **엔진 버전 업:** [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md)
