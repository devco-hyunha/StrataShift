# 코드 ↔ 문서 동기화 규칙

코드가 작성·변경되면 **같은 작업 안에서** 관련 문서를 갱신한다.  
“나중에 문서화”는 허용하지 않는다. (에이전트·PR 공통)

Cursor: `.cursor/rules/doc-sync.mdc` (alwaysApply)  
상세 코딩 규칙: [CODING_STANDARDS.md](CODING_STANDARDS.md)

---

## 1. 원칙

| 원칙 | 내용 |
|------|------|
| 동일 작업 | 코드 diff와 문서 diff를 **한 작업/커밋 단위**로 맞춤 |
| 정본 | 동작의 정본은 코드, **계약·배치·핀·Phase 완료**의 정본은 docs |
| Drift 금지 | 코드만 있고 문서가 옛 설명이면 미완료 |
| 최소 갱신 | 관련 문서만 고친다. 무관한 장문 재작성 금지 |

---

## 2. 변경 유형 → 필수 문서 매핑

| 코드/산출물 변경 | 반드시 갱신 |
|------------------|------------|
| 새 패키지·폴더·모듈 경로 | [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md), [MODULE_DESIGN.md](MODULE_DESIGN.md) |
| 공개 함수·CLI·API 시그니처 | [MODULE_DESIGN.md](MODULE_DESIGN.md), 필요 시 README 사용법 |
| 스트리밍/메모리 동작·제약 | [CONSTRAINTS.md](CONSTRAINTS.md), [ARCHITECTURE.md](ARCHITECTURE.md) |
| 양자화·헤더·정렬·인덱스 상수 | [docs/rea/](rea/README.md) 해당 파일, 코드 주석의 REA 링크 |
| PolyStrata 핀·지원 버전 | [REFERENCES.md](REFERENCES.md), [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) |
| Phase 완료·인수 항목 | [ACCEPTANCE.md](ACCEPTANCE.md), [ROADMAP.md](ROADMAP.md) 체크 |
| `.gitignore` / 커밋 제외 정책 | [GITIGNORE_POLICY.md](GITIGNORE_POLICY.md) |
| 새 코딩 관례·네이밍 | [CODING_STANDARDS.md](CODING_STANDARDS.md), `.cursor/rules/` |
| 사용자 진입점·상태 | [README.md](../README.md), 필요 시 [INDEX.md](INDEX.md) |
| 에이전트 절차·프롬프트 | [AGENTS.md](../AGENTS.md) |
| Goal 설계·AC·결과 | [docs/work/](work/README.md) + [AI_DEV_WORKFLOW.md](AI_DEV_WORKFLOW.md) |

구현 착수 후 추가될 것:

| 변경 | 갱신 |
|------|------|
| `pyproject.toml` 의존성 | README / DEV_SETUP(추후) |
| CHANGELOG | 릴리스 시 (도입 후) |

---

## 3. 작업 종료 체크리스트

코드 작업을 끝낼 때:

- [ ] 위 매핑에서 해당 행의 문서를 열었는가?
- [ ] 문서의 경로·함수명·CLI·핀이 **실제 코드와 일치**하는가?
- [ ] `HYPOTHESIS:` / `_TBD_` 중 코드로 확정된 것은 승격·삭제했는가?
- [ ] ACCEPTANCE/ROADMAP에 끝난 항목이 있으면 체크했는가?
- [ ] REA 상수 변경 시 `docs/rea/` ↔ `polystrata`/`quant` 양방향 링크가 있는가?

하나라도 No면 작업을 닫지 않는다.

---

## 4. 에이전트 행동

1. 코드 수정 전에 영향 문서 목록을 짧게 파악한다  
2. 코드 수정 후 **같은 턴에서** 문서를 패치한다  
3. 사용자에게 “문서도 갱신함”을 한 줄로 알린다 (어떤 파일)  
4. 문서만 필요하고 코드가 없으면 문서만; 코드만 있고 문서가 필요하면 **코드 머지 전에 문서 추가**

금지:

- “일단 코드만” 후 문서 TODO만 남기기  
- 존재하지 않는 모듈/CLI를 README에 단정 기록  
- 핀·경로를 문서와 코드에 서로 다르게 표기  

---

## 5. PR / 커밋 메시지 힌트

- 본문 또는 불릿에 `docs: …` 변경을 명시  
- 예: `feat(ingest): parse safetensors header` + `docs: MODULE_DESIGN + ACCEPTANCE Phase 1`

---

## 6. 한 줄

> 코드가 바뀌면 매핑표의 문서도 같은 작업에서 바꾼다. 문서 drift = 미완료.

관련: [CODING_STANDARDS.md](CODING_STANDARDS.md), [AGENTS.md](../AGENTS.md), [MODULE_DESIGN.md](MODULE_DESIGN.md)
