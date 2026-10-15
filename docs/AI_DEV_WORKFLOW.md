# AI 자동화 개발 워크플로

에이전트·인간이 **같은 루프**로 기능을 만든다.  
한 줄 요약: **목표 세분 → 설계 → 브랜치 → 코드(+문서) → 목표검증 → 테스트 → 리뷰 → 누락점검 → 커밋 → 푸시 → 줄기 머지**.

관련: [DOC_SYNC.md](DOC_SYNC.md), [CODING_STANDARDS.md](CODING_STANDARDS.md), [ACCEPTANCE.md](ACCEPTANCE.md), [ROADMAP.md](ROADMAP.md), [AGENTS.md](../AGENTS.md)

Cursor: `.cursor/rules/ai-dev-workflow.mdc` (alwaysApply)

---

## 1. 브랜치 모델

| 브랜치 | 역할 |
|--------|------|
| `main` | 안정 줄기. 직접 커밋 금지(기본). PR/머지로만 반영 |
| `develop` (선택) | 통합 줄기. 팀이 커지면 도입. 초기에는 `main`만 써도 됨 |
| `feat/<goal-id>-<short>` | 기능 |
| `fix/<goal-id>-<short>` | 버그 |
| `docs/<goal-id>-<short>` | 문서만 |
| `chore/<goal-id>-<short>` | 툴·ignore·CI |
| `rea/<goal-id>-<short>` | Phase 0 REA 산출 |

**goal-id:** `p0-01`, `p1-03`처럼 Phase+순번, 또는 이슈 번호.

예: `feat/p1-01-safetensors-header`

---

## 2. 필수 루프 (한 목표 = 한 브랜치)

```text
[0] 목표 세분화·설계 확정 (코드 금지)
[1] 브랜치 생성·체크아웃
[2] 코드 작성 + DOC_SYNC 문서 갱신
[3] 작업 목표 부합 검증 (AC 체크)
[4] 테스트 실행
[5] 코드 리뷰 (에이전트 셀프 + 가능하면 인간/2차 에이전트)
[6] 누락·추가 기능 검토 (스코프 안 / 백로그 분리)
[7] 전부 PASS → commit
[8] push
[9] 줄기(main|develop)로 이동 → merge (PR 권장)
[10] 로컬 정리·다음 목표로
```

**어느 단계든 FAIL → 커밋/머지 금지.** 수정 후 3→6 재실행.

---

## 3. 단계별 상세

### 0) 목표 세분화·설계

산출물(채팅 또는 `docs/work/<goal-id>.md`):

| 필드 | 내용 |
|------|------|
| Goal ID | `p1-01` |
| 한 줄 목표 | … |
| Phase | ROADMAP 정렬 |
| 범위 (In) | 이번 브랜치에서 할 일 |
| 비범위 (Out) | 명시적 제외 |
| 설계 요약 | 모듈·파일·데이터 흐름 |
| AC (Acceptance) | 체크 가능한 완료 조건 3~7개 |
| 테스트 계획 | 어떤 명령을 돌릴지 |
| 문서 영향 | DOC_SYNC 매핑 |
| 리스크 | OOM, REA 가설 등 |

설계가 승인(인간 한 줄 OK 또는 AC 자체 명확)되기 **전에 코드 작성 금지**.

### 1) 브랜치

```bash
git fetch
git checkout main   # 또는 develop
git pull
git checkout -b feat/<goal-id>-<short>
```

### 2) 구현

- [CONSTRAINTS](CONSTRAINTS.md) / [FOLDER_STRUCTURE](FOLDER_STRUCTURE.md) / [CODING_STANDARDS](CODING_STANDARDS.md) 준수
- [DOC_SYNC](DOC_SYNC.md)로 문서 동시 갱신
- 스코프 밖 리팩터·새 기능 추가 금지 (발견 시 §6 백로그)

### 3) 목표 부합 검증

- AC 항목을 하나씩 ✅/❌
- “동작은 하는데 AC에 없음” → AC 개정 또는 Out으로 이동 후 별도 Goal

### 4) 테스트

| 수준 | 언제 |
|------|------|
| 단위 | 파서·패커·순수 함수 |
| 가드 | `test_no_full_load` 등 제약 테스트 |
| 스모크 | CLI inspect/convert (Phase 해당 시) |
| 엔진 로드 | Phase 3·핀 업그레이드 시 |

테스트 실패 시 커밋 금지. 테스트가 아직 없으면 **이번 Goal AC에 “테스트 추가”를 포함**했는 mid 확인.

### 5) 코드 리뷰

체크:

- 전체 모델 적재 경로 없음
- 폴더·네이밍·의존 방향
- REA/핀 출처
- 시크릿·`.local`·가중치 미포함 (`git status`)
- DOC_SYNC 누락 없음

### 6) 누락·추가 기능 검토

| 발견 | 처리 |
|------|------|
| 이번 AC에 필요한 누락 | 같은 브랜치에서 보완 → 3~5 재실행 |
| 있으면 좋음 / 다음 Phase | 백로그·ROADMAP에만 기록, **이번 머지 차단 사유 아님** |
| 스코프 팽창 유혹 | 거부. 새 Goal ID 발급 |

### 7) 커밋

- 사용자/자동화가 **커밋을 지시한 뒤**에만 수행 (저장소 정책)
- Conventional commits 권장: `feat:`, `fix:`, `docs:`, `test:`, `chore:`
- 본문에 Goal ID, AC 요약, `docs:` 변경 명시

### 8) 푸시

```bash
git push -u origin HEAD
```

force push to `main` 금지.

### 9) 머지

- **PR 권장** (`gh pr create`) — CI·리뷰 게이트
- 초기 단독 자동화 시 fast-forward/merge commit 허용하되, 머지 전 §3~6 PASS 기록
- 충돌 시 줄기 재base/merge 후 테스트 재실행

### 10) 정리

```bash
git checkout main
git pull
git branch -d feat/<goal-id>-<short>   # 원격 삭제 정책에 따름
```

Goal 상태를 `done`으로 표시. ACCEPTANCE/ROADMAP 체크.

---

## 4. Definition of Done (DoD)

머지 가능 조건 — **전부 만족**:

- [ ] Goal AC 전부 PASS
- [ ] 합의된 테스트 PASS
- [ ] 코드 리뷰 이슈 해소 또는 명시적 defer(백로그 ID)
- [ ] DOC_SYNC 완료
- [ ] `.gitignore` 위반 파일 없음
- [ ] CONSTRAINTS 위반 없음
- [ ] 커밋 메시지에 Goal ID
- [ ] (해당 시) UPSTREAM 핀과 불일치 없음

---

## 5. AI 자동화에 추가로 넣은 가드레일

아래는 요청하신 루프 외에 **자동화 시 거의 필수**:

| 항목 | 이유 |
|------|------|
| **스코프 잠금** | 에이전트가 목표 밖으로 코드 확산하는 것 방지 |
| **설계 게이트** | 코드 전 AC 확정 |
| **DoD** | “느낌상 완료” 방지 |
| **백로그 분리** | 누락 검토 ≠ 무한 기능 추가 |
| **DOC_SYNC** | 코드·문서 drift 방지 |
| **비밀·대용량 가드** | `.local`/모델/키 푸시 방지 |
| **줄기 직접 커밋 금지** | main 오염 방지 |
| **인간 승인 포인트** | 첫 `main` 머지, public 전환, 핀 메이저 업, 강제 푸시 |
| **실패 시 중단** | 테스트/리뷰 FAIL이면 push·merge 안 함 |
| **작업 기록** | `docs/work/<goal-id>.md` 또는 PR 본문에 AC·결과 |
| **동시 에이전트** | 브랜치/worktree 분리, 같은 파일 동시 편집 금지 |
| **롤백** | 머지 후 문제 시 revert 커밋 (history rewrite 금지) |

선택(여유 시):

- CI (pytest + lint) on PR
- PR 템플릿 (`AC`, `테스트`, `DOC_SYNC`)
- `develop` 통합 브랜치
- Bugbot / 2nd-pass review 에이전트

---

## 6. Goal 기록 템플릿

`docs/work/<goal-id>.md`:

```markdown
# Goal <id>: <title>

- Phase:
- Branch: feat/<id>-...
- Status: planned | in_progress | review | done | blocked

## In / Out
## Design
## AC
- [ ] ...
## Test plan
## Doc sync
## Review notes
## Backlog spawned
## Result
```

`docs/work/`는 추적해도 되고, 민감하면 gitignore — 기본은 **추적(팀·자동화 재현용)**.

---

## 7. 에이전트용 한 줄

> 설계·AC 없이 코딩하지 않는다. 한 Goal·한 브랜치. DoD 전부 PASS 후에만 커밋→푸시→줄기 머지.  
> 새 아이디어는 백로그. 문서·테스트·리뷰를 건너뛰지 않는다.

---

## 8. 프롬프트 조각 (자동화 오케스트레이션)

```text
AI_DEV_WORKFLOW.md 루프를 엄수해. Goal ID는 <id>.
0단계에서 docs/work/<id>.md 를 채우고 AC를 확정한 뒤에만 브랜치를 만들고 코딩해.
DOC_SYNC·CONSTRAINTS·FOLDER_STRUCTURE를 지키고,
DoD 미달이면 commit/push/merge 하지 마.
누락은 같은 Goal에 필수인 것만 보완하고, 나머지는 Backlog로 남겨.
```
