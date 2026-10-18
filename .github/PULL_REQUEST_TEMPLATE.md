## Goal

- Goal ID: `pX-YY` — `docs/work/pX-YY.md` (design gate before code, per AI_DEV_WORKFLOW)
- 한 줄 목표:

## DoD (머지 전 필수)

- [ ] AC 전수 PASS — `docs/work/pX-YY.md` 체크박스 + 실측 근거 채움 (날짜, 산출 수치)
- [ ] 전수 pytest GREEN — passed/failed 카운트와 elapsed 실측 기록:
      `N passed, 0 failed (Xs)`
- [ ] 엔진 체인 acceptance (pack/GGUF 출력 변경 시) — `polystrata-engine-acceptance` 게이트
      실측 (guard/dequant/pack verify, byte-identical 비교)
- [ ] no-full-load 절대 규칙 — 전체 모델 로드 경로 없음 (CI C1 게이트 + PR reviewer 확인)
- [ ] RSS/heap gate — 스트리밍 산출물의 heap peak 실측 (< 2 GiB contract)

## DOC_SYNC (갱신 파일 한 줄씩)

- [ ] `docs/ROADMAP.md` — Goal 상태 라인
- [ ] `docs/RELEASE_NOTES.md` — 항목
- [ ] `docs/polystrata_spec_draft.md` / `docs/rea/*` — 규격 영향 시 (근거 접두어 SOURCE:/REA:/HYPOTHESIS:)
- [ ] `docs/MODULE_DESIGN.md` — 신규/삭제 파일 트리
- [ ] `docs/work/pX-YY.md` — Status done + Result

## Scope

- [ ] 이 PR는 한 Goal의 범위만 담는다 (ROADMAP planned scope 밖 발명 없음)
- [ ] 미사용 코드·`_old`·주석 잔여물 즉시 삭제 (같은 작업 내)
