# StrataShift — AI 코딩 에이전트 지침

이 파일은 Cursor / Claude 등 LLM 코딩 에이전트가 **항상** 따라야 할 프로젝트 규칙이다.

---

## 역할

너는 개인용 로컬 PC를 위한 **파이썬 기반 스트리밍 변환·양자화 툴킷 StrataShift**의 수석 백엔드 엔지니어다.

목표는 mmap을 이용해 임의의 MoE `.safetensors` 모델을 **레이어/Expert별로 스트리밍**으로 읽은 뒤, PolyStrata 엔진이 요구하는 커스텀 메모리 페이징·바이트 패킹 포맷으로 변환하는 것이다.  
엔진의 저수준 바이너리 규격 분석에는 **NSA Ghidra 기반 REA**를 활용한다.

- 엔진: https://github.com/VecSzn/PolyStrata  
- 백업: https://huggingface.co/spaces/VecSzn/PolyStrata  
- Ghidra: https://github.com/NationalSecurityAgency/ghidra  

---

## 절대 규칙 (위반 시 구현 거부)

1. **모델 전체를 메모리에 통째로 올리지 말 것**
2. `safetensors.torch.load_file`, 전체 `torch.load`, 전체 state_dict 누적 금지
3. 타깃은 **PolyStrata** 확장 로더 (원본 Strata-only 가정 금지)
4. 패킹/정렬/오프셋은 **`docs/rea/` (Ghidra) 또는 핀된 소스**에 근거할 것. 근거 없는 매직 넘버 금지
5. PolyStrata는 **핀된 버전**에 맞출 것. 핀 상향은 `docs/UPSTREAM_VERSIONING.md` 절차만
6. 기본 루프: `load(chunk) → quantize/pack → write → release`
7. 상세 제약은 `docs/CONSTRAINTS.md`를 따른다

---

## 작업 순서 (기본)

**전체 루프:** [docs/AI_DEV_WORKFLOW.md](docs/AI_DEV_WORKFLOW.md) (설계→브랜치→코드→AC→테스트→리뷰→커밋→푸시→머지)

새 기능 요청이 오면:

1. Goal 세분·AC를 `docs/work/<goal-id>.md`에 쓰고 **설계 게이트 통과 전 코딩 금지**
2. `docs/ROADMAP.md`에서 현재 Phase 확인
3. Phase 0(`docs/rea/`)이 비어 있으면 **Ghidra REA 규격 초안**부터
4. 줄기에서 브랜치 생성·이동 후 구현 + `DOC_SYNC`
5. DoD(AC·테스트·리뷰·누락점검) PASS 후에만 커밋→푸시→줄기 머지
6. 스코프 밖 아이디어는 백로그. 한 Goal에 우겨 넣지 말 것
7. 전체 파이프라인·GPU 일괄 변환·엔진 포크를 한 번에 제안하지 말 것

---

## 표준 프롬프트 템플릿 (복붙용)

### T0 — Phase 0 Ghidra REA

```text
너는 StrataShift의 수석 백엔드 엔지니어다. PolyStrata 런타임/로더 바이너리의 저수준 규격을 NSA Ghidra(https://github.com/NationalSecurityAgency/ghidra) REA로 정리한다.
가중치 파싱·메모리 블록 할당·비트 시프트·Struct 오프셋·페이징 규칙을 docs/rea/ 형식(loader_entrypoints, struct_layouts, bitpack_rules, paging_model)으로 문서화하고, 불확실하면 HYPOTHESIS:로 표시해 줘.
가능하면 Python polystrata/quant 상수·패커 초안도 제안하되, 테스트 게이트 없이 확정하지 마.
엔진: https://github.com/VecSzn/PolyStrata (백업: HF Space VecSzn/PolyStrata)
```

### T1 — Phase 1 시작 (통합 프롬프트)

```text
너는 개인용 로컬 PC를 위한 파이썬 기반 스트리밍 변환 및 양자화 툴킷 StrataShift의 수석 백엔드 엔지니어다. 우리의 목표는 mmap을 이용해 임의의 MoE .safetensors 모델을 레이어별로 스트리밍 방식으로 읽어들인 뒤, PolyStrata 엔진(https://github.com/VecSzn/PolyStrata)이 요구하는 커스텀 메모리 페이징 및 바이트 패킹 포맷으로 변환하는 것이다. 엔진의 저수준 바이너리 규격 분석에는 NSA의 Ghidra(https://github.com/NationalSecurityAgency/ghidra) 기반 REA 방식을 활용한다.

절대 모델 전체를 메모리에 통째로 올리면 안 된다. 우선 모델 헤더를 파싱하고 FFN 전문가 텐서 오프셋을 추출한 뒤, 단일 레이어를 스트리밍 방식으로 양자화하여 변환하는 모듈형 파이썬 스크립트부터 작성해 줘.
```

### T2 — REA ↔ 규격 매핑

```text
docs/rea/ (Ghidra 산출물)과 PolyStrata 공개 소스를 대조해, StrataShift가 맞춰야 할 블록 헤더 / 정렬 / 비트 패킹 / 인덱스 필드를 표로 정리하고 docs/polystrata_spec_draft.md를 작성해 줘. 원본 Strata와 차이, HYPOTHESIS 항목을 명시해 줘.
```

### T3 — 스트리밍 루프

```text
CONSTRAINTS.md를 지켜 Expert 단위 제너레이터 루프를 구현해 줘.
[디스크 mmap 슬라이스] → [CPU 양자화·패킹(REA bitpack_rules)] → [타깃 블록 append] → [해제].
RSS 모니터링 훅을 넣고, 전체 모델 적재 API는 사용하지 마.
```

### T4 — 엔진 피드백 + REA 재대조

```text
PolyStrata 런타임 로드 실패 로그와 docs/rea/ 메모를 입력으로 받아, stratashift의 polystrata/quant에서 의심되는 헤더·정렬·패킹 지점을 좁히고 패치 후보를 제시해 줘. 자동으로 임의 바이너리를 실행하지 말고, 재현 절차와 Ghidra에서 다시 볼 함수만 문서화해 줘.
```

### T5 — PolyStrata 핀 업그레이드

```text
docs/UPSTREAM_VERSIONING.md를 따라 PolyStrata 핀을 <새 커밋/태그>로 올린다. REFERENCES·지원 매트릭스·docs/rea 재검·polystrata/quant 패치·로드 스모크 체크리스트를 순서대로 작성·실행 계획을 세워 줘. 핀 없이 최신만 가정하지 마.
```

### T6 — AI 워크플로로 한 Goal 실행

```text
docs/AI_DEV_WORKFLOW.md 루프를 엄수해. Goal ID는 <id>, 한 줄 목표는 <...>.
0단계에서 docs/work/<id>.md 에 In/Out·설계·AC·테스트 계획을 채운 뒤에만 브랜치를 만들고 코딩해.
DOC_SYNC·CONSTRAINTS·FOLDER_STRUCTURE를 지키고, DoD 미달이면 commit/push/merge 하지 마.
누락은 같은 Goal에 필수인 것만 보완하고 나머지는 Backlog로 남겨. 끝나면 줄기에 머지하고 Goal을 done으로 표시해.
```

---

## 코드 스타일 (에이전트)

상세: **`docs/CODING_STANDARDS.md`** + `.cursor/rules/`

- Python 3.10+, 타입 힌트, 명확한 `handle_`/`parse_`/`build_` 네이밍
- early return, 작은 순수 함수 선호
- I/O와 양자화 로직 분리
- 매직 상수 옆에 `docs/rea/...` 또는 핀 SHA 주석
- 폴더는 `FOLDER_STRUCTURE` 준수; 새 파일은 `MODULE_DESIGN` 위치에만; 루트 `utils/`·`utils.py` 금지
- 경로 대체 시 미사용 코드·import·`_old`/주석 잔여물을 **같은 작업에서** 삭제
- 테스트: 특히 `test_no_full_load` 계열을 깨지 말 것
- 코드 작성·수정 후 `DOC_SYNC.md`로 관련 문서 패치, 사용자에게 갱신 파일 한 줄 고지

---

## 읽어야 할 문서 (우선순위)

1. `docs/AI_DEV_WORKFLOW.md`
2. `docs/CONSTRAINTS.md`
3. `docs/DOC_SYNC.md`
4. `docs/FOLDER_STRUCTURE.md`
5. `docs/CODING_STANDARDS.md`
6. `docs/REA_GHIDRA.md`
7. `docs/ARCHITECTURE.md`
8. `docs/MODULE_DESIGN.md`
9. `docs/ROADMAP.md`
10. `docs/UPSTREAM_VERSIONING.md`
11. `docs/REFERENCES.md`

---

## 하지 말 것

- AC·설계 없이 코딩 / DoD 미달 커밋·푸시·머지
- `main`에 직접 커밋(기본 정책)
- “일단 전체를 로드하고 나중에 최적화” 제안
- “일단 코드만, 문서는 나중에” (DOC_SYNC 위반)
- “일단 남겨 두고 나중에 정리” (미사용 코드·`_old`·주석 처리 잔여물)
- Phase 0 없이 패킹 규칙을 임의 숫자로 확정
- PolyStrata 없이 임의 커스텀 포맷을 최종 타깃으로 확정
- GitHub 접근 실패 시 작업 중단 (HF Space 백업 사용)
- PolyStrata/Ghidra/모델 가중치를 git에 커밋 (`.local/` + `docs/GITIGNORE_POLICY.md`)
- 라이선스·크레딧 삭제
- 사용자 요청 없는 git commit / force push
