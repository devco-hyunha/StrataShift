# StrataShift 코딩·파일/함수 생성 규칙

에이전트·기여자가 **새 파일·함수·모듈을 만들 때** 따르는 규칙이다.  
제품 제약은 [CONSTRAINTS.md](CONSTRAINTS.md).  
**폴더 골조**는 [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md).  
모듈 API·트리는 [MODULE_DESIGN.md](MODULE_DESIGN.md).

Cursor 상시 적용: `.cursor/rules/` (`folder-structure.mdc`, `stratashift-core.mdc`, `doc-sync.mdc`)

**코드가 바뀌면 문서도 같은 작업에서 갱신:** [DOC_SYNC.md](DOC_SYNC.md)  
**AI 자동화 루프(브랜치·테스트·머지):** [AI_DEV_WORKFLOW.md](AI_DEV_WORKFLOW.md)

---

## 0. 폴더 구조 (필수)

코드를 쓰기 전에 배치할 폴더를 고른다. 요약:

| 넣을 것 | 폴더 |
|---------|------|
| 라이브러리 | `src/stratashift/<area>/` |
| PoC | `scripts/` |
| 테스트 | `tests/` |
| 기획·규격 | `docs/` |
| REA | `docs/rea/` |

- 허용 최상위: `.cursor/`, `docs/`, `src/`, `scripts/`, `tests/` (+ 루트 설정 파일)
- **금지:** 루트 `utils/`, `lib/`, `core/`, `data/`, `tmp/`, `helpers/`
- 새 최상위·새 `<area>` 패키지 → **문서 먼저** (`FOLDER_STRUCTURE` + `MODULE_DESIGN`) 후 생성
- 영역 책임·승격 규칙 전문: [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)

---

## 1. 새 파일을 만들기 전 체크

다음을 **모두** 확인한 뒤에만 파일을 추가한다.

| # | 질문 | No이면 |
|---|------|--------|
| 0 | `FOLDER_STRUCTURE.md`상 올바른 폴더인가? | 폴더를 바꾸거나 문서를 먼저 수정 |
| 1 | `MODULE_DESIGN.md` 트리에 넣을 위치가 있는가? | 문서에 위치를 먼저 추가하거나 기존 모듈에 합친다 |
| 2 | 동일 책임의 파일이 이미 있는가? | **확장**하고 새 파일을 만들지 않는다 |
| 3 | 한 파일·한 책임(SRP)인가? | 책임을 쪼개거나 합친다 |
| 4 | Phase에 맞는가? (`ROADMAP`) | 상위 Phase 산출물부터 |
| 5 | 전체 모델 적재 경로를 유인하지 않는가? | 설계 거부 |

### 금지 (파일)

- 저장소 루트에 `utils.py`, `helpers.py`, `temp_*.py` 방치
- `src/stratashift` 밖에 비즈니스 로직 패키지 신설
- “나중에 정리”용 `_old`, `copy2` 파일

### 미사용 코드·잔여물 (같은 작업에서 정리)

경로를 바꾸거나 기능을 대체했으면 **같은 변경**에서 죽은 코드를 제거한다. 쌓아 두지 않는다.

| MUST | 내용 |
|------|------|
| MUST | 미사용 import / 함수 / 클래스 / 상수 / 스크립트를 같은 PR·작업에서 삭제 |
| MUST NOT | 주석 처리한 옛 구현, `# old:`, `_old` / `copy2`, “혹시 몰라” 스텁을 트리에 남김 |
| MUST NOT | `docs/work/` 히스토리를 코드 정리 명목으로 삭제 (별도 정책) |
| SHOULD | 죽은 코드는 보관하지 말고 git 히스토리로 되돌릴 수 있게 삭제 |
| SHOULD | 대체 완료된 `HYPOTHESIS:` / `_TBD_`는 [DOC_SYNC.md](DOC_SYNC.md)에 따라 승격·삭제 |

---

## 2. 파일 네이밍

| 대상 | 규칙 | 예 |
|------|------|-----|
| 패키지/모듈 | `snake_case.py` | `safetensors_header.py` |
| 디렉터리 | `snake_case` (소문자) | `polystrata/` |
| 테스트 | `test_<module_or_feature>.py` | `test_no_full_load.py` |
| 상수·규격 문서 | `snake_case.md` | `bitpack_rules.md` |
| Cursor 룰 | `kebab-case.mdc` | `python-modules.mdc` |

한 모듈 = 한 주요 개념.  
예: `pack.py`에 헤더 파싱을 넣지 않는다 → `ingest/` / `polystrata/headers.py`.

---

## 3. 함수·심볼 네이밍

| 종류 | 규칙 | 예 |
|------|------|-----|
| 순수 변환/계산 | `snake_case`, 동사로 시작 | `parse_header`, `build_expert_map` |
| 부수효과(I/O) | 동사 명확히 | `write_weight_block`, `append_index_entry` |
| 이벤트/핸들러 | `handle_` 접두 | `handle_convert_command` |
| 불리언 | `is_` / `has_` / `should_` | `is_expert_tensor`, `has_sharded_files` |
| 상수 | `UPPER_SNAKE` | `HEADER_MAGIC`, `DEFAULT_ALIGNMENT` |
| 비공개 | `_` 접두 | `_read_u64`, `_validate_offsets` |

### 함수 설계

- **`const` 스타일 함수**(모듈 레벨 `def` / 할당) 선호, 불필요 클래스 지양
- **early return**으로 중첩 감소
- 한 함수 = 한 일. 대략 **40줄 초과 시** 분리 검토
- I/O와 순수 계산 분리: `quant.pack`은 파일 경로를 열지 않음
- 공개 API는 타입 힌트 필수 (Python 3.11+: `list[str]`, `X | None`)
- 텍스트 파일 읽기·쓰기(`read_text`/`write_text`/텍스트 모드 `open`)는 항상 `encoding="utf-8"` — 한국어 Windows 기본 cp949 대비, `tests/test_text_encoding.py`가 막는다 (p6-11)
- `Any` 남용 금지. 불가피하면 이유 주석

```python
# ❌ BAD — 전체 적재 + 혼합 책임
def convert(path: str) -> None:
    tensors = load_file(path)  # 금지
    for name, t in tensors.items():
        open(out, "ab").write(pack(t))

# ✅ GOOD — 청크 + 책임 분리
def convert_expert_stream(path: str, experts: list[ExpertRef], spec: PolyStrataSpec) -> None:
    for expert in experts:
        view = mmap_tensor_slice(path, expert.offsets)
        if view is None:
            continue
        packed = quantize_and_pack(view, spec)
        write_weight_block(out_fd, packed, build_block_header(expert, spec))
        del view, packed
```

---

## 4. 새 함수를 추가할 때

1. **기존 함수로 불가능한지** 먼저 확인
2. 공개할지(`__all__` / 패키지 export) 비공개인지 결정
3. CONSTRAINTS 위반 여부 확인 (전체 load, 누적 버퍼 등)
4. REA/매직 상수는 `docs/rea/` 또는 핀 SHA 주석 필수

```python
# REA: docs/rea/bitpack_rules.md §shift_q4  (binary sha256: ...)
WEIGHT_Q4_SHIFT = 4
```

5. Phase 1+ 이면 **테스트**를 같은 PR/작업에 추가  
   - 파서 → `tests/test_header_parse.py`  
   - 패킹 → `tests/test_pack_matches_rea.py`  
   - 적재 금지 → `tests/test_no_full_load.py` 가드 유지
6. **문서 동기화** — [DOC_SYNC.md](DOC_SYNC.md) 매핑표에 따라 관련 docs를 같은 작업에서 수정

---

## 5. import·의존성

```
cli → 하위 모듈
stream → ingest (오프셋만)
quant → polystrata.spec 만 (I/O 금지)
ingest ↛ quant / stream
```

- 순환 import 금지
- 무거운 선택 의존은 함수 내부 lazy import (사유 주석)
- 새 서드파티 추가는 `pyproject.toml` + 이 문서/README에 한 줄 근거

---

## 6. 에러·로깅

- 복구 가능: 명시적 예외 (`ValueError`, 프로젝트 `StrataShiftError`)
- 삼키지 말 것 (`except: pass` 금지)
- 사용자 CLI 메시지는 짧고 행동 가능해야 함
- 디버그 로그에 텐서 **전체**를 덤프하지 말 것 (메타·shape·offset만)

---

## 7. 주석·독스트링

- “무엇을”이 아니라 **왜 / 규격 출처**를 적는다
- 공개 함수: 한 줄 summary + Args/Returns (필요 시)
- TODO는 Phase·이슈 없이 남기지 않는다. 있으면 `ROADMAP`에 연결

---

## 8. 에이전트용 한 줄 요약

> 폴더 = `FOLDER_STRUCTURE` → 파일 = `MODULE_DESIGN` → snake_case + early return + 청크만 + REA 주석 + 테스트 + **DOC_SYNC** + **미사용 코드 동시 삭제**.  
> 임의 최상위/`utils/`·전체 load·근거 없는 매직 넘버·문서 drift·죽은 코드 방치 = 머지 거부.

관련: [DOC_SYNC.md](DOC_SYNC.md), [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md), [AGENTS.md](../AGENTS.md), [CONSTRAINTS.md](CONSTRAINTS.md), [MODULE_DESIGN.md](MODULE_DESIGN.md)
