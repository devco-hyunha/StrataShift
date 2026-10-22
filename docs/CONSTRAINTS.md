# StrataShift 필수 준수 규칙 (Constraints)

이 문서는 **하드 제약**이다. 편의상 우회하는 API·스크립트·“임시 전체 로드”는 병합 금지.

---

## C1. 전체 모델 메모리 적재 금지

| 규칙 | 내용 |
|------|------|
| MUST NOT | 모든 텐서를 `torch.load` / `safetensors.torch.load_file` 등으로 한 번에 RAM에 올림 |
| MUST NOT | 전체 Expert를 리스트/딕셔너리에 누적한 뒤 일괄 양자화 |
| MUST | 청크(기본: Expert 또는 레이어 1단위)로 읽고 즉시 쓰고 해제 |
| MUST | 16GB~64GB RAM 개인용 로컬 PC에서 변환 중 OOM이 나지 않도록 설계 |
| SHOULD | 프로세스 RSS 상한을 CLI 플래그로 강제 (`--max-rss-gb`). Linux·Windows에서 측정, 측정 불가 OS는 조용히 통과하지 않고 거절 (p6-10) |

**허용되는 적재 범위**

- safetensors **헤더** (보통 수 MB 이하)
- **현재 처리 중인** 단일 텐서/Expert 슬라이스
- 양자화 출력 버퍼 (동일 단위)
- 인덱스·매니페스트 메타데이터

## C2. PolyStrata 규격 준수

| 규칙 | 내용 |
|------|------|
| MUST | 타깃은 **PolyStrata** 확장 모델 로더 (원본 Strata-only 가정 금지) |
| MUST | 텐서 정렬, 블록 헤더, 비트 패킹을 **Ghidra REA 및/또는 엔진 소스**로 검증된 규칙에 맞춤 |
| MUST | 사용한 규격 버전·바이너리 핀을 출력 매니페스트/`REFERENCES`에 기록 |
| MUST | PolyStrata 핀 변경 시 [UPSTREAM_VERSIONING.md](UPSTREAM_VERSIONING.md) 절차(REA 재검·지원 표)를 따른다 |
| SHOULD | 공개 소스와 REA 결과가 다르면 차이를 문서화하고 런타임으로 확정 |
| MUST NOT | 핀 없이 “항상 최신 PolyStrata”를 지원한다고 가정·표기 |

## C3. 스트리밍 I/O만 허용

```
load(chunk) → transform(chunk) → write(chunk) → release(chunk)
```

- 제너레이터/이터레이터 기반 루프를 기본으로 한다.
- 랜덤 액세스는 **오프셋 기반 mmap 슬라이스**로만 수행한다.
- “빠른 변환”을 이유로 전체 파일을 `bytes`로 읽는 경로를 추가하지 않는다.

## C4. Ghidra REA · LLM 에이전트 연동

| 규칙 | 내용 |
|------|------|
| MUST (Phase 0) | PolyStrata 런타임/로더를 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)로 분석해 규격 초안을 `docs/rea/`에 남긴다 |
| MAY | Ghidra 디컴파일·Struct 표를 LLM에 넘겨 비트 시프트/오프셋 → Python 패커 자동 초안 생성 |
| MAY | Phase 3에서 엔진 로그 + REA 재대조로 패치 제안 |
| MUST NOT | 미검증 REA/LLM 패치를 테스트 게이트 없이 프로덕션 경로에 강제 적용 |
| MUST NOT | 사용자 환경의 임의 바이너리를 설명 없이 실행하는 자동화 |
| MUST | 불확실한 필드는 `HYPOTHESIS:`로 표시 |

절차 상세: [REA_GHIDRA.md](REA_GHIDRA.md)

## C5. 의존성·환경

- 변환 경로는 **GPU 필수 아님** (CPU 양자화 기본)
- CUDA 사용 시에도 “모델 전체 VRAM 적재” 금지
- 외부 변환 도구에 전체 모델을 넘겨 RAM에서 펼치게 하는 래핑 금지
- Ghidra는 **개발/분석 의존**이며 런타임 변환 CLI의 필수 런타임 의존이 아님

## C6. 코드 리뷰 체크리스트 (PR)

- [ ] 전체 `load_file` / 전체 state_dict 적재 호출 없음
- [ ] 루프 단위가 Expert 또는 동등 이하 청크임
- [ ] `del` / context manager / 명시적 close로 버퍼 해제
- [ ] PolyStrata 규격 상수 출처가 `docs/rea/` 또는 REFERENCES 핀과 연결됨
- [ ] REA/`HYPOTHESIS` 변경 시 ACCEPTANCE·테스트 갱신
- [ ] [DOC_SYNC.md](DOC_SYNC.md) — 코드 변경에 대응하는 문서가 같은 PR에 있음
- [ ] [AI_DEV_WORKFLOW.md](AI_DEV_WORKFLOW.md) DoD — AC·테스트·리뷰 PASS, Goal ID·브랜치 규칙
- [ ] 메모리 모니터 또는 수동 RSS 검증 기록이 있음 (Phase 2+)

## C7. 위반 시 처리

제약이 깨진 코드는:

1. 머지 차단
2. 스트리밍 경로로 재작성
3. ACCEPTANCE 회귀 테스트에 “anti-full-load” 가드 추가
4. (규격) REA 문서와 불일치 시 `docs/rea`부터 수정 후 코드 반영

---

관련: [ARCHITECTURE.md](ARCHITECTURE.md), [REA_GHIDRA.md](REA_GHIDRA.md), [ACCEPTANCE.md](ACCEPTANCE.md)
