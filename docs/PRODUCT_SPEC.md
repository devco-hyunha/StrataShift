# StrataShift 제품·기술 기획 명세서

| 항목 | 값 |
|------|-----|
| 프로젝트명 | StrataShift |
| 문서 버전 | 0.2.0 |
| 상태 | Draft (Ghidra REA / Phase 0 반영) |
| 타깃 엔진 | PolyStrata |
| 기반 원본 | Strata (Niko1221 외) |
| REA 툴킷 | Ghidra (NSA) |

---

## 1. 문제 정의

최신 MoE(Mixture-of-Experts) 모델은 파라미터 규모가 커서, 개인용 로컬 PC의 VRAM/RAM으로는 **전체 가중치를 메모리에 적재한 뒤 변환·양자화**하기 어렵다.  
PolyStrata는 디스크 스트리밍 인퍼런스를 지원하지만, Hugging Face 표준 포맷을 PolyStrata 런타임 포맷으로 옮기는 **저사양 브릿지**가 필요하다.  
런타임 바이트 규격은 공개 문서만으로는 부족할 수 있으므로, **Ghidra REA**로 로더의 정렬·오프셋·비트 패킹을 확정한다.

## 2. 제품 목표

개인용 로컬 PC에서:

1. (Phase 0) Ghidra로 PolyStrata 로더의 저수준 바이트 규격을 추출·문서화한다.
2. MoE 모델 헤더·텐서 메타데이터를 **VRAM/RAM에 올리지 않고** 파싱한다.
3. Expert/레이어 단위로 스트리밍 양자화·비트 패킹을 수행한다.
4. PolyStrata 로더가 즉시 소비할 수 있는 **가중치 블록 + 바이너리 인덱스**를 출력한다.
5. 엔진 런타임 오류 + REA 데이터를 피드백해 변환 스크립트를 패치하는 루프를 만든다.

## 3. 비목표 (Out of Scope, v0.x)

- PolyStrata 엔진 자체 포크·재구현
- 클라우드 GPU 기반 일괄 변환 서비스
- 전체 모델 in-memory 양자화 경로
- 원본 Strata 전용(비확장) 로더만 타깃으로 하는 변환
- Ghidra 대체 상용 RE 툴 필수화 (Ghidra를 표준으로 함)

## 4. 사용자·환경

| 구분 | 가정 |
|------|------|
| 사용자 | 로컬에서 대형 MoE를 돌리려는 개발자·파워유저 |
| OS | Windows / Linux (PolyStrata 지원 환경과 정렬) |
| RAM | 16GB~64GB (변환; 최소 16GB, 권장 32GB+) |
| VRAM | 필수 아님 (변환 단계는 CPU + 디스크 중심) |
| 디스크 | 원본 모델 + 출력물 여유 공간 (SSD 권장) |
| REA | Ghidra를 돌릴 수 있는 개발 워크스테이션 (Phase 0) |

## 5. 입출력 계약

### 5.1 입력 (Source)

- Hugging Face 표준 MoE 체크포인트
- 우선: `.safetensors` (단일/샤드)
- 후속: GGUF

### 5.2 분석 입력 (REA)

- PolyStrata 런타임/로더 바이너리 (및 가능 시 소스)
- Ghidra 프로젝트·디컴파일 산출물 → `docs/rea/`

### 5.3 출력 (Target)

- PolyStrata 런타임 가중치 블록(커스텀 비트 패킹·양자화)
- 바이너리 인덱스 (오프셋, 레이어/Expert ID, shape, dtype/패킹 메타)
- 변환 매니페스트 (소스 해시, 양자화 프리셋, 엔진·REA 규격 버전)

### 5.4 성공 조건 (제품 수준)

- Phase 0: 로더 Struct/bitpack/paging 규칙이 문서·상수로 추적 가능
- 변환 중 프로세스 RSS가 **평탄**하게 유지 (단일 Expert 피크 수준)
- 출력물이 PolyStrata 로더에 주입되어 로드 성공
- OOM / 전체 텐서 일괄 load 경로 없음

## 6. 기능 요구사항

### FR-0 (Phase 0) Ghidra REA 규격 추출

- PolyStrata 바이너리를 Ghidra에 임포트·디컴파일
- 바이트 정렬, 오프셋, 비트 패킹, Struct 필드 문서화
- LLM 에이전트와 연동해 Python 변환 상수/패커 초안 생성 (인간 게이트 필수)

### FR-1 메타데이터 전용 파싱

- safetensors 헤더만 읽어 텐서 이름, dtype, shape, data_offsets 추출
- mmap으로 데이터 영역은 필요 시에만 매핑

### FR-2 Expert/레이어 스트리밍 변환

- FFN Expert 텐서를 식별·정렬
- 제너레이터 루프: load → quantize/pack → write → release

### FR-3 PolyStrata 규격 매핑

- Phase 0 REA(+소스) 기준으로 블록 헤더, 정렬, 비트 패킹 일치
- 규격 버전을 매니페스트에 기록

### FR-4 메모리 관측

- 변환 중 RAM(및 가능 시 VRAM) 샘플링 로그
- 임계치 초과 시 경고/중단 옵션

### FR-5 (Phase 3) 엔진 피드백 루프

- 로더 에러·segfault 로그를 수집
- LLM 에이전트 + Ghidra REA 데이터로 스크립트 패치 유도

## 7. 비기능 요구사항

| ID | 요구 |
|----|------|
| NFR-1 | 전체 모델 적재 금지 (하드 제약) |
| NFR-2 | 단일 Expert/레이어 처리 메모리 상한 설정 가능 |
| NFR-3 | 재현 가능: 동일 입력·프리셋 → 동일 출력 해시 |
| NFR-4 | CLI 우선, 라이브러리 API는 모듈로 분리 |
| NFR-5 | Python 3.11+ (p6-11 확정, `tomllib`; 2026-10-22 사용자 결정) |
| NFR-6 | REA 산출물·바이너리 핀이 추적 가능 (`docs/rea`, REFERENCES) |
| NFR-7 | PolyStrata는 핀·지원 매트릭스로 관리하고, 업그레이드는 UPSTREAM_VERSIONING 절차를 따른다 |

## 8. 리스크

| 리스크 | 영향 | 완화 |
|--------|------|------|
| 로더 난독화·심볼 부재 | REA 지연 | 문자열/XREF·런타임 로그 병행, 소스 교차 |
| PolyStrata 규격 변경 | 변환 실패 | 바이너리 SHA 핀, 매니페스트 버전 |
| GitHub 접근 제한 | 소스/바이너리 확보 실패 | HF Space zip/바이너리 백업 |
| REA 오판 (잘못된 시프트/오프셋) | segfault·로드 실패 | HYPOTHESIS 표기, Phase 3 로드로 검증 |
| 양자화 품질 | 추론 품질 저하 | 프리셋·캘리브레이션 옵션 (후속) |

## 9. 관련 문서

- [REA_GHIDRA.md](REA_GHIDRA.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [ROADMAP.md](ROADMAP.md)
- [CONSTRAINTS.md](CONSTRAINTS.md)
- [REFERENCES.md](REFERENCES.md)
