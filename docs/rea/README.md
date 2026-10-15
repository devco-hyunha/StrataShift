# docs/rea — Ghidra REA 산출물

Phase 0에서 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)로 PolyStrata 로더를 분석한 결과를 이 폴더에 고정한다.

절차: [../REA_GHIDRA.md](../REA_GHIDRA.md)

## 파일 체크리스트

| 파일 | 상태 | 설명 |
|------|------|------|
| `binary_pin.md` | **p0-01 고정** | PolyStrata `c566ff4` zip SHA256 · Ghidra 12.1.4 SHA256 · JDK 21 · 프로젝트 루트 제약 |
| `loader_entrypoints.md` | **p0-01 골격** | 소스 기반 진입점 표 (A~E, file:line 인용). 디컴파일 심볼 0건 |
| `struct_layouts.md` | **p0-01 골격** | GGUF 헤더 · blob 오프셋 · `index.txt` 19열 · `STRP` (소스+바이트 이중 확인) |
| `bitpack_rules.md` | **p0-01 골격** | Q2_0 2-bit LSB-first · QK64/18B · scale 폭 fp16↔f32 · SIMD unpack 순서 |
| `paging_model.md` | **p0-01 골격** | blob=페이징 단위 · stride 1,382,400 · ring byte budget · VMM granularity |

p0-01은 **소스 정찰만**으로 고정했다 (Ghidra 임포트·디컴파일은 p0-02).
불확실한 항목은 본문에 `HYPOTHESIS:` 접두어를 붙인다.
확정은 Phase 3 런타임 로드 성공 또는 소스 교차 검증 후에만 한다 (CONSTRAINTS C4).

## 근거 표기 규칙

- `SOURCE: <file>:<line>` — 핀된 PolyStrata 커밋 `c566ff4787c8ab188ecf33c6f1907a9f90d9f152`의 소스
- `VERIFIED:` — 실데이터의 바이트 대조 (실행 아님)
- `DECOMPILED:` — p0-02에서 Ghidra로 얻은 주소·심볼 (현재 0건)
- `HYPOTHESIS:` — 근거 없는 추정. 매직 넘버를 이 접두어 없이 확정하지 않는다
