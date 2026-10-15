# docs/rea — Ghidra REA 산출물

Phase 0에서 [Ghidra](https://github.com/NationalSecurityAgency/ghidra)로 PolyStrata 로더를 분석한 결과를 이 폴더에 고정한다.

절차: [../REA_GHIDRA.md](../REA_GHIDRA.md)

## 파일 체크리스트

| 파일 | 상태 | 설명 |
|------|------|------|
| `binary_pin.md` | _TODO_ | 바이너리 경로·SHA256·버전 |
| `loader_entrypoints.md` | _TODO_ | 파싱/할당 함수 주소·심볼 |
| `struct_layouts.md` | _TODO_ | Struct 필드 오프셋 |
| `bitpack_rules.md` | _TODO_ | 시프트·마스크·패킹 폭 |
| `paging_model.md` | _TODO_ | Expert/레이어 페이징 |

불확실한 항목은 본문에 `HYPOTHESIS:` 접두어를 붙인다.  
확정은 Phase 3 런타임 로드 성공 또는 소스 교차 검증 후에만 한다.
