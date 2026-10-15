# StrataShift 코어 아키텍처

## 1. 한 줄 요약

**Ghidra REA로 규격 확정 → 디스크 mmap 청크 → CPU 양자화·패킹 → PolyStrata 블록**  
전 구간에서 “모델 전체 적재” 경로를 구조적으로 차단한다.

## 2. 파이프라인 개요

```
┌─────────────┐   ┌──────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐
│  Source     │   │  Analysis Core   │   │  StrataShift Core        │   │  Target             │
│  HF MoE     │   │  REA / Ghidra    │──▶│  parse → stream → pack   │──▶│  PolyStrata runtime │
│  .safetensors│   │  align/offset/   │   │  (mmap, chunk, release)  │   │  weights + index    │
└──────┬──────┘   │  bitpack rules   │   └──────────────────────────┘   └─────────────────────┘
       │          └──────────────────┘
       └──────────────────────────────────────────▶ (메타·청크 입력)
```

### 레이어 역할

| 레이어 | 역할 |
|--------|------|
| 입력단 (Source) | HF MoE `.safetensors` / GGUF |
| 분석 코어 (REA / Ghidra) | 엔진 바이너리에서 바이트 정렬·오프셋·비트 패킹 룰 추출 |
| 처리 코어 (StrataShift) | mmap 저-RAM 스트리밍, Expert 단위 양자화·패킹 |
| 출력단 (Target) | PolyStrata 런타임 가중치 블록 + 바이너리 인덱스 |

### 변환 단계 상세

| 단계 | 역할 | 메모리 원칙 |
|------|------|-------------|
| 0. REA Spec | Ghidra로 로더 규격 고정 | 분석 워크스테이션 (변환 RSS와 분리) |
| 1. Header Parse | 텐서 이름·shape·offset 추출 | 헤더만 RAM |
| 2. Expert Map | FFN Expert / 레이어 인덱스 구성 | 메타데이터만 |
| 3. Stream Load | 단일 Expert/레이어 mmap 슬라이스 | 청크 상한 |
| 4. Quantize & Pack | REA 규칙 기반 CPU 비트 패킹·양자화 | 입력+출력 버퍼만 |
| 5. Stream Write | 타깃 블록·인덱스 append | 버퍼 flush 후 해제 |
| 6. Release | 참조·버퍼 명시적 해제 | RSS 평탄 유지 |

## 3. 논리 컴포넌트

```
stratashift/
├── cli/                 # CLI 엔트리
├── ingest/              # 소스 포맷 리더 (safetensors, 후속 GGUF)
├── meta/                # 헤더 파서, Expert 맵, 매니페스트
├── stream/              # mmap 청크 리더 / 라이터
├── quant/               # 양자화·비트 패킹 (REA 규칙 반영)
├── polystrata/          # 엔진 규격: 블록 헤더, 정렬, 인덱스 (REA 산출물)
├── monitor/             # RAM/VRAM 관측
└── feedback/            # Phase 3: 엔진 로그 + REA 재대조 → 패치 루프

docs/rea/                # Ghidra 분석 산출물 (Phase 0)
```

### 컴포넌트 책임

| 모듈 | 책임 | 금지 |
|------|------|------|
| `docs/rea` | Ghidra Struct/bitpack/paging 정본 | 추측을 검증 없이 코드에 하드코딩 |
| `ingest` | 포맷별 텐서 뷰 제공 | `load_full_model()` |
| `meta` | 오프셋·그래프 메타 | 가중치 본문 복사 |
| `stream` | 청크 I/O, 제너레이터 | 전체 텐서 리스트 적재 |
| `quant` | 단위 텐서 변환 (REA 패킹) | 배치로 전 레이어 축적 |
| `polystrata` | 헤더/정렬/패킹 규격 | Strata-only 가정 |
| `monitor` | 사용량 샘플·게이트 | — |
| `feedback` | 런타임 오류 + REA → 패치 힌트 | 미검증 자동 강제 적용 |

## 4. 데이터 흐름 (제너레이터 루프)

의사코드:

```text
spec     ← load_polystrata_spec(from docs/rea + polystrata constants)  # Phase 0
manifest ← parse_safetensors_header(path)          # mmap header only
experts  ← build_expert_map(manifest)

for expert in experts:
    view ← mmap_tensor_slice(path, expert.offsets) # one unit
    packed ← quantize_and_pack(view, spec.preset)  # CPU, REA bitpack
    append_weight_block(out_fd, packed, expert.meta, spec.headers)
    append_index_entry(index_fd, expert, offsets, spec.index)
    del view, packed
    maybe_gc()
    assert_rss_within(budget)

finalize_index(index_fd, manifest, spec)
write_conversion_manifest(..., spec.version)
```

## 5. PolyStrata 규격 레이어 (REA 정본)

StrataShift는 **원본 Strata**가 아니라 **PolyStrata 확장 모델 로더**를 타깃으로 한다.  
규격의 1차 추출 수단은 **Ghidra REA**이며, 공개 소스가 있으면 교차 검증한다.

매핑 항목:

1. 가중치 블록 헤더 레이아웃 (magic, version, flags, sizes)
2. 텐서 정렬(alignment) 및 패딩
3. 커스텀 비트 패킹 / 양자화 dtype 인코딩 (시프트·마스크)
4. Expert 페이징·인덱스 키 구조
5. 런타임이 기대하는 파일 분할 규칙 (단일 vs 샤드)

절차: [REA_GHIDRA.md](REA_GHIDRA.md)  
핀: [REFERENCES.md](REFERENCES.md)

## 6. 메모리 모델

```
Peak ≈ header_meta + max(single_expert_raw) + pack_buffer + I/O_buffer
```

- `max(single_expert_raw)` 가 설계 상한의 주 요인
- 동시 청크 수 = 1 (기본)
- Ghidra 분석은 변환 프로세스와 **별도** (변환 RSS에 포함하지 않음)

## 7. REA ↔ LLM ↔ 런타임 피드백

```
PolyStrata binary
       │
       ▼
 Ghidra REA  ──docs/rea/*──▶  polystrata/quant 상수·패커
                                    │
                         convert → runtime load
                                    │ error / segfault
                                    ▼
                          feedback + REA 재대조
                                    │
                                    ▼
                          LLM 패치 제안 → 테스트 게이트 → merge
```

## 8. 확장 포인트

- GGUF ingest 어댑터
- 양자화 프리셋 플러그인
- Ghidra 스크립트/헤드리스 내보내기 자동화
- 캘리브레이션 샘플 스트림 (선택)

## 9. 관련 문서

- [REA_GHIDRA.md](REA_GHIDRA.md)
- [MODULE_DESIGN.md](MODULE_DESIGN.md)
- [CONSTRAINTS.md](CONSTRAINTS.md)
- [ROADMAP.md](ROADMAP.md)
