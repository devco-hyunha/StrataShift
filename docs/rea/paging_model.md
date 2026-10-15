# paging_model — Expert/레이어 단위 로드 경계

Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152`.

## 1. 페이징의 단위 = expert blob

`SOURCE: include/strata/kernels/cpu/expert.hpp:49`, `src/prefill/prefill.cpp:220`

- blob = **1 expert의 gate_up+down codes + scales** = 1,382,400 B (Q2_0 pack)
- `prefill.cpp:220`: "**a slot is one WHOLE expert blob**" — 384 slots = 506 MiB
- Q8_0 pack은 blob 5,222,400 B → 같은 slot 수 = 1,912 MiB. **budget는 BYTES, slot 수는 pack에서 파생**

## 2. 디스크 → 아레나 (스트리밍 로드)

`SOURCE: src/core/load_main.cpp:19-21,47-60`

| 파라미터 | 값 | 근거 |
|----------|-----|------|
| blob | 1,382,400 B | `load_main.cpp:19` (architecture §3.2, `pack_layer.py` assert) |
| blobs/layer | 512 | `load_main.cpp:21` |
| chunk | 16 MiB (`--chunk-mb`) | `load_main.cpp:20` |
| threads | 8 | `load_main.cpp:20` |
| target | 33.97 GB: cold ≤60 s, warm ≤10 s | `load_main.cpp:2-4` |
| layer별 checksum | 있음 | `load_main.cpp:4` |

`SOURCE: src/core/concurrent_main.cpp:88`, `src/core/overlap_main.cpp:77`

```cpp
f.seekg((std::streamoff) i * 1382400);   // expert i는 blob stride로 접근
```

→ **expert `i`는 파일에서 contiguous한 1,382,400 B 블록**이며, 전체 모델 적재 API는 없다.
`SOURCE: tools/pack_layer.py:24-26` — expert `e`는 각 role tensor 안의 contiguous run
(`ne[0]`이 fastest, expert index가 `[in,out,experts]`의 **마지막 축**).

## 3. VRAM expert cache (residency)

`SOURCE: include/strata/core/expert_cache.hpp:1-31`, `src/core/expert_cache.cpp:368-379,536-565`

- `(layer, expert) → slot index or -1` (`kNotResident = -1`, `int32_t`)
- slot 오프셋 배열 `off_` (`off_[i+1]` = slot i의 끝) → `slots_within(bytes)`는
  `upper_bound`로 **mapped bytes 안에서 사는 slot 수**를 계산
- 정적 residency plan은 `STRP` 프로필에서 읽는다 (ranked by routing frequency,
  `struct_layouts.md` §5에서 바이트 검증)
- 실측 sizing: `h_expert` 0.6447 @ 4,105 slots, 5,413 MiB (steady state free 6,414 MiB)
- `h_layer` 0.0456 → **per-layer grouped kernel 불가** (10개 전부 resident인 pair는 4.6%)

## 4. Prefill ring (스트리밍 버퍼)

`SOURCE: src/prefill/prefill.cpp:215-262`

- ring은 **byte budget** (`ring_bytes()`), slot 수는 blob 크기에서 파생
- Q2_0: fused 1,024 slots / non-fused 384 slots, `ring_cap()` 512 (fused Q2_0는 1024)
- pinned share < 0.9이면 ring을 1/4로 줄인다 (`ring_bytes()`)
- `STRATA_PREFILL_RING` / `STRATA_RING_BYTES` override

## 5. GPU VMM (가상 주소 예약·매핑)

`SOURCE: src/core/vmm.cpp:14-72,82-94`

- `cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` / `cuMemSetAccess`를
  `cudaGetDriverEntryPointByVersion(12000)`로 런타임 resolve (CUDA 12.5+ versioned query)
- 할당 단위는 `cuMemGetAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)`의 `gran`
  → `create(&h, gran, …)`, `n = ceil(bytes, gran)`
- K/V pool과 cache는 **static destructor에서 release** (never destroyed)
- `HYPOTHESIS:` `gran`의 실측값은 런타임 값 (GPU 의존) — p0-02에서 실측 기록

## 6. StrataShift 변환기가 따라야 할 경계

1. **단위 = expert blob.** 레이어/Expert 경계 밖으로 읽지 말 것 (C1·C2).
2. **stride addressing:** `offset = layer_off + e × BLOB` — 재계산하지 말고 manifest의
   `layers[].offset` / `offsets`를 읽는다 (`pack_layer.py:100-103`).
3. **chunk는 byte budget로 잡는다** (engine이 16 MiB / ring bytes를 쓰는 이유 = blob 크기가
   pack마다 다르기 때문).
4. **manifest가 plane 크기를 지시한다** — loader는 재계산하지 않는다 (`pack_index.py:62-70`).
5. `HYPOTHESIS:` StrataShift의 출력 포맷이 engine의 `experts.bin` + `index.txt`와 **동일**해야
   engine이 로드한다. 이 동등성은 Phase 3 로드 테스트로만 확정한다 (C4).

## 7. DECOMPILED 검증 (p0-02, Ghidra 12.1.4)

- `DECOMPILED: load_main.o main @ 00100690` — `load_experts(path, dst, size, 0x151800, 0x200, …)`
  호출: blob **1,382,400** (`0x151800`), blobs/layer **512** (`0x200`), chunk 기본 `0x1000000`
  (16 MiB). §2 표 **일치**.
- `DECOMPILED: concurrent_main.o` thread lambda `lVar += 0x151800`, `overlap_main.o main @ 001005c0`
  `+ 0x151800` (2곳) → expert i stride addressing (§2 seekg **일치**).
- `DECOMPILED: expert_cache.o ExpertCache::slots_within @ 00103200` — `off_` 배열(base+0xd0/0xd8)
 과 `bytes/off` 산수 → mapped bytes 내 slot 수 계산 (§3 `upper_bound` 계약과 같은 구조).
- `DECOMPILED: vmm.o vmm_granularity @ 00100510` — `cuMemGetAllocationGranularity`를
  `cudaGetDriverEntryPointByVersion(…, 12000)`로 resolve (§5 **일치**). `gran` 실측값은
  런타임(GPU) 의존 → `HYPOTHESIS:` 유지 (본 환경 GPU 없음 — 실측 불가, 확정 금지).
- 충돌 없음.
