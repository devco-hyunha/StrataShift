# p6-02 — HIP enablement · Strix Halo 실측 (설계 게이트)

- Phase: 6 — 엔진 실행 확장 (ROADMAP planned scope 신설, 2026-10-20)
- Branch: `feat/p6-02-hip-enablement` (pin 상향 + REA 재검) / `docs/p6-02-scope`
- Status: **design (2026-10-20)** — 설계 게이트 문서. AC1~AC3은 PC/sandbox 진행, AC4~AC6은 Strix Halo + 인간 게이트.
- 실행 주체 분담:
  - **헤르메스(sandbox)**: AC1 pin 상향 실무·AC2 REA 재검·AC3 hipcc 컴파일 전용 게이트
  - **사용자 (Strix Halo)**: AC4 ROCm 설치·gfx1151 실측·AC5 mini HIP RUN·AC6 실모델 HIP RUN
  - **사용자**: AC7 ACCEPTANCE.md 인간 서명

## 한 줄 목표

PolyStrata upstream의 AMD ROCm/HIP port(PR #7 병합 커밋)로 pin을 상향하고, Strix Halo
(gfx1151)에서 `strata-poly` HIP 빌드·RUN 게이트를 실측으로 통과/기록하여, Strix Halo의
반복 오류(실행 경로 부재)를 근본적으로 해소한다.

## 근거 (실측, 2026-10-20)

1. pin `c566ff4`는 CUDA 전용 빌드 — `.local/polystrata/PolyStrata-main/CMakeLists.txt`
   실측 grep `STRATA_ENABLE_HIP`·`CMAKE_HIP_ARCHITECTURES`·`gfx` **0건**.
2. 엔진 RUN 계층 CUDA/HIP 전용 (`SOURCE: model_family.hpp:52-62`) — CPU 빌드 `strata-gguf`
   는 가드·parse까지 (p5-06 R7). Strix Halo iGPU(RDNA 3.5, gfx1151)는 NVIDIA 아님 →
   CUDA 물리 불가. p5-08 실행 장소 이동 기록(2026-10-19 Strix 반복 오류 → TK1).
3. upstream 실사(2026-10-19 WebSearch): PR #7 `feat(hip): AMD ROCm/HIP port` — RX 7900 XT
   (gfx1100) 실증, hipBLAS shim(`cublasGemmStridedBatchedEx` 등), VMM 가드
   `!defined(STRATA_USE_HIP)` → `dmalloc` fallback, INT8 KV(`--kv int8`), maintainer
   merge-as-beta 코멘트 + README AMD beta 표기. `CMAKE_HIP_ARCHITECTURES` 기본 gfx906.
4. maintainer 경고 실측: "GLM, Gemma 4, Tiel never ran on AMD" — 우리 Gemma4/Mixtral HIP
   RUN은 Strix에서 첫 실측 = beta 검증 → 인간 서명 게이트 필수.
5. 지금 PC(RTX 3050 Ti)/TK1(RTX 4070 Super)는 NVIDIA → HIP RUN 물리 불가. hipcc는
   컴파일 전용 게이트(p0-02 AC2 전례, C4 준수)에 한해 PC에서 실행 가능.

## In/Out

- In: upstream main 병합 커밋 SHA, pin zip 교체 절차(UPSTREAM_VERSIONING §3), ROCm toolchain
  (hipcc), Strix Halo 사양(RDNA 3.5 gfx1151), p5-08 AC4 RUN 인자 규격, 수집 패키지
  `.local/test/scripts/run-*-hip.sh` (WSL/Linux 동형 스크립트 실재).
- Out: AMD GPU 없는 machine의 HIP RUN(원천 불가), pin 전수 REA 재생(§3 판정 기준 준수),
  Mixtral 클래스 코드(분리: p6-01), StrataShift 저장소에 엔진 코드 커밋(금지 유지).

## 설계 (pseudocode)

```
1. [PC] AC1: upstream main에서 PR #7 병합 커밋 확인 → 병합 SHA 기록
   → `.local/polystrata` zip/트리를 새 pin으로 교체 → REFERENCES.md·지원 매트릭스 갱신
   (UPSTREAM_VERSIONING §3 순서 1-2)
2. [PC] AC2: REA 재검 — HIP 발현 지점 전수:
   - `vmm.cpp:14-72` 가드(`!defined(STRATA_USE_HIP)`) + dmalloc fallback 경로
   - `gguf_reader.hpp:565-590` split 검증 라인은 arch-중립 확인
   - `CMakeLists.txt` HIP 타깃(STRATA_ENABLE_HIP)·CMAKE_HIP_ARCHITECTURES 기본값 실측
   - `mind_vram()` HIP 가드(PR #7 코멘트: dynamic eviction shrink) — VRAM 안전 확인
   → docs/rea/paging_model.md §5·struct_layouts §8 갱신, HYPOTHESIS 발현 조건 기록
3. [PC] AC3: hipcc 컴파일 전용 게이트 — `CMAKE_HIP_ARCHITECTURES=gfx1151`으로
   poly 계열 .o 컴파일(-c, 링크·실행 없음, C4 준수), warning 0 확인, .o SHA256 기록.
   gfx1151 unsupported kernel(sm_90 전용 cluster 등) 발현 시: kernel별 fallback 존재
   목록 실측(미발현 = AC3 FAIL root cause 기록, 인간 게이트 이관)
4. [Strix] AC4: ROCm 설치 확인 + `hipconfig`/`rocminfo` gfx1151 실측 +
   cuMemGetAllocationGranularity 동등(HIP VMM 지원 여부) 실측 —
   paging_model 백로그(2 MiB CUDA 실측)의 AMD 측 대응값 기록
5. [Strix] AC5: HIP 빌드 `strata-poly` 산출물 SHA 기록 → mini fixture HIP RUN rc=0
   (p5-08 AC4 규격 인자 동일: --tokens 1,2,3 --max-new 1 --max-context 1024
   --vram-reserve-mib 512) + VRAM/RSS 실측 로그 (79 GiB unified — residency plan 관찰)
6. [Strix] AC6: 실모델 RUN — Gemma4(p5-08 closure 산출 SHA 29fa0890…) +
   Mixtral(p6-01 AC5 산출) HIP RUN rc=0 + tok/s. beta 표기 유지.
7. [사용자] AC7: ACCEPTANCE.md GPU RUN · AMD beta 항목 인간 서명
```

## AC

- AC1 (헤르메스): 병합 커밋 SHA 실측 + `.local` pin 교체 + `REFERENCES.md`·
  `UPSTREAM_VERSIONING.md` 지원 매트릭스 현재 지원 갱신 (새 pin = `planned` → AC5 PASS 후
  `supported` 전환 가능). 같은 커밋에 `docs/rea/binary_pin.md` §1 갱신.
- AC2 (헤르메스): REA 재검 문서 — HIP 가드 전수 grep 실측(`STRATA_USE_HIP` hits file:line
  표) + VMM granularity AMD 대응 백로그 발현 조건 정문화. `HYPOTHESIS:` 접두어 규칙 유지.
- AC3 (헤르메스): hipcc 컴파일 전용 GREEN (warning 0) + .o SHA 기록. 실행·링크 없음 (C4).
- AC4 (Strix + 인간): gfx1151 실측 로그(rocminfo/hipconfig) + AMD VMM granularity 실측.
  미지원 발현 시 root cause 기록 + 인간 게이트.
- AC5 (Strix + 인간): mini HIP RUN rc=0 로그 원본 — `.local/test/` 스크립트 규격 동형
  (RESULT 자동 생성 유지).
- AC6 (Strix + 인간): 실모델 HIP RUN 로그 (Gemma4/Mixtral) — rc·tok/s·VRAM/RSS.
- AC7 (사용자): ACCEPTANCE.md 서명 — AMD beta 항목(속도 기대치 명시) + 실험 빌드 SHA.
- 전수 pytest GREEN (pin 상향으로 engine-form tool 경로 `gguf_reader.py`·
  `canonical_xcheck`가 바뀌면 같은 Goal에서 DOC_SYNC) + `test_no_full_load` 무파괴.

## 테스트 계획

- 헤르메스: pin 상향 후 `scripts/` engine chain 재실행(sandbox 전수) — engine tool의
  HIP 빌드 의존 없음 확인(가드·parse 계층 CUDA/HIP 아님 실측 유지).
- Strix: AC5 mini + AC6 실모델 RUN 로그 원본 첨부(RESULT 파일 전달 — p5-08 TK1 전례 동형).
- regression: pin 상향 전후 our convert 산출 sha256 동일해야 함(변환기는 엔진 pin과
  무관 — 상수 변경 시 REA 근거로 패치).

## 리스크

- `CMAKE_HIP_ARCHITECTURES` 기본 gfx906 — gfx1151 target이 engine kernels에 지원되지
  않으면 AC3에서 컴파일 FAIL 발현 (PR #7 실증은 gfx1100 — Strix 미실증).
- Strix Halo 반복 오류(2026-10-19)의 root cause가 HIP 부재가 아닌 OS/environment 문제
  였을 가능성 — AC4에서 ROCm 실측으로 판별 기록.
- AMD VMM (`cuMem*` 미지원 → dmalloc fallback) 시 expert cache dynamic shrink가 꺼진
  경로(PR #7 코멘트 `mind_vram` guard) — 긴 prompt에서 VRAM 고갈 risk, Strix 79 GiB
  unified 메모리는 fallback에 유리 — 실측 확인.
- pin 상향은 REA 전수 재검 게이트 — 상수·패커 divergence 발현 시 p3-08(p3-09) 게이트
  재실행 필요 (UPSTREAM_VERSIONING §3 순서 3-5).

## Resume-Notes

(2026-10-20 설계 게이트) — ROADMAP Phase 6 planned scope 등록. 순서: AC1~AC3 (PC/sandbox)
선행 → Strix 방문은 AC4~AC6 RUN 3건 + AC7 서명만 남는다. Strix 사양 실측(ROCm 버전,
gfx1151 target 지원)이 AC4의 첫 게이트.
