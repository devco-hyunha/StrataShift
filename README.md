# StrataShift

**개인용 로컬 PC를 위한 PolyStrata 호환 MoE 스트리밍 변환·양자화 브릿지 툴킷**

Hugging Face 표준 MoE 모델(`.safetensors` / GGUF)을 **디스크 스트리밍** 방식으로 읽어들여, [PolyStrata](https://github.com/VecSzn/PolyStrata) 엔진이 즉시 로드할 수 있는 런타임 가중치 블록·바이너리 인덱스로 변환합니다.  
엔진 바이트 규격은 [Ghidra](https://github.com/NationalSecurityAgency/ghidra) 기반 REA로 확정합니다.  
모델 전체를 RAM/VRAM에 올리지 않습니다.

---

## 핵심 목표

| 항목 | 내용 |
|------|------|
| 타깃 엔진 | PolyStrata (확장형 MoE 스트리밍 인퍼런스) |
| 입력 | HF MoE (`.safetensors`, GGUF) |
| 분석 | Ghidra REA → 정렬·오프셋·비트 패킹 규격 |
| 출력 | PolyStrata 런타임 가중치 블록 + 바이너리 인덱스 |
| 하드웨어 | 16GB~64GB RAM 개인용 로컬 PC (막대한 VRAM 불필요) |
| 방식 | mmap + 레이어/Expert 단위 순차 스트리밍 |

## 절대 규칙

1. **전체 모델 메모리 적재 금지** — 청크 단위 디스크 스트리밍만 허용
2. **PolyStrata 규격 준수** — 원본 Strata가 아닌 PolyStrata 확장 로더 규격
3. **REA 기반 규격** — Ghidra로 추출한 패킹/오프셋 규칙을 변환기에 반영
4. **OOM 방지** — 단일 Expert/레이어 처리 후 즉시 메모리 해제

## 빠른 문서 안내

| 문서 | 설명 |
|------|------|
| [docs/INDEX.md](docs/INDEX.md) | 문서 읽는 순서 |
| [docs/PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md) | 제품·기술 기획 명세서 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 코어 아키텍처·파이프라인 |
| [docs/REA_GHIDRA.md](docs/REA_GHIDRA.md) | Phase 0 Ghidra REA 절차 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Phase 0~3 개발 로드맵 |
| [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md) | 메모리·규격·REA 안전 제약 |
| [docs/REFERENCES.md](docs/REFERENCES.md) | 엔진·Ghidra·미러 참조 |
| [docs/UPSTREAM_VERSIONING.md](docs/UPSTREAM_VERSIONING.md) | PolyStrata 버전 핀·따라가기 |
| [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md) | 고정 폴더 구조·배치 규칙 |
| [docs/GITIGNORE_POLICY.md](docs/GITIGNORE_POLICY.md) | Git 제외·로컬 엔진/REA 배치 |
| [docs/MODULE_DESIGN.md](docs/MODULE_DESIGN.md) | 모듈·디렉터리·API 설계 |
| [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) | 파일·함수 생성·코딩 규칙 |
| [docs/DOC_SYNC.md](docs/DOC_SYNC.md) | 코드 변경 시 문서 동기화 |
| [docs/AI_DEV_WORKFLOW.md](docs/AI_DEV_WORKFLOW.md) | AI 자동화: 목표→브랜치→리뷰→머지 |
| [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) | 단계별 인수 기준 |
| [AGENTS.md](AGENTS.md) | AI 코딩 에이전트용 지침·프롬프트 |
| [`.cursor/rules/`](.cursor/rules/) | Cursor 상시/글로브 룰 |

## 현재 상태

- **Phase:** Phase 0~4 **완료** (REA 백로그 해소 p0-06 포함), Phase 5 (v1.0.0 공개화) 진행 중
- **구현:** Phase 4 전수 — HF naming→engine GGUF 매핑(p4-01), multi-shard 디렉토리 입력(p4-02), 실모델 Qwen1.5-MoE-A2.7B 27 GB 스모크 R1~R8 PASS(p4-03, heap peak 116 MB < 2 GiB), dense 프리셋 Q4_0~Q8_0(p3-03~p3-09), packaging+console entry(p5-01). 전수 pytest 90+ GREEN. 잔여: Q2_K(upstream MAPPINGS 부재), GPU 의존 백로그, ACCEPTANCE 인간 서명
- **다음:** Phase 5 — CLI polish(p5-02), 사용자 문서(p5-03), CI 게이트(p5-04), v1.0.0 릴리즈(p5-05)

## Quickstart (실측, p5-03)

```
$ pip install .
$ stratashift --help
usage: stratashift [-h] {inspect,convert,verify} ...
$ stratashift inspect .local/models/mixtral-tiny.safetensors
tensors=21 experts=16 layers=[0, 1] data_start=2224
$ stratashift convert .local/models/mixtral-tiny.safetensors --out QSQ \
    --emit-gguf QSQ/model.gguf
rc=0, rss_peak_bytes=71,454,720, heap_peak_bytes=33,567,636, slice_heap_peak_bytes=20,758,102
$ stratashift verify QSQ
{"ok": true, "format": "p2", "pool_bytes": 0}   → exit 0
```

HF naming 모델은 `--hf-emit` 플래그 없이 fused HF emit 경로로 자동 감지된다 (p5-02 AC1).
`GGUF → convert → verify` engine-form hop도 실측 PASS: `convert QSQ/model.gguf --out QSQ2 &&
verify QSQ2` → pool_bytes=16,451,840, sha256 재계산 PASS, exit 0 (p5-02 AC2).

`--imatrix`는 calibration 산물(`imatrix.gguf`)을 consume해서 expert·dense 자리를 imatrix-weighted
Q4_K/Q4_0으로 양자화한다 (p6-05). 사용자 ruling 2026-10-21: Q2_0는 하지마 — `quantize_q2_0`는
imatrix를 consumer에서 무시하므로(Q4_0/Q4_K에만 imatrix branch 실재), expert 자리를 gate/up Q4_K
(`h % 256 = 0`) + down Q4_0(`ff % 256 = 192 남` → be=32 ladder)로 재단한다. plane unit = expert,
vector length = engine row length. opt-in flag — `--imatrix` 없으면 plain RTN golden byte 불변:

```
$ stratashift convert .local/models/gemma4-26b-a4b-it --out QSQ \
    --emit-gguf QSQ/model.gguf --imatrix .local/out/p6-04-imatrix-20261021/imatrix.gguf \
    --dense-type Q4_K
rc=0, expert tensors Q4_K/Q4_0, dense engine-key lookup consume (gate 열림 중, docs/work/p6-05.md)
```

`--imatrix`는 fused HF 경로·single-file 경로 양쪽에 배선됐다 (p6-05 refuse gate 제거). plane shape
mismatch는 refuse exit 3 (never-invented). 상세 contract: [docs/rea/imatrix_contract.md](docs/rea/imatrix_contract.md) §6.

모델 전체를 RAM에 올리지 않는다 — mmap 스트리밍 + 단일 chunk 피크. 상세 플래그 표: [docs/USAGE.md](docs/USAGE.md). 엔진 실 acceptance(harness 체인)은 `strata-gguf`/`strata-dequant`/`strata_pack` 체인이 정본 — 스킬 `polystrata-engine-acceptance` 참조.

## 엔진 구동 + API 종점 (실측, p5-09)

`convert --emit-gguf --hf-emit`은 HF `tokenizer.json`이 실재하면 GGUF 헤더에
`tokenizer.ggml.*` 메타(12 keys + `tokenizer.chat_template`)를 발행한다 (vocab·merges·
token_type·special id 실측 전사, never-invented 게이트). PolyStrata `serve/server.py`가
우리 산출물을 OpenAI `/v1/chat/completions`로 serve한다 (WSL CUDA build 실측 — host는
`.local` 전용 엔진 트리, 저장소 커밋 안 함):

```
python scripts/p5_api_smoke.py                                  # mini 스모크 (RC 0)
python scripts/p5_api_smoke.py --model <emitted.gguf> \
  --model-dir <hf-dir>                                          # 실model 스모크 (RC 0)
# R3 engine extract() → R4 run config → R5 /health 200 → R6 POST /v1/chat/completions
#   → HTTP 200 + 텍스트 왕복 + engine RSS/Swap/GPU 실측 → R7 teardown rc=0
```

실model(gemma4, probe.gguf 7,877,904,096 B·688 tensors·tokenizer keys 14) 실측: load 후
VmRSS 2,467,764 kB·VmSwap 0 kB·GPU 3,746 MiB, 생성 후 VmRSS 3,789,720 kB·VmSwap 0 kB·
GPU 3,787 MiB, chat completion HTTP 200 (usage total 51). 상세: [docs/work/p5-09.md](docs/work/p5-09.md).

## 참조

- PolyStrata: https://github.com/VecSzn/PolyStrata
- Hugging Face Space (소스·바이너리 백업): https://huggingface.co/spaces/VecSzn/PolyStrata
- Ghidra (REA): https://github.com/NationalSecurityAgency/ghidra
- 기반 엔진: Strata (Niko1221 및 기여자들)

## 라이선스

프로젝트 라이선스는 구현 착수 시 PolyStrata / Strata / Ghidra 관련 고지와의 호환성을 검토한 뒤 확정합니다.
