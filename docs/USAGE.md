# USAGE

StrataShift CLI 사용법. 모든 플래그 표는 2026-10-18 `--help` 실측 산출 (발명 없음).

## 설치

```
pip install .
stratashift --help
```

런타임 의존은 numpy뿐 — `safetensors`/`torch`는 CONSTRAINTS C1로 금지 (full-load API).

## `inspect` — 헤더 파싱 + expert 맵 (전체 로드 없음)

```
stratashift inspect [--json] model
```

| 인자 | 의미 |
|------|------|
| `model` | `.safetensors` 단일 파일 또는 multi-shard HF 디렉토리 (`index.json` + shards, p4-02) |
| `--json` | expert 맵·geometry를 JSON으로 출력 |

실측 예 (mixtral-tiny, 8.6 MB):

```
$ stratashift inspect .local/models/mixtral-tiny.safetensors
tensors=21 experts=16 layers=[0, 1] data_start=2224
```

## `convert` — 스트리밍 Q2_0 pack + 엔진 GGUF emit

```
stratashift convert --out OUT [--max-rss-gb N] [--emit-gguf PATH]
                    [--guard-keys JSON_PATH] [--hf-emit]
                    [--dense-type {F32,Q4_0,Q4_K,Q5_K,Q6_K,Q3_K,Q8_0}]
                    model
```

| 인자 | 의미 |
|------|------|
| `model` | HF `.safetensors` (단일/multi-shard) 또는 engine-form 입력 |
| `--out` | pack 산출 디렉토리 (experts.bin, index.txt, manifest.json, conversion.json) |
| `--max-rss-gb` | RSS 상한 초과 시 안전 중단 (exit 3) |
| `--emit-gguf` | 엔진 GGUF v3 (Q2_0 tensors)를 이 경로에 추가 산출 |
| `--guard-keys` | 컴파일된 architecture guard keys JSON 프리셋 경로 — never invented (struct_layouts §1) |
| `--hf-emit` | HF naming 입력의 emit 경로 강제. HF naming은 플래그 없이도 fused HF emit 경로로 **자동 감지**된다 (p5-02 AC1, dispatch 연결) |
| `--dense-type` | dense tensor 타입: F32 pass-through (기본) 또는 양자화 프리셋 (p3-03~p3-09) |

실측 예 (HF fused + BF16, mixtral-tiny — 플래그 없이 auto-detection):

```
$ stratashift convert .local/models/mixtral-tiny.safetensors --out QSQ \
    --emit-gguf QSQ/model.gguf
rc=0, rss_peak_bytes=71,454,720, heap_peak_bytes=33,567,636,
slice_heap_peak_bytes=20,758,102   (< 2 GiB gate PASS)
```

## `verify` — pack 로컬 self-check (두 계약, p5-02 AC2)

```
stratashift verify OUT
```

`OUT/manifest.json`의 키로 계약 자동 dispatch:

- **engine-form** (`tensors`/`experts`/`align`/`n_layers`/`n_experts_per_layer`): index.txt
  19-field rows 재계산 + plane sums + pool 재계산, experts.bin 크기 = layers × NE × blob,
  각 layer blob 경계, 산출 해시 재계산 (ACCEPTANCE Phase 3, p4 harness 계약).
- **p2 CLI pack** (`packed_rows`/`pool_bytes`/`experts_bin_sha256` — `convert --out` 산출):
  header pool/tensors vs manifest, row codes+scales = dst_bytes, dst_off 순차 재계산,
  experts.bin 크기 = pool, sha256 재계산, model 기준 in_bounds.

실측 (mixtral-tiny):

```
$ stratashift verify QSQ
{"ok": true, "format": "p2", "pool_bytes": 0, "experts_bin_sha256": "e3b0c442…b855"}  → exit 0
$ stratashift convert QSQ/model.gguf --out QSQ2 && stratashift verify QSQ2
{"ok": true, "pool_bytes": 16451840, "sha256": {…4 file 해시 재계산 PASS}} → exit 0
```

exit: PASS 0 / FAIL non-zero. engine-form GGUF는 `convert <gguf> --out`이 engine-form
manifest를 쓴다 (GGUF→convert→verify hop 실측 PASS, p5-02 AC2b).

## 지원 범위: 변환 vs 엔진 실행 (p5-06 AC1, pin `c566ff4` 실측 근거)

두 계층을 구분한다 — 정본: [docs/rea/arch_families.md](rea/arch_families.md).

- **변환 지원(StrataShift)**: 임의 MoE `.safetensors`/GGUF 입력 — HF naming(Qwen1.5 fused,
  Qwen2MoE/Mixtral per-expert) → engine-form GGUF v3 산출. geometry는 헤더에서 derived
  (never invented). p4-01~p4-03 실측: qwen4exp fixture 엔진 체인 PASS, Qwen1.5-MoE-A2.7B
  실모델 스모크 PASS.
- **엔진 실행 지원(PolyStrata)**: `general.architecture` 문자열 → compiled architecture 클래스
  dispatch (`SOURCE: src/poly/engine.cpp:808-816`): `glm5-next`/`qwen35moe`/`gemma4`/`deepseek4`
  4 family, 밖은 load 거부. 실행 계층은 shape를 파일(prefix 키)에서 읽는다
  (`SOURCE: engine.cpp:646-653,674-697`) — 하드코딩 프리셋 아님. `qwen4exp`는 guard
  (artifact) 계층 프리셋 (`check_architecture` gguf_reader.hpp:796-812)이며 실행 family가 아니다.
- 제품 정직성: Qwen1.5-MoE·Mixtral 아키텍처 클래스는 엔진에 실재하지 않는다
  (`src/poly/` grep 0건, p4-03 실측 load 거부) — 변환 산출물의 엔진 실 acceptance는
  reference verifier(harness)가 정본. 엔진 실행까지의 acceptance는 GPU·인간 게이트.

## exit codes

| code | 의미 |
|------|------|
| 0 | 성공 |
| 1 | 런타임 실행 (파일 부재·엔진 가드 FAIL 등, stderr 한 줄 메시지) |
| 2 | usage error (argparse) |
| 3 | `--max-rss-gb` 상한 초과 안전 중단 |

## 산출물 구조

- `experts.bin` — Q2_0 expert blob 스트림 (plane 분리, struct_layouts §3)
- `dense.bin` / `embd.bin` — dense·embedding plane
- `index.txt` — engine 19-field rows + header
- `manifest.json` — `convert <gguf>`는 engine-form, `convert <safetensors> --out`은 p2-form
  (verify가 키로 계약 dispatch, p5-02 AC2) / `conversion.json` — 스트리밍 메타 (RSS/heap peak, sha256)
- `model.gguf` — `--emit-gguf` 산출 엔진 GGUF v3

엔진 실 acceptance는 harness 체인(`strata-gguf` guard + `strata-dequant` CHECKSUM +
`strata_pack` build/verify)가 정본 — 스킬 `polystrata-engine-acceptance` 참조.

## tokenizer 메타 emit + 엔진 API 종점 (p5-09)

`convert --emit-gguf --hf-emit`은 HF 디렉토리에 `tokenizer.json`이 실재하면 GGUF 헤더에
`tokenizer.ggml.*` 12 keys + `tokenizer.chat_template`을 발행한다. 파일 부재 시 기존 경로
유치(메타 0 keys). convert 보고 JSON에 `tokenizer_meta_keys` 필드가 count를 실측한다.
never-invented 게이트: vocab_size guard measure-equal·special text measured table 외 refuse·
merge in-vocab·non-BPE refuse (p5-08 rope gate와 동형).

PolyStrata `serve/server.py`(OpenAI `/v1/chat/completions`) serve 스모크는
`scripts/p5_api_smoke.py`가 정본 (pinned 엔진 트리·CUDA build는 `.local` 전용 — 저장소
커밋 안 함):

```
python scripts/p5_api_smoke.py                    # mini 스모크 — RC 0
python scripts/p5_api_smoke.py --model emitted.gguf --model-dir hf-dir   # 실model — RC 0
```

단계: R3 engine `extract()` tokenizer dir → R4 run config → R5 server `/health` 200 →
R6 POST `/v1/chat/completions` HTTP 200 텍스트 왕복 (engine RSS/Swap/GPU 실측 포함) →
R7 teardown rc=0. 실model 실측: VmRSS 2,467,764 kB(load)→3,789,720 kB(gen), VmSwap 0 kB,
GPU 3,746→3,787 MiB, usage total 51. 정본: [docs/work/p5-09.md](work/p5-09.md).
