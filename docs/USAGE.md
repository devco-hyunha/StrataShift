# USAGE

StrataShift CLI 사용법. 모든 플래그 표는 2026-10-18 `--help` 실측 산출 (발명 없음).

## 설치

Python **3.11 이상** (p6-11, `tomllib`). 3.10에서는 `pip install`이 버전 이유로 거절된다.

```
pip install .
stratashift --help
```

런타임 의존은 numpy뿐 — `safetensors`/`torch`는 CONSTRAINTS C1로 금지 (full-load API).

테스트(엔진 연동 포함)를 돌릴 때는 선택 의존성 `test`를 함께 설치한다:

```
pip install -e ".[test]"
pytest
```

`test` = `pytest`, `pyyaml`(핀 gguf-py `metadata.py`), `tqdm`(gguf-py 쓰기 경로), `regex`(엔진
`tools/strata_tokenizer.py`). gguf-py는 pip `gguf`가 아니라 핀 llama.cpp 압축 해제본을 경로로 쓴다
(아래 "엔진 경로").

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
                    [--imatrix GGUF_PATH] [--engine-src DIR] [--yes] [--dry-run]
                    model
```

| 인자 | 의미 |
|------|------|
| `model` | HF `.safetensors` (단일/multi-shard) 또는 engine-form 입력 |
| `--out` | pack 산출 디렉토리 (experts.bin, index.txt, manifest.json, conversion.json) |
| `--max-rss-gb` | RSS 상한 초과 시 안전 중단 (exit 3). RSS는 Linux `VmRSS`, Windows Working Set으로 잰다. 잴 수 없는 OS에서는 시작 전에 거절한다 (exit 2, p6-10) |
| `--emit-gguf` | 엔진 GGUF v3 (Q2_0 tensors)를 이 경로에 추가 산출 |
| `--guard-keys` | 컴파일된 architecture guard keys JSON 프리셋 경로 — never invented (struct_layouts §1) |
| `--hf-emit` | HF naming 입력의 emit 경로 강제. HF naming은 플래그 없이도 fused HF emit 경로로 **자동 감지**된다 (p5-02 AC1, dispatch 연결) |
| `--dense-type` | dense tensor 타입: F32 pass-through (기본) 또는 양자화 프리셋 (p3-03~p3-09) |
| `--imatrix` | calibration 산물(`imatrix.gguf`) consume — expert 자리를 Q4_K(gate/up, h%256=0)/Q4_0(down, be=32 ladder) + dense engine-key lookup. plane unit = expert, vector length = engine row length. opt-in — flag 없으면 plain RTN golden byte 불변. Q2_0는 imatrix를 무시하므로 emit 금지 (p6-05 ruling). plane shape mismatch → refuse exit 3 |
| `--yes` | 없는 프로그램을 묻지 않고 설치 (`STRATASHIFT_YES=1`과 같음). Windows 전용 (p6-12) |
| `--dry-run` | 없는 프로그램과 설치할 것만 보여 주고 끝냄. 변환하지 않음, 종료 코드 0 (p6-12) |
| `--engine-src` | emit에 쓰는 핀 엔진 소스 폴더(`tools/gguf_writer.py`가 있는 곳). 환경 변수보다 먼저. 아래 "엔진 경로" (p6-09) |

실측 예 (HF fused + BF16, mixtral-tiny — 플래그 없이 auto-detection):

```
$ stratashift convert .local/models/mixtral-tiny.safetensors --out QSQ \
    --emit-gguf QSQ/model.gguf
rc=0, rss_peak_bytes=71,454,720, heap_peak_bytes=33,567,636,
slice_heap_peak_bytes=20,758,102   (< 2 GiB gate PASS)
```

## 엔진 경로 (p6-09)

엔진 관련 경로는 `src/stratashift/polystrata/engine_paths.py` 한 곳에서 정한다.
순서: ① 명령줄 옵션 → ② 환경 변수 → ③ 저장소 안 기본값(`.local/`, git에 안 들어감).

| 경로 | ① 옵션 | ② 환경 변수 | ③ 기본값 |
|------|--------|-------------|----------|
| 엔진 소스 | `convert --engine-src` | `STRATASHIFT_ENGINE_SRC` | `.local/polystrata/PolyStrata-main` |
| 엔진 도구 | (`--engine-src`/`tools`) | `STRATASHIFT_ENGINE_TOOLS` | `<엔진 소스>/tools` |
| 엔진 빌드 | — | `STRATASHIFT_ENGINE_BUILD` | `.local/polystrata/build` |
| gguf-py | — | `STRATASHIFT_GGUF_PY` | `<엔진 빌드>/_deps/strata_llamacpp-src/gguf-py`, 없으면 `.local/llama.cpp/llama.cpp-3cf03257…/gguf-py` (엔진 `CMakeLists.txt:1314` 핀) |
| 임시 폴더 | — | `STRATASHIFT_SCRATCH` | 시스템 임시 폴더 아래 `stratashift` |

- `--engine-src`를 주면 `STRATASHIFT_ENGINE_TOOLS`가 있어도 `<옵션>/tools`를 쓴다.
- 엔진 도구를 못 찾으면 emit은 종료 코드 2로 멈추고, 찾아본 경로와 바꿀 옵션·환경 변수를 알린다.
  `inspect`·`convert`(emit 없이)·`verify`는 엔진 없이 동작한다.
- 엔진 실행 파일은 Windows에서 `.exe`를 붙이고 `<빌드>/`, `<빌드>/Release/` 순으로 찾는다.
- 임시 파일 위치는 출력 바이트에 영향이 없다 (p6-09 AC2, 기준 해시 31/31).
- 예전 고정 경로를 쓰던 리눅스(WSL) 환경은 환경 변수만 맞추면 그대로 동작한다:

```bash
export STRATASHIFT_ENGINE_SRC=/opt/data/polystrata-src
export STRATASHIFT_ENGINE_BUILD=/opt/data/polystrata-build
export STRATASHIFT_SCRATCH=/opt/data/cache/scratch
```

## Windows에서 설치·실행 (p6-13)

WSL·도커 없이 PowerShell 또는 명령 프롬프트로 한다. 1~4는 자동 설치가 해 주지만, 실패하면 아래 "자동 설치"의
"손으로 설치하기"를 따른다.

1. Python 3.11 이상: `winget install -e --id Python.Python.3.11` (설치 뒤 새 창)
2. 저장소 폴더에서 설치: `pip install -e ".[test]"` (변환만 할 때는 `pip install -e .`)
3. 변환: `stratashift convert <model> --out <dir> --emit-gguf <out.gguf>` — 핀 엔진 소스가 없으면 한 번 묻고 받는다
4. 엔진 실행 파일 빌드
   - `stratashift build-engine`: CPU용 `strata-gguf.exe`·`strata-dequant.exe` (CUDA 필요 없음)
   - `stratashift build-engine --gpu`: `strata-poly.exe`까지 (CUDA 13.0 필요, `--cuda-arch`를 안 주면 `nvidia-smi`에서 읽음)
   - 핀 llama.cpp 소스·CMake·VS 2022 Build Tools(C++)·CUDA가 없으면 한 번 묻고 설치한다.
     `--dry-run`으로 무엇을 설치하고 어떤 명령으로 빌드할지 먼저 볼 수 있다.
   - 실행 파일은 `.local/polystrata/build/`(`STRATASHIFT_ENGINE_BUILD`)에 생긴다.
   - `stratashift build-engine --gpu --ext`: 엔진에 없는 아키텍처(지금은 `mixtral`)를 얹은 우리 실행 파일
     `strata-shift-poly.exe` (p6-01). 핀 엔진을 `.local/polystrata/ext-src`로 복사해 `engine_ext/`를 넣고
     `.local/polystrata/build-ext`에서 빌드한다. 공식 `strata-poly.exe`는 그대로다. 자세한 것은 `engine_ext/README.md`.
5. GPU 실행: CUDA 13 DLL은 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64`에 있다.
   CUDA를 막 설치했다면 **새 창**을 열어 PATH를 반영한다(안 하면 `cudart64_13.dll`을 못 찾는다). 예:
   `.local\polystrata\build\strata-poly.exe --model mini.gguf --tokens 1,2,3 --max-new 1 --max-context 1024 --vram-reserve-mib 512`
6. 테스트: `pytest`. 엔진 체인 테스트(`test_engine_smoke` 등)는 700 MB fixture를 변환하므로 파일 하나에 20분 넘게 걸릴 수 있다.

이 PC 실측(2026-10-22, RTX 3050 Ti Laptop 4 GB, 드라이버 595.79): CPU 빌드 46초, CUDA 설치 587초,
`strata-poly` mini fixture rc 0(0.6초 로드, expert 12/12 VRAM, 프롬프트 59.1 tok/s).
시스템 설정(페이지 파일·large pages)은 자동으로 바꾸지 않는다. 필요하면 사용자가 직접 바꾼다.

리눅스(WSL)에서 예전 고정 경로를 쓰던 환경은 위 "엔진 경로"의 환경 변수 3개만 맞추면 된다.

## 자동 설치 (p6-12)

엔진이 필요한 명령을 실행하면 필요한 것이 있는지 먼저 확인한다. 없으면 무엇을 설치할지(크기·시간·관리자 권한)
보여 주고 **한 번 묻고** 설치한 뒤 다시 확인한다. 엔진이 필요 없는 명령(`inspect`, emit 없는 `convert`, `verify`)은
아무것도 설치하지 않는다.

| 명령 | 필요한 것 | 자동 설치 방법 |
|------|-----------|----------------|
| `convert --emit-gguf` | 핀 엔진 소스 (`c566ff4`) | GitHub 커밋 zip을 받아 **내용 해시**(`905d413e…`, 파일 3,261개)를 확인하고 `.local/polystrata/PolyStrata-main`에 푼다. 다르면 받은 파일을 지우고 멈춘다 |
| `build-engine` / `build-engine --gpu` (p6-13) | 핀 엔진 소스, 핀 llama.cpp 소스(ggml), CMake, VS 2022 Build Tools(C++) / + CUDA 13.0 | `winget install -e --source winget ... --id Kitware.CMake` / `Microsoft.VisualStudio.2022.BuildTools` / `Nvidia.CUDA --version 13.0` (엔진 `setup.py`와 같은 인자) |

- `--dry-run`: 설치하지 않고 계획만 본다. `--yes`(또는 `STRATASHIFT_YES=1`): 묻지 않는다.
  입력이 터미널이 아니면 묻지 못하므로 `--yes` 없이는 멈춘다 (종료 코드 2).
- 이미 있는 것은 다시 설치하지 않는다. 설치가 끝나면 winget 종료 코드가 아니라 다시 찾아서 확인한다.
  PATH가 아직 안 바뀌었으면 새 창에서 다시 실행한다.
- 자동 설치는 저장소 안 기본 위치(`.local/`)에만 한다. `--engine-src`나 환경 변수로 준 경로가 없으면 설치하지 않고 멈춘다.
- 리눅스(WSL)는 자동 설치하지 않는다. 없는 것과 손으로 설치하는 방법만 알린다.
- winget이 없으면 Microsoft Store의 "앱 설치 관리자(App Installer)"를 설치하거나 아래처럼 손으로 설치한다.

손으로 설치하기:

- 핀 엔진 소스: `https://github.com/VecSzn/PolyStrata/archive/c566ff4787c8ab188ecf33c6f1907a9f90d9f152.zip`을 받아
  `.local/polystrata/PolyStrata-main`으로 푼다(zip 안의 최상위 폴더 이름만 바꾼다). 백업: HF Space `VecSzn/PolyStrata`
  — 내용 해시가 핀과 같을 때만 쓴다.
- CMake: `winget install -e --id Kitware.CMake` 또는 cmake.org 설치 파일
- VS 2022 Build Tools: "C++를 사용한 데스크톱 개발" 워크로드
- CUDA 13.0: https://developer.nvidia.com/cuda-downloads (드라이버 580 이상, 엔진 `setup.py:199`)

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
- 제품 정직성: Qwen1.5-MoE·Mixtral 아키텍처 클래스는 공식 엔진에 실재하지 않는다
  (`src/poly/` grep 0건, p4-03 실측 load 거부) — 변환 산출물의 엔진 실 acceptance는
  reference verifier(harness)가 정본. 엔진 실행까지의 acceptance는 GPU·인간 게이트.
- Mixtral은 우리 실행 파일 `strata-shift-poly`(`build-engine --gpu --ext`, `engine_ext/`)에서 실행한다 (p6-01).
  emit은 guard `architecture: mixtral`로 HF 이름(`block_sparse_moe.*`)을 엔진 이름으로 바꾸고, q/k 행을
  엔진 회전 방식에 맞게 재배열한다. 실측은 mini fixture RUN뿐이고 8x7B 실모델은 AC5(미실측), 토크나이저 emit은 아직 없다.

## exit codes

| code | 의미 |
|------|------|
| 0 | 성공 |
| 1 | 런타임 실행 (파일 부재·엔진 가드 FAIL 등, stderr 한 줄 메시지) |
| 2 | usage error (argparse), 측정 불가 OS에서 `--max-rss-gb` (p6-10), emit에 엔진 도구 없음 (p6-09), 필요한 프로그램 설치 거절·실패 (p6-12) |
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
