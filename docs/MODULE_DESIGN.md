# StrataShift 모듈·디렉터리 설계

Phase 0 REA 산출물과 Phase 1+ 구현이 공유하는 디렉터리·책임 경계.  
**폴더를 어디에 두는지**는 [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md).  
파일·함수를 **언제·어떻게** 만드는지는 [CODING_STANDARDS.md](CODING_STANDARDS.md).

---

## 1. 제안 디렉터리 트리

```text
StrataShift/
├── README.md
├── AGENTS.md
├── pyproject.toml                 # Phase 1에서 추가
├── docs/
│   ├── PRODUCT_SPEC.md
│   ├── ARCHITECTURE.md
│   ├── REA_GHIDRA.md              # Ghidra 절차
│   ├── ROADMAP.md
│   ├── CONSTRAINTS.md
│   ├── REFERENCES.md
│   ├── MODULE_DESIGN.md
│   ├── ACCEPTANCE.md
│   ├── polystrata_spec_draft.md   # Phase 0~1에서 작성
│   └── rea/                       # Phase 0 Ghidra 산출물
│       ├── README.md
│       ├── binary_pin.md
│       ├── loader_entrypoints.md
│       ├── struct_layouts.md
│       ├── bitpack_rules.md
│       └── paging_model.md
├── src/
│   └── stratashift/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   └── main.py            # argparse + 명령 분기 + 종료 코드 (p6-08: 로직은 아래 영역으로 이동)
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── safetensors_header.py  # + `numel(shape)` (p6-08)
│       │   ├── shard_index.py      # HF multi-shard index.json + shard 헤더 해석 (p4-02)
│       │   └── gguf.py            # 후속
│       ├── meta/
│       │   ├── __init__.py
│       │   ├── expert_map.py
│       │   ├── hf_gguf_map.py     # HF naming → engine GGUF 키/shape 매핑 (layout.cpp:60-110 p4-01; Gemma4 어휘 GEMMA4_LAYER/GLOBAL_DENSE + arch 파라미터, gemma4.cpp 정본 p5-06) + tokenizer 메타 빌더 `load_hf_tokenizer`/`build_tokenizer_meta` (p5-09: HF tokenizer.json 실측 → `tokenizer.ggml.*` GGUF 키; SPECIAL_TYPE_MEASURED 표 = reference GGUF 헤더 id별 실측)
│       │   ├── hf_emit_order.py   # HF→engine emit 순서 표 (데이터만, p6-08 cli에서 이동)
│       │   ├── gemma4_rope.py     # `gemma4_rope_freqs` — rope_freqs.weight 발행 값 (엔진 gemma4.cpp:207 rope gate, p5-08; p6-08 이동)
│       │   ├── imatrix.py         # imatrix streaming accumulator + GGUF loader (p6-03 AC2 — `in_sum2`/`in_sum`/`counts` additive planes, imatrix.cpp:655-664 전사; expert usage normalization arxiv 2406.08155 §3.3; `load_imatrix_stats`는 calibration product GGUF 단일 tensor slice mmap read — model body 아님, C1-safe)
│       │   └── manifest.py        # index.txt/manifest.json writer + `sha256_file` (p6-08)
│       ├── stream/
│       │   ├── __init__.py
│       │   ├── block_writer.py
│       │   ├── reader.py          # SliceReader/TensorReader — seek+read(mmap 아님, OOM RC=137 이력), p6-08
│       │   ├── convert_pack.py    # `convert --out` Q2_0 pack (safetensors / GGUF), p6-08
│       │   ├── emit_common.py     # 엔진 도구 확인, GGUF flush-at-EOF 마무리
│       │   ├── emit_gguf.py       # 단일 파일 emit (p3-02~p3-09 dense gate, p6-03 imatrix)
│       │   ├── emit_hf.py         # HF emit 오케스트레이션 (입력 열기·guard·metadata·레이어 루프·마무리)
│       │   ├── emit_hf_ctx.py     # EmitContext/ArchTables — 단계 함수가 공유하는 상태
│       │   ├── emit_hf_dense.py   # dense·fused attn_q/GDN·1-D·globals·PLE·rope 단계
│       │   └── emit_hf_experts.py # fused gate_up/down·per-expert 단계 (형식 선택은 quant/select.py만)
│       ├── quant/
│       │   ├── __init__.py
│       │   ├── pack.py            # docs/rea/bitpack_rules 반영
│       │   ├── dtype.py           # `to_f32` — BF16/F16 → F32 (순수, p6-08 이동)
│       │   ├── select.py          # 형식 이름+양자화 함수 선택 단일 지점: `select_dense_quant`·`select_pack_quant`·`select_expert_quant`(엔진 load 게이트 실패 시 None → emit 거부) (p6-08 AC5)
│       │   └── pack_imatrix.py    # imatrix-consuming producers (p6-03 AC3/p6-05 — Q4_K `quantize_row_q4_K_impl` + Q4_0 `quantize_row_q4_0_impl` 전사, `make_qx_quants`/`make_qkx3_quants`/`make_qp_quants` weighted search, `plane_tile` chunk-aligned plane slice; I/O 없음, dequant 왕복 gate)
│       ├── polystrata/
│       │   ├── __init__.py
│       │   ├── headers.py         # docs/rea/struct_layouts 반영
│       │   ├── alignment.py
│       │   ├── index.py
│       │   ├── strp.py            # STRP expert-profile writer/parser (struct_layouts §5)
│       │   ├── engine_paths.py    # 엔진 소스·도구·빌드·gguf-py·임시 폴더 경로 단일 지점: `resolve_engine_paths(engine_src)`(옵션→환경 변수→`.local/` 기본값)·`engine_binary`(`.exe`, `Release/`)·`scratch_dir`·`describe_missing` (p6-09)
│       │   ├── engine_tools.py    # 핀된 엔진 tools(gguf_writer) 늦은 import — `load(engine_src)` 첫 호출 때 한 번, `available()` (p6-08 이동, p6-09 지연 로드)
│       │   ├── gguf_stream_writer.py # StreamingGGUFWriter (p4-01 AC5, p6-08 이동)
│       │   └── verify_pack.py     # `verify` 자체 점검 (p2 / engine manifest, p6-08 이동)
│       ├── monitor/
│       │   ├── __init__.py
│       │   ├── rss.py             # `process_rss_bytes(pid)`/`current_rss_bytes()` — Linux VmRSS·Windows Working Set(ctypes), 측정 불가면 None (p6-10)
│       │   ├── heap.py            # `heap_gate` tracemalloc start/stop 짝 (p5-02 AC4, p6-08 이동)
│       │   └── sampler.py         # RSS+heap 샘플·`--max-rss-gb` 한도 판정 (p6-08)
│       ├── install/               # p6-12: 실행할 때 없는 프로그램 확인 → 한 번 묻기 → 설치 → 다시 확인 (I/O 전용)
│       │   ├── __init__.py
│       │   ├── plan.py            # 항목(엔진 소스·CMake·MSVC·CUDA)·묶음(`emit`, `engine-cpu`, `engine-gpu` — p6-13)·계획 문구·winget 명령 (순수)
│       │   ├── probe.py           # `find_cmake`·`find_vcvars`(vswhere)·`find_nvcc`(버전)·`probe(keys, paths)` (엔진 setup.py 탐색 순서)
│       │   ├── fetch.py           # 핀 엔진 소스 내려받기, 내용 해시(`ENGINE_TREE_SHA256`) 검증, 안전한 압축 풀기
│       │   ├── build.py           # `build-engine`: vcvars64 → CMake(Ninja) 구성·빌드, 한 번 재시도, `gpu_arch`(nvidia-smi) (p6-13)
│       │   └── ensure.py          # `ensure_need(need, paths, yes, dry_run)` → None(계속)/0(dry run)/2(거절·실패)
│       └── feedback/              # Phase 3
│           ├── __init__.py
│           └── log_ingest.py
├── scripts/
│   ├── poc_parse_header.py        # Phase 1 PoC
│   └── export_rea_notes.md        # REA → 에이전트 전달용 메모 템플릿
└── tests/
    ├── test_header_parse.py
    ├── conftest.py                # p6-09: 엔진 tools/ref를 sys.path에 한 번, `engine_paths` fixture
    ├── test_engine_paths.py       # p6-09 AC1: 옵션→환경 변수→기본값 순서, `.exe`, `Release/`
    ├── test_build_engine.py       # p6-13: 빌드 명령 조립·재시도·dry-run·CLI (실행은 가짜)
    ├── test_install.py            # p6-12: 계획·묻기·dry-run·내용 해시·winget 없음·리눅스 (winget·네트워크는 가짜)
    ├── test_rss.py                # p6-10: RSS 측정·None 샘플·`--max-rss-gb` 측정 불가 거절
    ├── test_text_encoding.py      # p6-11 AC2: 텍스트 I/O `encoding` 누락 0 (AST 검사)
    ├── test_select_quant.py       # p6-08 AC5: 형식 이름·양자화 함수 짝, 사다리 경계
    ├── test_no_full_load.py
    ├── test_pack_matches_rea.py   # Phase 2: 패킹 단위 벡터
    ├── test_stream_roundtrip.py
    └── test_packaging.py          # p5-01: pyproject contract + 실 설치 게이트 + C1 import graph
```

---

## 2. 핵심 인터페이스 (초안)

### 2.1 REA 규격 로드

```python
# stratashift.polystrata (constants sourced from docs/rea)

interface PolyStrataSpec:
    version: str
    binary_sha256: str
    header_layout: dict
    alignment: int
    bitpack: dict          # shifts, masks, width
    paging: dict

const load_spec = (rea_dir: str) -> PolyStrataSpec
# MUST: 출처 경로·핀을 매니페스트에 남김
```

### 2.2 헤더 파싱

```python
# stratashift.ingest.safetensors_header

interface TensorMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]  # begin, end

interface ModelMeta:
    path: str
    tensors: list[TensorMeta]
    header_size: int

const parse_header = (path: str) -> ModelMeta
# MUST: 헤더만 읽고 가중치 본문을 복사하지 않음
```

### 2.3 Expert 맵

```python
# stratashift.meta.expert_map

interface ExpertRef:
    layer_id: int
    expert_id: int
    tensor_names: list[str]
    offsets: list[tuple[int, int]]

const build_expert_map = (meta: ModelMeta) -> list[ExpertRef]
```

### 2.4 스트리밍 변환

```python
# stratashift.stream / quant / polystrata

const iter_expert_views = (path: str, experts: list[ExpertRef])
const quantize_and_pack = (view, spec: PolyStrataSpec) -> bytes
const write_weight_block = (fd, packed: bytes, header) -> int
const write_index_entry = (fd, expert: ExpertRef, offset: int) -> None
```

### 2.5 CLI

```text
stratashift inspect  <model.safetensors | shard dir> [--json]
stratashift convert  <model | shard dir> --out <dir> --preset <name> [--max-rss-gb N]
stratashift convert  <model | shard dir> --out <dir> --emit-gguf <out.gguf> --hf-emit [--guard-keys <preset.json>]
stratashift verify   <out_dir>   # Phase 3: PolyStrata 로드 스모크
```

dir 입력은 multi-shard(p4-02 `ingest/shard_index.py`). `--hf-emit`은 HF naming→engine
GGUF 매핑(p4-01 `meta/hf_gguf_map.py`). guard keys는 모델 디렉토리의 `config.json`에서
산출: `head_dim` = `hidden_size // num_attention_heads` (명시 `head_dim` 키 우선),
`embedding_length`/`block_count`/`expert_count`/`expert_feed_forward_length`는 tensor
shape·layer 수 산출 — 미측정 키는 가저하지 않는다 (never invented).

---

## 3. 모듈 간 의존 방향

```
docs/rea  ──(인간/에이전트)──▶  polystrata / quant 상수
cli → ingest / meta / stream / quant / polystrata / monitor
stream → ingest / meta / quant / polystrata / monitor  (emit 단계 함수가 읽기→양자화→쓰기를 잇는다)
stream 은 형식 선택을 하지 않는다 — 타입 이름과 양자화 함수는 `quant.select` 한 곳에서 짝으로 받는다
quant → polystrata.spec (패킹 규칙만)
feedback → polystrata / quant + docs/rea 재대조  (Phase 3)
```

순환 의존 금지. `ingest`가 `quant`를 import하지 않는다.  
Ghidra 자체는 Python 패키지 의존이 아니다.

---

## 4. 테스트 전략

| 테스트 | Phase | 목적 |
|--------|-------|------|
| REA 문서 존재·핀 필드 | 0 | 산출물 게이트 |
| 헤더 파싱 단위 | 1 | shape/offset 정확성 |
| anti-full-load | 1+ | 금지 API·피크 RSS |
| pack matches REA 벡터 | 2 | 시프트/마스크 회귀 |
| RSS 평탄 | 2 | 모니터 로그 |
| 런타임 주입 | 3 | PolyStrata 로드 성공 |

---

## 5. 패키징

- 빌드: `pyproject.toml` + src layout
- 최소 의존 (예정): `numpy`, (선택) safetensors 헤더 유틸 — **전체 load API 사용 금지**
- GPU·Ghidra는 기본 pip 의존에 넣지 않음
