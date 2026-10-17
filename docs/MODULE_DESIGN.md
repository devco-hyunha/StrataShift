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
│       │   └── main.py            # convert / inspect / verify
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── safetensors_header.py
│       │   ├── shard_index.py      # HF multi-shard index.json + shard 헤더 해석 (p4-02)
│       │   └── gguf.py            # 후속
│       ├── meta/
│       │   ├── __init__.py
│       │   ├── expert_map.py
│       │   ├── hf_gguf_map.py     # HF naming → engine GGUF 키/shape 매핑 (layout.cpp:60-110, p4-01)
│       │   └── manifest.py
│       ├── stream/
│       │   ├── __init__.py
│       │   ├── mmap_reader.py
│       │   └── block_writer.py
│       ├── quant/
│       │   ├── __init__.py
│       │   └── pack.py            # docs/rea/bitpack_rules 반영
│       ├── polystrata/
│       │   ├── __init__.py
│       │   ├── headers.py         # docs/rea/struct_layouts 반영
│       │   ├── alignment.py
│       │   ├── index.py
│       │   └── strp.py            # STRP expert-profile writer/parser (struct_layouts §5)
│       ├── monitor/
│       │   ├── __init__.py
│       │   └── rss.py
│       └── feedback/              # Phase 3
│           ├── __init__.py
│           └── log_ingest.py
├── scripts/
│   ├── poc_parse_header.py        # Phase 1 PoC
│   └── export_rea_notes.md        # REA → 에이전트 전달용 메모 템플릿
└── tests/
    ├── test_header_parse.py
    ├── test_no_full_load.py
    ├── test_pack_matches_rea.py   # Phase 2: 패킹 단위 벡터
    └── test_stream_roundtrip.py
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
stratashift inspect  <model.safetensors> [--json]
stratashift convert  <model> --out <dir> --preset <name> [--max-rss-gb N]
stratashift verify   <out_dir>   # Phase 3: PolyStrata 로드 스모크
```

---

## 3. 모듈 간 의존 방향

```
docs/rea  ──(인간/에이전트)──▶  polystrata / quant 상수
cli → ingest / meta / stream / quant / polystrata / monitor
stream → ingest (offsets only)
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
