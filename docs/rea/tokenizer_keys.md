# tokenizer metadata key contract (p5-09, pin c566ff4)

GGUF 헤더의 `tokenizer.ggml.*` 키 어휘 — 엔진 읽기 코드 실측 + reference GGUF 헤더
실측으로 고정. emit 구현: `src/stratashift/meta/hf_gguf_map.py::build_tokenizer_meta`,
emit 연결: `src/stratashift/cli/main.py::_cmd_emit_gguf_hf` (2026-10-20).

## 1. 엔진 읽기 계약 (SOURCE, pin c566ff4787c8ab188ecf33c6f1907a9f90d9f152)

- `tools/strata_tokenizer.py:175-190` `from_gguf`:
  - 필수 키 `tokenizer.ggml.tokens`, `tokenizer.ggml.merges` — 부재 시 raise
    "GGUF is missing %s; it does not carry a tokenizer"
  - `tokenizer.ggml.model` ∈ `MODELS` = {"gpt2": None, "gemma4": "gemma4"} (line 106)
    — model 문자열이 pre-tokenizer를 함의; `tokenizer.ggml.pre`는 model이 key에
    없는 경우의 fallback (line 190 `md.get("tokenizer.ggml.pre", "qwen35")`)
  - special ids: `tokenizer.ggml.`로 시작하고 `_token_id`로 끝나는 모든 키 전수 수집
    (line 186-187)
  - `tokenizer.ggml.token_type` (line 189): 3 = CONTROL(`parse_special`일 때만
    매칭), 4 = USER_DEFINED(항상 매칭) — `__init__` docstring 138-146 실측
    (oracle 왕복 3건 차이로 classes 분리 확정), 6 = BYTE token (`token_bytes`
    410-411: `<0xNN>` len 6 piece는 16진수 byte로 decode)
- `src/poly/main.cpp:915-916`: `--eos-ids` 부재 시 `tokenizer.ggml.eos_token_id`
  / `tokenizer.ggml.eot_token_id` / `tokenizer.ggml.eom_token_id`를 stop id로 읽음
- serve 읽기 경로 (p5-09 전제 실측): `serve/models.py:115-119` tokenizer dir에
  `vocab.json` 없으면 모델 unavailable; `serve/server.py:6853-6858`
  `load_tokenizer` → `Tokenizer.from_dir`; `extract()`(strata_tokenizer.py:432+)는
  GGUF 헤더에서 tokenizer dir을 만들어 쓴다.

## 2. reference GGUF 헤더 실측 (VERIFIED, 2026-10-20)

HF `unsloth/gemma-4-26B-A4B-it-GGUF`의 `gemma-4-26B-A4B-it-Q8_0.gguf` 선두
40 MB를 Range 요청(206, bytes 0-41943039/26859861728)으로 확보하고 엔진
`tools/gguf_reader.py`(read-only)로 parse:

| key | type | 값 (실측) |
|---|---|---|
| `tokenizer.ggml.model` | string | `gemma4` |
| `tokenizer.ggml.tokens` | array:string | 262,144 — HF `tokenizer.json` vocab id-map과 **완전 일치** |
| `tokenizer.ggml.merges` | array:string | 514,906 — HF merges(pair `["a","b"]`)와 `"a b"` 표기로 **완전 일치**, dedupe 0건, parts에 space 0건 |
| `tokenizer.ggml.scores` | array:f32 | 전수 `-1000.0` |
| `tokenizer.ggml.token_type` | array:i32 | {3:16, 1:261865, 4:7, 6:256} — byte token 256 = type 6 |
| `tokenizer.ggml.bos_token_id` | i32 | 2 |
| `tokenizer.ggml.eos_token_id` | i32 | **106** (`<turn|>`) |
| `tokenizer.ggml.mask_token_id` | i32 | 4 |
| `tokenizer.ggml.padding_token_id` | i32 | 0 |
| `tokenizer.ggml.unknown_token_id` | i32 | 3 |
| `tokenizer.ggml.add_bos_token` | bool | True |
| `tokenizer.ggml.add_space_prefix` | bool | False |
| `tokenizer.chat_template` | string | 18,922 B — HF `chat_template.jinja`(18,681 B)과 36 diff 행(converter 변환 산물; 우리 emit은 HF 원문) |

`tokenizer.ggml.pre` 키 **없음** — 모델 문자열이 pre를 함의하는 계약 (§1)과 정합.

**special id별 token_type 실측 전사** (reference 헤더, id별 판독 →
`SPECIAL_TYPE_MEASURED`로 emit; 형상 규칙 발명 없음):
`<pad>`0=3, `<eos>`1=**1(NORMAL)**, `<bos>`2=3, `<unk>`3=3, `<mask>`4=3,
`<|tool>`46=3, `<tool|>`47=3, `<|tool_call>`48=4, `<tool_call|>`49=4,
`<|tool_response>`50=4, `<tool_response|>`51=4, `<|"|>`52=4, `<|think|>`98=3,
`<|channel>`100=4, `<channel|>`101=4, `<|turn>`105=3, `<turn|>`106=3,
`<|image>`255999=3, `<|audio>`256000=3, `<|image|>`258880=3, `<|audio|>`258881=3,
`<image|>`258882=3, `<audio|>`258883=3, `<|video|>`258884=3.
GGUF eos는 turn 종결 id(106)이고 `<eos>` piece 자체는 NORMAL — 즉 엔진 GGUF는
turn 종결을 eos로 선언한다.

## 3. HF 체크포인트 실측 (VERIFIED, 2026-10-20, AC1 다운로드 `google/gemma-4-26B-A4B-it`)

- `tokenizer.json` 32,169,626 B: model.type=BPE, vocab 262,144 (id 0..n-1 완전
  커버리지 실측 True), merges 514,906 pairs, added_tokens 24건 (전부 vocab에
  존재 실측), byte tokens `<0xNN>` 256건, single-char piece 19,227건
  (ASCII 32..126 커버 94/95 — space는 SentencePiece `▁`로 표기; engine raw
  encoder `_encode_raw` 284-295의 byte fallback은 piece 단위이므로 vocab의
  single-char 커버리지가 round-trip 전제), normalizer Replace `" "→▁`,
  pre_tokenizer Split(`MergedWithPrevious`), decoder Sequence(Replace `▁→" "` +
  ByteFallback)
- `tokenizer_config.json` 3,082 B: `bos_token=<bos>`(id 2), `eos_token=<eos>`(1),
  `eot_token=<turn|>`(106), `mask_token=<mask>`(4), `pad_token=<pad>`(0),
  `unk_token=<unk>`(3)
- `chat_template.jinja` 18,683 B (BOM/CRLF 포함 헤딩 실측)

## 4. emit 규칙 (our builder gate — never-invented)

- 파일 실재 게이트: `tokenizer.json` 없으면 tokenizer emit 생략 (engine API
  unavailable — `strata-poly --tokens` RUN은 tokenizer 불필요, p5-08 AC4 실측)
- vocab id 0..n-1 완전 커버리지 게이트 (실측 True가 전제), guard `{arch}.vocab_size`
  가 있으면 measure-equal 아니면 refuse
- merge pair는 vocab 안에 있어야 한다 (gate refuse; engine `__init__` 168-170과
  동형 — raw(pre=gemma4) vocab는 merge 통과를 허용하나 우리 게이트는 실측 정합만 허용)
- unmeasured special(text가 `SPECIAL_TYPE_MEASURED`에 없음) refuse
- non-BPE model type refuse
- emit 키 set = §2 reference 실측 그대로 (pre 없음), 값 타입: tokens/merges =
  array:string, scores = array:f32(-1000.0), token_type = array:i32, ids = i32,
  add_bos/add_space_prefix = bool, chat_template = string(HF 원문)
- tensor payload byte·offset 불변(메타만 증가) — 왕복 테스트가 byte 비교로 강제
  (`tests/test_tokenizer_emit.py`)

## 5. parser 합성 (our side)

`src/stratashift/ingest/gguf.py`가 array-of-string(elem type 8) decode 경로를
가지게 됐다 (2026-10-20: tokenizer emit GGUF의 우리 parser round-trip이 KeyError 8로
잡힌 것을 같은 작업에서 수정 — 엔진 writer `gguf_writer.py:130-136` array:string
encoding과 동형 decode).
