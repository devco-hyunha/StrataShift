# Mixtral 실행 클래스 계약 (p6-01 AC1) — 실측 전사

정본: engine pin `c566ff4` 읽기 전용 실측 + Mixtral HF 실측 fixture. 발명 없음 — 모든
line·key·값은 실측 산출. HYPOTHESIS는 발현 조건만 표기.

## 1. `Engine` base hook 집합 (실측 `src/poly/engine.hpp`)

architecture가 공급하는 virtual hook (engine.hpp 실측 line — virtual 선언 순서):

| hook | 실측 line | Mixtral 필요 여부 (HF 실측 구조) |
|---|---|---|
| `load_geometry()` | 275 pure virtual | ✅ (shape 키 전사) |
| `load_layer(int)` | 278 pure virtual | ✅ |
| `recurrent_floats()` | 289 pure virtual | `return 0` (recurrent state 없음) |
| `alloc_caches()` | 298 pure virtual | ✅ (rope KV rings, window 없음) |
| `card_row_bytes()`/`save_card_rows`/`load_card_rows` | 300-302 pure virtual | ✅ |
| `assign_scratch()` | 304 pure virtual | ✅ |
| `widest()` | 306 pure virtual | ✅ |
| `streams()` | 308 pure virtual | `return 1` |
| `layer_head(Layer&, il, n)` | 310 pure virtual | rms → qkv lin → rope → attn → out lin |
| `layer_tail(Layer&, il, n)` | 312 pure virtual | SiLU(gate)·up → down expert 합 |
| `output_rows(first, rows, h)` | 314 pure virtual | final norm → output mat |
| `route(il, logits, token, ids, w, tmp)` | 318 pure virtual | top-2 softmax (§4 전사) |
| `moe_shared(L, il, n)` | 326 pure virtual | early return (shared expert 없음 — HF 실측) |
| `sees()` | 201 default false | 기본 false (vision 없음) |
| `begin_pass`/`check_draft_block`/`load_drafter`/`draft_step` | 279-287 | 미override (drafter 없음) |
| `snaps_spread`/`guess_vocab`/`finish_logits`/`embed_scale`/`extra_bytes`/`load_extra`/`block_attend`/`block_rows`/`head_key`/`turn`/`shift_rows`/`tensor_alias`/`meta_alias` | 292-361 default | 기본 유지 (logit softcap 없음, tie 없음 — `tie_word_embeddings: false` 실측) |

`route_card` (321)는 default — base `route` CPU 경로로 전진.

## 2. `gelu_` activation 경로 (실측)

- `engine.hpp:503`: `bool gelu_ = false;` — "the routed experts' activation: GELU of the
  gate, else SiLU within the layer's limits"
- `engine.cpp:1915,1943,2227,2248`: `gelu_`가 GPU expert rows(`geglu_rows`)와
  `cpu_experts()`에 전파 → base가 SiLU/GELU 양쪽 경로 실재
- Mixtral: HF `hidden_act: "silu"` 실측 (config.json) ⇒ `gelu_ = false` 유지
  (gemma4는 `load_geometry`에서 `gelu_ = true` 설정 — `gemma4.cpp:33-41` 실측)

## 3. `load_moe(il, shared)` 실측 (`engine.cpp` — engine.hpp:384 선언)

- `L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, n_expert_)`
- `shared==true` 시 `ffn_gate_shexp/ffn_up_shexp/ffn_down_shexp` upload — Mixtral은
  **shared expert 없음** (HF fixture: layer key 어휘에 shexp 없음, §5) ⇒ `shared=false`
  경로. engine base 실측 기본은 `shared = true` ⇒ Mixtral `load_layer`는 `load_moe(il, false)`
  명시 호출 (base 기본값과 다름 — 실측 전사)
- expert tensor gate (실측): shape `[n_in, rows, n_expert_]` 3-D 검증,
  `ffn_down_exps.weight` [n_ff_exp, n_embd, n_expert], `ffn_gate_exps.weight`
  [n_embd, n_ff_exp, n_expert] + `ffn_up_exps.weight` 분리형, 또는 fused
  `ffn_gate_up_exps.weight` [n_embd, 2*n_ff_exp, n_expert] — gate/up 타입 상이 시
  throw (실측 상한: "have different types, which this engine does not run" — 우리
  emit의 pack 상수 Q2_0로 동일 타입 강제)
- `L.d_bytes = payload(td)/n_expert_` — expert별 stride 분리 (base가 산출, 아님 발명)

## 4. `route()` 전사 기준 (실측 `gemma4.cpp:586-605` route body)

gemma4는 top-8 + `1/sqrt(n_embd)` scale + `kMinWeightSum` denom + `escale` 곱.
Mixtral HF 실측 규격(`num_experts_per_tok: 2`, gate [8,4096] 실측): plain softmax →
top-2 → sum renorm. 전사 기준 결정:

- base `route` 규약(318 시그니처) 유지, `n_used_ = 2` (shape에서 읽힘 — 발명 아님)
- gemma4의 `1/sqrt(n_embd)` scale: Mixtral HF gate는 plain softmax (transformers
  `BlockSparseMoe` gate 실측 규격) — scale 미적용 → `escale` 없음 (gemma4는 Block에
  `escale` vector 실재; Mixtral fixture에 escale key 없음) → `denom = sum` 경로
- ~~HYPOTHESIS: escale 부재 기본값~~ → **VERIFIED (p6-01, 2026-10-23)**: `escale`은 gemma4 `Block`
  전용 필드이고 base `load_moe`에는 없다. escale 없는 family인 `Qwen35::route`(qwen35.cpp:335-353)는
  `w[k] /= denom`만 하고, `route_card`는 `route_rows(..., escale=nullptr, norm=true)`(qwen35.cpp:355-358).
  Mixtral 클래스는 이 두 함수를 그대로 따른다 (`engine_ext/src/poly/mixtral.cpp`).

## 5. Mixtral HF 실측 fixture (`tests/fixtures/mixtral_header.json`, probe 실측)

`scripts/p6_mixtral_probe.py` 산출 (2026-10-20, HF CDN, header-only — C1):

- config: hidden_size 4096, layers 32, heads 32, kv_heads 8, head_dim 128,
  local_experts 8, experts_per_tok 2, intermediate_size 14336 (moe), rope_theta
  1e6, vocab 32000, silu, sliding_window None, tie false
- shards 19개, total 93,405,713,504 B, index `metadata.total_size` 93,405,585,408 B·
  entries 995 (p4-02 `shard_index.parse_index` gate 형식과 일치 — 실측)
- layer key 어휘 (shard1 51 tensors 실측): `self_attn.{q,k,v,o}_proj`,
  `input_layernorm`, `post_attention_layernorm`, `block_sparse_moe.gate.weight`
  [8,4096], `block_sparse_moe.experts.{0..7}.{w1,w2,w3}.weight` (w1 [14336,4096]
  gate, w2 [4096,14336] down, w3 [14336,4096] up), global `token_embeddings`·
  `norm` (index entries에서)

## 6. emit key 어휘 (engine 소스 실측 기준, AC2 발명 금지 게이트)

`load_moe`가 읽는 key: `ffn_gate_inp.weight`, `ffn_gate_exps/ffn_up_exps/ffn_down_exps.weight`
(분리형), `ffn_gate_up_exps.weight`(fused, 선택). prefix = `general.architecture` 문자열
(우리가 `mixtral`로 emit; engine dispatch 실측 `engine.cpp:808-818` throw line 앞에
`name == "mixtral"` 한 줄 추가 — AC3 scope).

## 7. RoPE 경로 (p6-01 AC3, 실읽기)

- 엔진 `head_norm_rope`(kernels.hpp:149-160)는 NeoX pairing(i, i+n_rot/2)인데 head마다 RMS norm을 **항상**
  한다 (kernels.cu:933-948: `scale = 1/sqrt(ss/dim + eps)`가 `w=null`이어도 곱해짐). Mixtral은 q/k norm이 없어 못 쓴다.
- `dsv4_rope`(kernels.hpp:219-232, kernels_dsv4.cu:111-137)는 `norm=false`면 norm 없이 normal pairing(i, i+1)을
  돌린다. `Rope{theta_scale, freq_scale=1, ext_factor=0, attn_factor=1}`이면 `yarn()`(kernels_dsv4.cu:89-101)이
  `cos(θ)`, `sin(θ)`만 낸다. `at`을 주면 토큰 t의 위치는 `at[2t]` — 엔진 `at_`은 토큰마다 {pos, turn} 두 수라
  `at + 1`을 주면 turn(회전 위치)을 읽는다 (qwen35.cpp:270의 `at + 1`과 같은 뜻).
- HF Mixtral 가중치는 half-split 회전 기준이라, emit이 q/k 행을 head 안에서 `[0, d/2, 1, d/2+1, …]` 순서로
  재배열한다(`meta/mixtral_rope.py`). q와 k에 같은 순열이면 q·k가 그대로다(`tests/test_mixtral_map.py`
  `test_row_order_turns_half_split_into_normal_pairs`가 수치로 확인).

## AC 상태

- AC1 전사 (이 문서) ✅ + fixture ✅ (probe rc=0). AC2(매핑 코드)·AC3(컴파일 게이트)는
  이 계약 문서의 grounds로 발화 가능.
