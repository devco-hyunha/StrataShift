# arch_families — PolyStrata 아키텍처 레지스트리·Gemma 4 처리 경로

Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` (GitHub `main`).
모든 항목은 **핀된 소스 인용** (`SOURCE:`)이다. Ghidra 디컴파일 산출물은 없음 — 이 주제는
소스 계층(`src/poly`, `include/strata/artifact`) 전용이며, 바이너리 로드 검증은 p4-03
reference verifier로 이미 수행된 범위(엔진 guard 밖 geometry는 load 거부)에 포함되지 않는다.

## 1. 엔진 아키텍처 레지스트리 (실행 계층)

`SOURCE: src/poly/engine.cpp:808-816` — `Engine::open()`은 GGUF의 `general.architecture`
문자열로 실행 클래스를 dispatch한다:

| `general.architecture` | 클래스 | 근거 |
|------------------------|--------|------|
| `glm5-next` / `glm5next` | `Glm5Next` | `engine.cpp:814` |
| `qwen35moe` | `Qwen35` | `engine.cpp:815` |
| `gemma4` | `Gemma4` | `engine.cpp:816` |
| `deepseek4` | `DeepSeek4` | `engine.cpp:817` |
| 그 밖의 이름 | `throw` — "is a '<name>' model, which this engine does not run" | `engine.cpp:818` |
| `mixtral` | `Mixtral` — **우리 실행 파일 `strata-shift-poly`에만** (공식 `strata-poly`는 throw 그대로) | `engine_ext/overlay.json` (p6-01) |

클래스는 `Engine` 파생 hook 집합으로 architecture를 구현한다
(`docs/BACKENDS.md` §143-158 — pin 소스 tree 실재): `load_geometry()`, `load_layer(il)`,
`load_moe()`, `route()`, `moe_shared()`, `layer_head()/layer_tail()`, `block_attend()`,
`sees()/whole()` (vision), `output_rows()`, `finish_logits()`, `embed_scale()`,
`recurrent_floats()`, `alloc_caches()`, `assign_scratch()`, `widest()`, `streams()`.
`src/poly/gemma4.hpp`의 `Gemma4 final : public Engine`이 그 실례다 (`SOURCE: src/poly/gemma4.hpp:19-51`).

**핵심**: 실행 지원은 "프리셋 값 등록"이 아니라 **architecture 클래스가源码에 compiled된
것**이 전제다. 소스 tree에 `gemma4.cpp` 36 hits·`glm5next.cpp` 33·`qwen35.cpp` 30가 실재
하고, Qwen1.5-MoE·Mixtral 아키텍처 클래스는 실재하지 않는다.

## 2. architecture prefix 키读取 (실행 계층은 shape를 파일에서 읽는다)

`SOURCE: src/poly/engine.cpp:646-653` — `Engine::meta_in(key)`은
`general.architecture`의 값 + `"."` prefix로 GGUF 메타데이터를 읽는다. 즉 키 prefix는
architecture 문자열 자체이며, 하드코딩된 `qwen4exp.`이 아니다.

`SOURCE: src/poly/engine.cpp:674-697` — `load_shape()`: `block_count`,
`embedding_length`, `vocab_size`, `expert_count`, `expert_used_count`,
`expert_feed_forward_length`, `expert_shared_feed_forward_length`,
`nextn_predict_layers`, `attention.layer_norm_rms_epsilon`를 **파일에서** 읽는다
(`shape_i`/`shape_f` 경유, prefix는 arch 값).

`Gemma4::load_geometry()`가 추가读取하는 키 (`SOURCE: src/poly/gemma4.cpp:48-82`):
`attention.sliding_window`, `attention.key_length`, `attention.key_length_swa`,
`rope.freq_base`, `rope.freq_base_swa`, `final_logit_softcapping`,
`attention.sliding_window_pattern` (must), `attention.head_count`,
`attention.head_count_kv` (per-layer 배열 허용 — `of_layer`),
`attention.shared_kv_layers` (must = 0), `embedding_length_per_layer_input` (must = 0),
`attention.value_length`, `attention.value_length_swa` (key/value 길이 일치 요구).

Gemma4의 layer weight tensor 어휘 (`SOURCE: src/poly/gemma4.cpp:197-211`,
`gemma4.cpp` 키 문자열 전수 실측):
`attn_norm.weight`, `attn_q.weight`, `attn_k.weight`, `attn_v.weight`,
`attn_output.weight`, `attn_q_norm.weight`, `attn_k_norm.weight`,
`ffn_norm.weight`, `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` (dense MLP)
+ routed experts `ffn_gate_exps.scale`, `ffn_up_exps.scale`, `ffn_down_exps.scale`,
`ffn_gate_inp.weight/scale`.

`SOURCE: src/poly/gemma4.hpp:3-9` (주석): 한 residual stream, 6 layer 중 5는 sliding
window + ring KV, 6번째는 full attention·wide heads; FFN = dense GELU MLP + routed GELU
experts; logits tanh cap; drafter는 별도 파일의 model (`gemma4-assistant`).

## 3. artifact 계층 guard와 실행 계층의 분리 (p4-01/03 정정)

`SOURCE: include/strata/artifact/gguf_reader.hpp:796-812`:

- `check_architecture(f)`: `model_family_from_arch`로 `qwen35moe`는
  `check_architecture_qwen35moe` (:607)로 키 전수 검증; 그 밖의 family는
  `read_geometry` (:639)가 `arch != "qwen4exp"`를 **name으로 거부**
  ("architecture is 'X', this engine requires 'qwen4exp'"), qwen4exp는 compiled
  `kCompiledGeometry`와 파일 값 교차검증.
- `check_engine_architecture(g)` (:809-812): `check_architecture` 후
  `model_family_not_supported(model_family(g))` (`model_family_runs` = Qwen4Exp 전용).

정리:

| 계층 | qwen4exp | qwen35moe | gemma4 | Qwen1.5-MoE / Mixtral |
|------|---------|-----------|--------|----------------------|
| artifact guard (`strata-gguf` 등 도구) | 통과 | 통과 (`check_architecture_qwen35moe`) | 거부 (name) | 거부 (name) |
| 실행 레지스트리 (`Engine::open`) | Qwen4Exp 실행 | Qwen35 실행 | Gemma4 실행 | throw (클래스 부재) |

**`HYPOTHESIS:` 없음 — 전항 소스 인용.** p4-01 §근거 2·p4-03 AC3의 표현
("compiled 프리셋 한 곳의 값")은 이 표의 artifact 계층에 한정된 정본으로 읽을 것.
실행의 전제조건은 architecture 클래스 compiled이며, geometry shape는 실행 계층에서
파일로부터 자유读取된다 (`engine.cpp:674-697`, `BACKENDS.md:184-186`).

## 4. Gemma 4가 동작하는 방식 (엔진 README·BACKENDS 실측)

- Gemma 4 26B-A4B는 `setup.sh --family gemma4`로 Unsloth UD 양자화 GGUF를 HF에서
  다운로드해 실행 (`README.md:393`, `docs/BACKENDS.md:19,42,57`).
- Gemma4는 CUDA/HIP build 전용 실행 (`model_family_not_supported` 메시지에서 SYCL은
  development) (`SOURCE: model_family.hpp:52-62`).
- "임의의 모델 family" 등록 절차는 PolyStrata源码에 architecture 클래스
  (hook·layer tensor 어휘·routing)를 추가·编译하는 것 — 프리셋 값 patch로 도달 불가능.

## 5. StrataShift 적용점

1. HF emit의 `general.architecture` prefix는 `guard_keys["architecture"]`로
   선택되는 구조가 이미 실재 (`stream/emit_hf.py::_emit_hf`, `write_guard_metadata`) — architecture 문자열 발명 금지.
2. Gemma4 emit을 지원하려면 실행 계층의 키 어휘 (§2)를 실측 fixture로 확정한 뒤
   `meta/hf_gguf_map.py`에 매핑 추가 — Qwen1.5/Mixtral과 같이 "실행 클래스가 없는
   family"는 변환 산출물이 engine에 load되지 못한다 (p4-03 acceptance 유지).
   p6-01부터 Mixtral은 `engine_ext/`의 `Mixtral` 클래스를 얹은 `strata-shift-poly`에서 실행한다
   (`MIXTRAL_LAYER_DENSE`, 계약: [mixtral_class_contract.md](mixtral_class_contract.md)).
3. Gemma4 HF 헤더(per-layer input, sliding window pattern, shared KV)는 미측정
   — 실측하기 전 매핑 금지 (never-invented, AGENTS/CONSTRAINTS C4).