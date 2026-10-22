# imatrix accumulation contract (p6-03 AC2)

정본 근거: llama.cpp `tools/imatrix/imatrix.cpp` 원문 (master pin, 2026-10-21 WebFetch
76 KB 확보) + QuantMoE-Bench arxiv 2406.08155 원문 (§3.1·3.3) + engine pin c566ff4 소비 계약.

## §1 축적 계약 (imatrix.cpp 전사, VERIFIED 원문 2026-10-21)

`imatrix.cpp:655-664` (expert tensor accumulation, matrix unit `ex`):

```c
float * acc = e.values.data() + ex * ne0;
float * act = has_act ? e.activations.data() + ex * ne0 : nullptr;
e.counts[ex]++;
touched[ex] = 1;
for (int64_t j = 0; j < ne0; ++j) { acc[j] += x[j] * x[j]; }
if (act) { for (int64_t j = 0; j < ne0; ++j) { act[j] += x[j]; } }
```

- `values[j]` = input column j의 sum-of-squares (`in_sum2` tensor로 저장)
- `activations[j]` = input column j의 sum (`in_sum` tensor, legacy는 없음)
- `counts[ex]` = matrix unit이 touched된 batch 수
- dense tensor도 same form (`:724-733`, mat_id = `ne2`/`ne3` 단위)
- **additive accumulation**: 각 forward chunk는 `x*x` rows를 plane에 더하기만 한다. full
  model accumulation이 아니라 unit-plane 축적 → our layer/expert chunk streaming은
  C1(전체 로드 금지)와 충돌하지 않는다. 이것이 our producer의 정문 근거다.

## §2 GGUF emit contract (imatrix.cpp :966-988 전사, VERIFIED)

tensor `name`의 저장 형식 (`nval = values.size()`, `nmat = counts.size()`):

- `in_sum2`: GGML_TYPE_F32, shape `[nval/nmat, nmat]` = `[ne0, nmat]`
- `counts`: GGML_TYPE_F32, shape `[1, nmat]`
- `in_sum`: shape `[nact/nmat, nmat]`, activations 비어 있으면 emit 생략

`fused_experts`는 matrix index = expert index (imatrix.cpp expert path), dense는 `ne2`·`ne3`
mat_id (`:724` `mat_id = (i3 % ne3) * ne2 + (i2 % ne2)`).

## §3 engine 소비 계약 (pin c566ff4 VERIFIED, p6-03 근거 2)

`src/kernels/native_expert_parity.cpp:407-409`:
`ggml_quantize_requires_imatrix((ggml_type) type) ? imatrix.data() : nullptr` — engine C++
quantize path가 imatrix를 인자 받는 계약 실재. 우리 Python producer는 separate producer rule
(AC3 HYPOTHESIS — dequant 왕복 gate 유지, engine rule 발명 없음).

## §4 MoE expert balance (arxiv 2406.08155 §3.3 전사, VERIFIED)

`usage = normalize( Σ_i G(W_l X_i) )` — calibration data의 top-k routing counts가 expert
importance. low-utilized experts는 calibration 부족 (Figure 2 실측 시각화) → our accumulator는
per-expert `counts`를 보존하고, forward sampling이 expert-balanced하게 routing을 발화하는 전제를
포함한다 (MoEQuant EBSS 전제와 같은 방향, 발명 없음 — 근거 citation).

## §5 our producer contract

`meta/imatrix.py` (pure arithmetic, no I/O): `new_stats(ne0, nmat)`,
`add_activation_rows(stats, mat_idx, x)` (§1 전사), `normalized_imatrix(stats, mat_idx) =
in_sum2[mat]/count` (engine ggml imatrix vector 형식 — mean-of-squares는 llama.cpp quantize
소비자 imatrix.cpp의 표준 normalization convention, consumer 확인 AC4에서 재대조),
`expert_usage(counts)` (§4 전사).

forward chain은 `calibration text → tokenize → embed → layer-streaming forward`로
layer L의 `activation_in`를 stream하고, layer L의 tensor t가 `x`를 소비할 때 §1로 accumulate한다.
HYPOTHESIS: our Python forward가 engine CUDA path와 동일 activation을 발하는지는 engine RUN
log 왕복 대조로 판정 (AC4 gate, 발명 금지).

## §6 emit consume contract (p6-05 실측, VERIFIED 2026-10-21)

delivery product (`p6-04-imatrix-20261021/imatrix.gguf`) 전수 실측:

- plane key는 engine key 그대로 (`blk.N.*` 295 base, HF key 0종) — `load_imatrix_stats`가
  engine name으로 lookup한다.
- plane unit = matrix unit = expert (imatrix.cpp pin :744 counts per unit, :971 emit shape
  `[nval/nmat, nmat]`). ggml ne0-fastest → element (i, j) at flat `i + j*ne0` (matrix-major;
  pin 3cf03257 :325/:616). `load_imatrix_stats` returns shape `(nmat, ne0)` with row j =
  unit j vector (no transpose).
- 정합 실측: `ffn_gate_up_exps.weight` plane vector length 2816 = engine row length h
  (Q4_K gate `2816 % 256 = 0` 통과), `ffn_down_exps.weight` length 704 = engine row length ff
  (Q4_K `704 % 256 = 192 남` 거부 → be=32 Q4_0 통과), dense `ffn_down.weight` 2112
  (`2112 % 256 = 64 남` → Q4_0).
- tile contract: stream element `r*row_len + c`는 `plane[c]`를 쓴다 (`broadcast_imatrix`).
  `payload_chunks`는 2^19 elements이고 `2^19 % 2816 = 512 남` 실측 — 청크가 row 중간에서
  시작하므로 `plane_tile`은 `offset % row_len` rolled tile로 slice한다.
- expert stream 정합: expert 청크 = expert row stream = 정수 row 개수 (gate row = ff rows
  of h, down row = h rows of ff) → expert 청크 경계 = row 경계, `sigma2` row-wide contract
  정합.
- producer 정합 실측: our `make_qx_quants` transcribe는 원문 (`signed_max`, `is` sweep -9..9,
  `sumlx² > best·suml2`)과 8 blocks 전수 `d/amax = 0.129 = 1/7.77` 일치.

