# bitpack_rules — 시프트·마스크·패킹 폭

Pin: PolyStrata `c566ff4787c8ab188ecf33c6f1907a9f90d9f152`.

## 1. Q2_0 (타깃 엔코딩)

`SOURCE: include/strata/artifact/dequant.hpp:10-14` (contract는 `docs/q2_0-contract.md`에 고정)

```
block = { fp16 d; uint8 qs[16] }   = 64 elements, 18 bytes
code {0,1,2,3} -> symbol {-1, 0, +1, +2}
v = (code - 1) * d
4 codes per byte, LSB-first: byte index j/4, bits (j%4)*2
QK2_0 = 64  (PROVEN from artifact offset brackets, not assumed)
```

- **패킹 폭**: 2 bit/code, 4 code/byte, **LSB-first**
- **시프트/마스크**: `code(j) = (qs[j/4] >> ((j%4)*2)) & 3`
- **스칼라**: fp16 `d` (block당 1개)
- **구조적 불변식**: 출력값은 반드시 `{-d, 0, +d, +2d}` — bit 순서·부호·stride 버그 검출용
  (`dequant.hpp:16-20`)

`SOURCE: fp16→fp32 변환은 subnormal/infinity를 처리하는 구현이 정본` (`dequant.hpp:22-40`).
naive shift는 subnormal에서 틀린다.

## 2. ACTIVATION 측 폭 (weight와 다른 계약)

`SOURCE: include/strata/kernels/cpu/expert.hpp:20-24,44-46`

- weight block: `QK = 64` weights / **fp16** scale
- activation chunk: `QKA = 32` elements / **자기 scale**
- **한 weight block은 두 activation chunk에 걸치고 scale가 두 개다.**
  weight block당 scale 1개를 쓰는 것이 "자연스럽게 보이는 실수"다.
- `MAXC = H/QKA = 80` activation chunks

## 3. SIMD unpack (x86)

`SOURCE: expert.hpp:12-16`

- `vpmultishiftqb` (AVX512-VBMI)가 2-bit code를 8개씩 unpack.
  **인수 순서 (control, data)**는 경험적으로 결정 — Intel 가이드는 반대이고,
  가이드대로 하면 64 code 중 43이 틀린다.
- `vpdpbusd` (AVX512-VNNI) = int8 MAC 4개/命令.
- row accumulator는 **float vector**로 유지하고 **row 끝에서 1번만** reduce
  (block마다 reduce하면 compute-bound 21 GB/s로 떨어진다).

## 4. 스케일 폭 규칙 (pack ↔ engine)

`SOURCE: tools/pack_index.py:31-40`

- pack에서 scale plane은 **2 B (fp16)**일 수 있고, engine에서는 **항상 4 B (f32)**.
  303 tensor 중 90개(Q2_0 전체 58, Q4_0, Q5_0, Q8_0, IQ4_NL)가 fp16 scale,
  213개(IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K)가 f32 scale.
- 판정 필드는 manifest의 `scales_fp16` — loader가 이 필드를 읽지 않고 4 B를 가정하면
  scale plane이 2배가 되고 loader는 **거부**한다 (실제로 그렇게 발견됨).
- fp16 → f32 widening은 **정확** (fp16 ⊂ f32). BF16 re-rounding은 precision을 잃는다.

## 5. StrataShift가 지켜야 할 것 (Phase 2 입력)

1. Q2_0 block 폭 18 B / QK 64 — 매직 넘버 아님, `dequant.hpp`+artifact bracket 근거.
2. code unpack은 LSB-first, `(j%4)*2` shift, mask `& 3`, symbol `code-1`.
3. pack 변환은 codes/scales를 **plane으로 분리**해 재배치 (`struct_layouts.md` §3).
4. scale는 pack에서 fp16, engine form에서 f32 — 폭을 재계산하지 말고 필드로 읽는다.
5. `HYPOTHESIS:` AVX512 순서·bitplane 매핑(`qp/psum/pscale`)은 CPU 커널 전용.
   StrataShift 변환기는 **스칼라 contract만** 구현하고 SIMD 매핑은 p0-02 디컴파일 후 판단한다.

## 6. DECOMPILED 검증 (p0-02, Ghidra 12.1.4)

`DECOMPILED: strata-dequant main (inline, static·stripped)` — Q2_0 scalar unpack 루프:

```c
// decomp line ~830 (strata-dequant.decomp.txt)
out[j] = (float)(int)((uint)*(byte*)(qs + (j >> 2) + 2)
                     >> ((byte)(j & 3) * 2) & 3U - 1) * d;   // loop j = 0..0x40
```

- `DECOMPILED:` `qs = block + 2` (fp16 scale bytes 0-1) → **18 B block, QK 64** (§1과 일치)
- `DECOMPILED:` byte index `j >> 2`, bit offset `(j & 3) * 2`, mask `& 3`, symbol `code - 1`, `* d`
  → §1의 `code(j) = (qs[j/4] >> ((j%4)*2)) & 3`, `v = (code-1)*d` **일치**
- `DECOMPILED:` fp16→fp32: `f = (float)(u | (e + 0x70) * 0x800000)` — exponent bias 트릭 (§1의
  subnormal 처리 계약과 같은 구현 계열)
- `DECOMPILED:` Q4_K 계열 nibble unpack `((b & 3) << 4 | b2 & 0xf) - 0x20` 실재 (K-quant path)
- SIMD(`vpmultishiftqb`)는 AVX512 커널이 `.cu`/per-file kernel에 있고 CPU 빌드에 미포함 →
  §5의 `HYPOTHESIS:` 유지 (p0-02 스코프 밖)

**결론**: Q2_0 스칼라 contract는 SOURCE + DECOMPILED 이중 근거로 고정. SIMD 매핑은 미확정.
