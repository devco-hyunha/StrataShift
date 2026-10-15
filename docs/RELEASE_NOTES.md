# RELEASE_NOTES

## v0.3.0-runtime (후보, Phase 3 최소 연동) — 2026-10-16

- GGUF v3 파서: 헤더 24 B(magic/version/n_tensors/n_kv), tensor 레코드 순서
  `name→n_dims→shape(u64×nd)→dtype→offset` (gguf_reader.hpp:463-469 VERIFIED).
- 아키텍처 가드: compiled geometry 키 전체 수용 (`embedding_length=2560`, `block_count=48` 등).
- 엔진 로드 스모크 (`scripts/p3_engine_smoke.py`, tests/test_engine_smoke.py):
  strata-gguf --check PASS, strata-dequant PASS(Q2_0 symbolic invariant 0 violations),
  CHECKSUM 엔진↔Python 교차 일치, 엔진 golden build+verify PASS,
  우리 convert 산출물 experts.bin/dense.bin/embd.bin/index.txt golden과 byte-identical,
  엔진 verify가 OUR pack을 acceptance, 2회 convert sha256 재현.
- IQ4_NL dense 경로: S4 + IQ4NL codebook(16값 −127..113, canonical_xcheck.py:571),
  `iq4_nl_planes` 평면 빌더, PLE `per_layer_token_embd.weight`(shape[0]=160) fixture 수용.
- `stratashift verify` 로컬 self-check PASS.
- pytest 21/21 GREEN (GGUF no-full-load 게이트 2건 추가: 헤더 파싱 peak < 2 MB,
  단일 expert-role slice peak = slice + margin).

## v0.2.0-stream (Phase 2 완료) — 2026-10-15

- safetensors 헤더 파싱, Expert 단위 스트리밍 루프, Q2_0 blob relayout(pack_layer 산수),
  메모리 안정 모니터(`--max-rss-gb`), CLI `inspect`/`convert`.
