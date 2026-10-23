# engine_ext — 핀 PolyStrata 엔진에 얹는 StrataShift 아키텍처

PolyStrata는 컴파일된 아키텍처만 실행합니다(`src/poly/engine.cpp` dispatch: glm5-next, qwen35moe, gemma4,
deepseek4). StrataShift는 그 밖의 MoE 모델도 변환하므로, 아키텍처 클래스를 더한 실행 파일을 따로 만듭니다.
핀 엔진 트리의 사본에 이 폴더의 파일을 넣어 빌드합니다. upstream에는 올리지 않고, 공식 `strata-poly`는
그대로 둡니다(사용자 결정 2026-10-23, [docs/work/p6-01.md](../docs/work/p6-01.md)).

| | |
|---|---|
| 실행 파일 | `strata-shift-poly` (공식: `strata-poly`) |
| 더한 아키텍처 | `mixtral` (`src/poly/mixtral.*`) |
| 핀 | PolyStrata `c566ff4`, 트리 해시 `905d413e…` ([binary_pin.md](../docs/rea/binary_pin.md)) |
| 라이선스 | PolyStrata는 MIT — [NOTICE.md](NOTICE.md) |

## 빌드

```bash
stratashift build-engine --gpu --ext
```

1. `pin_files.json`에 적힌 핀 트리 파일의 sha256을 잽니다. 하나라도 다르면 빌드하지 않습니다.
2. 핀 트리를 `.local/polystrata/ext-src`로 새로 복사합니다. 핀 트리 자체는 고치지 않습니다.
3. `overlay.json`의 `copy` 파일을 사본에 넣고 `edits`를 적용합니다. anchor 줄은 파일에 정확히 한 번 있어야
   합니다. dispatch 줄은 `#ifdef STRATASHIFT_EXT` 안에 있고, 이 정의는 `strata-shift-poly`에만 붙습니다.
4. `.local/polystrata/build-ext`에서 `strata-shift-poly` 타깃만 빌드합니다.

실행할 때는 `strata-poly`처럼 CUDA DLL 폴더가 `PATH`에 있어야 합니다([USAGE.md](../docs/USAGE.md)).

## 핀을 올릴 때

이 클래스들이 기대는 엔진 파일은 `pin_files.json`에 적힌 것뿐입니다. 옛 트리와 새 트리에서 그 파일만
비교하고, 클래스와 anchor를 고친 뒤 새 해시를 적습니다.

```bash
python scripts/engine_ext_diff.py OLD_TREE NEW_TREE
```

```bash
python scripts/engine_ext_diff.py --hashes NEW_TREE
```

첫 명령은 그 파일들의 unified diff만 보여 주고, 둘째는 `pin_files.json`의 `files` 표를 출력합니다.
핀 이동 절차 자체는 [UPSTREAM_VERSIONING.md](../docs/UPSTREAM_VERSIONING.md)를 따릅니다.
