# binary_pin — 분석 대상 핀

Goal: `p0-01` · Pin: PolyStrata `c566ff4` · Ghidra `12.1.4`

## 1. PolyStrata (분석 대상 소스)

| 필드 | 값 |
|------|-----|
| 확보 경로 | GitHub `main` zip (GitHub API `commits/main`로 커밋 확인) |
| 저장소 | https://github.com/VecSzn/PolyStrata |
| 커밋 SHA | `c566ff4787c8ab188ecf33c6f1907a9f90d9f152` |
| 커밋 일시 | 2026-10-15T03:48:30Z (`readme: Tiel Coder's workstation numbers …`) |
| zip 파일 | `.local/polystrata/polystrata-main-c566ff4.zip` (git 미추적) |
| zip SHA256 | `3ecc0e4c922cbef0a86b7dc8317c67a5545ddd06c430994ba56afde8ffa86b24` |
| zip 크기 | 25,263,408 B · 3,728 entries |
| 압해제 위치 | `.local/polystrata/PolyStrata-main/` |
| 바이너리 버전 | 미확정 — zip에 사전 빌드 실행 파일 없음 (`*.exe/*.so/*.dll` 0개) |

**zip에 들어 있는 `.bin`은 가중치/프로필 데이터지 실행 파일이 아니다**
(`data/expert-profile.bin`, `data/draft_vocab*.bin` 등). 따라서 REA의 정본 대상은
**핀된 소스에서 빌드한 로더 ELF**다.

- **HYPOTHESIS 해소(2026-10-19, p5-07 AC3)**: 분석 대상은 `CMakeLists.txt`로 빌드한 로더
  계열이고, CUDA 빌드 산출물 SHA256을 §5에 실측 기록한다.

## 5. p5-07 AC3 CUDA 빌드 산출물 (2026-10-19, RTX 3050 Ti PC WSL)

빌드 환경: WSL Ubuntu 26.04 · cmake(glibc 2.43) · CUDA toolkit 13.3(nvcc V13.3.73,
`cuda_13.3.r13.3`) · `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
-DCMAKE_CUDA_ARCHITECTURES=86`(RTX 3050 Ti = Ampere GA107 sm_86) · Release · Ninja.
소사 사본 `/opt/p5-07/PolyStrata-main`, 빌드 `/opt/p5-07/build-cuda13` (저장소 밖, git 미추적).

| 산출물 | SHA256 | 비고 |
|--------|--------|------|
| `strata` | `4f2e01c41ed292fcd903f87b480695958ded37a31d2e43714df04c3891e3995c` | CUDA 엔진 링크 (`[155/155] Linking CXX executable strata`) |
| `strata-gguf` | `ed40f3db16aa6f94daece183d782f587161e9dc392d978af876943b92dd617be` | GGUF 리더(CUDA 빌드 트리) |

- pin 소스 SHA256 §1과 동일 트리에서 빌드. C4: 산출물은 엔진 공식 경로로만 실행(REA 분석
  대상 아님 — CUDA RUN 게이트는 p5-07 AC4/AC5에서 공식 실행).

### 5-1. p6-13 Windows 네이티브 빌드 (2026-10-22, 같은 RTX 3050 Ti PC, `stratashift build-engine`)

빌드 환경: Windows 11 · VS 2022 Build Tools MSVC 19.44.35229 · CMake 4.4.4 · VS 동봉 Ninja ·
CUDA 13.0.2(winget `Nvidia.CUDA --version 13.0`) · `-DSTRATA_BUILD_TESTS=OFF -DSTRATA_GGML_DIR=<핀 llama.cpp 3cf03257>` ·
GPU 빌드는 `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` · Release. 소스: 핀 트리(내용 해시 `905d413e…`, p6-12).
빌드 폴더 `.local/polystrata/build` (git 미추적).

| 산출물 | SHA256 | 크기 |
|--------|--------|------|
| `strata-gguf.exe` | `2bd9cf041e295faed8526f1dce57e7ee53a5fe191457ab32c1ab301dab1042e8` | 71,168 B (CPU, CUDA 없이) |
| `strata-dequant.exe` | `5e394fe2de9c796297dc620d594ca791759b02b004fdf697f7e661be0510bc80` | 48,128 B (CPU, CUDA 없이) |
| `strata-poly.exe` | `dadabebfcc8976076e22884411bdd7baacad766e879fca68735d592420a3ef82` | 64,691,712 B (CUDA 13.0, sm_86) |

### 5-2. p6-01 `strata-shift-poly` (2026-10-23, 같은 PC, `stratashift build-engine --gpu --ext`)

핀 트리(`905d413e…`)를 `.local/polystrata/ext-src`로 복사하고 `engine_ext/`(Mixtral 클래스, `overlay.json` 편집 3곳,
CMake 조각)를 얹어 `.local/polystrata/build-ext`에서 `strata-shift-poly` 타깃만 빌드. 환경·스위치는 §5-1과 같다.
기대 엔진 파일 8개의 sha256은 `engine_ext/pin_files.json`과 일치(빌드 전 검사). 공식 `strata-poly.exe`는 다시
빌드하지 않았고 sha256이 §5-1 값 그대로다.

| 산출물 | SHA256 | 크기 |
|--------|--------|------|
| `strata-shift-poly.exe` | `850c2b10fce359eff4fe8c95630c5871792a6427b60a220828856a078ecb674f` | 64,700,416 B (CUDA 13.0, sm_86) |

- 첫 빌드에서 `strata_kernels`의 `s2_gemv_fast.cu` 전처리(`cmd /c cl.exe … > *.ii`)가 34분 동안 멈췄다(자식 `cl.exe` 없음).
  nvcc를 끝내자 `build-engine`의 한 번 재시도가 이어서 빌드를 마쳤다. 코드 문제가 아니라 이 PC의 일시 현상으로 보고 기록만 한다.
- CONSTRAINTS C4: zip 안의 바이너리·모델 가중치 **실행 금지** (해시·hexdump만).

## 2. Ghidra (도구)

| 필드 | 값 |
|------|-----|
| 릴리즈 | Ghidra **12.1.4** (`Ghidra_12.1.4_build`, 2026-09-21) |
| zip | `/opt/data/tools/ghidra_12.1.4_PUBLIC_20260921.zip` |
| zip SHA256 | `ddac49f903da9d5bac833e5cc79395098b9c33cfd3279be5f31bd00387d2d4db` — GitHub 릴리즈에 공개된 값과 **일치 확인** |
| 설치 경로 | `/opt/data/tools/ghidra_12.1.4_PUBLIC/` (저장소 밖, git 미추적) |
| headless | `/opt/data/tools/ghidra_12.1.4_PUBLIC/support/analyzeHeadless` (기동 확인) |
| JDK | Temurin **21.0.12.1+1** `/opt/data/tools/jdk-21/` (Ghidra 12.1 요구는 Java 21 — GettingStarted.md 명문화) |
| JDK tar.gz SHA256 | `ce79869e1307ed8ee1e2baa86a412b1eb5b75d10a01006d788a6f968bcfaee94` |

## 3. 프로젝트 루트 (환경 제약)

Ghidra 12.1은 **`.`로 시작하는 경로 요소를 거부한다**
(`NamingUtilities.checkName` → "Path element starting with '.' is not permitted").
`.local/ghidra-projects/`는 **사용 불가**가 실증으로 확인되었다 (REA_GHIDRA §2-1 갱신).

- 프로젝트 루트: `/opt/data/ghidra-projects/` (저장소 밖 → git 미추적)
- 스모크 테스트: `analyzeHeadless /opt/data/ghidra-projects StrataShift-Smoke -import /bin/ls -NoAnalysis`
  → `REPORT: Import succeeded` (`.gpr`/`.rep` 생성)

## 4. p0-02 빌드 산출물 (분석 대상, 2026-10-15)

빌드 환경: cmake **3.31.6** · g++ **14.2.0** (C++20, Release) · CUDA 없음 →
CPU 타깃 + CUDA 헤더(pip `nvidia-cuda-runtime/nvcc/cublas-cu12` 12.9.x)로 **컴파일 전용** `.o`
(링크·실행 없음 — C4 준수). 소스 트리 사본 `/opt/data/polystrata-src`, 빌드
`/opt/data/polystrata-build` (저장소 밖, git 미추적).

| 산출물 | SHA256 | 크기 | 비고 |
|--------|--------|------|------|
| `strata-gguf` | `514ba6453e94cded724909fd8afc15bc82b95ccc05f31976036d542ea697344e` | 97,000 B | CMake CPU 타깃 (GGUF 리더) |
| `strata-dequant` | `e1d39994f8d5dbb23bdfc337425077e2dcfa7b211d537d2e7174fac7dc6aed37` | 74,368 B | CMake CPU 타깃 (Q2_0 언팩) |
| `strata-plan` | `f20f8e0a6d6557c2ab131ba69da836aa39500b17ed0ce0bfc2e15227d53b1162` | 48,504 B | CMake CPU 타깃 |
| `weights.o` | `4002a300cc30111f611b11e4a9b84c530a00c88dd038bc58ecf5611d57e6bea3` | 59,280 B | 컴파일 전용 |
| `expert_cache.o` | `07f620860c7e98fa8bfd038467b640d8d6ad1ddfb423603e18a47bdd937d8af4` | 82,392 B | 컴파일 전용 |
| `vmm.o` | `810ecf38275c747437c75d61adb4567a73522eca4ebadf2f9d7d6738bd66687d` | 9,960 B | 컴파일 전용 |
| `load_main.o` | `368f091ff3cefa9c82986479c4ce0abb3bea04f439d7dd067923e1aeddd7b880` | 15,160 B | 컴파일 전용 |
| `concurrent_main.o` | `6d7f6df6e3597b05da6b5213f0d287bdd9dde6e47a5ebd2eaf9791a8f3c00751` | 22,864 B | 컴파일 전용 |
| `overlap_main.o` | `f47490c6fe281ee36d10f5050978907d3c3349c558ae06ae66e3f369cec1e1f0` | 16,272 B | 컴파일 전용 |

- Ghidra 프로젝트: `StrataShift-PolyStrata-c566ff4` (`/opt/data/ghidra-projects/`) — 9개 전수
  임포트·Auto-analyze·Decompiler 성공 (headless 로그 `/opt/data/ghidra-decomp/decomp3.log`).
- 디컴파일 덤프: `/opt/data/ghidra-decomp/*.decomp.txt` (postScript `DecompileDump.java`).
- 환경 제약 실측: 압해제 실행비트 소실은 shebang뿐 아니라 **네이티브 decompiler 바이너리**
  (`Ghidra/Features/Decompiler/os/linux_x86_64/decompile`, `sleigh`)에도 적용 — `chmod +x` 필요
  (p0-01의 §2 메모보다 넓음. REA_GHIDRA §2-1 반영).
- 분석일: **2026-10-15** (REFERENCES Ghidra 핀 표의 `TBD(p0-02)` 채움).
