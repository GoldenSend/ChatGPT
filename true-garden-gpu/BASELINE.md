# Baseline: what is verified, and by whom

Prepared 2026-10-23 by the cloud lead session. This session had **no PS4 contact and no IdeaPad access**. Every row says where its evidence comes from.

Labels: `ARTIFACT_VERIFIED` (reproduced in this folder by `reproduce.sh`), `USER_REPORTED` (owner prompt, 2026-10-13 context packet, or 2026-10-14 lead handoff; not re-checked here), `PUBLIC_CLAIM`, `INFERENCE`, `UNKNOWN`.

## 1. Inherited physical baseline (`USER_REPORTED`, not re-checked)

| Item | Recorded value | Source |
|---|---|---|
| Hardware | PS4 #1 CUH-2215B, firmware 12.50, Liverpool GPU `1002:9923`, ~6.83 GiB RAM, no swap | packet 10-13 |
| OS / kernel | Debian 13 on PNY USB; PNY `bzImage` rolled back to stock `fc5dc10d...` on 2026-10-14 06:49 UTC | handoff 10-14 |
| CPU oracle / fallback | llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`, Jaguar SSE/AVX/F16C, no AVX2/FMA | packet 10-13 |
| Model | Qwen3-0.6B-Q4_K_M, sha256 `b0638f08417a2d3c8652760462eb5407c6e30173cf9608ad0820757a281eea0e` | packet 10-13 |
| CPU speed | tg64 ~14.17 t/s (4 thr), ~19.65 (6), ~24.3 (8); a later report 24.79 | packet 10-13 |
| Hybrid 003 / 004 | one Q6_K `Vcur-0` projection on the GPU, 151,936 and 303,872 logits bit-identical to CPU; ~3.0 t/s vs ~7.6 t/s CPU | packet 10-13 |
| GPU-016 (layer 0, 2 tokens) | token 0 7/7 PASS; token 1 635/1024 bad, max abs 0.0085 | handoff 10-14 |
| GPU-021b (hardened attention shader) | token 0 7/7; token 1 573/1024 bad; attention error ~5x smaller, pre-FFN ~2x smaller | owner memory, 2026-10-14 08:26 UTC |
| GPU speed | ~46 ms per layer for token 1 (~0.8 t/s extrapolated to 28 layers) | handoff 10-14 |
| "Three qualified physical layers, four unproven" | no matching receipt in the cloud files | owner prompt (cannot be reconciled from here) |
| Layer-2 FFN > 5000; Q8 investigation with 12 changed codes | no matching receipt in the cloud files | owner prompt |
| Vulkan | process exit 137 (SIGKILL, likely OOM); not a working backend | packet 10-13 |

Status: **full 28-layer GPU inference, autoregressive GPU decode and any GPU speed advantage are NOT PROVEN.** Nothing in this folder changes that.

## 2. Offline baseline established by this session (`ARTIFACT_VERIFIED`)

Host: x86-64 cloud container, 4 cores, gcc 13.3.0, glibc 2.39, Python 3.13, numpy 2.5.3, Mesa 25.2.8 LLVMPipe (LLVM 20.1.2). Not a PS4. Not Liverpool.

### 2.1 Oracle build
- llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`, `-march=btver2`, `GGML_AVX=ON GGML_F16C=ON GGML_SSE42=ON GGML_AVX2=OFF GGML_FMA=OFF` (exact flags in `reproduce.sh`). This selects the same ggml kernels as the PS4 CPU build.
- `flash_attn=auto` resolves to flash attention **on** for the CPU backend: an auto dump is byte-identical to an FA=on dump.
- 1 vs 4 threads: byte-identical dumps.

### 2.2 Synthetic models (real weights unreachable: Hugging Face is blocked from the cloud container)
Qwen3-0.6B layer-0 geometry, 1 layer, vocab 512, deterministic. Regenerated twice from scratch with identical hashes.

| File | sha256 |
|---|---|
| synth-f32-s1.gguf | `54de976108c35564530add6cd295a4129b4f6edc8a38dacd333e7f752031be63` |
| synth-f32-s2.gguf | `699b5057081d3bc6885c76c6e25817155aa723df8cad36ea81a9b19108e93213` |
| synth-f32-s3.gguf | `2e70bf85791efdebdcb670b43a70881900420deccc86a5581999c0c9819170cf` |
| synth-q4km-s1.gguf | `49e72c7497c71994ac5ddaae705f00c215acc072b49e914eaa79380b4cda5f2d` |
| synth-q4km-s2.gguf | `039a2690a76385798453b246163ee4d12507c59cde80de89df2c56f6aaa5f71b` |
| synth-q4km-s3.gguf | `9116401ebae1279c1bb031eda78293e2dfcf396a11a7313ab816134bb021edf9` |

### 2.3 Results

| Check | Result | File |
|---|---|---|
| numpy emulator vs llama.cpp, every layer-0 stage, 3 models, 11 steps | bit-exact | `RESULTS/emu_vs_oracle.txt` |
| GLSL helpers (FP16 RNE, rcp, sqrt, expf, ggml_v_expf, SiLU, RMSNorm) vs CPU | PASS (2 expf cases where glibc itself misrounds, ~4e-7 rate) | `RESULTS/gles_exact_lib.txt` |
| GLSL flash attention vs `FLASH_ATTN_EXT`, 59 steps, n_kv 1..48 | 59/59 bit-exact; RTZ negative control fails 55/55 steps with n_kv >= 2 | `RESULTS/gles_fa_vs_oracle.txt` |
| GLSL Q8_K + Q4_K/Q6_K matvec vs `MUL_MAT`, 413 matvecs | 413/413 bit-exact; row-serial negative control 14/14 differ | `RESULTS/gles_matvec_vs_oracle.txt` |
| Whole layer 0 chained on the GPU driver | 59 steps x 17 stages bit-exact | `RESULTS/gles_layer_vs_oracle.txt` |
| Which GPU-style deviations break token 1 | table, 12 cases per variant | `RESULTS/sensitivity_summary.md` |

State of the candidate: **PASS_OFFLINE (LLVMPipe, synthetic weights) / HOLD_HARDWARE.** It has never run on Liverpool.

## 3. Source revisions

| Item | Revision |
|---|---|
| This folder | branch `claude/project-thread-uovzkj` of GoldenSend/ChatGPT (see `CHECKPOINT.json` for the commit) |
| llama.cpp | `23b0202a189c44a54625aadcb37a946dd1d6278d` (cloned by `reproduce.sh`) |
| gguf-py | from the same llama.cpp checkout |
| GPU-016/021 runner, its shaders, its CPU oracle script | **not available here** (on the IdeaPad, `~/Projects/atom-ps4-linux-compat-001/ps4-takeover-reconcile-20261013-002/`) |

File hashes for the shaders, emulator, tools and tests are in `CHECKPOINT.json`.
