# True Garden: Liverpool GPU numerical lane

Offline engineering for running Qwen3 layer 0 on the PS4 Liverpool GPU (GLES 3.2 compute) so that it matches the PS4's CPU llama.cpp **bit for bit**. Prepared 2026-10-23 in an isolated cloud container. **GPU-022 then ran on the PS4 (owner-approved): layer 0, synthetic weights, 7 decode steps, 182/182 outputs bit-exact on Liverpool.** Real weights are next.

## The short version

- Physical runs GPU-016 and GPU-021b pass token 0 and fail token 1 (`USER_REPORTED`). Token 0 cannot see Q, K, the softmax or RoPE at all, so it is not evidence that those parts are right (`NUMERICAL-ANALYSIS.md` section 4).
- llama.cpp quantizes every matvec input to Q8_K. One code flip changes all 1024 outputs of the layer, so "more precise" GPU math does not converge to the CPU; only the **same rounding sequence** does (section 5).
- The likeliest remaining cause on hardware is the attention algorithm itself (FP16 Q, online softmax, FP16 accumulator rounded at each step), then matvec lane order (section 6, `RESULTS/sensitivity_summary.md`). The GPU-021 attention shader, once read, had the right softmax and accumulator but summed the Q·K dot product in index order; that alone breaks token 1 in 120/120 offline cases (section 7.1).
- `shaders/` reproduces llama.cpp's Jaguar arithmetic in GLSL. On Mesa LLVMPipe the whole layer, chained on the GPU driver, is **bit-exact for 59 decode steps x 17 stages**, and the negative controls fail as they should.
- On Liverpool (GPU-022, `GATE-GPU-022.md`): all shaders compile under radeonsi; the helpers are bit-exact except that fp32 denormals are flushed (confirmed bit for bit against a model; harmless for layer 0); the whole layer is bit-exact for 7 decode steps. Still unknown: the real weights, other layers, speed, and which oracle GPU-016/021 compared against.

## Layout

| Path | What |
|---|---|
| `BASELINE.md` | inherited claims vs what this session verified, hashes |
| `NUMERICAL-ANALYSIS.md` | the diagnosis, with evidence labels |
| `PERFORMANCE.md` | measured numbers (all historical) kept apart from estimates |
| `PATCHES/` | how the shader set replaces the runner's shaders (integration contract) |
| `TESTS/` | the tests and their negative controls (`TESTS/README.md`) |
| `RESULTS/` | committed outputs of every test |
| `shaders/` | ggml-exact GLSL ES 3.2 compute shaders plus a shared library |
| `emu/` | bit-exact numpy emulator of the Jaguar ggml kernels; glibc `expf` models |
| `tools/` | synthetic model writer, llama.cpp tensor dumper, headless GLES runners (`gles_run.c`, and `gles_run.py` for machines without a compiler), GPU-022 plan builder, gate, comparator and read-only preflight |
| `gate/` | `gpu022-bundle.tar.gz`: the replay bundle the PS4 runs (built and rehearsed in the cloud; sha256 in `CHECKPOINT.json`) |
| `experiments/` | the sensitivity study (one GPU-style deviation at a time) |
| `reproduce.sh` | rebuilds every result from scratch |
| `HANDOFF-CODEX.md`, `HANDOFF-CLAUDE.md`, `HANDOFF-AGY.md` | next work per agent (the Codex file is written for agy or opencode to run, since Codex usage ran out) |
| `GATE-GPU-022.md` | the exact hardware operation put to the owner, with `tools/gpu022_gate.sh` |
| `CHECKPOINT.json` | machine-readable state |

## Reproduce

Needs an x86-64 Linux host with gcc/g++, cmake, git, Python 3 with numpy, and Mesa with EGL + GLES 3.2 (LLVMPipe is enough; on Debian/Ubuntu: `libegl-dev libgles-dev mesa-utils libegl-mesa0`).

```sh
WORK=/path/to/scratch ./reproduce.sh
```

It clones llama.cpp at `23b0202a`, builds it with the Jaguar flags, writes the three synthetic models (their hashes must match `BASELINE.md`), dumps the oracle tensors, then runs every test. Each test prints `RESULT: PASS`; each negative control must print `FAIL` and is checked for that. About 10 minutes on 4 cores, mostly the llama.cpp build. It never contacts a PS4 and never needs the network after the clone.

To force a specific driver, set `LIBGL_ALWAYS_SOFTWARE=1` (LLVMPipe). `gles_run` prints `GL_RENDERER` to stderr; the tests discard stderr, so run one `gles_run` by hand to see which driver answered.

## Safety

No file here touches firmware, the bootloader, the kernel, storage or the GPU of the PS4. Running anything here on the PS4 is a physical GPU dispatch and needs the owner's fresh, operation-specific approval (see the gate in `HANDOFF-CLAUDE.md`). Offline results are labelled as offline everywhere; none of them is physical validation.
