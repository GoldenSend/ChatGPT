# Patches

## Why there is no diff here

The candidate correction is meant to replace shaders in the GPU-016/021 runner, but **the runner's source is not available in this repository or the project files** (it lives on the IdeaPad under `~/Projects/atom-ps4-linux-compat-001/ps4-takeover-reconcile-20261013-002/`, which this cloud session could not reach). A diff against an unseen file would be invented, so none is given.

The correction is therefore delivered as a complete, tested shader set in `../shaders/` plus the integration contract below. Porting it into the runner needs a session that can read the runner (its questions are Job 3 of `../HANDOFF-CODEX.md`). The hardware gate in `../HANDOFF-CLAUDE.md` does not need the port: it runs the shaders through this folder's own harness (`tools/gles_run.c` and `TESTS/`).

## The shader set

Every `.comp` file needs `ggml_exact_lib.glsl` concatenated in front of it, after a `#version 320 es` line (this is what `tools/gles_run.c` does).

| Shader | Replaces | Dispatch | Bindings |
|---|---|---|---|
| `ggml_rms_norm_mul.comp` | RMSNorm (+ weight multiply) for attn_norm, q_norm, k_norm, ffn_norm | `rows` groups x 128 | 0 Params `{int n; int rows; uint eps_bits; int has_w}`, 1 x f32, 2 w f32, 3 y f32 |
| `ggml_q8k_quantize.comp` | the Q8_K quantization of a matvec input (llama.cpp does this before every Q4_K/Q6_K matvec) | `n/256` groups x 256 | 0 x f32, 1 out: 273 uint per block = `d` bits, 256 int `qs`, 16 int `bsums` |
| `ggml_matvec_qk.comp` | Q/K/V/O, gate/up/down projections | `ceil(rows/64)` groups x 64 | 0 Params `{int rows; int nb; int qtype}` (4 = Q4_K, 6 = Q6_K), 1 raw GGUF blocks row-major, 2 Q8_K blocks, 3 y f32 |
| `ggml_rope_neox.comp` | RoPE (NEOX) for Q and K | `heads` groups x 64 | 0 Params `{int n_heads; int n_dims}`, 1 x, 2 table `[64 cos][64 sin]`, 3 y |
| `ggml_fa_f16.comp` | the whole attention (KQ, softmax, KQV) | `n_head` groups x 128 | 0 Params `{int n_kv; int n_head; int n_head_kv; uint scale_bits}`, 1 q f32, 2 K cache, 3 V cache (FP16 bits, two per uint, `[n_kv][n_head_kv][128]`), 4 o f32 |
| `ggml_eltwise.comp` | residual adds (`mode 0`), SwiGLU (`mode 1`), FP16 packing of K/V for the cache (`mode 2`) | `ceil(n/64)` groups x 64 | 0 a, 1 b, 2 y; uniforms `mode`, `n` |

Host-side duties (these are part of the contract, not optional):

1. **RoPE table**: build `cos[i], sin[i]` for `i < 64` on the CPU with ggml's own sequence: `theta_scale = powf(1e6f, -2.0f/128)`, `theta = (float)pos`, then per pair `cosf(theta)`, `sinf(theta)`, `theta *= theta_scale` (glibc `cosf`/`sinf`, float). `emu/ggml_emu.py: rope_neox_cache` is the reference.
2. **K/V cache**: append the `mode 2` FP16-packed `Krope` and `Vcur` rows each step. Never round with `packHalf2x16`.
3. **Scale**: `scale_bits` is the bit pattern of `1.0f / sqrtf(128.0f)` computed in float.
4. **Q8_K before every matvec**: the four matvec inputs (attn_norm, attention output, ffn_norm, swiglu) each go through `ggml_q8k_quantize.comp` first. Reuse one Q8_K buffer for q/k/v and for gate/up.

## Portability notes for radeonsi

- `ggml_q8k_quantize.comp` uses a 256-invocation workgroup. GLES 3.2 only guarantees 128 (`GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS`); radeonsi normally reports 1024, but check before the first run.
- All helpers in `ggml_exact_lib.glsl` that depend on exact fp32 products use `precise`. Whether radeonsi honours it against `v_fma`/`v_mad` contraction, and whether it flushes fp32 denormals, is untested. `TESTS/test_gles_exact_lib.py` is the probe; it must PASS on the target driver before any layer result means anything.
- Do not substitute `exp`, `inversesqrt`, `/`, `sin`, `cos` or `packHalf2x16` for the library functions; each one changes the rounding sequence.

## Rollback

The shader set is additive. Porting it into a runner copy in a new mission directory leaves the frozen 021 bundle (`liverpool-021-gate/021-BUNDLE.tar`, sha `edca7da9...`) untouched; rollback is deleting the new directory.
