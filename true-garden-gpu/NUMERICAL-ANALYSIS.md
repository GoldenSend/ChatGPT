# Numerical analysis: why token 0 passes and token 1 fails

Prepared 2026-10-23 by the cloud lead session. Everything below was run **offline** (x86-64 container, Mesa LLVMPipe). Nothing here was run on the PS4 or on Liverpool.

Evidence labels: `ARTIFACT_VERIFIED` (file in this folder, reproducible with `reproduce.sh`), `USER_REPORTED` (from the owner's prompt or the 2026-10-13 packet / 2026-10-14 handoff, not re-checked here), `PUBLIC_CLAIM`, `INFERENCE`, `UNKNOWN`.

## 1. Question

The hardware record (`USER_REPORTED`, handoff 2026-10-14) shows the same pattern in every physical run:

| Run | Token 0 | Token 1 final output |
|---|---|---|
| GPU-016 (RTZ `packHalf2x16`) | 7/7 checks pass | 635/1024 bad, max abs 0.0085 |
| GPU-021b (RNE + deterministic exp + correctly rounded reciprocal) | 7/7 pass | 573/1024 bad; attention error ~5x smaller, pre-FFN ~2x smaller |

The plan on record was to harden the remaining shaders (Q/K/V/O matvec, norms, RoPE) for precision. This analysis asks what the CPU oracle actually computes, why token 0 cannot fail the same way, and which deviations are able to produce the token-1 failure.

## 2. Method

1. **Oracle.** llama.cpp at the PS4 CPU baseline commit `23b0202a189c44a54625aadcb37a946dd1d6278d`, built with `-march=btver2` (AMD Jaguar: SSE4.2, AVX, F16C; no AVX2, no FMA), so ggml picks the same kernels as on the PS4 CPU. `tools/dump_layer0.cpp` dumps every tensor of every decode step. llama.cpp's default (`flash_attn = auto`) resolves to **flash attention on** for the CPU backend (`ARTIFACT_VERIFIED`: an auto dump is byte-identical to an FA=on dump). Results do not depend on thread count (1 vs 4 threads byte-identical).
2. **Model.** Real Qwen3-0.6B weights are not reachable from the cloud container (Hugging Face is blocked by the network policy). `tools/make_synth_qwen3.py` writes a 1-layer model with the exact Qwen3-0.6B layer-0 geometry (n_embd 1024, 16 Q heads, 8 KV heads, head_dim 128, n_ff 3072, QK-norm, NEOX RoPE base 1e6, eps 1e-6, tied embeddings) and random weights; `llama-quantize Q4_K_M` then gives layer 0 the same tensor types as the real file (Q4_K for q/k/o/gate/up, Q6_K for v/down/token_embd). **Absolute counts below are therefore patterns, not predictions for the real model.**
3. **Emulator.** `emu/ggml_emu.py` re-implements every op in numpy with the exact float32 rounding sequence of the Jaguar kernels, including SIMD lane assignment and horizontal-reduction order. It is **bit-identical to llama.cpp** on 18 stages x 11 decode steps x 3 models (`RESULTS/emu_vs_oracle.txt`). It is checked against llama.cpp itself, not against itself.
4. **Variants.** `experiments/sensitivity.py` swaps one op at a time for a GPU-style implementation that is numerically sound (often more accurate than ggml) but uses a different rounding sequence, and measures each decode step.

## 3. Where the oracle is discontinuous

A last-bit difference only matters if it reaches a step function. On the Jaguar oracle with flash attention (the default) there are four kinds:

| Discontinuity | Where in layer 0 | Step size |
|---|---|---|
| Q8_K activation quantization (`quantize_row_q8_K_ref`) before every Q4_K/Q6_K matvec | attn_norm -> Q/K/V; attention output -> O; ffn_norm -> gate/up; swiglu -> down | amax/127 of the 256-element block |
| FP16 rounding of K and V into the KV cache | K after RoPE, V | 1 FP16 ulp (2^-11 relative) |
| FP16 rounding of Q before the K.Q dot (`q_to_vec_dot`) | Q after RoPE | 1 FP16 ulp |
| FP16 VKQ accumulator: rounded after the online-softmax rescale and after every `V*vs` accumulation (`ggml_vec_scale_f16`, `ggml_vec_mad_f16`) | attention output | 1 FP16 ulp, per step |

Every other op (matvec float accumulation, double-accumulated RMSNorm, the RoPE theta chain with glibc `cosf`/`sinf`, glibc `expf`, `ggml_v_expf` SiLU, the correctly rounded divisions) is continuous: a different implementation only moves the result by a few ulps. That matters only if it moves an input across one of the step functions above.

## 4. Finding 1: token 0 cannot see the Q/K/softmax path (`ARTIFACT_VERIFIED` + derivation)

With one key (n_kv = 1) the ggml FA loop does: `s > M(-inf)` so `M = s`, `ms = expf(-inf) = 0`, `VKQ = f16(0*0) = 0`, then `VKQ = f16(0 + v0*1) = v0` (v0 is already FP16), `S = 0*0 + 1 = 1`, output `= v0 * (1/1) = v0`. **The output is the FP16 V row, independent of Q, K, the score, the scale, exp and softmax.** RoPE at position 0 has theta = 0, so cos = 1 and sin = 0 exactly: RoPE is the identity.

So a token-0 PASS on hardware tests only the V path, the matvecs, the norms and the FFN. It says nothing about Q, K, RoPE, the K.Q dot, exp or the softmax. Measured: across 9 GPU-style variants x 12 token pairs, **0 token-0 failures**, including variants that change 100% of the token-1 attention outputs (`RESULTS/sensitivity_summary.md`). The FA shader's RTZ negative control gives the same picture: n_kv = 1 bit-exact, every n_kv >= 2 step wrong (`RESULTS/gles_fa_vs_oracle.txt`).

## 5. Finding 2: one Q8_K flip fails the whole layer (`ARTIFACT_VERIFIED`)

Probe: run token 1 exactly, then move the single `ffn_norm` element nearest a Q8_K rounding boundary across it (an input change of 1.7e-6 to 5.1e-6), and propagate.

| Model | Codes changed at gate/up input | Downstream flips at down-proj input | l_out values changed | l_out max abs | Bad at atol 1e-5 |
|---|---|---|---|---|---|
| seed 1 | 1 | 49 | 1024/1024 | 1.43e-2 | 1024 |
| seed 2 | 1 | 69 | 1024/1024 | 1.30e-2 | 1022 |
| seed 3 | 1 | 47 | 1024/1024 | 1.59e-2 | 1024 |

One flipped code changes all 3072 gate/up outputs a little, which flips dozens of codes at the down-projection input, which moves every output by up to ~1e-2. The hardware's 635/1024 bad with max abs 0.0085 (`USER_REPORTED`) is the same order of magnitude, with real weights and the hardware gate's tolerance. The earlier note that the Q8 reference changed twelve quantization codes (`USER_REPORTED`) fits this mechanism.

**Consequence:** the strict final check is almost a step function of "did any code flip". Making the drift 2x or 5x smaller lowers the chance of a flip but does not shrink the failure once one happens. That is consistent with 635 -> 573 after the 021 hardening.

## 6. Finding 3: which GPU-style choices make token 1 fail (`ARTIFACT_VERIFIED`, synthetic weights)

12 token pairs over 3 models; "fails" = any l_out value off by more than 1e-5.

| Variant (one op computed GPU-style, rest exact) | Token 0 fails | Token 1 fails |
|---|---|---|
| exact (control) | 0/12 | 0/12 |
| standard fp32 attention (fp32 Q, two-pass softmax, fp32 weighted sum) | 0/12 | **12/12** |
| ggml FA but Q kept in fp32 (no FP16 rounding of Q) | 0/12 | **12/12** |
| ggml FA but fp32 VKQ accumulator (no FP16 rounding per step) | 0/12 | **12/12** |
| matvec with one row-serial float accumulator instead of ggml's 8 lanes | 0/12 | **6/12** |
| RMSNorm squares summed in fp32 (tree) instead of double | 0/12 | 1/12 |
| ggml FA but K.Q dot summed serially | 0/12 | 1/12 |
| RoPE with a model of hardware sin/cos (+-1 ulp) | 0/12 | 1/12 |
| RoPE theta computed directly instead of ggml's float chain | 0/12 | 0/12 |
| correctly rounded exp instead of glibc expf | 0/12 | 0/12 |
| SiLU with precise exp instead of ggml_v_expf | 0/12 | 0/12 |
| **all of the "precise" variants together** | 0/12 | **12/12** |

Reading:

1. **Attention semantics dominate.** If the GPU attention does not reproduce ggml's FP16 Q and FP16 per-step accumulator, token 1 fails every time.
2. **Matvec reduction order is second.** ggml's AVX Q4_K/Q6_K dot keeps 8 float lanes (plus a separate 4-lane mins accumulator for Q4_K) and reduces them in a fixed tree. A row-serial GPU kernel differs in the last bit on 70-90% of rows (`RESULTS/gles_matvec_vs_oracle.txt`, negative control). That is exactly the "token 0 projection differs at the last bit" observation on record (`USER_REPORTED`), and it fails token 1 half the time through the attention path.
3. **"More precise" is not "matching".** The all-precise variant, where every op is at least as accurate as ggml, still fails token 1 every time. The target has to be ggml's *sequence*, not accuracy.

## 7. What this says about the current hardware runner (`INFERENCE`, runner source not available here)

The 021b result (RNE packs, deterministic exp, correct reciprocal, still 573/1024) is what section 6 predicts if at least one of these is still true of the runner. They are ordered by how often each one failed token 1 offline:

1. The attention does not reproduce ggml FA exactly: FP16 Q, the online max/rescale order, and an FP16 VKQ rounded after the rescale and after each accumulate. A two-pass softmax with a single normalization fails 12/12 here even with perfect rounding.
2. The Q/K/V/O matvecs use a different float reduction order than ggml's 8 AVX lanes.
3. RMSNorm accumulates in fp32 rather than emulating ggml's double sum.
4. RoPE evaluates sin/cos on the GPU.

**Smallest missing input:** the 021 runner's shader sources and the script that produced its CPU oracle (see section 9). With those, `experiments/sensitivity.py` can be pointed at the runner's actual arithmetic instead of these generic variants.

## 8. The candidate correction: ggml-exact shaders (`ARTIFACT_VERIFIED` on LLVMPipe only)

`shaders/` implements every layer-0 op as the exact ggml/Jaguar sequence in GLSL ES 3.2:

- FP16 rounding in integer arithmetic (never `packHalf2x16`).
- glibc's generic `expf` ported to fp32-only double-float arithmetic.
- Correctly rounded `1/x`, `a/b` and `sqrt`, found by checking candidate neighbours with exact Dekker products, so they work whatever the hardware's rcp/sqrt accuracy.
- ggml's SSE2 `ggml_v_expf` for SiLU.
- Exact integer Q4_K/Q6_K x Q8_K lane sums with ggml's per-lane float accumulation and `hsum_float_8`.
- Q8_K quantization with a correctly rounded `-127/max` and `1/iscale`.
- RMSNorm with a double-float sequential sum.
- RoPE from a host-built cos/sin table (built with ggml's own theta chain and glibc), so the GPU never evaluates trig.
- Flash attention with FP16 Q and an FP16 accumulator.

Results on Mesa 25.2.8 LLVMPipe:

| Test | Result |
|---|---|
| FA shader vs llama.cpp `FLASH_ATTN_EXT`, 59 decode steps, n_kv 1..48, 3 models | 59/59 bit-exact |
| FA negative control (RTZ FP16) | n_kv = 1 passes, 55/55 steps with n_kv >= 2 fail (the hardware signature) |
| Q8_K + Q4_K/Q6_K matvec vs llama.cpp `MUL_MAT`, 413 matvecs | 413/413 bit-exact, Q8_K blocks identical |
| Matvec negative control (row-serial accumulator) | 14/14 matvecs differ |
| **Whole layer 0 chained on the GPU** (only the embedding row and the RoPE table come from the host) | **59 steps x 17 stages bit-exact** |
| Helper stress: FP16 RNE (4.4M), rcp (4.7M), sqrt (4.2M), ggml_v_expf (4.7M), SiLU (5.2M), RMSNorm (4096 rows, with ~3000x outlier channels) | 0 mismatches |
| glibc expf port (4.7M args in [-87.3, 4]) | 2 mismatches, both cases where glibc itself is not correctly rounded and the port returns the correctly rounded value (rate ~4e-7 per call) |

### Residual risks before any hardware claim

- **Liverpool/radeonsi is untested.** LLVMPipe is IEEE with denormals; radeonsi may flush fp32 denormals and may implement `/`, `sqrt` and `exp` approximately. The correctly-rounded helpers search +-2 or +-3 ulps around the hardware result, so they tolerate approximate rcp/sqrt, but denormal flushing and whether `precise` really stops `v_mad`/`v_fma` contraction need a radeonsi run (`TESTS/test_gles_exact_lib.py` is the probe) or an offline ISA dump.
- **expf ~4e-7 per call** (see table above). One call per head per KV step. A software-fp64 port would remove it.
- **RMSNorm** uses a ~48-bit double-float in place of ggml's 53-bit double; it can differ when the exact sum of squares sits within ~2^-47 of a float midpoint. No mismatches in 4096 stress rows.
- **Oracle provenance (section 9).**

## 9. Oracle provenance: the biggest open uncertainty (`UNKNOWN`)

These shaders target **llama.cpp 23b0202a on Jaguar with flash attention on**. If the strict oracle used by GPU-016/021 came from somewhere else, the target sequence is different:

- **Native build on the IdeaPad (AVX2+FMA):** different Q4_K dot kernel and lane layout, fused multiply-adds, different `ggml_v_expf`/SiLU path.
- **Flash attention off:** KQ via `mul_mat` with FP16 Q, `soft_max` with `ggml_v_expf`, then KQV with the softmax rounded to FP16. Not emulated here (`emu/ggml_emu.py: attn_nofa` raises).
- **A custom Python/numpy reference:** different again.

glibc's FMA vs non-FMA `expf` builds agreed on all 400,000 samples tested, so the host's libm is not the concern; the ggml build flags and the FA setting are. The authoritative production oracle is the PS4's own CPU llama.cpp (the Atom fallback), and that is what these shaders match.

## 10. Other leads from the prompt

- **Layer-2 FFN values above 5,000** (`USER_REPORTED`). Qwen-family models are publicly reported to carry "massive activations": a few residual channels in the thousands from early layers on (`PUBLIC_CLAIM`, Sun et al. 2024, "Massive Activations in Large Language Models"). A value above 5,000 is therefore not by itself evidence of a GPU bug; compare it with the CPU oracle at the same stage. Two GPU consequences (`INFERENCE`): FP16 storage of the residual would be catastrophic (FP16 ulp is 4.0 between 4096 and 8192), and a Q8_K block containing such a channel has a step of about 40, so every other value in that block is quantized coarsely and a flip there costs more. The RMSNorm stress test includes ~3000x outlier channels and stays bit-exact.
- **"Three qualified physical layers, four unproven"** (`USER_REPORTED`, prompt). The packet and handoff available here only document layer-0 physical runs (012, 014, 016, 021b). Nothing in the cloud files records a multi-layer physical run, so this cannot be reconciled from here.
- **Linux compute vs native Aether/GNM.** No Aether/GNM source is present in the repository or project files, so no comparison was possible. Nothing in this analysis depends on the API; it depends on reproducing ggml's arithmetic sequence.
- **Hybrid 003/004 bit-identical logits** (`USER_REPORTED`) do not show that the GPU matvec matches ggml's order (`INFERENCE`): `Vcur` reaches the rest of the model only through the FP16 V cache (and at token 0 the attention output is exactly f16(V)), so a last-bit fp32 difference is absorbed by the 13 bits FP16 drops, except in rare halfway cases. `Vcur` is the one projection where that holds; Q, K, O, gate, up and down feed Q8_K or the score directly.
- **Vulkan exit 137** (`USER_REPORTED`): not investigated. Exit 137 is SIGKILL, typically the OOM killer, and the PS4 has no swap (packet). Not needed for this work.
