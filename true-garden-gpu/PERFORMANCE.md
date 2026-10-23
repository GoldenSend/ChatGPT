# Performance

No performance was measured on the PS4 in this session, and LLVMPipe timings say nothing about Liverpool, so none are reported. This page keeps the historical measurements and the estimates strictly apart.

## Measured (historical, `USER_REPORTED`, not re-measured)

| What | Value | Source |
|---|---|---|
| CPU llama.cpp Qwen3-0.6B Q4_K_M tg64 | ~14.17 t/s (4 threads), ~19.65 (6), ~24.3 (8); later report 24.79 | context packet 2026-10-13 |
| Hybrid 004 (one Q6_K projection on the GPU) | ~3.00661 t/s vs paired CPU ~7.624 t/s | context packet 2026-10-13 |
| GPU-021b token 1, one layer | ~46 ms per layer | handoff 2026-10-14 |
| Same, extrapolated to 28 layers | ~0.8 t/s (an extrapolation, not a measurement) | handoff 2026-10-14 |

## Estimates (`INFERENCE`, not measured)

The shaders in `shaders/` are correctness references. They were written to be easy to check against ggml, not to be fast.

| Shader | Exactness cost compared with an ordinary shader | Biggest speed problem as written |
|---|---|---|
| `ggml_matvec_qk.comp` | ~none: the integer dot products are exact anyway; ggml's 8 float lanes plus `hsum_float_8` add a few instructions per block | one thread per row, so weight reads are not coalesced; use one workgroup per row (or per few rows) with the same 8-lane assignment |
| `ggml_q8k_quantize.comp` | small: correctly rounded `-127/max` and `1/iscale` once per block | none of note |
| `ggml_fa_f16.comp` | integer FP16 round per accumulator element per KV step; double-float `expf` once per head per KV step | thread 0 does the reduce and the `expf` serially; acceptable at short contexts |
| `ggml_rms_norm_mul.comp` | largest: thread 0 adds 1024 squares serially in double-float | serial sum; see the parallel exact design in `HANDOFF-CLAUDE.md` |
| `ggml_rope_neox.comp` | none on the GPU: the cos/sin table is built on the host | one tiny upload per token |
| `ggml_eltwise.comp` (SwiGLU) | `ggml_v_expf` plus correctly rounded division per element | none of note |

The ~46 ms per layer on record is dominated by something other than arithmetic (`INFERENCE`: one decode step of one Qwen3-0.6B layer is ~15M weights, i.e. ~30 MFLOP and ~9 MB of quantized weights; at the public Liverpool figures of ~1.8 TFLOPS and ~176 GB/s that is ~0.05 ms of memory traffic and ~0.2 ms of arithmetic even at 10% of peak). Dispatch, synchronization and transfer overhead per op are the likely cost, so a speed project should start by profiling the runner's per-dispatch overhead, not by removing exactness.

The physical speed path is unproven either way; any speed claim needs a measured run under the hardware gate.
