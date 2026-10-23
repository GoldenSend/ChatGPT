# Handoff: next Claude lead

Read `README.md`, then `NUMERICAL-ANALYSIS.md`. State at handoff: the ggml-exact shader set is **PASS_OFFLINE** (LLVMPipe, synthetic Qwen3-shaped weights, 59 decode steps x 17 stages bit-exact) and **PASS_HARDWARE_SYNTHETIC** (GPU-022 on Liverpool, 2026-10-23: layer 0, 7 decode steps, 182/182 bit-exact; fp32 denormals are flushed, harmless for layer 0). Real weights have not run on the GPU.

Hard limits (unchanged): no PS4 boot, reboot, payload send, storage write, kernel or firmware change, or GPU dispatch without the owner's fresh approval for that specific operation. The owner runs payload sends and boot-ID re-pins. Never touch GoldenPi, PS4 #3 or the stream PC.

## 1. First: verify the offline executor's results (`HANDOFF-CODEX.md`; agy or opencode, since Codex usage ran out)

- Model hash `b0638f08...` printed; the six synthetic hashes equal `BASELINE.md`.
- Real weights: `test_emu_vs_oracle.py` PASS and `test_gles_layer_vs_oracle.py` PASS on LLVMPipe. If the emulator fails on real weights, the shaders are unproven for them too; find the first differing stage before anything else.
- `PROVENANCE.md` answers. Decide from them:
  - **Old oracle = PS4 CPU llama.cpp, FA on**: the shaders target the right sequence. Go to section 3.
  - **Old oracle = IdeaPad native build (AVX2/FMA) or FA off or Python**: GPU-016/021 were compared against a sequence the PS4 CPU never computes. Re-target the gate to the PS4's own CPU llama.cpp (the production fallback, and what these shaders match) and record that 016/021 numbers are not comparable with GPU-022 numbers.
- Point `experiments/sensitivity.py` at the runner's real arithmetic (add one variant per runner shader from Job 3 answers) to confirm which runner choice explains 573/1024. This turns section 7 of the analysis from `INFERENCE` into `ARTIFACT_VERIFIED` offline.

## 2. Open engineering items (offline, any order)

1. **expf at 100%.** The fp32 double-float port differs from glibc in ~4e-7 of calls (cases where glibc misrounds). A software fp64 (integer-emulated double) port of glibc's generic `expf` would be exact; it is called once per head per KV step, so cost is irrelevant.
2. **Parallel exact RMSNorm.** Today thread 0 sums 1024 squares serially in double-float (~48 bits vs ggml's 53-bit double). Design: each product `x*x` of a float is exact in 48 bits; accumulate all products exactly in parallel (integer super-accumulator in shared memory), then round to float. ggml's sequential double sum differs from the exact sum by at most ~1024 * 2^-53 relative, far below a float ulp, so the float result `(float)(sum/n)` can differ only when the exact value lies within that bound of a float rounding boundary. Detect that case and fall back to the serial path for that row. Bit-exact and parallel; test with `test_gles_exact_lib.py` stress rows.
3. **Non-FA attention path.** `emu/ggml_emu.py: attn_nofa` raises `NotImplementedError`. Needed only if the PS4 worker runs with flash attention off (check its command line; llama.cpp's default resolves to on for CPU).
4. **radeonsi behaviour without hardware.** Read Mesa 24.x radeonsi/ACO or LLVM AMDGPU source for: fp32 denormal mode for compute on gfx7, whether `precise` (NIR `exact`) blocks `v_mad_f32`/`v_fma_f32` formation, and how `/` and `sqrt` lower. On GCN, `v_mad_f32` rounds like a separate multiply and add but does not support denormals, and `v_fma_f32` is fused; either one inside a `precise` path (fused, or with denormal operands) would break exactness.
5. **Speed.** Matvec as one workgroup per row with ggml's 8-lane assignment and a fixed reduce order; fuse Q8_K quantization with RMSNorm; batch all dispatches of a step into one context instead of one process per op. See `PERFORMANCE.md`: the 46 ms per layer on record is probably overhead, so profile before optimizing arithmetic.

## 3. GPU-022 hardware qualification gate (synthetic weights: DONE 2026-10-23, PASS; see GATE-GPU-022.md)

**The exact operation is `GATE-GPU-022.md`.** The owner approved it once (stock kernel, one attempt, no retry, boot ID pinned by the owner). The bundle `gate/gpu022-bundle.tar.gz` is built in the cloud by `tools/build_gate_bundle.sh`: a fixed dispatch plan, content-addressed inputs and LLVMPipe-verified expected outputs. The PS4 replays it with `tools/gles_run.py` (ctypes, GBM like the 021 runner) and compares with `tools/gate_compare.py` (standard library only), so it needs no numpy, compiler or headers. The owner runs the read-only preflight (`tools/gpu022_preflight.sh`), staging and the gate; the lead decides GO or HOLD from the preflight and writes the exact gate command (boot ID, `GLES_DRM_DEVICE`, `D_ALLOW`, `MESA_WRAPPER`).

Purpose: find out whether radeonsi on Liverpool reproduces the offline result. It is a qualification of the shader arithmetic, not a speed test and not production use.

### After a PASS: the real-weights gate (needs its own approval)
Same builder, real inputs: `LLAMA_CPP=... tools/build_gate_bundle.sh Qwen3-0.6B-Q4_K_M.gguf dumps/real_fixture NSTEPS out.tar.gz` on a host with numpy, gguf-py and LLVMPipe (not the IdeaPad today). Preconditions:
- Section 1 done: real-weight LLVMPipe layer PASS, provenance answered.
- Real-weight dumps for the fixture tokens come from llama.cpp `23b0202a` with the Jaguar flags (or from the PS4 worker itself, which needs a read-only CPU step on the PS4 in the approval).
- The model file hash matches the PS4 copy (`b0638f08...`).
- Owner present for the whole session.

Approval wording to request: "Approve GPU-023 real-weights layer-0 gate on the stock kernel, one attempt, HOLD on any stop condition." The claim after a PASS is exactly: "layer 0, N decode steps, bit-exact on Liverpool against <oracle>".

### Stop conditions (any one: stop the gate, no retry, report)
Process in D state; USB/root stall; lost SSH or Ethernet; kernel ring timeout, GPU fault or I/O error in `dmesg`; a `gles_run_guarded.sh` timeout; hash mismatch; changed boot ID; unexpected `GL_RENDERER`; temperature above the 021 aftercare record; anything that makes manual recovery uncertain.

### Aftercare (always, pass or fail)
Repeat the preflight block; compare boot ID and `dmesg` with the preflight; copy `step-*.txt`, `compile-isa.log` and the `RESULT` lines back to the IdeaPad mission directory and hash them. Leave the PS4 running as found; no shutdown unless the owner asks.

### Rollback
`rm -rf ~/ps4-llm-test/gpu-022` on the PS4 removes everything the gate added. No kernel, boot, service or configuration change is part of the gate.

## 4. Reporting rules

Offline PASS is never "GPU validated". After step C, the claim is exactly: "layer 0, N decode steps, bit-exact on Liverpool against <oracle>", nothing about full-model inference or speed.
