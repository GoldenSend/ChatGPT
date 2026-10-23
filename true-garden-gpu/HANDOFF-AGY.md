# Handoff: agy (independent verification and adversarial review)

Your job is to try to break the claims in this folder, offline. Assume the author was wrong somewhere. Report each finding with a file path and line, the evidence, and a label (`ARTIFACT_VERIFIED`, `INFERENCE`, `UNKNOWN`).

**Scope:** offline only, on the IdeaPad or any Linux box. No PS4 contact (no SSH to `ps4-garden`, no payload, no file copy, no GPU dispatch on the PS4). No `sudo`, no package installs. Work in a new directory, e.g. `~/Projects/atom-ps4-linux-compat-001/gpu-022-review/`; rollback is deleting it. If you also take Job 3 of `HANDOFF-CODEX.md` (provenance), keep its answers in that job's `PROVENANCE.md`, separate from this review.

Source: GoldenSend/ChatGPT, branch `claude/project-thread-uovzkj`, folder `true-garden-gpu/`; llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`. File hashes in `CHECKPOINT.json`.

## 1. Re-run, independently

```sh
WORK=$PWD/work LIBGL_ALWAYS_SOFTWARE=1 ./reproduce.sh 2>&1 | tee reproduce.log
```

Compare with `RESULTS/reproduce_from_scratch.txt`: same six model hashes, same `RESULT` lines, last line `ALL OFFLINE CHECKS DONE`. Any difference is a finding.

## 2. Attack the tests (most valuable)

1. **Is the oracle really llama.cpp?** Confirm `tools/dump_layer0.cpp` only reads tensors through `cb_eval` and that `reproduce.sh` builds with `GGML_NATIVE=OFF`, `GGML_AVX2=OFF`, `GGML_FMA=OFF`. Check `$WORK/llama.cpp/build-jaguar/CMakeCache.txt`.
2. **Does the layer test leak oracle values into the GPU chain?** In `TESTS/test_gles_layer_vs_oracle.py`, only `embd` (the layer input) and the host RoPE table may come from outside the shaders. Check that every later stage, including the K/V cache, is built from shader outputs.
3. **Are the right tensors compared?** `oracle()` and `find()` take the first manifest match by name and op. Check the `MUL_MAT@wo` rule and the `ROPE` picks against a `MANIFEST.tsv` (several tensors share names like `Qcur-0`).
4. **Do the negative controls exercise the path they claim?** The FA control swaps the RNE expression in the library; the matvec control replaces both `acc[l] = t + acc[l];` lines. Try your own mutation (e.g. change one constant in `glibc_expf`, or one lane in `hsum8`) and confirm the relevant test fails. A test that still passes after a real mutation is a finding.
5. **Is the emulator check circular?** `TESTS/test_emu_vs_oracle.py` must compare `emu/ggml_emu.py` against dumped llama.cpp tensors, never against itself.
6. **The expf known-class exclusion.** `TESTS/test_gles_exact_lib.py` does not count `expf` mismatches where glibc is not correctly rounded. Check that classification with an independent high-precision reference (`mpmath` or `decimal`); a mismatch excluded wrongly would hide a real shader error.

## 3. Review the shaders for latent bugs

`shaders/ggml_exact_lib.glsl` and each `.comp`:
- Missing `barrier()` or `memoryBarrierShared()` around shared-memory use; reads past buffer ends at the last workgroup.
- Integer overflow in the Q4_K/Q6_K lane sums (Q6_K scales are signed 8-bit; products with Q8_K values can reach large magnitudes).
- `precise` missing anywhere a float product or sum must not be contracted.
- Behaviour with denormal inputs, `-0.0`, `inf` and `NaN` (RMSNorm of an all-zero row, `expf` of `-inf` in the softmax).
- The `rcp_cr`/`div_cr`/`sqrt_cr` candidate searches: is +-2 / +-3 ulps enough if the hardware's starting estimate is worse than 1 ulp? Prove or find a counterexample.

## 4. Check the writing for overreach

Read `NUMERICAL-ANALYSIS.md`, `BASELINE.md` and `README.md` and flag any sentence that:
- implies anything ran on the PS4 or on Liverpool (nothing did);
- states an `INFERENCE` as fact, especially section 7 (what the runner does) and the PERFORMANCE estimates;
- generalises synthetic-weight counts to the real model.

## Report

`REVIEW.md` in your directory: findings first, ordered by severity, then the list of checks that found nothing. Do not change any file in the repository; propose diffs in the report instead.
