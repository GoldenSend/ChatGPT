# Handoff: offline executor on the IdeaPad (written for Codex; Codex usage is exhausted, so agy or opencode runs it)

## Who runs what

Codex usage ran out on 2026-10-23, so this file is written to be executed mechanically by **agy** (own Claude quota) or **opencode** (cheap OpenRouter model, e.g. GLM 5.3 Flash or DeepSeek V4.1 Flash). The owner launches the agent; nothing here needs judgment beyond following the steps.

| Job | Kind | Suggested executor |
|---|---|---|
| 1 Reproduce | run commands, copy output | opencode (cheap) |
| 2 Real weights | run commands, copy output | opencode (cheap) |
| 3 Provenance | read code, quote lines | agy (needs careful reading) |
| 4 Bundle | copy and hash files | opencode (cheap) |

Rules for the executor:
- Copy every `RESULT` line verbatim. Do not summarize a FAIL as a PASS, and do not interpret results; the next Claude lead does that.
- Never edit anything under `$M/repo/`. Never loosen, skip or rewrite a test.
- On the first unexpected output (a missing tool, a hash mismatch, a crash), stop, write what happened into `$M/RESULTS.md`, and end. At most one retry, and only for a network failure during `git clone`.

**Scope:** offline work on the IdeaPad. **No PS4 contact of any kind:** no SSH to `ps4-garden`, no payload send, no file copy to the PS4, no GPU dispatch anywhere except the IdeaPad's own driver. No `sudo`, no package installs (if something is missing, stop and list it for the owner). Never write inside existing mission directories; create the new one below.

**Known IdeaPad state (owner's report, 2026-10-23 ~21:00Z):** Python has no numpy and there are no EGL/GLES headers, so Jobs 1, 2 and 4 stop at their first step until the owner decides how to provide them (for example `sudo apt install python3-numpy libegl-dev libgles-dev`, or a venv inside `$M`). That decision is the owner's, not the executor's. `$M` already exists, with `repo/` from the GPU-022 preflight run: reuse it (`git -C $M/repo pull`) instead of creating it, and write new files only under `$M`.

## Source

- Repository: GoldenSend/ChatGPT, branch `claude/project-thread-uovzkj`, folder `true-garden-gpu/`. File hashes are in `CHECKPOINT.json`.
- llama.cpp: `23b0202a189c44a54625aadcb37a946dd1d6278d` (cloned by `reproduce.sh`).
- Real model: `Qwen3-0.6B-Q4_K_M.gguf`, sha256 `b0638f08417a2d3c8652760462eb5407c6e30173cf9608ad0820757a281eea0e` (recorded at `/home/goldensend/ps4-llm-test/`; check the hash before use and never modify it).

New mission directory (create it; it must not exist yet):

```sh
export M=$HOME/Projects/atom-ps4-linux-compat-001/gpu-022-offline
test -e $M && { echo "$M already exists: stop"; exit 1; }
```

## Job 1: reproduce the cloud results (about 10 min)

```sh
mkdir -p $M && cd $M
git clone -b claude/project-thread-uovzkj https://github.com/GoldenSend/ChatGPT.git repo   # owner's git credentials if the repo is private
cd repo/true-garden-gpu
LIBGL_ALWAYS_SOFTWARE=1 WORK=$M/work ./reproduce.sh 2>&1 | tee $M/reproduce-llvmpipe.log
```

Expected: the six model hashes printed equal those in `BASELINE.md`; every line `RESULT: PASS`, two lines `negative control, failed as required`, last line `ALL OFFLINE CHECKS DONE`. The build uses `-march=btver2`, so the CPU must support AVX, F16C, BMI1 and MOVBE (any recent x86-64 does).

Then run the GPU tests once more on the IdeaPad's **own** GPU driver (unset `LIBGL_ALWAYS_SOFTWARE`) and save the log. Run `$M/work/bin/gles_run --src shaders/ggml_exact_lib.glsl --src shaders/ggml_eltwise.comp --compile-only` first and record the `GL_RENDERER` line. A FAIL on a real hardware driver is useful early evidence about contraction or denormal flushing; record it, do not "fix" it by loosening a test.

## Job 2: real Qwen3-0.6B weights (about 15 min)

```sh
cd $M/repo/true-garden-gpu
W=$M/work; export LLAMA_CPP=$W/llama.cpp WORKDIR=$W
MODEL=/home/goldensend/ps4-llm-test/Qwen3-0.6B-Q4_K_M.gguf
sha256sum $MODEL                                    # must be b0638f08...
TOK48=$(python3 -c "import numpy as np; print(' '.join(map(str, np.random.default_rng(9).integers(0, 151936, 48))))")
DUMP_STOP_AFTER=l_out-0 $W/bin/dump_layer0 $MODEL $W/dumps/real_long 1 4 $TOK48   # ~2 MB per step
python3 TESTS/test_emu_vs_oracle.py $MODEL $W/dumps/real_long $TOK48 | tail -3
LIBGL_ALWAYS_SOFTWARE=1 python3 TESTS/test_gles_layer_vs_oracle.py $W/bin/gles_run $MODEL $W/dumps/real_long | tee $M/layer-real-llvmpipe.txt | tail -3
```

If the GPU-016/021 fixture tokens are found in Job 3, dump and test those exact tokens too (directory `real_fixture`). Save every `RESULT` line and the dump directory's `MANIFEST.tsv` hashes in `$M/RESULTS.md`.

Known limit: `emu/ggml_emu.py: load_weights` supports only Q4_K, Q6_K and F32 for layer 0. If the real file has another type there, stop and report the tensor name and type.

## Job 3: answer the provenance questions (read-only)

Read, without modifying, the GPU-015 software oracle, the 016 evidence and the 021 bundle in `~/Projects/atom-ps4-linux-compat-001/ps4-takeover-reconcile-20261013-002/` (`gpu-hardening-20261014/`, `liverpool-021-gate/`) and the hybrid worktree named in the 2026-10-13 packet. Answer in `$M/PROVENANCE.md`, quoting file paths and line numbers:

1. Where did the strict CPU oracle for 016/021 come from: the PS4's CPU llama.cpp, an IdeaPad build, or a Python/numpy reference?
2. If llama.cpp: which binary and commit, `CMakeCache.txt` values of `GGML_NATIVE`, `GGML_AVX2`, `GGML_FMA`, `GGML_F16C`, and was flash attention on or off?
3. The runner's shader list with sha256 for each, and for the attention shader: one pass or two? FP16 Q? Online max with rescale? FP16 or fp32 accumulator?
4. Matvec: which reduction order (one thread per row? shared-memory tree? how many partial sums?) and is the input quantized to Q8_K first?
5. How the runner creates its GLES context (C, Python, which libEGL), and whether it runs under `/home/goldensend/ps4-mesa2409-runtime-20261011/run-with-mesa24.sh`.
6. The exact token IDs used for token 0 and token 1.

## Job 4 (only after Jobs 1 to 3 pass): build the GPU-022 candidate bundle offline

Superseded for the synthetic gate: the cloud lead builds that bundle (`gate/gpu022-bundle.tar.gz`, `tools/build_gate_bundle.sh`), because the IdeaPad has no numpy. For the real weights, run the same builder only on a host that has numpy, gguf-py and Mesa LLVMPipe, and only if the lead asks: `LLAMA_CPP=... tools/build_gate_bundle.sh MODEL.gguf DUMPDIR NSTEPS $M/bundle/gpu023-bundle.tar.gz`. It must print `126/126`-style full reference agreement and `GATE RESULT: PASS`. Do not copy the bundle anywhere.

## Report back

`$M/RESULTS.md` (every RESULT line, renderer strings, hashes), `$M/PROVENANCE.md`, and the bundle's `SHA256SUMS`. Label each claim `ARTIFACT_VERIFIED`, `INFERENCE` or `UNKNOWN`.

## Rollback

Everything lives in `$M`. `rm -rf ~/Projects/atom-ps4-linux-compat-001/gpu-022-offline` removes it all. Nothing else on the IdeaPad is changed.
