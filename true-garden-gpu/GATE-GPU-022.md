# GPU-022 gate: the exact operation

**Question it answers:** does Liverpool (radeonsi) compute the ggml-exact shaders bit-exactly, the way LLVMPipe does? It is a qualification of shader arithmetic on synthetic Qwen3-0.6B-shaped weights. It is not a speed test, not the real model, and not production use.

**Approval:** the owner chose "Approve GPU-022 ABC" on 2026-10-23 at 20:42Z. It covers exactly the staging write and the one gate command below, for one session, on the boot whose ID is typed into that command. It covers the stock kernel only and allows no retry. Anything else (a second attempt, a reboot, a package install, a kernel change, the real model) needs its own yes.

**Changes since the approval card (same scope, smaller footprint):** the first IdeaPad run showed that the IdeaPad has no numpy and no EGL headers, and the PS4 check never ran (a script bug). The bundle is now built and verified in the cloud and committed to the branch. The PS4 replays a fixed list of dispatches with Python's standard library and the system GL libraries; nothing is compiled or installed there. The counts are unchanged (A compile only, B 7 dispatches, C 182 dispatches), but C is now one 7-step chain on the s1 synthetic model instead of 2 steps on s1 plus 5 on s2. The staging write is about 16 MB instead of about 40 MB, into the same new directory.

## Phase 1: read-only preflight (no approval needed, nothing written on the PS4)

The owner runs this on the IdeaPad (the cloud session cannot open a shell there):

```sh
M=~/Projects/atom-ps4-linux-compat-001/gpu-022-offline
cd $M/repo && git pull -q && bash true-garden-gpu/tools/gpu022_preflight.sh 2>&1 | tee ~/gpu022-preflight.txt
```

`tools/gpu022_preflight.sh` prints the bundle hash, then runs inspection commands over `ssh ps4-garden` and reads the GPU-021 bundle on the IdeaPad. It never copies anything to the PS4 and never touches `/key` or the initramfs. The lead decides GO or HOLD from the report:

| Check | Needed for GO |
|---|---|
| Kernel and cmdline | stock kernel `5.4.247-DFAUS-blkscrn_Fix_mt7668_hdmia`; not satadiag1 |
| `lsblk` | no Hitachi 320 GB line (that would mean satadiag1); root on the PNY USB |
| Boot ID | recorded; it becomes the pin |
| ext4 `errors_count`, root mount | 0; `/` mounted rw |
| D state, 10 samples over 20 s with wait channels | no user process in D. A kernel thread in D must be explained by its wait channel and unrelated to GPU and storage (mmc0 is probably the MT7668 SDIO Wi-Fi, which is unverified). That evidence alone sets `D_ALLOW` |
| `dmesg` | readable by the SSH user (the gate needs it); no GPU fault, ring timeout or new I/O error |
| SATA, mmc0, USB, I/O error counts | judged against the 2026-10-23 power-loss boot; a USB or I/O error on the root device means HOLD |
| Temperature | at or below 75 C |
| `python3`, `tar`, `gzip`, `/dev/dri`, DRM driver, `libEGL.so.1`, `libGLESv2.so.2`, `libgbm.so.1` | present (no installs) |
| `~/ps4-llm-test/gpu-022` | does not exist yet |
| GPU-021 runner and `run-with-mesa24.sh` | tells whether `MESA_WRAPPER` is needed and which DRM node 021 opened |

## Preflight result (2026-10-23 21:27Z, owner's run from the IdeaPad): GO

| Check | Observed (LIVE_OBS via the owner's report) | Decision |
|---|---|---|
| Kernel | `5.4.247-DFAUS-blkscrn_Fix_mt7668_hdmia`, stock cmdline | OK |
| Boot ID | `092d4a88-7f6f-4c4e-a1e3-238d727d6048`, up 1:03 | the pin |
| Storage | only `sda` (USB 3.2.1 FD, 57.8 GB), root `/dev/sda2` rw, ext4 errors 0; no Hitachi line (not satadiag1) | OK |
| Worker, temperature, memory, disk | atom-worker active; 49 C; 6644 MB available; 44 GB free | OK |
| D state | `jbd2/sda2-8`, `flush-8:0` and `events_freezable_power_` kworkers, each in 1 of 10 samples; wait channel not reported (`-`) | normal journal, writeback and disk-event waits on the USB root (INFERENCE). The gate now counts only a process in D for 4 s or more; `D_ALLOW` stays empty |
| Kernel log | readable; ATAPI `IDENTIFY` failures on `ata1` at 7-49 s (the optical drive, boot only); 937 `mmc0` sdhci timeout lines, the last at 533 s; no USB errors; GPU log only boot-time | `mmc0` is not a block device (no `mmcblk` in `lsblk`), so its lines are ignored (`DMESG_IGNORE`); everything else still stops the gate. New lines are found by timestamp because a log this full can wrap |
| GPU | `amdgpu` 3.35.0, PCI `1002:9923`, `/dev/dri/renderD128` | OK |
| GL libraries | none in `/usr/lib/x86_64-linux-gnu`; the opt-in Mesa 24.0.9 runtime `run-with-mesa24.sh` is present | `MESA_WRAPPER` is required, as for GPU-021. The gate checks the libraries load before step A |
| GPU-021 runner | opened `/dev/dri/renderD128` (GBM) and required `liverpool` in `GL_RENDERER` | same device; the gate now requires `liverpool` too |
| `~/ps4-llm-test/gpu-022` | absent | OK |

## Phase 2: what the approval covers

The bundle is `gate/gpu022-bundle.tar.gz` in this folder (sha256 in `CHECKPOINT.json`):

```
gpu022/
  tools/       gles_run.py (ctypes EGL/GLES runner), gles_run_guarded.sh (timeout), compile_check.sh,
               gpu022_gate.sh, gate_compare.py (standard library only)
  shaders/     the 7 ggml-exact compute shaders and their shared library
  plan/        plan_B.sh, plan_C.sh (the dispatch lists), labels.tsv
  in/          content-addressed host inputs: weights, embedding rows, RoPE tables, parameter blocks
  expected/    the LLVMPipe replay, verified bit-exact against llama.cpp 23b0202a tensors (C) and CPU references (B)
  SHA256SUMS
```

1. **Staging (one disk write, about 16 MB):** the owner runs, on the IdeaPad,

   ```sh
   ssh ps4-garden 'test ! -e ~/ps4-llm-test/gpu-022 && mkdir -p ~/ps4-llm-test/gpu-022' \
     && ssh ps4-garden 'tar -C ~/ps4-llm-test/gpu-022 -xzf -' < $M/repo/true-garden-gpu/gate/gpu022-bundle.tar.gz
   ```

   The gate itself checks every file against `SHA256SUMS` before it starts; a mismatch is a HOLD.

2. **The gate command, run once by the owner.** Typing the boot ID is the boot pin:

   ```sh
   ssh -t ps4-garden 'cd ~/ps4-llm-test/gpu-022/gpu022 && GLES_DRM_DEVICE=/dev/dri/renderD128 \
     MESA_WRAPPER=/home/goldensend/ps4-mesa2409-runtime-20261011/run-with-mesa24.sh \
     DMESG_IGNORE="\] mmc0: " tools/gpu022_gate.sh 092d4a88-7f6f-4c4e-a1e3-238d727d6048 ABC'
   ```

   `GLES_DRM_DEVICE` makes the runner open the GPU through GBM and `EGL_PLATFORM_GBM`, the way the GPU-021 runner reached Liverpool; `MESA_WRAPPER` puts the Mesa 24.0.9 runtime's libraries first, as for GPU-021.

| Step | GPU work | Pass condition |
|---|---|---|
| A | none: compiles and links the 7 shaders (`AMD_DEBUG=cs` saves the ISA) | all compile; the renderer is not software; workgroup sizes are within the driver limit |
| B | 7 dispatches: FP16 RNE, glibc expf, rcp, ggml_v_expf, sqrt, SiLU, RMSNorm (256 rows) | all 7 outputs bit-exact. A FAIL stops the gate; C does not run |
| C | 182 dispatches: layer 0 for 7 decode steps, 26 per step, one small dispatch per process. Each step's K/V cache is built from the GPU's own earlier outputs | every output bit-exact, compared after every step |

Every dispatch has a 120 s timeout. The health checks run before A, after A, after B and after every decode step.

**Stop conditions:** the gate halts on the first of these, writes `HOLD` or `FAIL`, runs aftercare and exits. There is no retry. A HOLD before step A says "nothing touched the GPU" (boot pin, bundle hash, health and library checks only).
- The boot ID differs from the pin (HOLD, exit 2).
- The bundle hash check fails (HOLD).
- The GL libraries do not load through `MESA_WRAPPER` (HOLD; `dlopen` only, no device opened).
- A process stays in D state for 4 s or more (in all of 5 samples 1 s apart) and is a user process, looks GPU- or storage-related, or is a kernel thread `D_ALLOW` does not match (HOLD). Short D waits for journal commits and writeback on the USB root are normal and do not stop it.
- A kernel-log line newer than the gate's start (by timestamp), not matched by `DMESG_IGNORE`, mentions the GPU, DRM, the ring, a timeout, a fault, a hang, an I/O error, a reset, USB or a blocked task (HOLD).
- The temperature is above 75 C (HOLD).
- A dispatch fails or times out (HOLD).
- Step A reports a software renderer, or a renderer without `liverpool` in its name (HOLD).
- An output differs from `expected/` (FAIL, exit 4). The log names the first differing output, its stage and decode step.
- `dmesg` cannot be read (exit 3, before anything runs).

The owner should also stop it by hand (Ctrl-C) if SSH or Ethernet drops.

**Aftercare (automatic, pass or fail):** boot ID, D state, temperature, load and memory are logged; `dmesg` is saved; outputs are hashed in `gpu022/gate-out-<time>/`. The owner copies that folder back with `scp -r ps4-garden:ps4-llm-test/gpu-022/gpu022/gate-out-* $M/` and sends `gate.log`. The PS4 is left running as found.

**Rollback:** `rm -rf ~/ps4-llm-test/gpu-022` removes everything the gate added. No kernel, boot, service, package or configuration change is part of it.

**What a PASS would show:** layer 0 of a Qwen3-0.6B-shaped model, 7 decode steps, bit-exact on Liverpool against llama.cpp `23b0202a` (Jaguar flags, flash attention on). It would not show anything about the real weights, full-model inference or speed. The next step after a PASS is the same gate with the real Qwen3-0.6B weights, which needs its own approval.

**Rehearsal (LLVMPipe, cloud, not hardware):** `RESULTS/gate_rehearsal_llvmpipe.txt`. From the final `.tar.gz`, with numpy made unimportable to mirror the PS4: ABC PASS (189 dispatches). A tampered bundle HOLDs before step A. A round-toward-zero FP16 conversion FAILs in B at `B1_f16r`, and with C alone it FAILs at decode step 0 (exit 4). `TESTS/test_gate_stops.sh` checks 16 stop paths with stand-ins for `ps`, `dmesg`, the wrapper and the runner: brief D waits pass, stuck ones HOLD, `D_ALLOW` cannot excuse storage threads, a GPU fault is caught after the log wraps, `mmc0` lines are ignored only when asked, a missing library or a non-Liverpool renderer HOLDs. The bundle builder first checks 126 of 126 reference-backed outputs bit-exact against llama.cpp and the CPU references.

**Rebuilding the bundle** (needs numpy, gguf-py and Mesa LLVMPipe; the IdeaPad has no numpy):

```sh
LLAMA_CPP=/path/llama.cpp tools/build_gate_bundle.sh models/synth-q4km-s1.gguf dumps/s1_long 7 gate/gpu022-bundle.tar.gz
```

The tarball's hash changes on every build (file times), so `CHECKPOINT.json` records the committed one.
