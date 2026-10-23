# GPU-022 gate: exact operation for owner approval

**Question it answers:** does Liverpool (radeonsi) compute the ggml-exact shaders bit-exactly, the way LLVMPipe does? It is a qualification of shader arithmetic on synthetic Qwen3-0.6B-shaped weights. It is not a speed test, not the real model, and not production use.

**Approval wording:** "Approve GPU-022 ABC" covers exactly the staging write and the one gate command below, for one session, on the boot whose ID is typed into that command. Anything else (a second attempt, a reboot, a package install, a kernel change, the real model) needs its own yes.

## Phase 1: before approval (read-only on the PS4)

Done by the lead through Remote Control on the HP Pavilion, IdeaPad, then `ssh ps4-garden`. Nothing on the PS4 is written.

| Check | Command | Needed for GO |
|---|---|---|
| Kernel | `uname -a; cat /proc/cmdline` | stock kernel (last record `fc5dc10d`, 2026-10-14). If it is satadiag1, stop and ask |
| HDD visible? | `lsblk -o NAME,SIZE,MODEL,SERIAL` | no Hitachi 320 GB line (that would mean satadiag1) |
| Boot ID | `cat /proc/sys/kernel/random/boot_id` | recorded; it becomes the pin |
| Hung tasks | `ps -eo pid,stat,comm \| awk '$2 ~ /^D/'` | prints nothing |
| Kernel log | `dmesg --level=err,warn \| tail -40` | no GPU fault, ring timeout or I/O error |
| Temperature | `cat /sys/class/hwmon/hwmon*/temp*_input` | at or below 75 C |
| Python | `python3 -c "import numpy; print(numpy.__version__)"` | present (no installs) |
| Build tools | `gcc --version; ls /usr/include/EGL/egl.h /usr/include/GLES3/gl32.h` | present, or an IdeaPad-built `gles_run` that runs here (`ldd`) |
| 021 driver setup | read the 021 runner's launch script in `liverpool-021-gate/` | tells whether `MESA_WRAPPER=/home/goldensend/ps4-mesa2409-runtime-20261011/run-with-mesa24.sh` is needed |

On the IdeaPad, in a new directory `~/Projects/atom-ps4-linux-compat-001/gpu-022-offline/`: clone this branch, run `LIBGL_ALWAYS_SOFTWARE=1 WORK=... ./reproduce.sh` (must end `ALL OFFLINE CHECKS DONE`), and pack the bundle:

```
gpu-022/
  true-garden-gpu/            this folder at the approved commit
  models/synth-q4km-s1.gguf   sha 49e72c74...      models/synth-q4km-s2.gguf  sha 039a2690...
  dumps/s1_fa1_t1/  dumps/s2_fa1/                   (llama.cpp 23b0202a, Jaguar flags)
  llama-gguf/gguf-py/
  gles_run                    (built for the PS4, or built on it during staging)
  SHA256SUMS
```

About 40 MB.

## Phase 2: what the approval covers

1. **Staging (one disk write):** copy the bundle to the new directory `~/ps4-llm-test/gpu-022/` on the PS4 and run `sha256sum -c SHA256SUMS`. A mismatch means HOLD. If `gles_run` is built on the PS4, it is built inside that directory with `gcc -O2 -o gles_run true-garden-gpu/tools/gles_run.c -lEGL -lGLESv2`.
2. **The gate command, run once by the owner.** Typing the boot ID is the boot pin:

```sh
cd ~/ps4-llm-test/gpu-022/true-garden-gpu
BUNDLE=~/ps4-llm-test/gpu-022 GLES_RUN_BIN=~/ps4-llm-test/gpu-022/gles_run \
  [MESA_WRAPPER=...] tools/gpu022_gate.sh <BOOT_ID> ABC
```

| Step | GPU work | Pass condition |
|---|---|---|
| A | none: compiles and links 7 shaders (`AMD_DEBUG=cs` saves the ISA) | all compile; renderer is not software; workgroup sizes are within the driver limit |
| B | 7 dispatches (FP16 RNE, expf, rcp, ggml_v_expf, sqrt, SiLU, RMSNorm on 4096 rows) | `RESULT: PASS`. A FAIL stops the gate; C does not run |
| C | 52 dispatches (2 decode steps), then 130 (5 steps); one small dispatch per process | every stage bit-exact at every step |

Every dispatch has a 120 s timeout. Between steps the script checks the stop conditions.

**Stop conditions:** the gate halts on the first of any of these, writes `HOLD`, runs aftercare and exits. There is no retry.
- The boot ID differs from the pin.
- Any process is in D state.
- New kernel messages mention the GPU, DRM, the ring, a timeout, a fault, a hang, an I/O error, a reset or USB.
- The temperature is above 75 C.
- A dispatch times out.
- A step fails.

The owner should also stop it by hand if SSH or Ethernet drops.

**Aftercare (automatic, pass or fail):** boot ID, D-state, temperature, load and memory are logged; `dmesg` is saved; outputs are hashed in `gate-out-<time>/`. The lead copies that folder to the IdeaPad mission directory. The PS4 is left running as found.

**Rollback:** `rm -rf ~/ps4-llm-test/gpu-022` removes everything the gate added. No kernel, boot, service, package or configuration change is part of it.

**What a PASS would show:** layer 0 of a Qwen3-0.6B-shaped model, 7 decode steps in total, bit-exact on Liverpool against llama.cpp `23b0202a` (Jaguar flags). It would not show anything about the real weights, full-model inference or speed. The next step after a PASS is the same gate with the real Qwen3-0.6B weights, which needs its own approval.

**Rehearsal:** the script was run end to end in the cloud on LLVMPipe (`ALLOW_SOFTWARE=1`): A, B and C passed, and the wrong-boot-ID and software-renderer stops both fired (`RESULTS/gate_rehearsal_llvmpipe.txt`).
