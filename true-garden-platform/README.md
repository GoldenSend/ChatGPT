# True Garden: PS4 Linux platform record (Tracks B and C)

Canonical engineering record for turning PS4 #1 True Garden (CUH-2215B, firmware 12.50, Debian 13) into a broadly capable PS4-specific Linux machine, built on what already works. Mission `PS4-NATIVE-LINUX-PLATFORM-001`. The GPU numerical work (Track A) lives beside this folder in `../true-garden-gpu/` (PR #2) and is not duplicated here.

Prepared 2026-10-23 in a cloud container. **Nothing in this folder has run on the PS4.**

## Where things stand

- **GPU (Track A, see PR #2):** layer 0 token 0 passes on Liverpool (7/7). Token 1 fails 573/1024 (GPU-021b). The exact blocker is that GPU arithmetic does not reproduce llama.cpp's rounding sequence, and Q8_K re-quantization amplifies any difference. A ggml-exact shader set is bit-exact offline. GPU-022 (owner-approved 20:42 UTC) is the hardware test of it. Speed is unproven: ~46 ms per layer vs 14-24 t/s on CPU.
- **Platform (Track B):** True Garden is a **Belize** console (live PCI IDs), so the best-documented modern PS4 kernels (Linux psX 6.18/7.2) are Baikal-only evidence for us. The support matrix has 9 areas. Proven today: USB-root Debian, SSH, CPU inference, GLES compute dispatch, temperature, recovery from power loss. Biggest gaps: GPU correctness and speed, Ethernet relink, internal HDD only on satadiag1, no fan visibility, recovery needs the owner at the console.
- **Distribution design (Track C):** design only. A thin, hash-pinned PS4 support layer on plain Debian: same boot chain, one patch queue, one isolated Mesa runtime with a self-test, read-only health checks, and a boot-set procedure with rollback.

## Files

| File | What |
|---|---|
| `SUPPORT-MATRIX.md` | capability-by-capability state for this console, with evidence class |
| `SOURCE-AUDIT.md` | kernels (DFAUS 5.4, crashniels, obsidianx, rmux 7.1.7, Linux psX), ps4-linux-loader v25, PS4 Mesa patches, recovery; commits and URLs |
| `DEPENDENCY-MAP.md` | what depends on what, shared single points of failure, what blocks Track A |
| `IMPROVEMENTS-RANKED.md` | R1-R12 by benefit, effort, risk, reversibility, with gates |
| `ARCHITECTURE.md` | Track C minimal platform, design only |
| `tools/readonly_platform_probe.sh` | one read-only SSH probe (P0-P10) for the `UNKNOWN` rows |
| `CHECKPOINT.md`, `RESULTS.json`, `NEXT.md` | mission state, contradiction ledger, 10-23 boot-symptom triage, next jobs |

## Four findings worth knowing first

1. **Southbridge is Belize, not Baikal.** A public table says CUH-2215B is Baikal. The live IDs say otherwise. Choose kernel sources accordingly.
2. **The loader puts the console's HDD key in the initramfs** (`/key/eap_hdd_key.bin`, built at boot). Never capture or upload `/key` or the in-memory initramfs.
3. **GPU clock may be at strap speed** (~200-300 MHz) after kexec on every ps4-linux kernel (rmux, unproven). It's a speed lead for after correctness.
4. **Never use the legacy `ps4fancontrol` tool.** Per rmux, its malformed write disables the fan PID loop until a power cycle.

## Safety

Everything here is read-only or offline. Any boot, kernel, initramfs, Mesa, VRAM, fan, service or storage change needs the owner's fresh approval for that exact operation, under the boot-set procedure in `ARCHITECTURE.md`. GoldenPi remains the only Atom authority.
