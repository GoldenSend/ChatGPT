# Hardware support matrix: PS4 #1 True Garden (CUH-2215B, Belize, FW 12.50)

Stack: Debian 13, kernel `5.4.247-DFAUS-blkscrn_Fix_mt7668_hdmia` (PNY bzImage stock `fc5dc10d…`), ps4-linux-loader v25 `linux-1024mb.elf`, isolated Mesa 24.0.9 for GPU work. Prepared 2026-10-23 from records only; **no new probe was run for this file.**

**States**
- `PROVEN` operational on this console (with the evidence date)
- `PARTIAL` works with a named defect or limit
- `EXTERNAL` supported in public source or on other PS4s, not validated here
- `BLOCKED` blocked by a named missing driver, kernel feature or defect
- `HW-LIMITED` the hardware can't do it, or it isn't exposed to Linux at all
- `UNKNOWN` no evidence yet; the probe ID points to `tools/readonly_platform_probe.sh`

**Evidence classes:** `LIVE_OBS`, `ARTIFACT_VERIFIED`, `USER_REPORTED`, `PUBLIC_CLAIM`, `INFERENCE`, `UNKNOWN` (as in the project rules). Rows dated before 2026-10-23 describe earlier boots. The 10-23 power loss means only the 10-23 recovery facts describe the current boot.

**Important distinction:** the hardware has a capability (for example, Liverpool has UVD/VCE video blocks) ≠ the installed Linux stack exposes it ≠ it works on this console.

## 1. Liverpool GPU (1002:9923, GCN 1.1 / GFX7, 18 CU)

| Capability | State | Evidence |
|---|---|---|
| amdgpu kernel driver, render node `/dev/dri/renderD128` | `PROVEN` (2026-10-12/14 runs) | GPU-012/014/016/021b opened the node and dispatched (`USER_REPORTED`, packet + handoff) |
| OpenGL ES 3.2 compute (radeonsi, isolated Mesa 24.0.9) | `PARTIAL` | Q4_K canary 1024/1024; Hybrid 003/004 logits bit-identical; layer 0 token 0 7/7; **token 1 573/1024 bad** (GPU-021b). Numerically incorrect for multi-token today (`USER_REPORTED`) |
| Bit-exact ggml-matching compute shaders | `PASS_OFFLINE` only | PR #2, LLVMPipe, synthetic weights; GPU-022 pending (`ARTIFACT_VERIFIED` offline) |
| Desktop OpenGL (GL 4.6 radeonsi) | `EXTERNAL` | Baikal consoles with patched Mesa (Linux psX, issue #2). Not tried here; True Garden runs headless |
| Vulkan (RADV) | `BLOCKED` | here: historical process exit 137 (SIGKILL, likely OOM) (`USER_REPORTED`). Elsewhere: RADV smoke tests pass on Baikal, but Linux psX rolled back a system Mesa because of Vulkan GPU-completion stalls and an HTILE hang (`PUBLIC_CLAIM`). Not production-safe on any PS4 yet |
| OpenCL / ROCm / HIP | `HW-LIMITED` (stack) | GFX7 is outside ROCm support; PS4 configs disable HSA/KFD and USERPTR (`PUBLIC_CLAIM`, Linux psX) |
| GPU synchronization (fences, readback) | `PARTIAL` | runs completed with aftercare PASS; earlier runs refused on D-state guard. No soak (`USER_REPORTED`) |
| GPU clock / power management | `UNKNOWN` (P6) | default loader cmdline has `amdgpu.dpm=0`; rmux reports SCLK stuck at strap ~200-300 MHz after kexec (`PUBLIC_CLAIM`, unproven) |
| Performance vs CPU | `BLOCKED` | ~46 ms per layer (~0.8 t/s extrapolated) vs 14-24 t/s CPU; hybrid 3.0 vs 7.6 t/s (`USER_REPORTED`). Blocked first on correctness, then on dispatch overhead and possibly the clock |
| UVD H.264 decode | `EXTERNAL` | 900-frame exact decode on Baikal with an experimental kernel transport + Mesa 26.2.5 (`PUBLIC_CLAIM`). Not in our 5.4 stack (`INFERENCE`) |
| VCE encode | `UNKNOWN` | not qualified anywhere public |
| HDMI scanout / KMS | `UNKNOWN` (P7) | the DM black-screen fix is in our kernel's name; headless use so far |

## 2. CPU (8x AMD Jaguar, 2 modules)

| Capability | State | Evidence |
|---|---|---|
| 8 cores online, SMP scheduling | `PROVEN` | llama.cpp 4/6/8-thread runs (`USER_REPORTED`) |
| ISA: SSE4.2, AVX, F16C; **no AVX2, no FMA3** | `PROVEN` | CPU llama.cpp build flags `-march=btver2`, AVX2/FMA off (`USER_REPORTED`); matches Jaguar spec |
| LLM CPU inference | `PROVEN` | Qwen3-0.6B Q4_K_M tg64 14.17/19.65/24.3 t/s at 4/6/8 threads (`USER_REPORTED`); atom-worker active 2026-10-23 (`USER_REPORTED`) |
| cpufreq / P-states | `UNKNOWN` (P5) | none exposed on the Baikal 6.18 kernel; out-of-tree Jaguar driver in rmux 7.1.7 (`PUBLIC_CLAIM`) |
| Sustained full-load thermals | `UNKNOWN` | ~48.5-49 °C idle readings (`USER_REPORTED`); no fan visibility, so no qualified sustained-load envelope |
| KVM (SVM) | `EXTERNAL` | `/dev/kvm` present on a Baikal 6.18 build (`PUBLIC_CLAIM`); ours `UNKNOWN` (P9) |

## 3. Memory (8 GiB GDDR5, unified)

| Capability | State | Evidence |
|---|---|---|
| Linux-visible RAM ~6.8 GiB | `PROVEN` | ~6.83 GiB (10-13), ~6.65 GiB available after the 10-23 recovery (`USER_REPORTED`) |
| Fixed 1 GiB VRAM carve-out | `PROVEN` (by design) | `linux-1024mb.elf` default; the loader removes it from E820 (`PUBLIC_CLAIM`, loader source) |
| Swap / zram | `UNKNOWN` (P8) | "no swap" on 10-13 (`USER_REPORTED`); zram is in some community kernels' configs |
| GPU buffers from system memory (GTT) | `EXTERNAL` | amdgpu TTM/GTT path, 512 MiB GART window on PS4 (`PUBLIC_CLAIM`); whether our runner uses VRAM or GTT is `UNKNOWN` |
| Runtime RAM/VRAM resizing | `HW-LIMITED` (stack) | no mechanism; only a boot-time carve-out change (`PUBLIC_CLAIM`) |
| DMA safety | `PARTIAL` | southbridge devices use 31-bit DMA masks; IOMMU is patched/restricted in PS4 trees (`PUBLIC_CLAIM`). No DMA fault seen in our records |
| Bandwidth | `UNKNOWN` | ~176 GB/s is the public spec; never measured under Linux here |

## 4. Storage

| Capability | State | Evidence |
|---|---|---|
| USB-root boot (PNY, ext4 `/dev/sda2`, xHCI 00:14.7) | `PROVEN` (10-23) | root rw, ext4 errors 0 after the power loss (`USER_REPORTED`) |
| Filesystem integrity after power loss | `PARTIAL` | errors 0 on one recovery; no fsck run (none authorised) |
| Internal HDD (Hitachi HTS545032B9SA00, 320 GB) enumeration | `PARTIAL` | **only on satadiag1** (10-13, irq 27, read-only set by hand). **Not on the current stock kernel** (`USER_REPORTED`) |
| Internal HDD read (partition table, data) | `BLOCKED` | not authorised; Sony partitions are encrypted (EAP/UFS stack not supported by any public Linux tree) |
| Standalone AHCI 00:14.2 (optical port) | `BLOCKED` | ATAPI `cmd 0xa1` IDENTIFY failures every boot (`USER_REPORTED`); Blu-ray drive not detected on any public tree |
| mmc0 (Belize SD/MMC host, 90cb) | `PARTIAL` | timeouts seen at the 10-23 boot (`USER_REPORTED`). `INFERENCE`: this host carries the SDIO Wi-Fi/BT, not root storage. Confirm with P3 |
| Persistent HDD write protection | `BLOCKED` | `blockdev --setro` does not survive a reboot; would need a udev rule or `libata.force` |
| Recovery | `PARTIAL` | original SanDisk = OS rollback; single AES archive of the old PNY on the HP; restore never exercised (`USER_REPORTED`) |

## 5. Networking

| Capability | State | Evidence |
|---|---|---|
| Built-in Ethernet (sky2, Belize) | `PARTIAL` | works; boot handoff toggles link at ~62 s and the chip never relinks; manual unbind/bind fix proven; auto-guard installed, **not yet proven on a failing boot** (`USER_REPORTED`) |
| SSH | `PROVEN` (10-23) | (`USER_REPORTED`) |
| Wi-Fi (MT7668 SDIO, per kernel name) | `UNKNOWN` (P3) | driver is in our kernel per release notes (`PUBLIC_CLAIM`); never used here |
| Bluetooth | `UNKNOWN` (P3) | works on other consoles with MT7668/Marvell (`PUBLIC_CLAIM`) |
| Throughput / reconnect reliability | `UNKNOWN` | never measured; a cable-pull test is a deliberate network disruption and needs approval |
| USB Ethernet fallback (r8152) | `EXTERNAL` | works on a Baikal 6.18 build (`PUBLIC_CLAIM`); config here `UNKNOWN` (P9) |

## 6. Media

| Capability | State | Evidence |
|---|---|---|
| HDMI display | `UNKNOWN` (P7) | not used; the DM black-screen fix is in this kernel |
| HDMI audio | `UNKNOWN` | "hdmia" in the kernel name is not explained by its release notes. Elsewhere: fixed by an IEC958 patch (rmux), 44.1 kHz clean / 48 kHz broken (Baikal) (`PUBLIC_CLAIM`) |
| Hardware video decode/encode | `BLOCKED` | UVD needs an experimental kernel transport not in 5.4 (`INFERENCE`) |
| USB audio | `EXTERNAL` | listed in the kernel's release notes (`PUBLIC_CLAIM`) |

## 7. Peripherals

| Capability | State | Evidence |
|---|---|---|
| USB 3 host ports (xhci-aeolia) | `PROVEN` | the root disk itself |
| USB mass storage, keyboard | `PROVEN` | root disk; owner logs in at the console (`USER_REPORTED`) |
| DualShock 4 (BT or USB) | `UNKNOWN` | works elsewhere on 5.x/6.x; disabled since 6.3 in one report (`PUBLIC_CLAIM`) |
| Front-panel LED, power button | `EXTERNAL` | drivers in rmux 7.1.7 (`PUBLIC_CLAIM`); ours `UNKNOWN` (P10) |
| Sensors | `PARTIAL` | one GPU/APU temperature via `hwmon1/temp1_input`; which driver provides it is `UNKNOWN` (P4) |

## 8. System management

| Capability | State | Evidence |
|---|---|---|
| Temperature reading | `PROVEN` | ~48.5-49 °C (`USER_REPORTED`, 10-13 and 10-23) |
| Fan RPM / threshold | `UNKNOWN` (P4) | a hwmon driver exists only in rmux 7.1.7 for Aeolia/Belize (`PUBLIC_CLAIM`). **Never use the legacy `ps4fancontrol` tool**: its malformed write disables the fan PID loop until a power cycle (`PUBLIC_CLAIM`) |
| Clean power-off | `PROVEN` once | 2026-10-14 06:49 UTC under approval (`USER_REPORTED`) |
| Reboot into Linux | `HW-LIMITED` | there is no native cold boot: every start is Orbis → exploit → GoldHEN → BinLoader → loader → kexec. "Reboot" returns to Orbis |
| Suspend / resume | `HW-LIMITED` (stack) | `/sys/power/state` absent on PS4 kernels (`PUBLIC_CLAIM`) |
| Hardware watchdog | `UNKNOWN` (P10) | the loader runs a watchdog thread in Orbis only before kexec (`PUBLIC_CLAIM`) |
| Remote recovery | `BLOCKED` | needs the owner at the console; no out-of-band management |
| Boot diagnostics | `PARTIAL` | dmesg/journal after boot; UART is possible but needs soldering (`PUBLIC_CLAIM`) |
| Unplanned power-loss survival | `PROVEN` once | 2026-10-23: relaunch worked, ext4 errors 0 (`USER_REPORTED`) |

## 9. General Linux capabilities

| Capability | State | Evidence |
|---|---|---|
| Debian 13 userspace, apt | `PROVEN` | the running system |
| systemd services | `PROVEN` | atom-worker, ssh, sky2 guard (`USER_REPORTED`) |
| Native toolchains (gcc, Python, numpy) | `PARTIAL` | llama.cpp was built for the PS4; numpy present per the GPU thread (`USER_REPORTED`) |
| Containers (namespaces, cgroup v2, overlayfs) | `UNKNOWN` (P9) | depends on the kernel config; check `/proc/config.gz` |
| Kernel age vs userspace | `PARTIAL` | `INFERENCE`: Debian 13 is built and tested against its 6.12 kernel; running it on 5.4 works today but newer systemd and container features may assume interfaces 5.4 lacks. Treat as a known compatibility debt, not a current failure |
| Filesystems | `PARTIAL` | ext4 and vfat proven; others `UNKNOWN` (P9) |

## Summary by state (current boot)

- `PROVEN`: USB-root Debian boot via the loader chain, SSH, CPU inference (the Atom worker), 8-core scheduling, ~6.8 GiB RAM, temperature read, power-loss recovery once, GLES compute dispatch.
- `PARTIAL`: GPU correctness (token 1), Ethernet relink, internal HDD (satadiag1 only), mmc0 health, recovery tooling.
- `BLOCKED`: practical GPU speedup, Vulkan, HDD data access, optical drive, hardware video, remote recovery.
- `HW-LIMITED`/stack-limited: native cold boot, suspend, ROCm/OpenCL, runtime VRAM resizing.
- `UNKNOWN` and cheap to settle with one read-only probe: southbridge revision, mmc0 identity, fan/hwmon drivers, cpufreq, GPU clock, swap/zram, kernel config (containers, KVM, filesystems), Wi-Fi/BT presence.
