# Platform improvements, ranked

Scored on expected benefit (B), effort (E), risk to the working system (R) and reversibility (Rev), each Low/Med/High. The rank weighs benefit against risk first, because the project protects the working baseline. **Nothing here is approved.** Each item names its gate. Read-only items still wait until the GPU thread is done with the PS4.

| # | Improvement | B | E | R | Rev | Gate | Why this rank |
|---|---|---|---|---|---|---|---|
| R1 | **One read-only platform probe** (`tools/readonly_platform_probe.sh`): southbridge revision, mmc0 identity, D-state wchan classification, hwmon/fan drivers, cpufreq, GPU clock registers if exposed, kernel config, zram/swap | High (settles ~10 `UNKNOWN` rows and the 10-23 boot symptoms) | Low | Low (reads only) | n/a | read-only scope already allowed; run after GPU-022 | cheapest information per minute; feeds every other item |
| R2 | **Prove the sky2 carrier guard** on the next natural boot (read `journalctl -u ps4-sky2-carrier-guard -b`) | High (remote reachability) | Low | Low | High (rollback documented) | none beyond the next owner-approved boot | the main remote-ops failure mode |
| R3 | **Persistent HDD write protection** for any future satadiag1 boot: a udev rule that sets the internal disk read-only by serial `100621PBN308GTH1R66S` | High (removes the "writable until re-applied" window) | Low | Low | High (delete one file) | PNY root write + owner approval; test only alongside an approved satadiag1 boot | closes a known safety gap before more SATA work |
| R4 | **Diff our stock 5.4 source vs satadiag1 vs feeRnt `1fdfbd9a`** for the composite AHCI and IRQ path (offline, IdeaPad or delegated agent) | High (explains why stock hides the HDD; decides whether satadiag1 is the right long-term patch) | Low | None | n/a | offline | turns an `INFERENCE` in `SOURCE-AUDIT.md` into evidence |
| R5 | **GPU clock measurement**, then if strap is confirmed, an isolated kernel candidate with the SCLK-force patch (rmux 0050, ported to 5.4) | High for Track A speed (possibly 2-4x on compute-bound parts) | Med | Med (GPU clock change, boot change) | High (bzImage rollback) | only after GPU-022 correctness; separate boot approval; never in the same window as a SATA change | biggest plausible GPU speed lever not yet tested; correctness first |
| R6 | **Fan visibility** (read-only RPM/threshold via a 5.4 port of rmux `ps4-fan`), no writes | Med-High (needed before any sustained CPU/GPU load claim) | Med | Low if read-only | High | kernel candidate + boot approval | thermal safety for longer GPU runs |
| R7 | **Persistent `blockdev`-style hygiene in scripts**: UUID/serial-based device selection everywhere (the 016 `sda2` trap) | Med | Low | None | High | offline | already practised in 021/022; codify |
| R8 | **zram swap** (if the kernel has it) | Med (OOM headroom; Vulkan's exit 137 was an OOM kill) | Low | Low | High | persistent service change: owner approval; must not disturb atom-worker | modest, reversible headroom |
| R9 | **Smaller VRAM carve-out** (`linux-512mb.elf` or `vram.txt=512`) to return ~0.5 GiB to Linux | Med (CPU path RAM) | Low | Med (display/GPU assumptions change) | High (one file) | boot change; only after the GPU path's VRAM/GTT use is measured | helps CPU inference; could starve GPU work |
| R10 | **Modern Belize kernel candidate** (crashniels 6.15.4 aeolia-belize or 5.15.15-obsidianx, as an *isolated* bzImage+modules on a spare boot stick) | Med (newer amdgpu, composite AHCI on by default, Wi-Fi/BT fixes) | High (Debian modules, initramfs, firmware cpio, regression testing) | Med-High (IRQ conflicts reported on 6.8+, Belize untested by the best-documented project) | Med (separate stick) | separate implementation plan + approvals | real long-term value, but no current blocker requires it |
| R11 | Vulkan/RADV compute backend (llama.cpp Vulkan) | Unknown | High | High (stalls/hangs on Liverpool reported even on Mesa 26.2) | High | after GLES path is proven and a Mesa candidate is qualified | public evidence says not ready |
| R12 | UVD video decode, HDMI audio, DS4 input, KVM guests | Low for current goals | Med-High | Med | High | per-feature plan | breadth, not blocking anything |

## Explicitly not recommended now

- Replacing the running kernel, Mesa, or initramfs in place (project rule; also no evidence any current blocker needs it).
- Basing on the Baikal 6.18/7.2 trees for this console: they are Baikal-tested only.
- Legacy `ps4fancontrol` (corrupts the EMC fan config).
- Any internal-HDD partition read, mount or decrypt: the Sony EAP/UFS stack has no public Linux support, and the key material is secret.
