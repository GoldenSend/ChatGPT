# CHECKPOINT: PS4-NATIVE-LINUX-PLATFORM-001 (Tracks B and C)

Written 2026-10-23 ~21:10 UTC by the cloud platform thread. **No PS4, IdeaPad or HP command was run.** All work is offline source audit and documentation. The GPU thread had sole use of the PS4 during GPU-022.

## 1. Fresh state (as recorded, not re-probed)

| Item | Value | Class |
|---|---|---|
| Boot | relaunched after the 2026-10-23 power loss via GoldHEN 12.50 + loader v25 | `USER_REPORTED` |
| Kernel | `5.4.247-DFAUS-blkscrn_Fix_mt7668_hdmia`, PNY bzImage stock `fc5dc10d…` since 10-14 | `USER_REPORTED` |
| Root | `/dev/sda2` ext4 on PNY USB, rw, errors 0 | `USER_REPORTED` |
| Services | SSH and atom-worker active | `USER_REPORTED` |
| Memory | ~6.65 GiB available just after recovery | `USER_REPORTED` |
| Boot symptoms | SATA IDENTIFY failures, recurring mmc0 timeouts, D-state count fluctuating | `USER_REPORTED`, not yet judged |
| Boot ID | new and not recorded here; every earlier boot ID and preflight is stale | rule |
| Southbridge | **Belize** (90ca/90cf), not Baikal | `INFERENCE` from 10-13 `LIVE_OBS` IDs + pci.ids |

## 2. Contradiction ledger

| # | Claim A | Claim B | Resolution |
|---|---|---|---|
| C1 | CUH-2215B = Baikal (sony-jaguar-devs table) | live PCI IDs 90ca/90cf = Belize | live IDs win; confirm revision with probe P1 |
| C2 | 10-13 packet: archived 5.4 source limits composite AHCI to the Aeolia xHCI ID | public feeRnt 5.4.247 HEAD `1fdfbd9a` inits composite AHCI for all IDs, with MSI vectors | open; diff the actual stock source (NEXT J2). Either way the stock kernel hides the HDD |
| C3 | Handoff 10-14: "unknown whether the ELF reads files from the PNY FAT" | loader v25 source reads bzImage/initramfs/bootargs/vram from `/mnt/usb0`, `/mnt/usb1`, then two HDD paths | answered from source (`PUBLIC_CLAIM`); our ELF's build commit is still `UNKNOWN` |
| C4 | Linux psX 6.18/7.2 "works": GL, Vulkan, Wi-Fi, SATA RW | those results are Baikal-only; Belize "unbooted" in their own table | not transferable to True Garden |
| C5 | GPU speed measurements so far | rmux: SCLK stuck at strap ~200-300 MHz after kexec on all ps4-linux kernels | unproven; measure (P6, then a gated test) |

## 3. Triage of the 10-23 boot symptoms (`INFERENCE`; the GPU-022 preflight samples these live)

| Symptom | Most likely source | What would make it serious | Read-only check |
|---|---|---|---|
| SATA IDENTIFY failures | the known ATAPI optical port on standalone AHCI `00:14.2` (`cmd 0xa1`), seen on every earlier boot | errors on a port that has a disk; any `sd`/`usb-storage` error on the PNY; ext4 errors > 0 | `dmesg` ata lines (P2): port number and command; `errors_count` |
| mmc0 timeouts | Belize SD/MMC host `90cb`, which carries the SDIO Wi-Fi/BT. Root is USB on `00:14.7` | timeouts continuing for hours, D-state threads with mmc wchan blocking unrelated work | P3: `mmc_host` → PCI function, SDIO card IDs |
| Fluctuating D-state | likely transient waits in the ATA error handler or mmc kworkers during those retries | D-state in `jbd2/sda2`, `usb-storage`, `kworker` with `blk_*`/`io_schedule` wchan, or in the GPU runner | D-state census with wchan, 3 samples (in the probe) |

**Update 2026-10-23 21:27Z (owner-run read-only preflight, boot `092d4a88`, relayed by the GPU thread):** the triage holds. ata1 ATAPI IDENTIFY timeouts at 7, 18 and 49 s are the optical port and occur only at boot. mmc0 logged 937 timeout lines, the last at 533 s, and has no `mmcblk` device, so it is not storage. The D-state tasks seen were jbd2/sda2-8, flush-8:0 and a freezable kworker, each in 1 of 10 samples: ordinary I/O waits. `wchan` is "-" on this kernel, so the probe now reads `/proc/PID/stack` for D tasks.

None of this justifies relaxing a guard. The GPU thread's gate already separates user processes, GPU/storage threads and known-unrelated threads.

## 4. What was produced

`SOURCE-AUDIT.md`, `SUPPORT-MATRIX.md`, `DEPENDENCY-MAP.md`, `IMPROVEMENTS-RANKED.md`, `ARCHITECTURE.md`, `tools/readonly_platform_probe.sh`, `RESULTS.json`, `NEXT.md`.

Test of the probe: `bash -n` passes; the remote part was dry-run inside the cloud container (not a PS4) with `WITH_GPU_SYSFS=1`, exit 0, all 14 sections printed, missing tools reported and skipped. That shows it is well-formed and fails soft. It is not evidence about the PS4.

## 5. Safety record

No PS4 contact, no device write, no boot, no GPU dispatch, no storage access. Findings sent to the GPU thread: southbridge identity, mmc0/SATA triage, the loader's HDD key in the initramfs, the SCLK and Mesa-identity leads.

## 6. State labels

- Track B: `PASS_OFFLINE` for the audit and documents; probe `HOLD` until the PS4 is free.
- Track C: `DESIGN_ONLY`.
- Nothing is `PASS_HARDWARE` or `PRODUCTION_ELIGIBLE` from this mission.
