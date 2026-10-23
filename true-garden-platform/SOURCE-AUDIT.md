# Source audit: PS4 kernels, loader, Mesa, recovery

Prepared 2026-10-23 (cloud session, no PS4 or IdeaPad contact). Sources were cloned or fetched from public GitHub at the commits below. Everything here is `PUBLIC_CLAIM` (what a project's own source or docs say) or `INFERENCE` (my reading of the source). None of it has been validated on True Garden unless a row says so.

## 0. The fact that decides which sources apply: True Garden is Belize

| Evidence | Class | Says |
|---|---|---|
| Read-only probe 2026-10-13 06:40 UTC: `00:14.2 [104d:90ca]`, `00:14.7 [104d:90cf]` | `USER_REPORTED` (context packet `LIVE_OBS`) | Sony southbridge functions 90ca/90cf |
| `pciutils/pciids` master: 90c8-90cf = **Belize**, 90d7-90de = **Baikal**, 908f-90a4 = Aeolia | `PUBLIC_CLAIM` | 90ca = Belize SATA AHCI, 90cf = Belize USB 3.0 xHCI |
| Linux psX issue #2 lists Baikal as `104d:90d7…90de` | `PUBLIC_CLAIM` | agrees with pci.ids |
| Built-in Ethernet works on True Garden through `sky2` (with a relink bug) | `USER_REPORTED` (handoff 10-14) | Belize uses the Marvell Yukon `sky2` MAC; Baikal does not (Baikal needed a new "MTS" driver) |
| sony-jaguar-devs/ps4-linux model table: "CUH-2215B: Baikal, 5.4.247" | `PUBLIC_CLAIM` | **contradicts** the live IDs |

**Verdict:** True Garden is a **Belize** console (`INFERENCE` from live PCI IDs; the community table is wrong for this unit or lists another revision). The loader also distinguishes "Belize2"; which Belize revision this is remains `UNKNOWN` until a read-only `dmesg`/sysctl look (probe item P1).

**Consequence:** the newest, best-documented work (Linux psX 6.18/7.2) is **Baikal-only on hardware**. Its own README marks Belize "reference source; unbooted". Do not read Baikal results as Belize results.

## 1. Kernel trees

| Tree (commit) | Base | Southbridges it targets | What it claims (`PUBLIC_CLAIM`) | Relevance to True Garden |
|---|---|---|---|---|
| **Running:** `5.4.247-DFAUS-blkscrn_Fix_mt7668_hdmia`, PNY bzImage sha `fc5dc10d…` | 5.4.247 | built as "baikal-dfaus", runs on our Belize | release notes (feeRnt tag `v5.4.247__baikal_mt76`): DM black-screen fix, initramfs/rescue fix, MT7668 Wi-Fi/BT, USB audio; tested on CUH-7202B | **current baseline, keep.** Exact source commit of our binary is `UNKNOWN` |
| feeRnt/ps4-linux-12xx `5.4.247-baikal-dfaus` @ `1fdfbd9a` | 5.4.247 | Baikal (also ran on CUH-2115B per notes) | the public HEAD of the branch our kernel came from | xHCI front-end calls the composite AHCI init for **every** Sony xHCI and asks for MSI vectors (`pci_alloc_irq_vectors`), the Baikal style. See section 2 |
| satadiag1 (`1721a19d…`, patch `87155b19…`) | our 5.4 | Belize change | booted once 2026-10-13: internal HDD enumerated on irq 27 | `USER_REPORTED` hardware PASS for enumeration only |
| crashniels/linux `ps4-linux-6.15.y-baikal` @ `b3b6b1e4`; also `6.15.4-aeolia-belize-crashniels` | 6.15.4 | Aeolia/Belize branch exists | AMDGPU works "with a workaround"; IRQ/APCIe conflict noted on 6.8+; CUH-1216 (Belize) tested | the closest **Belize** modern kernel with community boots |
| feeRnt `5.15.15-obsidianx-*` | 5.15.15 | **Belize** | DM black-screen fix, 88w8897 and MT7668 Wi-Fi/BT, Gladius fix, ZRAM; "will crash on Aeolia"; CUH-1216 tested | mature Belize LTS line |
| rmux "Strawberry" `zaebiz/7.1.7-Stable` (snapshot via vnkgdshrr/linux-rmuxnet; original repo 404 on 2026-10-17) | 7.1.7 | Aeolia/Belize (+ Baikal unification attempt) | 170 patches incl. Belize fan threshold/RPM hwmon, PS4 cpufreq, HDMI audio IEC958, Belize bridge retries, AHCI IRQ fixes, sky2 storm fix (Aeolia) | richest **feature** source for Belize; boot status on Belize per that maintainer, not here. Maintainer asks that others not base kernels on it |
| sony-jaguar-devs/ps4-linux `7.0-Clean` | 7.0 | all, "validated model by model" | HDMI hotplug, 1080p120, IEC958 HDMI audio, Aeolia sky2 storm fix | secondary reference |
| meerzulee/linux-ps4 @ `25ccb79f` (6.18.44, 54 patches) | 6.18.44 | **Baikal B1 only tested** | boot, USB root, HDMI 1080p60, MT7668 Wi-Fi, GL 4.6 and Vulkan 1.3 smoke tests; SATA degraded (IDENTIFY/read timeouts); fan not exposed; no suspend | method model (patch-based, immutable provenance, hashed manifests); **not a Belize kernel** |
| meerzulee/linux-psx @ `21380639` (7.2.11) | 7.2.11 | Baikal tested; `7.2-ps4` target for Aeolia/Belize "compiles; has not booted on any console" | adds UVD H.264 decode experiment, HDMI audio 44.1 kHz, internal SATA RW on a test image, MTS Ethernet | same: Baikal evidence only |

### Kernel findings that matter here
1. **SCLK may be stuck at strap clock** (`PUBLIC_CLAIM`, rmux 7.1.7 patch 0050, 2026-03-24): "after ps4-kexec performs a GFX soft-reset, the SCLK divider reverts to the fuse startup value… ~200-300 MHz… stuck at strap speed on every kernel version shipped." rmux removed the patch on 2026-08-01 "due to unuse" (patch 0163), without saying it was wrong. If true, every GPU timing so far ran at roughly a quarter to a third of the 800 MHz nominal clock. This is a **speed** lead for Track A after correctness, not a correctness lead. Needs a read-only clock register read or a measured A/B; both need the GPU thread's gate.
2. **Fan control:** rmux patch 0049 documents that the legacy `ps4fancontrol` userspace tool sends a zero-padded EMC config and **disables the fan PID loop until a hard power cycle**. Never run legacy `ps4fancontrol` on True Garden. A kernel hwmon driver (read-only RPM + threshold) exists for Aeolia/Belize in that tree only.
3. **cpufreq:** no cpufreq policy is exposed on the 6.18 Baikal kernel; rmux added a Jaguar P-state MSR driver (0135-0158). Our 5.4 state is `UNKNOWN` (probe P5).
4. **xHCI shutdown bug class:** Linux psX found that `xhci_hcd_pci_shutdown()` sent non-Belize consoles down the generic path and read the wrong driver data; on Belize the special path was used. Relevant only if we ever change kernels: reboot/power-off must be re-qualified per kernel.

## 2. SATA path (feeds the SATA lane; offline reading only)

- In every tree the internal HDD hangs off an **AHCI function embedded in the composite xHCI device** (`00:14.7`, BAR 2), set up by `ahci_init_one()` inside `xhci-aeolia.c`. The standalone AHCI function `00:14.2` (90ca) is a separate port and is where the known ATAPI `cmd 0xa1` IDENTIFY failures occur (optical drive, `USER_REPORTED` 10-13).
- 6.15/6.18/7.x trees: composite AHCI is initialised for every non-Aeolia xHCI, `ctlr->dev_id = 0`, Belize IRQs come from `apcie_assign_irqs()` (Baikal uses MSI vectors). The SATA PHY init routine is the "Belize SATA PHY init" sequence, gen-3 off for 90ca/909f/90d9.
- feeRnt 5.4.247 HEAD: composite init for every device, but IRQs via MSI vectors (`pci_alloc_irq_vectors(...MSIX|MSI)`, the `apcie_assign_irqs` call commented out).
- satadiag1 (handoff 10-14): Belize-specific branch, `dev_id = 0`, IRQ `pdev->irq + 1`; HDD appeared on irq 27 with real interrupts.
- `INFERENCE`: the Baikal-style MSI request on a Belize composite device is a plausible reason the stock 5.4 kernel never shows the HDD, and satadiag1's apcie-style IRQ is consistent with how every modern Belize tree does it. The 10-13 packet said the archived source restricted composite AHCI to the Aeolia ID; the public HEAD does not. Our binary's exact source is unknown, so **both readings stay open** until the IdeaPad source is diffed (job J2 in `NEXT.md`).
- Linux psX on Baikal saw IDENTIFY/read timeouts that delayed root discovery by ~96 s and used `libata.force=1.00:disable` to keep the internal disk out of the boot path. A boot-time SATA stall can delay USB root mount; that is a real recovery risk for any kernel that enables the internal port.

## 3. ps4-linux-loader (github.com/ps4-linux/ps4-linux-loader)

Tag `v25` = commit `9acef9fbf79097a2bb39d6c9c17228198bc445cc` (2026-07-25, "PS4 13.52 support + edid fix"); `v25.1` and later commits exist (HEAD `d0f4d4c0`, 14.00 support). Our payload `linux-1024mb.elf` sha `c813d169…` is `USER_REPORTED` as v25; that it was built from this exact commit is `UNKNOWN` (we cannot rebuild the private toolchain bit-for-bit from here).

Behaviour read from `linux/main-aio.c` at v25 (`PUBLIC_CLAIM`, answers the handoff's open question "does the ELF read files from the PNY FAT?"):
- Loads `bzImage`, `initramfs.cpio.gz` (both fatal if missing), `bootargs.txt` and `vram.txt` (optional) from, in order: `/mnt/usb0/`, `/mnt/usb1/`, `/data/linux/boot/`, `/user/system/boot/`. So the PNY FAT partition is read through Orbis's USB mount, and **a file on the internal HDD's Orbis data partition is a fallback**. Two USB sticks plugged in at once means usb0 wins, whatever that is.
- No `bootargs.txt` → built-in default cmdline (`panic=0 clocksource=tsc … radeon.dpm=0 amdgpu.dpm=0 … console=ttyS0,115200n8 console=tty0 video=HDMI-A-1:1920x1080@60`).
- `vram.txt` overrides the compiled default (1024 MB for `linux-1024mb.elf`), clamped to 32-4096.
- Reads the southbridge revision with `sysctlbyname("hw.sce_subsys_subid")` and passes it to the kexec stage (names: Aeolia A0/A1/A2, Belize A0/B0, Baikal A0/B0/B1, Belize2 A0).
- Starts a "reboot watchdog thread" in Orbis before kexec.
- `firmware_extract()` builds an extra cpio at boot containing GPU microcode (PFP/ME/CE/MEC/RLC/SDMA under `lib/firmware/amdgpu/`), the TV's EDID, and **the console's EAP HDD key at `/key/eap_hdd_key.bin`**. **Safety:** the running system's initramfs contents include a per-console secret. Never copy, hash-dump, back up to cloud, or paste anything under `/key` or the in-memory initramfs. The project rule on initramfs secrets applies to this file specifically.
- Linux psX documents the VRAM carve-out mechanics: the loader removes the VRAM region from the E820 map and programs the GPU framebuffer window and MTRRs, with a power-of-two assumption. So `linux-1024mb.elf` is why Linux sees ~6.8 GiB of the 8 GiB GDDR5. Changing it is a boot change (see `IMPROVEMENTS-RANKED.md`, item R9).

## 4. Graphics userspace (Mesa / libdrm)

| Source | Claims (`PUBLIC_CLAIM`) | Relevance |
|---|---|---|
| DionKill/ps4-video-archlinux `mesa-ps4/ps4-mesa.patch` @ `57e0f8ec` (author froyo); ported to Mesa 26.2.5 by Linux psX patch 0001 | Kernel exposes Liverpool under `FAMILY_KV`. Stock Mesa identifies it as **Kaveri**: 2 render backends instead of 4, raster config `0x00000002` instead of `0x2a00161a`. Patch adds `CHIP_LIVERPOOL`/`CHIP_GLADIUS`, LLVM processor name "bonaire", console raster layouts | Fixes **graphics** (rendering stripes seen on Baikal). Compute numerics: no evidence it matters (`INFERENCE`). Whether True Garden's isolated Mesa 24.0.9 carries this patch is `UNKNOWN` (J3) |
| Linux psX Mesa 26.2.5 dev5 candidate (patches 0001-0005) | Vulkan/RADV: HTILE D16 depth hang trigger, a compute-fill (`meta fill`) offset bound that changes ACO's lowering from FLAT to MUBUF; GPU-completion stalls still under investigation; system adoption rolled back after a Vulkan stall | **Vulkan on Liverpool is not production-safe** even on Baikal with a modern stack. Supports the current choice of GLES 3.2 compute for Track A |
| feeRnt 5.4.247-neocine release notes | "needs Mesa 25.1 or lower for GPU acceleration" on that 5.4 line | matches our choice of an isolated Mesa 24.0.9; do not upgrade Mesa system-wide on the 5.4 kernel |
| orbis-ports/mesa-ps4 | RADV for **Orbis** (no-DRM winsys, video-out WSI) | not a Linux path; ignore for this platform |
| striker91n/ps4-flatpak-mesa | Flatpak GL runtime with PS4 patches | packaging idea only |
| Linux psX `PS4-DYNAMIC-MEMORY.md` | amdgpu on PS4 has a 512 MiB GART window; USERPTR and HSA/KFD disabled in their config; GTT-backed buffers are the path to more GPU memory | no ROCm/HIP/OpenCL-via-KFD; GPU compute on PS4 Linux means GL/GLES compute or Vulkan |

## 5. Initramfs and recovery

- Our initramfs is private (it is assembled with Orbis-extracted firmware and, per section 3, the HDD key). The 10-13 sky2-handoff initramfs experiment black-screened and was rolled back (`USER_REPORTED`).
- No public "PS4 rescue initramfs" project with remote recovery was found that targets Belize and is maintained. The loader itself provides a fallback order (usb0 → usb1 → HDD paths), which is the only built-in multi-boot mechanism.
- Recovery today = physical: power-cycle, Orbis + GoldHEN + BinLoader, owner runs `send_loader.py`. The original SanDisk is the OS rollback. This is adequate for a lab; it is the main gap for "dependable system management" (see `ARCHITECTURE.md`).

## 6. Commits and URLs used

- pci.ids: `https://raw.githubusercontent.com/pciutils/pciids/master/pci.ids` (fetched 2026-10-23)
- meerzulee/linux-ps4 `25ccb79f45c228535a0e0c164b48e14deecf65d8`: `STATUS.md`, `docs/PATCH-PROVENANCE.md`, `patches/{5.4,6.18}-baikal/**`
- meerzulee/linux-psx `2138063970722631aa0a7cb7f034ba3e11b82507`: `README.md`, `docs/{PORT-7.2,RMUX-IMPORT,PS4-BOOTARGS,PS4-DYNAMIC-MEMORY}.md`, `userspace/mesa/**`, `platforms/ps4/patches/rmux-7.1.7/**`
- meerzulee/linux-psx issue #2 (CUH-2002A Baikal hardware report, 2026-10-20)
- ps4-linux/ps4-linux-loader tag `v25` = `9acef9fbf79097a2bb39d6c9c17228198bc445cc`; later fix `f70f43a` changes a 13.02 offset only (not 12.50)
- feeRnt/ps4-linux-12xx releases page; sony-jaguar-devs/ps4-linux README; ps4linux.com crashniels 6.6 article; fluxcoil.net PS4 wiki
- Not reachable from this container: gitlab.freedesktop.org, archive.mesa3d.org (so Mesa 24.0.9 radeonsi denormal handling was **not** checked here; PR #2's `test_gles_exact_lib.py` is the on-device probe for it)
