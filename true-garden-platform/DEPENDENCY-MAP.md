# Dependency map: what each capability rests on

Layers run from the bottom up. Arrows read "depends on". `[FROZEN]` means a working component the project rules protect, `[EXP]` an isolated experimental component, `[GAP]` a missing piece.

```
 Orbis FW 12.50 ──► exploit + GoldHEN 12.50 ──► BinLoader :9090
                                                   │  (owner runs send_loader.py from the IdeaPad)
                                                   ▼
 ps4-linux-loader v25  linux-1024mb.elf  [FROZEN, sha c813d169…]
   ├─ reads  /mnt/usb0|usb1 → /data/linux/boot → /user/system/boot :
   │         bzImage, initramfs.cpio.gz, bootargs.txt, vram.txt
   ├─ E820 carve-out 1 GiB VRAM, GPU FB window, MTRRs
   ├─ appends cpio: GPU microcode, EDID, /key/eap_hdd_key.bin  (SECRET)
   └─ passes FW version + southbridge id ──► kexec
                                                   ▼
 Kernel 5.4.247-DFAUS (stock fc5dc10d) [FROZEN]   |  satadiag1 1721a19d [EXP]
   ├─ ps4 platform: apcie (Belize glue 90cc), ICC, IRQ routing (pci=nocrs class)
   ├─ xhci-aeolia (00:14.7, 90cf) ─► USB root (PNY) ─► ext4 /dev/sda2 ─► everything
   │     └─ embedded AHCI (BAR2) ─► internal HDD   [stock: absent | satadiag1: irq 27]
   ├─ ahci (00:14.2, 90ca) ─► optical/ATAPI port   [IDENTIFY 0xa1 fails]
   ├─ sky2 (90c9) ─► eth0 ─► SSH, atom-worker reachability ─► Atom (GoldenPi authority)
   │     └─ ps4-sky2-carrier-guard.service [owner-approved, unproven on a failing boot]
   ├─ sdhci (90cb, mmc0) ─► SDIO Wi-Fi/BT (MT7668?)  [timeouts 10-23]
   └─ amdgpu (Liverpool 1002:9923, gmc_v7, 512 MiB GART) ◄─ microcode from loader cpio
         │   clock: amdgpu.dpm=0 by default; SCLK possibly at strap after kexec [UNKNOWN]
         ▼
 libdrm (Debian 13 system) ──► Mesa 24.0.9 radeonsi, isolated runtime [FROZEN]
         │   Liverpool identified as Kaveri unless PS4 family patch present [UNKNOWN]
         ▼
 EGL/GLES 3.2 compute ──► GPU runners 016/021/022 (renderD128) ──► layer-0 tensors
         ▼                                                          ▼
 PR #2 ggml-exact shaders [EXP, PASS_OFFLINE]       CPU oracle: llama.cpp 23b0202a, Jaguar flags
                                                    [FROZEN; also the production path]
 Debian 13 userspace (systemd, apt, Python/numpy) ── atom-worker.service [FROZEN, protected]
```

## Critical shared dependencies (single points of failure)

| Shared component | Who depends on it | Why it matters |
|---|---|---|
| Composite xHCI `00:14.7` | USB root **and** the internal HDD's AHCI | Any SATA experiment runs on the same device as the root disk. A stall there can take root down (the 10-12 D-state history; Baikal's ~96 s SATA delay). This is the reason SATA work and GPU work are never combined in one boot window |
| PNY FAT partition | bzImage, initramfs, bootargs, vram | One write here changes the next boot. Rollback = file copy, hash-checked |
| Loader cmdline / `bootargs.txt` | GPU clock policy (`amdgpu.dpm`), consoles, root selector | changing it is a boot change |
| Belize apcie IRQ routing | AHCI, xHCI, sky2, SDHCI | IRQ conflicts were the reported blocker for 6.8+ community kernels |
| sky2 link after handoff | SSH, atom-worker, every remote action | network loss = owner must go to the console |
| Isolated Mesa 24.0.9 | all GPU compute | system Mesa must not be replaced (project rule; also "Mesa ≤ 25.1" on 5.4 lines) |

## What blocks Track A (GPU), in order

1. **Numerical exactness on Liverpool.** Depends only on Mesa radeonsi codegen behaviour (rounding, `precise`, fp32 denormals, packHalf2x16) and the shader set. GPU-022 tests this directly. No kernel change is needed for it.
2. **Per-dispatch overhead** (estimated to dominate the ~46 ms/layer). Depends on the runner design and the Mesa/amdgpu submission path. No kernel change is needed to measure it.
3. **GPU clock.** If SCLK really sits at strap, the fix is a kernel patch (amdgpu, boot change). Measure first (probe P6, read-only), then decide.
4. **Memory.** Qwen3-0.6B weights (~0.4 GB) fit in either VRAM or GTT; larger models would compete with the 6.8 GiB Linux pool. A VRAM-size change is a loader/boot change and last in line.

Nothing in Track B is on the critical path for GPU-022. Items 3 and 4 become relevant only after correctness passes.
