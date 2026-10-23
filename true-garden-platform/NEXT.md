# NEXT

## One highest-value next action

**After GPU-022 finishes and the GPU thread releases the PS4,** the owner runs the read-only probe once from the IdeaPad and attaches the output to this thread:

```sh
bash true-garden-platform/tools/readonly_platform_probe.sh > probe-$(date -u +%Y%m%dT%H%M%SZ).txt
```

ETA: under 1 minute of PS4 time. It writes nothing on the PS4 and does not touch `/dev/dri`. It settles about ten `UNKNOWN` rows in `SUPPORT-MATRIX.md` and gives the 10-23 boot symptoms a wait-channel classification.

## Queue (offline unless stated)

| ID | Job | Who | Gate |
|---|---|---|---|
| J1 | read-only probe above | owner runs; lead reviews | after GPU-022 |
| J2 | diff stock 5.4 source (IdeaPad) vs satadiag1 vs feeRnt `1fdfbd9a` for `xhci-aeolia.c` / `ahci.c` / IRQ assignment; answer C2 | agy or opencode on the IdeaPad, read-only; lead verifies | none (offline) |
| J3 | find out whether the isolated Mesa 24.0.9 runtime carries the PS4 Liverpool family patch (read its build receipt and patch list, or `grep -a -c -i liverpool` over its gallium/radeonsi shared library; stock Mesa has no Liverpool chip name) | agy/opencode, offline on the IdeaPad | none |
| J4 | port rmux `ps4-fan` hwmon as **read-only** (no threshold write) and the SCLK-force patch onto the 5.4 source as an isolated candidate; compile only | agy/opencode | boot needs a separate owner approval |
| J5 | draft the persistent HDD read-only udev rule (by serial) for the next satadiag1 boot | lead | install is a PNY root write: owner approval |
| J6 | `BOOTSET.json` manifest + `send_loader` wrapper that checks the FAT set before sending (`ARCHITECTURE.md`) | lead, offline | none until used |

Superseded by this mission: nothing. Earlier checkpoints are unchanged.
