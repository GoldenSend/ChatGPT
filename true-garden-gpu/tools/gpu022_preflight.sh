#!/usr/bin/env bash
# GPU-022 read-only preflight, run by the owner on the IdeaPad. On the PS4 it
# only runs inspection commands over ssh: nothing is copied, written, installed
# or dispatched there, and /key and the initramfs are never touched. On the
# IdeaPad it only reads the GPU-021 bundle. Prints a report to send back.
# usage: bash tools/gpu022_preflight.sh 2>&1 | tee ~/gpu022-preflight.txt
set -u
PS4_HOST=${PS4_HOST:-ps4-garden}   # not $PS4: bash uses that name for its trace prompt
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== IdeaPad: $(hostname) | clone $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)"
B="$HERE/gate/gpu022-bundle.tar.gz"
echo "bundle: $(sha256sum "$B" 2>&1 | cut -c1-64) ($(du -h "$B" 2>/dev/null | cut -f1))"

echo; echo "=== PS4 preflight over ssh $PS4_HOST (read-only)"
ssh -o ConnectTimeout=15 "$PS4_HOST" 'bash -s' 2>&1 <<'EOS'
echo "user: $(whoami) | host: $(hostname) | date: $(date -u +%FT%TZ)"
echo "kernel: $(uname -a)"
echo "cmdline: $(cat /proc/cmdline)"
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id)"
echo "uptime: $(uptime)"
echo "root: $(findmnt -no SOURCE,UUID,OPTIONS /)"
echo "ext4 errors_count: $(cat /sys/fs/ext4/*/errors_count 2>/dev/null | tr '\n' ' ')"
echo "atom-worker: $(systemctl is-active atom-worker.service 2>&1)"
echo "--- lsblk"; lsblk -o NAME,SIZE,MODEL,TRAN,MOUNTPOINT
echo "--- D-state, 10 samples over 20 s (count pid ppid stat wchan comm)"
for i in 1 2 3 4 5 6 7 8 9 10; do ps -eo pid,ppid,stat,wchan:32,comm | awk 'NR > 1 && $3 ~ /^D/'; sleep 2; done | sort | uniq -c
echo "--- dmesg readable by $(whoami)? $(dmesg >/dev/null 2>&1 && echo yes || echo NO)"
echo "--- dmesg counts: ata $(dmesg 2>/dev/null | grep -ciE 'ata[0-9]+.*(identify|fail|error|exception)') | mmc0 $(dmesg 2>/dev/null | grep -c mmc0) | usb err/reset/disconnect $(dmesg 2>/dev/null | grep -ciE 'usb.*(error|reset|disconnect)') | I/O error $(dmesg 2>/dev/null | grep -ci 'i/o error')"
echo "--- dmesg ata (last 6)"; dmesg 2>/dev/null | grep -iE 'ata[0-9]+' | grep -iE 'identify|fail|error|exception|cmd' | tail -6
echo "--- dmesg mmc0 (last 4)"; dmesg 2>/dev/null | grep mmc0 | tail -4
echo "--- dmesg usb/sd errors (last 6)"; dmesg 2>/dev/null | grep -iE 'usb.*(error|reset|disconnect)|sd[a-z].*(error|reset)|i/o error' | tail -6
echo "--- dmesg GPU/DRM (last 15)"; dmesg 2>/dev/null | grep -iE 'amdgpu|radeon|\[drm\]|liverpool|gfx' | tail -15
echo "--- temps (milli-C): $(cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | tr '\n' ' ')"
echo "--- mem: $(free -m | sed -n 2p)"
echo "--- disk ~: $(df -h ~ | sed 1d)"
echo "--- python3: $(python3 --version 2>&1); tar: $(tar --version | head -1); gzip: $(command -v gzip)"
echo "--- DRM nodes: $(ls /dev/dri 2>&1 | tr '\n' ' ')"
for c in /sys/class/drm/card*/device/uevent; do grep -hE 'DRIVER|PCI_ID' "$c" 2>/dev/null; done | sort -u
echo "--- libs: $(ls /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/lib/x86_64-linux-gnu/libGLESv2.so.2 /usr/lib/x86_64-linux-gnu/libgbm.so.1 2>&1 | tr '\n' ' ')"
echo "--- existing: $(ls -d ~/ps4-llm-test ~/ps4-llm-test/gpu-022 2>&1 | tr '\n' ' ')"
W=/home/goldensend/ps4-mesa2409-runtime-20261011/run-with-mesa24.sh
echo "--- $W"; [ -f "$W" ] && sed -n 1,30p "$W" || echo "not present"
EOS

echo; echo "=== GPU-021 runner: how it opened Liverpool (read-only)"
D=$HOME/Projects/atom-ps4-linux-compat-001/ps4-takeover-reconcile-20261013-002
T=$D/liverpool-021-gate/021-BUNDLE.tar
tar -xOf "$T" --wildcards '*full_layer_ps4_supervised.py' 2>/dev/null | grep -nE "dri|render|card|os.open|LIBGL|MESA|environ|DRI_PRIME|AMD_" | head -20
echo; echo "=== GPU-021 attention shader (read-only, for the numerical analysis)"
tar -xOf "$T" --wildcards '*attn_two_token_gqa.comp' 2>/dev/null | head -160
echo; echo "=== GPU-021 Q4_K matvec reduction (key lines)"
tar -xOf "$T" --wildcards '*attn_q4k_q8k_avx_order.comp' 2>/dev/null | grep -nE "for|acc|sum|shared|barrier|precise|fma" | head -40
echo; echo "=== done"
