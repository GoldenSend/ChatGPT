#!/usr/bin/env bash
# GPU-022 phase 1, run by the owner on the IdeaPad from a fresh clone:
#   1. read-only PS4 preflight over ssh (inspection commands only; nothing is
#      copied to, written on, or run on the PS4's GPU)
#   2. read-only look at how the GPU-021 runner launched GLES
#   3. offline build + tests on LLVMPipe, then the GPU-022 bundle
# Writes only under $M (default: the folder that holds the clone).
# usage: bash tools/gpu022_prepare.sh      then paste $M/REPORT.txt back
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
M=${M:-$(cd "$HERE/../.." && pwd)}
PS4=${PS4:-ps4-garden}
R="$M/REPORT.txt"; : > "$R"
say() { echo "$*" | tee -a "$R"; }
section() { say ""; say "=== $*"; }

section "IdeaPad"
say "$(hostname) | $(uname -r) | $(ldd --version | head -1) | clone $(git -C "$HERE" rev-parse --short HEAD)"
for t in gcc g++ cmake git python3; do say "$t: $(command -v $t || echo MISSING)"; done
say "$(python3 -c 'import numpy; print("numpy", numpy.__version__)' 2>&1)"
say "$(ls /usr/include/EGL/egl.h /usr/include/GLES3/gl32.h 2>&1 | tr '\n' ' ')"

section "PS4 preflight (read-only)"
ssh -o ConnectTimeout=15 "$PS4" 'bash -s' 2>&1 <<'EOS' | tee -a "$R"
echo "user: $(whoami) | host: $(hostname)"
echo "kernel: $(uname -a)"
echo "cmdline: $(cat /proc/cmdline)"
echo "boot_id: $(cat /proc/sys/kernel/random/boot_id)"
echo "uptime: $(uptime)"
echo "--- lsblk"; lsblk -o NAME,SIZE,MODEL,SERIAL,MOUNTPOINT
echo "root: $(findmnt -no SOURCE,UUID,OPTIONS /)"
echo "ext4 errors_count: $(cat /sys/fs/ext4/*/errors_count 2>/dev/null | tr '\n' ' ')"
echo "atom-worker: $(systemctl is-active atom-worker.service 2>&1)"
echo "--- D-state, 10 samples over 20 s (count pid ppid stat wchan comm)"
for i in 1 2 3 4 5 6 7 8 9 10; do ps -eo pid,ppid,stat,wchan:32,comm | awk 'NR > 1 && $3 ~ /^D/'; sleep 2; done | sort | uniq -c
echo "--- dmesg readable by $(whoami)? $(dmesg >/dev/null 2>&1 && echo yes || echo NO)"
echo "--- dmesg counts: ata/SATA $(dmesg 2>/dev/null | grep -ciE 'ata[0-9]+.*(identify|fail|error|exception)'), mmc0 $(dmesg 2>/dev/null | grep -c mmc0), usb errors $(dmesg 2>/dev/null | grep -ciE 'usb.*(error|reset|disconnect)'), I/O errors $(dmesg 2>/dev/null | grep -ci 'i/o error')"
echo "--- dmesg ata/mmc samples"; dmesg 2>/dev/null | grep -iE 'ata[0-9]+.*(identify|fail|error)|mmc0' | tail -8
echo "--- dmesg GPU/DRM"; dmesg 2>/dev/null | grep -iE 'amdgpu|radeon|drm|liverpool|gfx' | tail -15
echo "--- dmesg err/warn, last 25"; dmesg --level=err,warn 2>&1 | tail -25
echo "--- temps (milli-C)"; cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | tr '\n' ' '; echo
echo "--- mem / disk"; free -m | sed -n 2p; df -h ~ | sed 1d
echo "--- tools"; for t in gcc python3; do echo "$t: $(command -v $t || echo MISSING)"; done
python3 -c 'import numpy; print("numpy", numpy.__version__)' 2>&1
ls /usr/include/EGL/egl.h /usr/include/GLES3/gl32.h 2>&1
ldd --version | head -1
echo "--- existing dirs"; ls -d ~/ps4-llm-test ~/ps4-llm-test/gpu-022 2>&1
echo "--- DRM"; ls /dev/dri 2>&1 | tr '\n' ' '; echo; cat /sys/class/drm/card*/device/uevent 2>/dev/null | grep -E 'DRIVER|PCI_ID' | sort -u
W=/home/goldensend/ps4-mesa2409-runtime-20261011/run-with-mesa24.sh
echo "--- $W"; [ -f "$W" ] && sed -n 1,40p "$W" || echo "not present"
EOS

section "How GPU-021 launched GLES (read-only)"
D=$HOME/Projects/atom-ps4-linux-compat-001/ps4-takeover-reconcile-20261013-002
T=$D/liverpool-021-gate/021-BUNDLE.tar
[ -f "$T" ] && tar -tf "$T" | head -60 | tee -a "$R"
[ -f "$T" ] && tar -xOf "$T" --wildcards '*.sh' '*.py' 2>/dev/null | grep -n -E 'run-with-mesa|LD_LIBRARY_PATH|MESA_|EGL_PLATFORM|libEGL|surfaceless|gbm|eglGetPlatformDisplay|boot_id' | head -30 | tee -a "$R"
grep -rn -E 'run-with-mesa|LD_LIBRARY_PATH|EGL_PLATFORM' "$D/liverpool-021-gate" "$D/gpu-hardening-20261014" --include='*.sh' --include='*.py' --include='*.md' 2>/dev/null | head -20 | tee -a "$R"

section "Historical GPU receipts on the IdeaPad (names only, read-only)"
ls -d "$HOME"/Projects/atom-ps4-linux-compat-001/*/ 2>/dev/null | sed "s#$HOME/##" | tail -40 | tee -a "$R"
find "$HOME/Projects/atom-ps4-linux-compat-001" "$HOME/ps4-llm-test" -maxdepth 4 -type f \
    \( -iname '*GPU06*' -o -iname '*layer*' -o -iname 'CHECKPOINT*' -o -iname 'RESULTS.json' \) 2>/dev/null \
    | sed "s#$HOME/##" | sort | tail -40 | tee -a "$R"

section "Offline build and tests on LLVMPipe (about 10 min)"
SKIP=0; { [ -f /usr/include/EGL/egl.h ] && [ -f /usr/include/GLES3/gl32.h ]; } || SKIP=1
LIBGL_ALWAYS_SOFTWARE=1 SKIP_GLES=$SKIP WORK="$M/work" "$HERE/reproduce.sh" > "$M/reproduce.log" 2>&1; rc=$?
grep -E 'RESULT|negative control|ALL OFFLINE|SKIP_GLES|synth-q4km' "$M/reproduce.log" | sed "s#$M/##" | tee -a "$R"
[ $rc = 0 ] || tail -15 "$M/reproduce.log" | tee -a "$R"
say "reproduce exit: $rc (SKIP_GLES=$SKIP)"

if [ $rc = 0 ]; then
    section "Bundle"
    B="$M/bundle"; mkdir -p "$B/models" "$B/dumps"
    tar -C "$(dirname "$HERE")" --exclude=__pycache__ --exclude=.work -cf - "$(basename "$HERE")" | tar -C "$B" -xf -
    for s in 1 2; do LLAMA_CPP="$M/work/llama.cpp" python3 "$HERE/tools/export_layer0_npz.py" "$M/work/models/synth-q4km-s$s.gguf" "$B/models/synth-q4km-s$s.npz" | sed "s#$M/##" | tee -a "$R"; done
    cp -r "$M/work/dumps/s1_fa1_t1" "$M/work/dumps/s2_fa1" "$B/dumps/"
    [ $SKIP = 0 ] && cp "$M/work/bin/gles_run" "$B/gles_run.ideapad"
    (cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    say "bundle: $(du -sh "$B" | cut -f1), $(wc -l < "$B/SHA256SUMS") files, SHA256SUMS $(sha256sum "$B/SHA256SUMS" | cut -c1-16)"
fi
section "done. Paste the contents of $R back to Claude."
