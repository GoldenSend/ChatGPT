#!/usr/bin/env bash
# GPU-022: hardware qualification of the ggml-exact shaders. Run it ON the PS4
# from the unpacked bundle (tools/build_gate_bundle.sh), only after the owner
# approved GPU-022 (GATE-GPU-022.md). Typing the expected boot ID is the boot
# pin: the gate refuses to start, and stops, if the running boot is not that one.
# Needs only bash, coreutils, python3 (stdlib) and the system libEGL/libGLESv2.
#
# usage: tools/gpu022_gate.sh EXPECTED_BOOT_ID STEPS        STEPS = A | AB | ABC
#   A  compile and link all shaders, no buffers, no dispatch
#   B  replay plan/plan_B.sh: 7 dispatches of the GLSL helpers on fixed inputs
#   C  replay plan/plan_C.sh: layer 0 chained on the GPU for 7 decode steps
#      (26 dispatches per step), synthetic Qwen3-0.6B-shaped weights
# Every output is compared with expected/ (a verified LLVMPipe replay that is
# bit-exact against llama.cpp) after part B and after every decode step.
# env: GLES_DRM_DEVICE (e.g. /dev/dri/renderD128: open the GPU through GBM, as
#      the GPU-021 runner did; unset = EGL surfaceless), MESA_WRAPPER (optional
#      launcher), GLES_TIMEOUT (s per dispatch, default 120), TEMP_MAX (C,
#      default 75), D_ALLOW (regex of kernel-thread names/wait channels allowed
#      to stay in D state because the preflight showed them unrelated to GPU and
#      storage; default none), DMESG_IGNORE (regex of kernel-log lines known from
#      the preflight to be unrelated, e.g. '\] mmc0: '; default none),
#      ALLOW_SOFTWARE=1 (rehearsal on LLVMPipe only), GATE_RECORD=1 (bundle
#      build only: replay without comparing)
# D state: a process counts only if it is in D in all of 5 samples 1 s apart
# (short D waits for journal commits and writeback on the USB root are normal).
# Stops on the first problem. Nothing is retried. A HOLD before step A means
# nothing touched the GPU.
# Exit 0 = every requested step PASS; 2 = HOLD (safety); 4 = FAIL (numerics).
set -u
EXPECTED=${1:?expected boot id}; STEPS=${2:?steps}
B="$(cd "$(dirname "$0")/.." && pwd)"
export GLES_RUN_BIN=${GLES_RUN_BIN:-$B/tools/gles_run.py} GLES_TIMEOUT=${GLES_TIMEOUT:-120}
TEMP_MAX=${TEMP_MAX:-75}
G="$B/tools/gles_run_guarded.sh"
RUN="$B/gate-out-$(date -u +%Y%m%dT%H%M%SZ)"; I="$B/in"; O="$RUN/out"; mkdir -p "$O"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN/gate.log"; }

maxtemp() { cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | sort -n | tail -1 | awk '{print int($1/1000)}'; }
snapshot() {  # $1 label; logs state, never stops
    log "--- $1: boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r)"
    log "    D-state: $(ps -eo pid,ppid,stat,wchan:32,comm | awk '$3 ~ /^D/ {printf "[%s] ", $0}')"
    log "    max temp C: $(maxtemp)  load: $(cut -d' ' -f1-3 /proc/loadavg)  mem avail MB: $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
}
aftercare() {
    snapshot "aftercare"
    dmesg > "$RUN/dmesg-after.txt" 2>/dev/null
    (cd "$RUN" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
    log "outputs: $RUN"
}
STARTED=0
hold() { [ "$STARTED" = 1 ] || set -- "$*" "(before step A: nothing touched the GPU)"; log "HOLD: $*"; aftercare; exit 2; }
fail() { log "RESULT: FAIL $*"; aftercare; exit 4; }
dstuck() {  # processes in D state in every one of 5 samples, 1 s apart
    local f="$RUN/dstate-samples.txt" i
    : > "$f"
    for i in 1 2 3 4 5; do
        ps -eo pid,ppid,stat,wchan:32,comm | awk 'NR > 1 && $3 ~ /^D/' >> "$f"
        [ "$i" = 5 ] || sleep 1
    done
    awk '{ n[$1]++; l[$1] = $0 } END { for (p in n) if (n[p] == 5) print l[p] }' "$f"
}
newdmesg() {  # kernel-log lines newer than the last one before the gate started
    dmesg 2>/dev/null | awk -v t0="$DMESG_T0" '
        match($0, /^\[ *[0-9]+\.[0-9]+\]/) { ts = substr($0, 2, RLENGTH - 2) + 0 }
        ts > t0 + 0'
}
health() {  # safety stop conditions
    [ "$(cat /proc/sys/kernel/random/boot_id)" = "$EXPECTED" ] || hold "boot id is not the pinned one"
    # A process stuck in D state stops the gate if it is a user process, looks
    # GPU- or storage-related, or is a kernel thread that D_ALLOW does not match.
    local d; d=$(dstuck | awk -v allow="${D_ALLOW:-}" '
        {
            gs = ($4 $5) ~ /amdgpu|radeon|drm|ttm|fence|gfx|sdma|ring|gpu|kgd|usb|scsi|blk|jbd2|ext4|flush|ata/
            if ($2 != 2 || gs || allow == "" || ($4 $5) !~ allow) print
        }')
    [ -z "$d" ] || hold "process stuck in D state for 4 s or more: $d"
    local t; t=$(maxtemp); [ -z "$t" ] || [ "$t" -le "$TEMP_MAX" ] || hold "temperature ${t} C above ${TEMP_MAX} C"
    local new; new=$(newdmesg | { if [ -n "${DMESG_IGNORE:-}" ]; then grep -vE "$DMESG_IGNORE"; else cat; fi; } \
        | grep -iE 'gpu|radeon|amdgpu|drm|ring|timeout|fault|hang|error|i/o|reset|usb|blocked for more than|hung_task' || true)
    [ -z "$new" ] || hold "new kernel messages: $new"
}
NDISP=0
run() {  # label shader gles_run-args...
    local label=$1 shader=$2; shift 2
    "$G" --src "$B/shaders/ggml_exact_lib.glsl" --src "$B/shaders/$shader" "$@" 2>> "$RUN/dispatch-stderr.log" \
        || hold "dispatch $label failed (exit $?)"
    NDISP=$((NDISP + 1))
}
concat() { local dst=$1; shift; cat "$@" > "$dst" || hold "concat $dst failed"; }
compare() {  # part [maxstep]
    [ "${GATE_RECORD:-0}" = 1 ] && return 0
    python3 "$B/tools/gate_compare.py" "$O" "$B/expected" "$B/plan/labels.tsv" "$@" > "$RUN/compare-$1.txt" 2>&1
    local rc=$?; tail -4 "$RUN/compare-$1.txt" | tee -a "$RUN/gate.log"; return $rc
}
checkpoint() {  # called by plan_C.sh after every decode step
    health
    local s=${1#C_step}
    log "    after decode step $s: $NDISP dispatches"
    compare C "$s" || fail "at decode step $s (first differing output above)"
}

dmesg > "$RUN/dmesg-before.txt" 2>/dev/null || { echo "cannot read dmesg; run as a user that can"; exit 3; }
# new kernel messages are found by timestamp, not line count (the ring buffer may wrap)
DMESG_T0=$(awk 'match($0, /^\[ *[0-9]+\.[0-9]+\]/) { t = substr($0, 2, RLENGTH - 2) + 0 } END { print t + 0 }' "$RUN/dmesg-before.txt")
snapshot "preflight"
health
log "steps $STEPS, runner $GLES_RUN_BIN, display ${GLES_DRM_DEVICE:-surfaceless}, timeout ${GLES_TIMEOUT}s, wrapper '${MESA_WRAPPER:-none}', D_ALLOW '${D_ALLOW:-}', DMESG_IGNORE '${DMESG_IGNORE:-}'"
if [ "${GATE_RECORD:-0}" != 1 ]; then
    (cd "$B" && sha256sum -c --quiet SHA256SUMS) > "$RUN/bundle-check.txt" 2>&1 || hold "bundle hash mismatch (see bundle-check.txt)"
fi
# the GL libraries must load (dlopen only: no device is opened, no context is made)
${MESA_WRAPPER:+"$MESA_WRAPPER"} python3 -c 'import ctypes, sys
for n in sys.argv[1:]: ctypes.CDLL(n)' libEGL.so.1 libGLESv2.so.2 ${GLES_DRM_DEVICE:+libgbm.so.1} > "$RUN/libcheck.txt" 2>&1 \
    || hold "GL libraries do not load: $(tail -1 "$RUN/libcheck.txt")"
STARTED=1

case "$STEPS" in *A*)
    log "=== A: compile only"
    AMD_DEBUG=cs "$B/tools/compile_check.sh" "$G" > "$RUN/step-a.txt" 2> "$RUN/step-a-stderr.log"
    rc=$?; tee -a "$RUN/gate.log" < "$RUN/step-a.txt"
    grep -m1 GL_RENDERER "$RUN/step-a-stderr.log" | tee -a "$RUN/gate.log"
    if [ "${ALLOW_SOFTWARE:-0}" != 1 ]; then
        grep -qiE 'llvmpipe|softpipe|swrast' "$RUN/step-a-stderr.log" && hold "software renderer: this would not test Liverpool"
        grep -m1 GL_RENDERER "$RUN/step-a-stderr.log" | grep -qi liverpool || hold "renderer is not Liverpool (the GPU-021 runner required it too)"
    fi
    [ $rc = 0 ] || hold "step A failed"
    health ;;
esac
case "$STEPS" in *B*)
    log "=== B: GLSL helpers, $(grep -c '^run ' "$B/plan/plan_B.sh") dispatches"
    . "$B/plan/plan_B.sh"
    health
    compare B || fail "in part B (the driver differs from IEEE somewhere); part C not run" ;;
esac
case "$STEPS" in *C*)
    log "=== C: layer 0 chained on the GPU, $(grep -c '^run ' "$B/plan/plan_C.sh") dispatches"
    . "$B/plan/plan_C.sh" ;;
esac
log "GATE RESULT: PASS for steps $STEPS ($NDISP dispatches)"
aftercare
exit 0
