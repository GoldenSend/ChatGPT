#!/usr/bin/env bash
# GPU-022: hardware qualification of the ggml-exact shaders. Run it ON the PS4,
# from the staged true-garden-gpu folder, only after the owner approved GPU-022
# (GATE-GPU-022.md). Typing the expected boot ID is the boot pin: the gate
# refuses to start, and stops, if the running boot is not that one.
#
# usage: gpu022_gate.sh EXPECTED_BOOT_ID STEPS        STEPS = A | AB | ABC
#   A  compile and link all shaders, no buffers, no dispatch
#   B  helper exactness on the GPU: 7 dispatches (TESTS/test_gles_exact_lib.py, n=65536)
#   C  layer 0 chained on the GPU vs llama.cpp: 2-step dump (52 dispatches), then
#      5-step dump (130 dispatches), synthetic Qwen3-0.6B-shaped weights
# env: GLES_RUN_BIN (built gles_run), BUNDLE (dir with models/, dumps/, llama-gguf/),
#      MESA_WRAPPER (optional launcher), GLES_TIMEOUT (s per dispatch, default 120),
#      TEMP_MAX (C, default 75), ALLOW_SOFTWARE=1 (offline rehearsal on LLVMPipe only)
# Stops on the first problem. Nothing is retried. Exit 0 = every requested step PASS.
set -u
EXPECTED=${1:?expected boot id}; STEPS=${2:?steps}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUNDLE=${BUNDLE:?set BUNDLE}; : "${GLES_RUN_BIN:?set GLES_RUN_BIN}"
export GLES_RUN_BIN GLES_TIMEOUT=${GLES_TIMEOUT:-120} LLAMA_CPP="$BUNDLE/llama-gguf"
TEMP_MAX=${TEMP_MAX:-75}
G="$HERE/tools/gles_run_guarded.sh"
OUT="$BUNDLE/gate-out-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT/work"; export WORKDIR="$OUT/work"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/gate.log"; }

snapshot() {  # $1 label; prints state, never stops
    log "--- $1: boot $(cat /proc/sys/kernel/random/boot_id) kernel $(uname -r)"
    log "    D-state: $(ps -eo pid,stat,comm | awk '$2 ~ /^D/ {printf "%s ", $0}')"
    log "    max temp C: $(maxtemp)  load: $(cut -d' ' -f1-3 /proc/loadavg)  mem avail MB: $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
}
maxtemp() { cat /sys/class/hwmon/hwmon*/temp*_input 2>/dev/null | sort -n | tail -1 | awk '{print int($1/1000)}'; }
aftercare() {
    snapshot "aftercare"
    dmesg > "$OUT/dmesg-after.txt" 2>/dev/null
    (cd "$OUT" && sha256sum ./*.txt ./*.log > SHA256SUMS 2>/dev/null)
    log "outputs: $OUT"
}
hold() { log "HOLD: $*"; aftercare; exit 2; }
health() {  # stop conditions, checked before and after every step
    [ "$(cat /proc/sys/kernel/random/boot_id)" = "$EXPECTED" ] || hold "boot id is not the pinned one"
    local d; d=$(ps -eo pid,stat,comm | awk '$2 ~ /^D/'); [ -z "$d" ] || hold "process in D state: $d"
    local t; t=$(maxtemp); [ -z "$t" ] || [ "$t" -le "$TEMP_MAX" ] || hold "temperature ${t} C above ${TEMP_MAX} C"
    local now; now=$(dmesg 2>/dev/null | wc -l)
    if [ "$now" -gt "$DMESG0" ]; then
        local new; new=$(dmesg | tail -n +"$((DMESG0 + 1))" | grep -iE 'gpu|radeon|amdgpu|drm|ring|timeout|fault|hang|error|i/o|reset|usb' || true)
        [ -z "$new" ] || hold "new kernel messages: $new"
    fi
}

dmesg > "$OUT/dmesg-before.txt" 2>/dev/null || { echo "cannot read dmesg; run as a user that can"; exit 3; }
DMESG0=$(wc -l < "$OUT/dmesg-before.txt")
snapshot "preflight"
health
log "steps $STEPS, timeout ${GLES_TIMEOUT}s per dispatch, wrapper '${MESA_WRAPPER:-none}'"

case "$STEPS" in *A*)
    log "=== A: compile only"
    AMD_DEBUG=cs "$HERE/tools/compile_check.sh" "$G" > "$OUT/step-a.txt" 2> "$OUT/step-a-stderr.log"
    rc=$?; cat "$OUT/step-a.txt" | tee -a "$OUT/gate.log"
    grep -m1 GL_RENDERER "$OUT/step-a-stderr.log" | tee -a "$OUT/gate.log"
    if grep -qiE 'llvmpipe|softpipe|swrast' "$OUT/step-a-stderr.log" && [ "${ALLOW_SOFTWARE:-0}" != 1 ]; then
        hold "software renderer: this would not test Liverpool"
    fi
    [ $rc = 0 ] || hold "step A failed"
    health ;;
esac
case "$STEPS" in *B*)
    log "=== B: helper exactness"
    python3 "$HERE/TESTS/test_gles_exact_lib.py" "$G" "$WORKDIR" 65536 > "$OUT/step-b.txt" 2>&1
    rc=$?; tail -12 "$OUT/step-b.txt" | tee -a "$OUT/gate.log"
    health
    [ $rc = 0 ] || hold "step B did not pass (the driver differs from IEEE somewhere); step C not run" ;;
esac
case "$STEPS" in *C*)
    for d in "s1:s1_fa1_t1" "s2:s2_fa1"; do
        m=${d%%:*}; dump=${d#*:}
        log "=== C: layer 0, model $m, dump $dump"
        python3 "$HERE/TESTS/test_gles_layer_vs_oracle.py" "$G" "$BUNDLE/models/synth-q4km-$m.gguf" "$BUNDLE/dumps/$dump" > "$OUT/step-c-$dump.txt" 2>&1
        rc=$?; tail -8 "$OUT/step-c-$dump.txt" | tee -a "$OUT/gate.log"
        health
        [ $rc = 0 ] || hold "step C ($dump) did not pass"
    done ;;
esac
log "GATE RESULT: PASS for steps $STEPS"
aftercare
exit 0
