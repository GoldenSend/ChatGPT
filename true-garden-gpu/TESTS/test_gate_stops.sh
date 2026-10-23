#!/usr/bin/env bash
# Stop-path tests for tools/gpu022_gate.sh, run against a finished bundle on
# LLVMPipe with stand-ins for ps, dmesg, the Mesa wrapper and the GL runner.
# Every case must end with the exit code and log line listed for it, leave
# nothing in /dev/shm and leave its log in the bundle's gate-out-* folder.
# usage: TESTS/test_gate_stops.sh gate/gpu022-bundle.tar.gz
set -u
TGZ=$(realpath "$1")
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
BID=$(cat /proc/sys/kernel/random/boot_id)
SHIM="$W/shim"; mkdir -p "$SHIM"
REAL_PS=$(command -v ps); REAL_DMESG=$(command -v dmesg)

cat > "$SHIM/ps" <<EOF
#!/usr/bin/env bash
# FAKE_PS: off | always | sometimes (D line in 2 of every 3 calls)
# FAKE_IO: progress (the fake diskstats count grows with every call) | stall
n=\$(( \$(cat "$W/ps.count" 2>/dev/null || echo 0) + 1 )); echo \$n > "$W/ps.count"
[ "\${FAKE_IO:-}" = progress ] && echo "   8       0 sda \$((1000 + n)) 0 0 0 \$((500 + n)) 0 0 0 0 0 0" > "$W/diskstats"
"$REAL_PS" "\$@"
case "\${FAKE_PS:-off}" in
    always) echo "\$FAKE_PS_LINE" ;;
    sometimes) [ \$((n % 3)) = 0 ] || echo "\$FAKE_PS_LINE" ;;
esac
EOF
cat > "$SHIM/dmesg" <<EOF
#!/usr/bin/env bash
# FAKE_DMESG=DIR: first call prints DIR/before.txt, later calls DIR/after.txt
[ -n "\${FAKE_DMESG:-}" ] || exec "$REAL_DMESG" "\$@"
if [ -e "$W/dmesg.called" ]; then cat "\$FAKE_DMESG/after.txt"; else touch "$W/dmesg.called"; cat "\$FAKE_DMESG/before.txt"; fi
EOF
cat > "$SHIM/fake_gles_run" <<'EOF'
#!/usr/bin/env bash
echo "GL_RENDERER: $FAKE_RENDERER | GL_VERSION: OpenGL ES 3.2 Mesa 24.0.9" >&2
echo "compiled OK; local size 64x1x1; GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 1024"
EOF
cat > "$SHIM/bad_wrapper" <<'EOF'
#!/bin/sh
[ "$1" = python3 ] && { echo "OSError: libEGL.so.1: cannot open shared object file" >&2; exit 1; }
exec "$@"
EOF
chmod +x "$SHIM"/*
echo "   8       0 sda 1000 0 0 0 500 0 0 0 0 0 0" > "$W/diskstats"

# kernel logs: 100 lines before; after = a wrapped buffer (same line count) plus one new line
mk_dmesg() {  # dir new-line
    mkdir -p "$1"
    for i in $(seq 1 100); do printf '[%5d.000000] filler %d\n' "$i" "$i"; done > "$1/before.txt"
    { for i in $(seq 2 100); do printf '[%5d.000000] filler %d\n' "$i" "$i"; done; echo "$2"; } > "$1/after.txt"
}
mk_dmesg "$W/dm_gpu" "[  150.000000] amdgpu 0000:00:01.0: GPU fault detected: 146 0x0c80440c"
mk_dmesg "$W/dm_mmc" "[  150.000000] mmc0: Timeout waiting for hardware interrupt."

PASS=0; BAD=0
case_() {  # name expected-exit expected-log-regex steps env...
    local name=$1 want=$2 re=$3 steps=$4 bid=${CASE_BID:-$BID}; shift 4
    rm -rf "$W/x" "$W/ps.count" "$W/dmesg.called"; mkdir -p "$W/x"; tar -C "$W/x" -xzf "$TGZ"
    echo "   8       0 sda 1000 0 0 0 500 0 0 0 0 0 0" > "$W/diskstats"
    local decoy=""
    if [ -n "${CASE_DECOY:-}" ]; then (exec -a "$W/x/gpu022/tools/gles_run.py" sleep 60) & decoy=$!; fi
    (cd "$W/x/gpu022" && env PATH="$SHIM:$PATH" LIBGL_ALWAYS_SOFTWARE=1 "$@" tools/gpu022_gate.sh "$bid" "$steps") > "$W/log" 2>&1
    local rc=$?
    [ -z "$decoy" ] || kill "$decoy" 2>/dev/null
    ls -d /dev/shm/gpu022-* > /dev/null 2>&1 && { echo "     (left a run dir in /dev/shm)"; rc=99; rm -rf /dev/shm/gpu022-*; }
    ls -d "$W"/x/gpu022/gate-out-*/gate.log > /dev/null 2>&1 || { echo "     (no gate-out-*/gate.log in the bundle)"; rc=98; }
    if [ "$rc" = "$want" ] && grep -qE "$re" "$W/log"; then
        PASS=$((PASS + 1)); echo "ok   $name: exit $rc"
    else
        BAD=$((BAD + 1)); echo "FAIL $name: exit $rc (want $want, /$re/)"; grep -E "HOLD|RESULT|FAIL" "$W/log" | head -5
    fi
}
JBD="  281     2 D    -                                jbd2/sda2-8"
FL="  7169     2 D    -                                kworker/u16:2+flush-8:0"
KW="  8517     2 D    -                                kworker/4:1+events_freezable_power_"
USR="  9001  1200 D    -                                python3"
GPUK="  412     2 D    -                                amdgpu_dm_hpd_rx_offload_wq"
SW="ALLOW_SOFTWARE=1"
DS="GATE_DISKSTATS=$W/diskstats"
case_ "baseline A"                               0 "GATE RESULT: PASS" A $SW
case_ "baseline ABC (run dir in RAM, copied)"    0 "GATE RESULT: PASS for steps ABC .189 dispatches" ABC $SW
case_ "jbd2 in D 2 of 3 samples"                 0 "GATE RESULT: PASS" A $SW $DS FAKE_IO=stall FAKE_PS=sometimes "FAKE_PS_LINE=$JBD"
case_ "flush stuck, disk completing I/O"         0 "note: in D for 4 s while the disks completed.*flush-8:0" A $SW $DS FAKE_IO=progress FAKE_PS=always "FAKE_PS_LINE=$FL"
case_ "flush stuck, no I/O completing"           2 "HOLD: process stuck in D state.*no block I/O completed meanwhile.*flush-8:0.*nothing touched the GPU" A $SW $DS FAKE_IO=stall FAKE_PS=always "FAKE_PS_LINE=$FL"
case_ "user process stuck, no I/O"               2 "HOLD: process stuck in D state.*python3" A $SW $DS FAKE_IO=stall FAKE_PS=always "FAKE_PS_LINE=$USR"
case_ "user process stuck, I/O completing"       0 "GATE RESULT: PASS" A $SW $DS FAKE_IO=progress FAKE_PS=always "FAKE_PS_LINE=$USR"
case_ "GPU kthread stuck, I/O completing"        2 "HOLD: process stuck in D state.*GPU-related.*amdgpu" A $SW $DS FAKE_IO=progress FAKE_PS=always "FAKE_PS_LINE=$GPUK"
case_ "unexplained kthread stuck, no I/O"        2 "HOLD: process stuck in D state.*events_freezable" A $SW $DS FAKE_IO=stall FAKE_PS=always "FAKE_PS_LINE=$KW"
case_ "kthread stuck, no I/O, D_ALLOW matches"   0 "GATE RESULT: PASS" A $SW $DS FAKE_IO=stall FAKE_PS=always "FAKE_PS_LINE=$KW" D_ALLOW=events_freezable
case_ "storage kthread, no I/O, D_ALLOW tries"   2 "HOLD: process stuck in D state.*jbd2" A $SW $DS FAKE_IO=stall FAKE_PS=always "FAKE_PS_LINE=$JBD" D_ALLOW=jbd2
case_ "two reasons reported together"            2 "HOLD: process stuck.*; new kernel messages:.*GPU fault" A $SW $DS FAKE_IO=stall FAKE_PS=always "FAKE_PS_LINE=$JBD" FAKE_DMESG="$W/dm_gpu"
case_ "GPU fault after ring-buffer wrap"         2 "HOLD: new kernel messages:.*GPU fault" A $SW FAKE_DMESG="$W/dm_gpu"
case_ "mmc0 timeout, not ignored"                2 "HOLD: new kernel messages:.*mmc0" A $SW FAKE_DMESG="$W/dm_mmc"
case_ "mmc0 timeout, DMESG_IGNORE"               0 "GATE RESULT: PASS" A $SW FAKE_DMESG="$W/dm_mmc" 'DMESG_IGNORE=\] mmc0: '
case_ "GPU fault, DMESG_IGNORE for mmc0"         2 "HOLD: new kernel messages:.*GPU fault" A $SW FAKE_DMESG="$W/dm_gpu" 'DMESG_IGNORE=\] mmc0: '
CASE_DECOY=1 case_ "GPU runner still alive"      2 "HOLD: a GPU runner process is still alive" A $SW
case_ "GL libraries do not load"                 2 "HOLD: GL libraries do not load.*nothing touched the GPU" A $SW MESA_WRAPPER="$SHIM/bad_wrapper"
case_ "software renderer"                        2 "HOLD: software renderer" A
case_ "hardware renderer, not Liverpool"         2 "HOLD: renderer is not Liverpool" A GLES_RUN_BIN="$SHIM/fake_gles_run" "FAKE_RENDERER=AMD TAHITI (DRM 3.35.0)"
case_ "Liverpool renderer (as seen 2026-10-23)"  0 "GATE RESULT: PASS" A GLES_RUN_BIN="$SHIM/fake_gles_run" "FAKE_RENDERER=AMD Radeon Graphics (radeonsi, liverpool, LLVM 19.1.7, DRM 3.35, 5.4.247-DFAUS-blkscrn_Fix_mt7668_hdmia)"
CASE_BID=00000000-0000-0000-0000-000000000000 case_ "wrong boot id" 2 "HOLD: boot id is not the pinned one" A $SW
echo "RESULT: $([ $BAD = 0 ] && echo PASS || echo FAIL) ($PASS ok, $BAD wrong)"
[ $BAD = 0 ]
