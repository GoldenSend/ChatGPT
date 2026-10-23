#!/usr/bin/env bash
# Stop-path tests for tools/gpu022_gate.sh, run against a finished bundle on
# LLVMPipe with stand-ins for ps, dmesg, the Mesa wrapper and the GL runner.
# Every case must end with the exit code and log line listed for it.
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
n=\$(( \$(cat "$W/ps.count" 2>/dev/null || echo 0) + 1 )); echo \$n > "$W/ps.count"
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
    (cd "$W/x/gpu022" && env PATH="$SHIM:$PATH" LIBGL_ALWAYS_SOFTWARE=1 "$@" tools/gpu022_gate.sh "$bid" "$steps") > "$W/log" 2>&1
    local rc=$?
    if [ "$rc" = "$want" ] && grep -qE "$re" "$W/log"; then
        PASS=$((PASS + 1)); echo "ok   $name: exit $rc"
    else
        BAD=$((BAD + 1)); echo "FAIL $name: exit $rc (want $want, /$re/)"; grep -E "HOLD|RESULT|FAIL" "$W/log" | head -5
    fi
}
JBD="  281     2 D    -                                jbd2/sda2-8"
KW="  8517     2 D    -                                kworker/4:1+events_freezable_power_"
USR="  9001  1200 D    -                                python3"
SW="ALLOW_SOFTWARE=1"
case_ "baseline A"                         0 "GATE RESULT: PASS" A $SW
case_ "jbd2 in D 2 of 3 samples"           0 "GATE RESULT: PASS" A $SW FAKE_PS=sometimes "FAKE_PS_LINE=$JBD"
case_ "jbd2 stuck in D"                    2 "HOLD: process stuck in D state.*jbd2.*nothing touched the GPU" A $SW FAKE_PS=always "FAKE_PS_LINE=$JBD"
case_ "user process stuck in D"            2 "HOLD: process stuck in D state.*python3" A $SW FAKE_PS=always "FAKE_PS_LINE=$USR"
case_ "unexplained kthread stuck"          2 "HOLD: process stuck in D state.*events_freezable" A $SW FAKE_PS=always "FAKE_PS_LINE=$KW"
case_ "kthread stuck, D_ALLOW matches"     0 "GATE RESULT: PASS" A $SW FAKE_PS=always "FAKE_PS_LINE=$KW" D_ALLOW=events_freezable
case_ "storage kthread, D_ALLOW tries"     2 "HOLD: process stuck in D state.*jbd2" A $SW FAKE_PS=always "FAKE_PS_LINE=$JBD" D_ALLOW=jbd2
case_ "GPU fault after ring-buffer wrap"   2 "HOLD: new kernel messages:.*GPU fault" A $SW FAKE_DMESG="$W/dm_gpu"
case_ "mmc0 timeout, not ignored"          2 "HOLD: new kernel messages:.*mmc0" A $SW FAKE_DMESG="$W/dm_mmc"
case_ "mmc0 timeout, DMESG_IGNORE"         0 "GATE RESULT: PASS" A $SW FAKE_DMESG="$W/dm_mmc" 'DMESG_IGNORE=\] mmc0: '
case_ "GPU fault, DMESG_IGNORE for mmc0"   2 "HOLD: new kernel messages:.*GPU fault" A $SW FAKE_DMESG="$W/dm_gpu" 'DMESG_IGNORE=\] mmc0: '
case_ "GL libraries do not load"           2 "HOLD: GL libraries do not load.*nothing touched the GPU" A $SW MESA_WRAPPER="$SHIM/bad_wrapper"
case_ "software renderer"                  2 "HOLD: software renderer" A
case_ "hardware renderer, not Liverpool"   2 "HOLD: renderer is not Liverpool" A GLES_RUN_BIN="$SHIM/fake_gles_run" "FAKE_RENDERER=AMD TAHITI (DRM 3.35.0)"
case_ "Liverpool renderer"                 0 "GATE RESULT: PASS" A GLES_RUN_BIN="$SHIM/fake_gles_run" "FAKE_RENDERER=AMD LIVERPOOL (DRM 3.35.0, 5.4.247, LLVM 15.0.7)"
CASE_BID=00000000-0000-0000-0000-000000000000 case_ "wrong boot id" 2 "HOLD: boot id is not the pinned one" A $SW
echo "RESULT: $([ $BAD = 0 ] && echo PASS || echo FAIL) ($PASS ok, $BAD wrong)"
[ $BAD = 0 ]
