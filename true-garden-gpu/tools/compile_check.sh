#!/usr/bin/env bash
# Hardware gate step A: compile and link every shader on the current driver and
# print its compute limits. Allocates no buffers and dispatches nothing.
# usage: compile_check.sh GLES_RUN_BINARY
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
rc=0
for f in "$HERE"/shaders/*.comp; do
    printf '%-26s ' "$(basename "$f")"
    "$1" --src "$HERE/shaders/ggml_exact_lib.glsl" --src "$f" --compile-only || rc=1
done
echo "RESULT: $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
